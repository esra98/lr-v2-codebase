#!/usr/bin/env python
"""Check a dataset made by scripts/build_dataset.py against its source.

Every check prints PASS or FAIL. Checks that decode video are run on a random
sample of clips; the others cover every clip.

Usage:
    python scripts/verify_dataset.py --dataset data/tr-lrs-clean-ptm
"""

import argparse
import filecmp
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av  # noqa: E402
import numpy as np  # noqa: E402

from avsr.data.manifest import FilterConfig, is_hallucinated_phrase, read_clip_table, read_manifest  # noqa: E402
from avsr.data.recurrence import has_internal_loop  # noqa: E402
from avsr.preprocess import landmarks as lm  # noqa: E402
from avsr.preprocess import video_io  # noqa: E402
from avsr.preprocess.crop import crop_mouth  # noqa: E402
from avsr.preprocess.pipeline import SIDE_FILES, BuildOptions  # noqa: E402
from avsr.preprocess.ptm_bake import build_mask, clip_rng, mask_frames  # noqa: E402

FAILED = []


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
    if not ok:
        FAILED.append(name)


def stream_facts(path):
    with av.open(path) as container:
        facts = []
        for stream in container.streams:
            ctx = stream.codec_context
            if stream.type == "video":
                facts.append(("video", ctx.name, ctx.width, ctx.height, ctx.pix_fmt, float(stream.average_rate)))
            else:
                facts.append(("audio", ctx.name, ctx.sample_rate, ctx.layout.name))
        return facts


def expected_frames(raw, cache, row, options):
    """The frames the clip should contain, rebuilt from the raw video."""
    frames = video_io.read_video(os.path.join(raw, row["session"], row["clip_id"] + ".mp4"))
    with np.load(os.path.join(cache, "landmarks", row["session"], row["clip_id"] + ".npz")) as c:
        landmarks = lm.interpolate_missing(c["landmarks"], c["detected"])
    crop = crop_mouth(frames, landmarks, options.out_size, options.roi_scale, options.smoothing, align=options.align)
    clean = crop
    if row["masked"] == "yes":
        mask = build_mask(options.ptm_grid, options.ptm_alpha, options.ptm_partitions, options.ptm_fill)
        crop = mask_frames(crop, mask, clip_rng(options.seed, row["session"], row["clip_id"]))
    return crop, clean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--manifests", default="", help="Also check these manifests")
    parser.add_argument("--sample", type=int, default=60, help="Clips per split for the checks that decode video")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    table = read_clip_table(dataset + ".clips.tsv")
    with open(dataset + ".settings.json") as f:
        settings = json.load(f)
    options = BuildOptions(**{k: v for k, v in settings.items() if k in BuildOptions.__dataclass_fields__})
    kept = [r for r in table if r["verdict"] == "kept"]
    rejected = [r for r in table if r["verdict"] != "kept"]
    print(f"{dataset}: {len(table)} source clips examined, {len(kept)} kept, {len(rejected)} left out\n")

    # ---- layout -----------------------------------------------------------
    on_disk = Counter()
    strays = []
    for entry in sorted(os.listdir(dataset)):
        path = os.path.join(dataset, entry)
        if not os.path.isdir(path):
            strays.append(entry)
            continue
        for name in os.listdir(path):
            stem, ext = os.path.splitext(name)
            if ext in (".mp4",) + SIDE_FILES:
                on_disk[(entry, stem)] += 1
            else:
                strays.append(os.path.join(entry, name))
    report("dataset folder holds session folders with mp4, aac and txt only", not strays, f"{len(strays)} other files")
    expected = {(r["session"], r["clip_id"]) for r in kept}
    report("every kept clip is present", expected <= set(on_disk), f"{len(expected - set(on_disk))} missing")
    report("no left-out clip is present", set(on_disk) <= expected, f"{len(set(on_disk) - expected)} extra")
    incomplete = [k for k, n in on_disk.items() if n != 3]
    report("every clip has its three files", not incomplete, f"{len(incomplete)} incomplete")
    sessions_src = set(os.listdir(args.raw))
    report("session names are those of the source", {s for s, _ in on_disk} <= sessions_src)

    # ---- files copied unchanged -------------------------------------------
    changed = Counter()
    for r in kept:
        for ext in SIDE_FILES:
            src = os.path.join(args.raw, r["session"], r["clip_id"] + ext)
            dst = os.path.join(dataset, r["session"], r["clip_id"] + ext)
            if not (os.path.isfile(dst) and filecmp.cmp(src, dst, shallow=False)):
                changed[ext] += 1
    report("transcripts are byte-identical to the source", not changed[".txt"], f"{changed['.txt']} differ")
    report("aac files are byte-identical to the source", not changed[".aac"], f"{changed['.aac']} differ")

    # ---- labels -----------------------------------------------------------
    cfg = FilterConfig()
    bad = [r for r in kept if is_hallucinated_phrase(r["text"]) or has_internal_loop(r["text"].split(), cfg.recurrence)]
    report("no kept transcript is an invented phrase or a loop", not bad, f"{len(bad)} found")
    verdicts = {(r["session"], r["clip_id"]): r["verdict"] for r in table}
    flagged = [k for k in on_disk if verdicts.get(k, "").startswith("recurrent")]
    report("no clip with a recurrent transcript is present", not flagged, f"{len(flagged)} found")
    counts = Counter(r["text"] for r in kept)
    print("      most frequent kept sentences:", ", ".join(f"{t} ({n})" for t, n in counts.most_common(5)))

    # ---- split and masking flags ------------------------------------------
    holdout = set(settings["holdout_sessions"])
    wrong_split = [r for r in kept if (r["split"] == "test") != (r["session"] in holdout)]
    report("test clips are exactly the clips of the held-out sessions", not wrong_split)
    wrong_mask = [r for r in kept if (r["masked"] == "yes") != (options.ptm and r["split"] == "train")]
    report("only training clips are marked as masked", not wrong_mask)
    by_split = Counter(r["split"] for r in kept)
    print("      clips per split:", dict(by_split))

    # ---- video and audio content, on a sample ------------------------------
    rng = random.Random(args.seed)
    sample = []
    for split in ("train", "val", "test"):
        rows = [r for r in kept if r["split"] == split]
        sample += rng.sample(rows, min(args.sample, len(rows)))
    want_video = ("video", "h264", options.out_size, options.out_size, "yuv420p", 25.0)
    layout_bad = frames_bad = audio_bad = unreadable = 0
    diff_inside, diff_outside, leak, dark = [], [], [], []
    for r in sample:
        dst = os.path.join(dataset, r["session"], r["clip_id"] + ".mp4")
        src = os.path.join(args.raw, r["session"], r["clip_id"] + ".mp4")
        try:
            facts, source_facts = stream_facts(dst), stream_facts(src)
            video = video_io.read_video(dst).astype(np.float32)
        except (OSError, ValueError, av.error.FFmpegError):
            unreadable += 1
            continue
        if len(facts) != 2 or facts[0] != want_video or facts[1] != source_facts[1]:
            layout_bad += 1
        if len(video) != int(r["num_frames"]) or len(video) != len(video_io.read_video(src)):
            frames_bad += 1
            continue
        if not np.array_equal(video_io.read_audio(dst), video_io.read_audio(src)):
            audio_bad += 1
        want, clean = expected_frames(args.raw, args.cache, r, options)
        masked_pixels = np.any(want != clean, axis=-1) if r["masked"] == "yes" else np.zeros(want.shape[:3], bool)
        error = np.abs(video - want).mean(-1)
        if masked_pixels.any():
            diff_inside.append(error[masked_pixels].mean())
            dark.append(video[masked_pixels].mean())
            share = masked_pixels.reshape(len(video), -1).any(1).mean()
            leak.append(abs(share - round(options.ptm_alpha * len(video)) / len(video)))
        diff_outside.append(error[~masked_pixels].mean())
    n = len(sample)
    report("every sampled clip can be decoded", unreadable == 0, f"{unreadable} of {n}")
    report("mp4: H.264 96x96 25 fps video as stream 0, source audio as stream 1", layout_bad == 0, f"{layout_bad} of {n}")
    report("frame count equals the source and the clip table", frames_bad == 0, f"{frames_bad} of {n}")
    report("audio inside the mp4 is sample-identical to the source", audio_bad == 0, f"{audio_bad} of {n}")
    report(
        "video matches the mouth crop rebuilt from the raw clip",
        bool(diff_outside) and max(diff_outside) < 6.0,
        f"mean difference {np.mean(diff_outside):.2f}, worst clip {max(diff_outside):.2f} of 255" if diff_outside else "",
    )
    if options.ptm:
        report("training clips in the sample carry a mask", len(dark) == sum(r["masked"] == "yes" for r in sample))
        report(
            "masked cells are black and where the seed puts them",
            bool(dark) and max(dark) < 12.0 and max(diff_inside) < 12.0,
            f"mean level inside the cells {np.mean(dark):.1f} of 255" if dark else "",
        )
        report(
            f"the mask covers {options.ptm_alpha:.0%} of the frames of each clip",
            bool(leak) and max(leak) < 1e-9,
        )

    # ---- manifests --------------------------------------------------------
    if args.manifests:
        table_index = {f"{r['session']}/{r['clip_id']}": r for r in kept}
        problems, sessions = 0, {}
        for split in ("train", "val", "test"):
            rows = read_manifest(os.path.join(args.manifests, f"{split}.tsv"))
            sessions[split] = {r.session for r in rows}
            for row in rows:
                entry = table_index.get(row.utt_id)
                if entry is None or entry["split"] != split or int(entry["num_frames"]) != row.num_frames:
                    problems += 1
        report("manifests agree with the clip table", problems == 0, f"{problems} rows differ")
        report("test sessions do not occur in training or validation", not sessions["test"] & (sessions["train"] | sessions["val"]))

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else str(len(FAILED)) + ' CHECK(S) FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
