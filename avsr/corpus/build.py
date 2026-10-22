"""From the per-session results to one dataset: clip table, split, manifests, checks.

The clip table has the columns of the first corpus (``avsr.data.manifest.CLIP_FIELDS``),
so the manifests, the loader and the training code are unchanged. Every
utterance of every ``segments.tsv`` gets one row: rejected by the text rules,
rejected by the cutter, or kept with its split.

Partition-Time Masking is not written into the clips; it is applied during
training (``augment.video.ptm``), which draws a new mask every epoch.
"""

import json
import os
import random
from collections import Counter

from avsr.corpus import cut as cutter
from avsr.corpus import segment as seg
from avsr.corpus.status import SessionStatus, atomic_json, list_sessions, session_dir
from avsr.data.manifest import CLIP_FIELDS, is_hallucinated_phrase, kept_rows, read_clip_table, write_clip_table, write_manifest
from avsr.data.recurrence import RecurrenceConfig, has_internal_loop
from avsr.text.labels import parse_label_file
from avsr.text.normalize import FOREIGN_LETTERS, TURKISH_ALPHABET, normalize_text


def assign_splits(clip_ids, is_test, val_fraction):
    """Split per clip: a whole test session, or the last clips as validation.

    Validation clips are a block at the end of the session rather than a
    random sample, because neighbouring clips come from the same speech.
    """
    if is_test:
        return {c: "test" for c in clip_ids}
    num_val = int(round(len(clip_ids) * val_fraction))
    first_val = len(clip_ids) - num_val
    return {c: ("val" if i >= first_val else "train") for i, c in enumerate(clip_ids)}


def session_rows(session, work_dir, holdout, val_fraction):
    """Clip-table rows of one session, from its segments.tsv and clips.tsv."""
    segments = seg.read_segments(os.path.join(work_dir, "segments.tsv"))
    results = cutter.read_clip_results(os.path.join(work_dir, "clips.tsv"))
    rows = []
    for s in segments:
        r = results.get(s["utt_id"])
        verdict = s["verdict"] if s["verdict"] != "kept" else (r["verdict"] if r else "not_cut")
        rows.append(
            {
                "session": session,
                "clip_id": s["utt_id"],
                "verdict": verdict,
                "split": "",
                "masked": "",
                "num_frames": r["num_frames"] if r else "",
                "mean_conf": s["mean_conf"],
                "detection_rate": r["detection_rate"] if r else "",
                "yaw": r["yaw"] if r else "",
                "mouth_motion": r["mouth_motion"] if r else "",
                "text": s["text"],
                "error": r["error"] if r else "",
            }
        )
    kept = [row["clip_id"] for row in rows if row["verdict"] == "kept"]
    splits = assign_splits(kept, session in holdout, val_fraction)
    for row in rows:
        if row["verdict"] == "kept":
            row["split"], row["masked"] = splits[row["clip_id"]], "no"
    return rows


def build(work_root, dataset_dir, holdout, val_fraction=0.05, manifests_dir=None, sessions=None, log=print):
    """Write ``<dataset>.clips.tsv``, ``<dataset>.settings.json`` and the manifests."""
    dataset_dir = os.path.abspath(dataset_dir)
    manifests_dir = manifests_dir or os.path.join(dataset_dir + "-manifests")
    sessions = sessions or [s for s in list_sessions(work_root) if os.path.isfile(os.path.join(session_dir(work_root, s), "clips.tsv"))]
    missing = sorted(set(holdout) - set(sessions))
    if missing:
        raise ValueError(f"held-out sessions without clips: {missing}")
    rows, settings = [], {}
    for session in sessions:
        work = session_dir(work_root, session)
        status = SessionStatus(work)
        entry = status.get(cutter.STAGE) or {}
        if entry.get("dataset") not in (None, dataset_dir):
            log(f"warning: {session} was cut into {entry.get('dataset')}, not {dataset_dir}")
        rows.extend(session_rows(session, work, set(holdout), val_fraction))
        settings.setdefault("cut", status.get(cutter.STAGE))
        settings.setdefault("segment", status.get("segment"))
        settings.setdefault("transcribe", status.get("transcribe"))
    settings.update(holdout_sessions=sorted(holdout), val_fraction=val_fraction, sessions=sessions, ptm=False)
    table_path = dataset_dir + ".clips.tsv"
    write_clip_table(table_path, rows)
    atomic_json(dataset_dir + ".settings.json", settings)
    counts = Counter(r["verdict"] for r in rows)
    log(f"{len(rows)} utterances, kept {counts['kept']} ({counts['kept'] / max(len(rows), 1):.1%}) -> {table_path}")
    for split in ("train", "val", "test"):
        split_rows = kept_rows(rows, split)
        write_manifest(os.path.join(manifests_dir, f"{split}.tsv"), split_rows)
        hours = sum(r.num_frames for r in split_rows) / 25 / 3600
        log(f"{split:5s}: {len(split_rows):6d} utterances, {hours:6.2f} h, {len({r.session for r in split_rows})} sessions")
    return rows


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(dataset_dir, manifests_dir=None, sample=300, seed=0, log=print):
    """Check a built dataset. Returns the list of failed check names."""
    import av
    import numpy as np

    from avsr.preprocess import video_io

    dataset_dir = os.path.abspath(dataset_dir)
    manifests_dir = manifests_dir or dataset_dir + "-manifests"
    failed = []

    def report(name, ok, detail=""):
        log(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}")
        if not ok:
            failed.append(name)

    table = read_clip_table(dataset_dir + ".clips.tsv")
    kept = [r for r in table if r["verdict"] == "kept"]
    report("clip table has kept clips", bool(kept), f"{len(kept)} of {len(table)}")

    # files: every kept clip present, nothing else in the session folders
    expected = {(r["session"], r["clip_id"]) for r in kept}
    present, extra = set(), []
    for session in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, session)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            stem, ext = os.path.splitext(name)
            if ext not in (".mp4", ".txt") or (session, stem) not in expected:
                extra.append(f"{session}/{name}")
            elif ext == ".mp4" and os.path.isfile(os.path.join(folder, stem + ".txt")):
                present.add((session, stem))
    report("every kept clip has mp4 and txt", present == expected, f"{len(expected - present)} missing")
    report("no file that is not a kept clip", not extra, extra[0] if extra else "")

    # splits and manifests
    by_split = Counter(r["split"] for r in kept)
    report("every kept clip has a split", all(r["split"] in ("train", "val", "test") for r in kept), str(dict(by_split)))
    test_sessions = {r["session"] for r in kept if r["split"] == "test"}
    other = {r["session"] for r in kept if r["split"] != "test"}
    report("test sessions are whole sessions", not (test_sessions & other), str(sorted(test_sessions)))
    for split in ("train", "val", "test"):
        path = os.path.join(manifests_dir, f"{split}.tsv")
        ok = os.path.isfile(path)
        if ok:
            from avsr.data.manifest import read_manifest

            ids = {m.utt_id for m in read_manifest(path)}
            want = {f"{r['session']}/{r['clip_id']}" for r in kept if r["split"] == split}
            ok = ids == want
        report(f"{split} manifest agrees with the clip table", ok)
    report("no clip is masked", all(r["masked"] == "no" for r in kept))

    # text
    alphabet = set(TURKISH_ALPHABET + FOREIGN_LETTERS + " ")
    bad_alpha = [r for r in kept if set(r["text"]) - alphabet]
    report("kept texts use the alphabet only", not bad_alpha, bad_alpha[0]["text"][:40] if bad_alpha else "")
    letters = Counter(c for r in kept for c in r["text"] if c in "Iİ")
    report("both dotted and dotless I occur", letters["I"] > 0 and letters["İ"] > 0, f"I {letters['I']}, İ {letters['İ']}")
    rec = RecurrenceConfig()
    bad_text = [r for r in kept if is_hallucinated_phrase(r["text"]) or has_internal_loop(r["text"].split(), rec)]
    report("no kept text is an invented phrase or a loop", not bad_text, bad_text[0]["text"][:40] if bad_text else "")

    # decoded content of a sample
    rng = random.Random(seed)
    chosen = []
    for split in ("train", "val", "test"):
        rows = [r for r in kept if r["split"] == split]
        chosen += rng.sample(rows, min(sample, len(rows)))
    problems = Counter()
    for r in chosen:
        base = os.path.join(dataset_dir, r["session"], r["clip_id"])
        try:
            with av.open(base + ".mp4") as c:
                v = c.streams.video[0]
                a = c.streams.audio[0] if c.streams.audio else None
                if v.codec_context.name != "h264" or v.width != v.height or float(v.average_rate) != 25.0:
                    problems["video stream"] += 1
                if a is None or a.codec_context.name != "aac" or a.sample_rate != video_io.SAMPLE_RATE:
                    problems["audio stream"] += 1
            frames = video_io.read_video(base + ".mp4")
            if len(frames) != int(r["num_frames"]):
                problems["frame count"] += 1
            audio = video_io.read_audio(base + ".mp4")
            # AAC pads the stream to whole 1024-sample blocks; the loader trims it.
            extra = len(audio) - len(frames) * video_io.SAMPLES_PER_FRAME
            if not 0 <= extra <= 4096:
                problems["audio length"] += 1
            utt = parse_label_file(base + ".txt")
            if normalize_text(utt.text) != r["text"]:
                problems["label text"] += 1
            duration = len(frames) / 25.0
            if any(w.start < 0 or w.end > duration + 0.1 for w in utt.words):
                problems["word times"] += 1
        except Exception as e:  # noqa: BLE001
            problems[f"unreadable ({type(e).__name__})"] += 1
    detail = ", ".join(f"{k} {v}" for k, v in problems.items()) or f"{len(chosen)} clips"
    report("sampled clips decode with the expected streams, lengths and labels", not problems, detail)
    return failed


def describe(table_path, log=print):
    """A short summary of a clip table: counts per verdict and per split."""
    table = read_clip_table(table_path)
    counts = Counter(r["verdict"] for r in table)
    log(f"{len(table)} utterances")
    for k, v in counts.most_common():
        log(f"  {k:22s} {v:7d} ({v / len(table):.1%})")
    kept = [r for r in table if r["verdict"] == "kept"]
    for split in ("train", "val", "test"):
        rows = [r for r in kept if r["split"] == split]
        hours = sum(int(r["num_frames"]) for r in rows) / 25 / 3600
        log(f"{split:5s}: {len(rows):6d} clips, {hours:.2f} h, {len({r['session'] for r in rows})} sessions")
