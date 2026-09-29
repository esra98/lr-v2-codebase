#!/usr/bin/env python
"""Build the training dataset from the raw corpus.

The result has the layout of the source:

    <session>/<clip>.mp4   video (stream 0) and audio (stream 1)
    <session>/<clip>.aac   audio
    <session>/<clip>.txt   transcript with word times and confidences

and differs from it in three ways:

- clips with unreliable labels are left out (recurrent or invented
  transcripts, low confidence, text that does not fit the clip) and so are
  clips without a detectable face; the others keep their original numbers;
- the video is the 96x96 mouth crop instead of the 224x224 face;
- Partition-Time Masking is written into the video of the training clips.

Validation and test clips are not masked, so that error rates are measured
on real video. The split is fixed here and recorded in the clip table: the
sessions given with --holdout-sessions are the test set, and the last clips
of every other session are the validation set.

Audio and transcript files are copied unchanged, and the audio inside the mp4
is the original stream. The dataset folder holds nothing else. Two files are
written next to it: <out>.clips.tsv with the verdict for every source clip,
and <out>.settings.json.

The run can be interrupted and started again; finished clips are skipped.

Usage:
    python scripts/build_dataset.py --out data/tr-lrs-clean-ptm --holdout-sessions 2025-2026-birlesim-3 video-10
    python scripts/build_dataset.py --out data/sample --sessions video-10 video-4 --holdout-sessions video-10 --limit 300
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avsr.data.manifest import (  # noqa: E402
    FilterConfig,
    judge_duration,
    judge_session_labels,
    load_session_labels,
    read_clip_table,
    write_clip_table,
)
from avsr.preprocess.pipeline import SIDE_FILES, BuildOptions, ClipResult, build_clip  # noqa: E402

_detector = None


def _init_worker():
    global _detector
    # One thread per worker; the pool provides the parallelism.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from avsr.preprocess.landmarks import FaceLandmarker

    _detector = FaceLandmarker()


def _work(job):
    raw, out, cache, session, clip_id, options, speech_end, text, filter_cfg, overwrite = job
    utt = SimpleNamespace(speech_end=speech_end)

    def check(num_frames):
        return judge_duration(num_frames, utt, text, filter_cfg)

    return build_clip(raw, out, cache, session, clip_id, options, _detector, check=check, overwrite=overwrite)


def run_pool(jobs, workers, work=_work, initializer=_init_worker):
    """Run jobs in a process pool.

    Returns (results, unfinished). ``unfinished`` is non-empty only when a
    worker process died, which makes the whole pool unusable.
    """
    results, unfinished = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except BrokenProcessPool:
                unfinished.append(job)
            except Exception as e:
                results.append(ClipResult(job[3], job[4], verdict="error", error=f"{type(e).__name__}: {e}"))
    return results, unfinished


def run_jobs(jobs, workers, work=_work, initializer=_init_worker, batch_size=64):
    """Run jobs and survive crashing workers.

    A native crash inside a worker takes down the pool and every clip that was
    queued in it. Those clips are retried in small batches, and a batch that
    crashes again is run one clip at a time, so that only the clip that
    actually causes the crash is lost.
    """
    results, unfinished = run_pool(jobs, workers, work, initializer)
    for i in range(0, len(unfinished), batch_size):
        batch_results, suspects = run_pool(unfinished[i : i + batch_size], workers, work, initializer)
        results.extend(batch_results)
        for job in suspects:
            single, crashed = run_pool([job], 1, work, initializer)
            results.extend(single)
            for bad in crashed:
                results.append(ClipResult(bad[3], bad[4], verdict="error", error="worker process crashed"))
    return results


def remove_clip(out, session, clip_id):
    """Delete the files of a clip that does not belong in the dataset."""
    for ext in (".mp4",) + SIDE_FILES:
        path = os.path.join(out, session, clip_id + ext)
        if os.path.isfile(path):
            os.remove(path)


def assign_splits(session, clip_ids, holdout_sessions, val_fraction):
    """Split of every clip: the whole session is test, or its last clips are validation.

    Validation clips are a block at the end of the session rather than a
    random sample, because neighbouring clips come from the same speech.
    """
    if session in holdout_sessions:
        return {clip_id: "test" for clip_id in clip_ids}
    num_val = int(round(len(clip_ids) * val_fraction))
    first_val = len(clip_ids) - num_val
    return {clip_id: "val" if i >= first_val else "train" for i, clip_id in enumerate(clip_ids)}


def build_session(session, args, options, filter_cfg, workers):
    """Build one session and return its rows of the clip table."""
    clips = load_session_labels(args.raw, session)
    label_verdicts = judge_session_labels(session, clips, filter_cfg)
    if args.limit:
        # Recurrence was judged on the whole session; only the building is limited.
        clips = clips[: args.limit]
    candidates = [clip_id for clip_id, _, _ in clips if not label_verdicts[clip_id]]
    splits = assign_splits(session, candidates, set(args.holdout_sessions), args.val_fraction)
    unmasked = replace(options, ptm=False)

    rows, jobs = {}, []
    for clip_id, utt, text in clips:
        row = {"session": session, "clip_id": clip_id, "text": text, "error": "", "split": "", "masked": ""}
        row["mean_conf"] = f"{utt.mean_conf:.4f}" if utt else ""
        row.update(num_frames="", detection_rate="", yaw="", mouth_motion="")
        reason = label_verdicts[clip_id]
        if reason:
            row["verdict"] = reason
        else:
            masked = options.ptm and splits[clip_id] == "train"
            row.update(split=splits[clip_id], masked="yes" if masked else "no")
            clip_options = options if masked else unmasked
            jobs.append(
                (args.raw, args.out, args.cache, session, clip_id, clip_options, utt.speech_end, text, filter_cfg, args.overwrite)
            )
        rows[clip_id] = row

    for result in run_jobs(jobs, workers) if jobs else []:
        rows[result.clip_id].update(
            verdict=result.verdict,
            num_frames=result.num_frames,
            detection_rate=f"{result.detection_rate:.3f}",
            yaw=f"{result.yaw:.4f}",
            mouth_motion=f"{result.mouth_motion:.5f}",
            error=result.error,
        )
    for clip_id, row in rows.items():
        if row["verdict"] != "kept":
            row.update(split="", masked="")
            remove_clip(args.out, session, clip_id)
    return [rows[clip_id] for clip_id, _, _ in clips]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw", help="Extracted source corpus (only read)")
    parser.add_argument("--out", required=True, help="Folder of the new dataset")
    parser.add_argument("--cache", default="data/cache", help="Where face landmarks are cached")
    parser.add_argument("--sessions", nargs="*", help="Sessions to build (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="Build only the first N clips of each session")
    parser.add_argument("--holdout-sessions", nargs="+", required=True, help="Sessions that form the test set")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Share of each other session used for validation")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true", help="Rebuild clips that already exist")
    defaults = BuildOptions()
    parser.add_argument("--no-ptm", action="store_true", help="Do not write Partition-Time Masking into the clips")
    parser.add_argument("--out-size", type=int, default=defaults.out_size)
    parser.add_argument("--roi-scale", type=float, default=defaults.roi_scale)
    parser.add_argument("--smoothing", default=defaults.smoothing, choices=["none", "moving_average", "kalman", "one_euro"])
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    args.raw, args.out = os.path.abspath(args.raw), os.path.abspath(args.out)
    if args.out == args.raw or args.out.startswith(args.raw + os.sep) or args.raw.startswith(args.out + os.sep):
        sys.exit("--out must not be inside the raw data folder, or contain it")
    options = BuildOptions(
        out_size=args.out_size, roi_scale=args.roi_scale, smoothing=args.smoothing, ptm=not args.no_ptm, seed=args.seed
    )
    options.validate()
    filter_cfg = FilterConfig()

    all_sessions = sorted(
        d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)) and not d.startswith(".")
    )
    sessions = args.sessions or all_sessions
    missing = (set(sessions) | set(args.holdout_sessions)) - set(all_sessions)
    if missing:
        sys.exit(f"Sessions not found in {args.raw}: {sorted(missing)}")
    if not 0.0 < args.val_fraction < 0.5:
        sys.exit("--val-fraction must be between 0 and 0.5")
    if not set(sessions) - set(args.holdout_sessions):
        sys.exit("Every session is held out for testing; nothing is left for training")

    settings_path, table_path = args.out + ".settings.json", args.out + ".clips.tsv"
    settings = asdict(options)
    settings.update(holdout_sessions=sorted(args.holdout_sessions), val_fraction=args.val_fraction, limit=args.limit)
    if os.path.isfile(settings_path) and not args.overwrite:
        with open(settings_path) as f:
            previous = json.load(f)
        changed = [k for k in settings if previous.get(k) != settings[k]]
        if changed:
            sys.exit(
                f"{args.out} was built with different settings ({', '.join(changed)}). "
                "Choose another --out, or use --overwrite to rebuild it."
            )
    os.makedirs(args.out, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    table = {}
    if os.path.isfile(table_path):
        table = {(r["session"], r["clip_id"]): r for r in read_clip_table(table_path)}
    print(
        f"{len(sessions)} session(s) -> {args.out}, {args.workers} workers, "
        f"PTM {'in training clips' if options.ptm else 'off'}, test sessions {sorted(args.holdout_sessions)}",
        flush=True,
    )

    totals, start = Counter(), time.time()
    for number, session in enumerate(sessions, 1):
        session_start = time.time()
        rows = build_session(session, args, options, filter_cfg, args.workers)
        for row in rows:
            table[(row["session"], row["clip_id"])] = row
            totals[row["verdict"]] += 1
        # Written after every session, so an interrupted run loses at most one.
        write_clip_table(table_path, [table[k] for k in sorted(table)])
        kept = sum(r["verdict"] == "kept" for r in rows)
        elapsed = (time.time() - start) / 60
        left = elapsed / number * (len(sessions) - number)
        print(
            f"[{number}/{len(sessions)}] {session}: kept {kept} of {len(rows)} clips "
            f"in {(time.time() - session_start) / 60:.1f} min | about {left:.0f} min left",
            flush=True,
        )

    total = sum(totals.values())
    print(f"\nclips examined: {total}")
    print(f"kept          : {totals['kept']} ({totals['kept'] / max(total, 1):.1%})")
    for reason, n in totals.most_common():
        if reason != "kept":
            print(f"  left out, {reason:22s} {n:7d} ({n / total:.1%})")
    print(f"clip table: {table_path}")
    if totals["error"]:
        sys.exit(f"{totals['error']} clips failed with an error; see the clip table")


if __name__ == "__main__":
    main()
