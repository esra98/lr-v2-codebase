#!/usr/bin/env python
"""Build the corpus from full-length recordings, stage by stage.

    python scripts/corpus.py ingest                       # videos/*.mp4 -> videos/sources.tsv
    python scripts/corpus.py download --ids ID [ID ...]   # YouTube sessions at 1080p
    python scripts/corpus.py transcribe [--sessions S ...] [--backend faster_whisper --model large-v3]
    python scripts/corpus.py segment [--sessions S ...]   # transcript.json -> segments.tsv
    python scripts/corpus.py cut --dataset dataset/tr-lrs-v3 [--sessions S ...] [--workers 4]
    python scripts/corpus.py build --dataset dataset/tr-lrs-v3 --holdout S1 S2   # clip table, manifests
    python scripts/corpus.py verify --dataset dataset/tr-lrs-v3 [--sample 300]
    python scripts/corpus.py run --dataset dataset/tr-lrs-v3 [--sessions S ...]   # transcribe, segment, cut, pipelined
    python scripts/corpus.py status                       # what is done per session

Every stage is resumable: finished sessions and chunks are skipped, and
interrupting a run loses at most the item in progress. Sessions are named
after the files in videos/; a session with no video but a transcript (for
example copied from an earlier run) can still be segmented.
"""

import argparse
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avsr.corpus import build as builder  # noqa: E402
from avsr.corpus import cut as cutter  # noqa: E402
from avsr.corpus import segment as seg  # noqa: E402
from avsr.corpus import sources, transcribe  # noqa: E402
from avsr.corpus.status import SessionStatus, list_sessions, options_hash, session_dir  # noqa: E402

STAGES = ("transcribe", "segment", "cut")


def video_sessions(videos_dir):
    return [r["session"] for r in sources.read_sources(os.path.join(videos_dir, "sources.tsv"))]


def chosen_sessions(args, available):
    if args.sessions:
        missing = sorted(set(args.sessions) - set(available))
        if missing:
            sys.exit(f"unknown sessions: {missing}")
        return list(args.sessions)
    return list(available)


def cmd_ingest(args):
    rows = sources.register(args.videos, hash_files=args.hash)
    hours = sum(float(r["hours"]) for r in rows)
    print(f"{len(rows)} sessions, {hours:.1f} h -> {os.path.join(args.videos, 'sources.tsv')}")


def cmd_download(args):
    ids = args.ids
    if not ids:
        ids = [r["session"][3:] for r in sources.read_sources(os.path.join(args.videos, "sources.tsv"))
               if r["origin"] == "youtube" and int(r["height"]) < 720]
    done = sources.download_youtube(ids, args.videos, ytdlp=args.ytdlp, min_free_gb=args.min_free_gb)
    sources.register(args.videos)
    print(f"{len(done)} of {len(ids)} sessions downloaded")


def cmd_transcribe(args):
    cfg = transcribe.TranscribeConfig(backend=args.backend, model=args.model, chunk_minutes=args.chunk_minutes,
                                      device=args.device, compute_type=args.compute_type)
    for session in chosen_sessions(args, video_sessions(args.videos)):
        video = os.path.join(args.videos, session + sources.VIDEO_EXT)
        transcribe.transcribe_session(session, video, session_dir(args.work, session), cfg, out_name=args.out_name)


def cmd_segment(args):
    cfg = seg.SegmentConfig()
    totals = {}
    for session in chosen_sessions(args, list_sessions(args.work)):
        work = session_dir(args.work, session)
        transcript = os.path.join(work, args.transcript)
        if not os.path.isfile(transcript):
            print(f"skip {session}: no {args.transcript}")
            continue
        status = SessionStatus(work)
        options = {"cfg": cfg, "transcript": args.transcript, "transcript_bytes": os.path.getsize(transcript)}
        if status.is_done("segment", options) and os.path.isfile(os.path.join(work, "segments.tsv")) and not args.force:
            print(f"skip {session}: segmented")
            continue
        summary = seg.run(session, work, cfg, transcript_name=args.transcript)
        status.mark("segment", options, **summary)
        for k, v in summary["counts"].items():
            totals[k] = totals.get(k, 0) + v
    if totals:
        total = sum(totals.values())
        print(f"\nutterances: {total}")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:7d} ({v / total:.1%})")


def _line_buffered():
    """Progress lines must reach a log file as they happen, also from workers."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


# Which worker process this is, set by _init_worker. The first
# ``gpu_workers`` processes run MediaPipe on the GPU delegate, the others on
# the CPU: the integrated GPU saturates at about four workers, and the CPU
# cores left over carry a few more.
_WORKER = {"index": 0, "gpu_workers": None}


def _init_worker(counter, gpu_workers):
    with counter.get_lock():
        _WORKER["index"] = counter.value
        counter.value += 1
    _WORKER["gpu_workers"] = gpu_workers


def _worker_pool(workers, gpu_workers):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    counter = multiprocessing.Value("i", 0)
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(counter, gpu_workers))


def _cut_one(job):
    session, video, work, dataset, options, retry_errors = job
    import cv2
    from dataclasses import replace

    _line_buffered()
    cv2.setNumThreads(2)
    if _WORKER["gpu_workers"] is not None:
        options = replace(options, delegate="gpu" if _WORKER["index"] < _WORKER["gpu_workers"] else "cpu")
    print(f"{session}: worker {_WORKER['index']}, detector on {options.delegate}")
    try:
        counts = cutter.cut_session(session, video, work, dataset, options, retry_errors=retry_errors)
        return session, dict(counts), ""
    except Exception as e:  # noqa: BLE001
        return session, {}, f"{type(e).__name__}: {e}"


def cmd_cut(args):
    options = cutter.CutOptions(variant=args.variant, out_size=args.out_size, roi_scale=args.roi_scale,
                                smoothing=args.smoothing, crf=args.crf).validate()
    jobs = []
    for session in chosen_sessions(args, video_sessions(args.videos)):
        work = session_dir(args.work, session)
        if not os.path.isfile(os.path.join(work, "segments.tsv")):
            print(f"skip {session}: not segmented")
            continue
        jobs.append((session, os.path.join(args.videos, session + sources.VIDEO_EXT), work, args.dataset, options, args.retry_errors))
    print(f"{len(jobs)} sessions -> {args.dataset}, {args.workers} workers, variant {options.variant} {options.out_size}px")
    totals = {}
    if args.workers <= 1:
        results = map(_cut_one, jobs)
    else:
        pool = _worker_pool(args.workers, args.gpu_workers)
        results = pool.map(_cut_one, jobs)
    for session, counts, error in results:
        if error:
            print(f"FAILED {session}: {error}")
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    total = sum(totals.values())
    if total:
        print(f"\nutterances cut: {total}")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:7d} ({v / total:.1%})")


def cmd_run(args):
    """Transcribe on the GPU one session at a time; segment and cut each session as soon as it is done."""
    tcfg = transcribe.TranscribeConfig(backend=args.backend, model=args.model, chunk_minutes=args.chunk_minutes,
                                       device=args.device, compute_type=args.compute_type)
    scfg = seg.SegmentConfig()
    options = cutter.CutOptions().validate()
    rows = {r["session"]: r for r in sources.read_sources(os.path.join(args.videos, "sources.tsv"))}
    sessions = chosen_sessions(args, list(rows))
    pool = _worker_pool(args.workers, args.gpu_workers)
    futures = []
    for session in sessions:
        video = os.path.join(args.videos, session + sources.VIDEO_EXT)
        work = session_dir(args.work, session)
        transcribe.transcribe_session(session, video, work, tcfg)
        status = SessionStatus(work)
        sopts = {"cfg": scfg, "transcript": "transcript.json", "transcript_bytes": os.path.getsize(os.path.join(work, "transcript.json"))}
        if not (status.is_done("segment", sopts) and os.path.isfile(os.path.join(work, "segments.tsv"))):
            status.mark("segment", sopts, **seg.run(session, work, scfg))
        if int(rows[session]["height"]) < args.min_height:
            print(f"{session}: {rows[session]['height']}p, not cut until a larger file is downloaded")
            continue
        futures.append(pool.submit(_cut_one, (session, video, work, args.dataset, options, False)))
    totals = {}
    for future in futures:
        session, counts, error = future.result()
        if error:
            print(f"FAILED {session}: {error}")
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    pool.shutdown()
    total = sum(totals.values())
    if total:
        print(f"\nutterances cut: {total}")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:7d} ({v / total:.1%})")


def cmd_build(args):
    sessions = args.sessions or None
    builder.build(args.work, args.dataset, args.holdout, args.val_fraction, args.manifests, sessions)


def cmd_verify(args):
    failed = builder.verify(args.dataset, args.manifests, sample=args.sample)
    builder.describe(os.path.abspath(args.dataset) + ".clips.tsv")
    if failed:
        sys.exit(f"{len(failed)} check(s) failed: {', '.join(failed)}")
    print("all checks passed")


def cmd_status(args):
    sessions = sorted(set(list_sessions(args.work)) | set(video_sessions(args.videos)))
    print(f"{'session':28s} {'video':6s} " + " ".join(f"{s:12s}" for s in STAGES))
    for session in sessions:
        has_video = os.path.isfile(os.path.join(args.videos, session + sources.VIDEO_EXT))
        status = SessionStatus(session_dir(args.work, session))
        cells = []
        for stage in STAGES:
            entry = status.get(stage)
            cells.append((entry.get("model") or entry["time"][:10]) if entry else "-")
        print(f"{session:28s} {'yes' if has_video else 'no':6s} " + " ".join(f"{c:12s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--work", default="work")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="probe the videos and write sources.tsv")
    p.add_argument("--hash", action="store_true", help="also compute missing sha256 checksums (slow)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("download", help="download YouTube sessions at up to 1080p")
    p.add_argument("--ids", nargs="*", help="YouTube video ids (default: every yt- session below 720p)")
    p.add_argument("--ytdlp", default=".venv/bin/yt-dlp")
    p.add_argument("--min-free-gb", type=float, default=15)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("transcribe", help="speech recognition of the recordings")
    p.add_argument("--sessions", nargs="*")
    defaults = transcribe.TranscribeConfig()
    p.add_argument("--backend", default=defaults.backend, choices=["faster_whisper", "whisper_timestamped"])
    p.add_argument("--model", default=defaults.model)
    p.add_argument("--chunk-minutes", type=float, default=defaults.chunk_minutes)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--compute-type", default=defaults.compute_type)
    p.add_argument("--out-name", default="transcript.json", help="file name under work/<session>/")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("segment", help="transcript.json -> segments.tsv with text verdicts")
    p.add_argument("--sessions", nargs="*")
    p.add_argument("--transcript", default="transcript.json", help="file name under work/<session>/")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("cut", help="cut the kept utterances into clips")
    p.add_argument("--sessions", nargs="*")
    p.add_argument("--dataset", required=True, help="output folder, e.g. dataset/tr-lrs-v3")
    p.add_argument("--workers", type=int, default=4, help="sessions cut in parallel")
    p.add_argument("--gpu-workers", type=int, default=4, help="how many of the workers detect on the GPU delegate")
    cut_defaults = cutter.CutOptions()
    p.add_argument("--variant", default=cut_defaults.variant, choices=["mouth", "face"])
    p.add_argument("--out-size", type=int, default=cut_defaults.out_size)
    p.add_argument("--roi-scale", type=float, default=cut_defaults.roi_scale)
    p.add_argument("--smoothing", default=cut_defaults.smoothing, choices=["none", "moving_average", "kalman", "one_euro"])
    p.add_argument("--crf", type=int, default=cut_defaults.crf)
    p.add_argument("--retry-errors", action="store_true", help="cut the utterances with verdict 'error' again")
    p.set_defaults(func=cmd_cut)

    p = sub.add_parser("run", help="transcribe, segment and cut, session by session")
    p.add_argument("--sessions", nargs="*")
    p.add_argument("--dataset", required=True)
    p.add_argument("--workers", type=int, default=4, help="sessions cut in parallel while the next is transcribed")
    p.add_argument("--gpu-workers", type=int, default=4, help="how many of the workers detect on the GPU delegate")
    p.add_argument("--min-height", type=int, default=720, help="sessions with smaller video are transcribed but not cut")
    p.add_argument("--backend", default=defaults.backend, choices=["faster_whisper", "whisper_timestamped"])
    p.add_argument("--model", default=defaults.model)
    p.add_argument("--chunk-minutes", type=float, default=defaults.chunk_minutes)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--compute-type", default=defaults.compute_type)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("build", help="clip table, split and manifests from the cut sessions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--holdout", nargs="+", required=True, help="whole sessions that form the test set")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--manifests", default=None, help="default: <dataset>-manifests")
    p.add_argument("--sessions", nargs="*", help="default: every cut session")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("verify", help="check a built dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifests", default=None)
    p.add_argument("--sample", type=int, default=300, help="clips per split whose content is decoded")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("status", help="stages done per session")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    _line_buffered()
    args.func(args)


if __name__ == "__main__":
    main()
