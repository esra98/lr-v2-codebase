"""The full-length source videos: ``videos/<session>.mp4`` and ``videos/sources.tsv``.

A session is named after the recording: ``2024-2025-birlesim-72`` for an
assembly session, ``yt-<video id>`` for a YouTube upload. The table records
where each file came from and what ffprobe says about it, so the dataset
description can state the source material exactly.
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time

from avsr.corpus.status import atomic_write

SOURCE_FIELDS = ["session", "origin", "url", "width", "height", "fps", "audio_hz", "hours", "bytes", "sha256"]
VIDEO_EXT = ".mp4"


def probe(path):
    """Video size, frame rate, audio rate and duration of a media file."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,sample_rate:format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    num, _, den = video["r_frame_rate"].partition("/")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": round(int(num) / int(den or 1), 3),
        "audio_hz": int(audio["sample_rate"]) if audio else 0,
        "duration": float(info["format"]["duration"]),
    }


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_of(session):
    """('youtube', url) for ``yt-<id>`` sessions, ('tbmm', '') otherwise."""
    if session.startswith("yt-"):
        return "youtube", f"https://www.youtube.com/watch?v={session[3:]}"
    return "tbmm", ""


def read_sources(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_sources(path, rows):
    lines = ["\t".join(SOURCE_FIELDS)]
    for r in sorted(rows, key=lambda r: r["session"]):
        lines.append("\t".join(str(r.get(k, "")) for k in SOURCE_FIELDS))
    atomic_write(path, "\n".join(lines) + "\n")


def read_checksums(videos_dir):
    """``sha256sum`` output next to the videos, if present: name -> digest.

    A digest is only returned for files older than the checksum file, since a
    file replaced afterwards (a re-download) no longer matches it.
    """
    path = os.path.join(videos_dir, "SHA256SUMS")
    sums = {}
    if os.path.isfile(path):
        written = os.path.getmtime(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name = parts[1].lstrip("*")
                file = os.path.join(videos_dir, name)
                if os.path.isfile(file) and os.path.getmtime(file) <= written:
                    sums[name] = parts[0]
    return sums


def register(videos_dir, table=None, hash_files=False, log=print):
    """Bring ``sources.tsv`` up to date with the files in ``videos_dir``.

    Files that are unchanged (same size) keep their row, including the
    checksum; new or changed files are probed again. ``hash_files`` computes
    missing checksums, which reads every file once.
    """
    table = table or os.path.join(videos_dir, "sources.tsv")
    previous = {r["session"]: r for r in read_sources(table)}
    checksums = read_checksums(videos_dir)
    rows = []
    for name in sorted(os.listdir(videos_dir)):
        if not name.endswith(VIDEO_EXT) or name.startswith("."):
            continue
        session = name[: -len(VIDEO_EXT)]
        path = os.path.join(videos_dir, name)
        size = os.path.getsize(path)
        old = previous.get(session)
        if old and int(old.get("bytes") or 0) == size:
            row = dict(old)
        else:
            facts = probe(path)
            origin, url = origin_of(session)
            row = {
                "session": session, "origin": origin, "url": url,
                "width": facts["width"], "height": facts["height"], "fps": facts["fps"],
                "audio_hz": facts["audio_hz"], "hours": round(facts["duration"] / 3600, 2),
                "bytes": size, "sha256": "",
            }
            log(f"probed {session}: {facts['width']}x{facts['height']} {facts['fps']} fps, {row['hours']} h")
        if not row.get("sha256"):
            row["sha256"] = checksums.get(name, "")
        if not row["sha256"] and hash_files:
            log(f"hashing {session} ...")
            row["sha256"] = sha256_file(path)
        rows.append(row)
    write_sources(table, rows)
    return rows


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

# H.264 video up to 1080p with AAC audio, merged into one mp4. AV1 and VP9
# streams are refused by YouTube for this client, and a JavaScript runtime
# is needed for the URL challenge, or every download fails with HTTP 403.
YTDLP_FORMAT = "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][ext=mp4]+ba/b[height<=1080]"


def ytdlp_command(video_id, out_template, ytdlp="yt-dlp", js_runtime="node"):
    return [
        ytdlp, "--js-runtimes", js_runtime, "--continue", "--no-overwrites", "--no-playlist",
        "--retries", "10", "--fragment-retries", "10",
        "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
        "-o", out_template, f"https://www.youtube.com/watch?v={video_id}",
    ]


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def download_youtube(video_ids, videos_dir, ytdlp="yt-dlp", min_free_gb=15, min_height=720, log=print):
    """Download sessions into ``videos/_incoming`` and move each finished file into place.

    A session whose file already exists with at least ``min_height`` rows is
    skipped. Waits while the disk has less than ``min_free_gb`` free, since a
    1080p session is up to 6 GB and is merged from two parts.
    """
    incoming = os.path.join(videos_dir, "_incoming")
    os.makedirs(incoming, exist_ok=True)
    done = []
    for video_id in video_ids:
        final = os.path.join(videos_dir, f"yt-{video_id}{VIDEO_EXT}")
        if os.path.isfile(final) and probe(final)["height"] >= min_height:
            log(f"skip {video_id}: already {probe(final)['height']}p")
            done.append(video_id)
            continue
        while free_gb(videos_dir) < min_free_gb:
            log(f"waiting: {free_gb(videos_dir):.0f} GB free, need {min_free_gb}")
            time.sleep(300)
        out = os.path.join(incoming, f"yt-{video_id}{VIDEO_EXT}")
        if not os.path.isfile(out):
            result = subprocess.run(ytdlp_command(video_id, os.path.join(incoming, "yt-%(id)s.%(ext)s"), ytdlp))
            if result.returncode != 0 or not os.path.isfile(out):
                log(f"FAILED {video_id}")
                continue
        height = probe(out)["height"]
        if height < min_height:
            log(f"UNEXPECTED {video_id}: {height}p, left in {incoming}")
            continue
        os.replace(out, final)
        log(f"DONE {video_id} {height}p")
        done.append(video_id)
    return done
