"""Cut the kept utterances of a session out of its recording.

One pass over the recording: frames are decoded in order, and for every
utterance of ``segments.tsv`` with verdict ``kept`` the frames nearest to its
25 fps grid are taken, face landmarks are detected on them, and the mouth
region (or the face) is cropped and written with the utterance's audio as
``<dataset>/<session>/<utt>.mp4`` and ``.txt``. Landmarks are cached under
``work/<session>/landmarks/`` so a later crop variant needs no detection.

Memory: a 1080p utterance of 10 s would hold 1.8 GB of frames, so every frame
is shrunk, once the face has been seen, to the scale at which the face is
``keep_face_px`` high. The crop is at most that large, so nothing is lost.

Speed: landmarks are detected on every ``detect_every``-th output frame and
interpolated in between (the crop track is smoothed anyway), and only a
square of ``region_scale`` face heights around the tallest face is stored,
so a worker holds a few hundred megabytes rather than gigabytes.

Detection: MediaPipe looks for faces on a 192x192 letterboxed copy of its
input, so in a 16:9 frame a face must be about a quarter of the frame height
to be found, and the speaker at the rostrum of a 1080p recording (200 to 300
pixels) is missed in a large share of the frames. :class:`FaceFinder` scans
square tiles of a downscaled frame every few frames and, in between, looks
for each known face inside a square window around its last position, where
it fills the detector's input. Landmarks always come from the full frame.

Resume: every finished utterance is a line of ``work/<session>/clips.tsv``;
a run skips those lines, so an interrupted session continues where it
stopped and never writes an utterance twice.
"""

import csv
import os
import shutil
import subprocess
import time
import traceback
from collections import Counter, deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from avsr.corpus import segment as seg
from avsr.corpus.status import SessionStatus, atomic_write, options_hash
from avsr.preprocess import landmarks as lm
from avsr.preprocess import video_io
from avsr.preprocess.crop import crop_mouth

STAGE = "cut"
CLIP_FIELDS = ["utt_id", "verdict", "num_frames", "detection_rate", "num_faces", "face_px", "yaw", "mouth_motion", "error"]


@dataclass
class CutOptions:
    variant: str = "mouth"  # mouth | face
    out_size: int = 96
    # Mouth: ROI side as a fraction of the face height (0.63 = Auto-AVSR area).
    # Face: side of the square as a fraction of the face height.
    roi_scale: float = 0.63
    smoothing: str = "kalman"
    align: str = "none"
    crf: int = 18
    fps: float = 25.0
    min_frames: int = 12
    max_frames: int = 600
    min_detection_rate: float = 0.8
    # Median face height in source pixels below which the shot is a wide
    # view of the chamber rather than the speaker.
    min_face_px: float = 50.0
    # Face height, in pixels of the stored frames, that is kept at most.
    keep_face_px: int = 0  # 0 = derived from out_size / roi_scale
    # Faces looked for in every frame. Close-ups in the chamber usually show
    # a listener or two behind the speaker.
    num_faces: int = 4
    # A face track must be present in this share of the frames to be a
    # candidate, and at least this fraction as tall as the tallest candidate.
    min_track_share: float = 0.5
    min_relative_height: float = 0.5
    # Face discovery (see FaceFinder): tiles of a copy of the frame shrunk to
    # ``scan_height`` rows are scanned every ``scan_every`` frames; known
    # faces are followed inside ``track_roi`` face heights of their last place.
    scan_height: int = 540
    scan_every: int = 16
    track_roi: float = 2.5
    # Landmarks are detected on every n-th output frame and interpolated.
    detect_every: int = 2
    # Side of the stored square around the tallest face, in face heights
    # (0 = store the whole shrunk frame). The mouth crop is 0.63 heights.
    region_scale: float = 3.0
    # Pause when the disk holding the dataset has less than this free.
    min_free_gb: float = 5.0
    # Where MediaPipe runs its models: cpu | gpu (OpenGL ES; falls back to
    # the CPU). A runtime choice, not part of the result's identity.
    delegate: str = "gpu"

    RUNTIME_FIELDS = ("delegate",)

    def identity(self):
        """The options that define the result (used for the status hash)."""
        return {k: v for k, v in asdict(self).items() if k not in self.RUNTIME_FIELDS}

    def validate(self):
        if self.variant not in ("mouth", "face"):
            raise ValueError("variant must be mouth or face")
        if self.out_size % 2:
            raise ValueError("out_size must be even")
        if self.keep_face_px <= 0:
            self.keep_face_px = int(np.ceil(self.out_size / self.roi_scale * 1.05))
        return self


@dataclass
class ClipResult:
    utt_id: str
    verdict: str
    num_frames: int = 0
    detection_rate: float = 0.0
    num_faces: int = 0
    face_px: float = 0.0
    yaw: float = 0.0
    mouth_motion: float = 0.0
    error: str = ""

    def to_row(self):
        d = asdict(self)
        d["detection_rate"] = f"{self.detection_rate:.3f}"
        d["face_px"] = f"{self.face_px:.1f}"
        d["yaw"] = f"{self.yaw:.4f}"
        d["mouth_motion"] = f"{self.mouth_motion:.5f}"
        return d


# ---------------------------------------------------------------------------
# sequential frame source
# ---------------------------------------------------------------------------


class FrameSource:
    """Decodes a recording once, in order, handing out frames by time span."""

    def __init__(self, path, lookback=8):
        import av

        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.time_base = float(self.stream.time_base)
        rate = self.stream.average_rate
        self.src_fps = float(rate) if rate else 25.0
        self.dt = 1.0 / self.src_fps
        self._decoder = self.container.decode(self.stream)
        self._recent = deque(maxlen=lookback)  # (time, av frame) already handed out
        self._pending = None  # the first frame past the previous span
        self.exhausted = False

    def _next(self):
        try:
            frame = next(self._decoder)
        except StopIteration:
            self.exhausted = True
            return None
        t = (frame.pts if frame.pts is not None else 0) * self.time_base
        return t, frame

    def frames_between(self, start, end):
        """Yield (time, av.VideoFrame) with start <= time < end, in order."""
        for t, frame in list(self._recent):
            if start <= t < end:
                yield t, frame
        if self._pending is not None:
            t, frame = self._pending
            if t >= end:
                return
            self._pending = None
            if t >= start:
                self._recent.append((t, frame))
                yield t, frame
        while True:
            item = self._next()
            if item is None:
                return
            t, frame = item
            if t >= end:
                self._pending = item
                return
            if t >= start:
                self._recent.append((t, frame))
                yield t, frame

    def seek(self, t):
        """Continue decoding from the last keyframe at or before ``t`` seconds."""
        self.container.seek(int(max(t, 0.0) / self.time_base), stream=self.stream, backward=True, any_frame=False)
        self._decoder = self.container.decode(self.stream)
        self._recent.clear()
        self._pending = None
        self.exhausted = False

    def close(self):
        self.container.close()


def target_times(start, num_frames, fps):
    return start + (np.arange(num_frames) + 0.5) / fps


class FrameSelector:
    """Online nearest-frame selection from a source grid onto the output grid."""

    def __init__(self, start, num_frames, fps, src_dt):
        self.targets = target_times(start, num_frames, fps)
        self.k = 0
        self.half = src_dt / 2.0

    def take(self, t):
        """How many output frames source frame at time ``t`` serves (0, 1 or more)."""
        count = 0
        while self.k < len(self.targets) and self.targets[self.k] < t + self.half:
            count += 1
            self.k += 1
        return count

    @property
    def done(self):
        return self.k >= len(self.targets)


# ---------------------------------------------------------------------------
# finding every face of a frame
# ---------------------------------------------------------------------------


def tile_offsets(length, side):
    """Start positions of square tiles of ``side`` covering ``length``, overlapping by half."""
    if length <= side:
        return [0]
    count = int(np.ceil((length - side) / (side / 2.0))) + 1
    return [int(round(x)) for x in np.linspace(0, length - side, count)]


def merge_faces(faces, max_distance=0.5):
    """Drop duplicates: faces whose mouth centres are within ``max_distance`` face heights."""
    kept = []
    for pts in sorted(faces, key=lambda f: -float(np.linalg.norm(f[lm.FOREHEAD] - f[lm.CHIN]))):
        height = float(np.linalg.norm(pts[lm.FOREHEAD] - pts[lm.CHIN]))
        mouth = pts[lm.MOUTH_OUTER].mean(axis=0)
        if all(np.linalg.norm(mouth - k[lm.MOUTH_OUTER].mean(axis=0)) > max_distance * height for k in kept):
            kept.append(pts)
    return kept


class FaceFinder:
    """All faces of consecutive frames, found with a detector that needs a square input.

    ``detector.detect_all(image)`` returns landmark arrays in pixels of the
    image it is given. A scan runs the detector on overlapping square tiles of
    the frame shrunk to ``scan_height`` rows; it happens on the first frame,
    every ``scan_every`` frames, after a gap of more than ``max_gap`` seconds
    in the frame times, and whenever no face is known. Every face known from
    the previous frame (or just scanned) is then re-detected at full
    resolution inside a square of ``track_roi`` face heights around it, and
    the detection nearest to its previous mouth position is taken.
    """

    def __init__(self, detector, scan_height=540, scan_every=8, track_roi=2.5, max_gap=1.0, max_jump=0.6):
        self.detector = detector
        self.scan_height = scan_height
        self.scan_every = scan_every
        self.track_roi = track_roi
        self.max_gap = max_gap
        self.max_jump = max_jump
        self.reset()

    def reset(self):
        self.known = []
        self.last_t = None
        self.since_scan = 0

    def scan(self, rgb):
        h, w = rgb.shape[:2]
        s = min(1.0, self.scan_height / float(h))
        small = cv2.resize(rgb, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
        sh, sw = small.shape[:2]
        side = min(sh, sw)
        found = []
        for y in tile_offsets(sh, side):
            for x in tile_offsets(sw, side):
                for pts in self.detector.detect_all(small[y : y + side, x : x + side]):
                    found.append((pts + np.array([x, y], dtype=np.float32)) / s)
        return found

    def redetect(self, rgb, pts):
        """The face ``pts`` in this frame, from a square window around its last place."""
        h, w = rgb.shape[:2]
        height = float(np.linalg.norm(pts[lm.FOREHEAD] - pts[lm.CHIN]))
        centre = (pts[lm.FOREHEAD] + pts[lm.CHIN]) / 2.0
        side = int(round(min(max(self.track_roi * height, 64.0), h, w)))
        x0 = int(round(min(max(centre[0] - side / 2.0, 0), w - side)))
        y0 = int(round(min(max(centre[1] - side / 2.0, 0), h - side)))
        found = self.detector.detect_all(rgb[y0 : y0 + side, x0 : x0 + side])
        if not found:
            return None
        offset = np.array([x0, y0], dtype=np.float32)
        mouth = pts[lm.MOUTH_OUTER].mean(axis=0)
        best = min(found, key=lambda f: float(np.linalg.norm(f[lm.MOUTH_OUTER].mean(axis=0) + offset - mouth)))
        if float(np.linalg.norm(best[lm.MOUTH_OUTER].mean(axis=0) + offset - mouth)) > self.max_jump * height:
            return None
        return best + offset

    def detect_all(self, rgb, t=None):
        if t is not None and self.last_t is not None and t - self.last_t > self.max_gap:
            self.reset()
        seeds = list(self.known)
        if not seeds or self.since_scan >= self.scan_every:
            seeds = merge_faces(seeds + self.scan(rgb))
            self.since_scan = 0
        faces = merge_faces([f for f in (self.redetect(rgb, pts) for pts in seeds) if f is not None])
        self.known, self.last_t = faces, t
        self.since_scan += 1
        return faces

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def extract_session_audio(video, wav_path, sample_rate=video_io.SAMPLE_RATE):
    """16 kHz mono 16-bit WAV of the whole recording, for memory-mapped slicing."""
    if os.path.isfile(wav_path):
        return wav_path
    tmp = wav_path + ".tmp.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", tmp],
        check=True,
    )
    os.replace(tmp, wav_path)
    return wav_path


def open_wav(wav_path):
    """Memory-mapped int16 samples of a 16-bit mono WAV written by ffmpeg."""
    with open(wav_path, "rb") as f:
        header = f.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"{wav_path}: not a WAV file")
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{wav_path}: no data chunk")
            name, size = chunk[:4], int.from_bytes(chunk[4:8], "little")
            if name == b"data":
                offset = f.tell()
                break
            f.seek(size + (size & 1), 1)
    count = (os.path.getsize(wav_path) - offset) // 2 if size in (0, 0xFFFFFFFF) else size // 2
    return np.memmap(wav_path, dtype=np.int16, mode="r", offset=offset, shape=(count,))


def audio_slice(samples, start, num_frames, sample_rate=video_io.SAMPLE_RATE):
    first = int(round(start * sample_rate))
    need = num_frames * video_io.SAMPLES_PER_FRAME
    chunk = np.asarray(samples[max(first, 0) : max(first, 0) + need], dtype=np.float32) / 32768.0
    return video_io.fit_audio_to_video(chunk, num_frames)


# ---------------------------------------------------------------------------
# choosing the speaker among the faces of a clip
# ---------------------------------------------------------------------------


def _track_faces(faces_per_frame, max_jump=0.6, max_gap=10):
    """Greedy frame-to-frame association of faces by mouth position.

    Returns tracks as dicts ``{frame index: landmarks}``. A face continues
    the track whose last mouth centre is nearest, if that distance is below
    ``max_jump`` face heights and the track was seen within ``max_gap`` frames.
    """
    tracks = []
    for t, faces in enumerate(faces_per_frame):
        used = set()
        for track in tracks:
            if t - track["last_t"] > max_gap:
                continue
            last = track["last"]
            reach = max_jump * float(np.linalg.norm(last[lm.FOREHEAD] - last[lm.CHIN]))
            best, best_d = None, None
            for i, pts in enumerate(faces):
                if i in used:
                    continue
                d = float(np.linalg.norm(pts[lm.MOUTH_OUTER].mean(axis=0) - last[lm.MOUTH_OUTER].mean(axis=0)))
                if d <= reach and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                track["frames"][t] = faces[best]
                track["last"], track["last_t"] = faces[best], t
        for i, pts in enumerate(faces):
            if i not in used:
                tracks.append({"frames": {t: pts}, "last": pts, "last_t": t})
    return tracks


def select_speaker(faces_per_frame, min_track_share=0.5, min_relative_height=0.5):
    """Pick one face per frame: the track whose mouth moves most.

    Tracks present in fewer than ``min_track_share`` of the frames, and
    tracks much smaller than the tallest candidate, are not considered. The
    speaker is the remaining track with the largest variation of the lip
    opening, which separates the person talking from listeners beside them.

    Returns (per-frame landmarks or None, number of candidate tracks).
    """
    num_frames = len(faces_per_frame)
    tracks = _track_faces(faces_per_frame)
    if not tracks:
        return [None] * num_frames, 0
    candidates = [tr for tr in tracks if len(tr["frames"]) >= min_track_share * num_frames]
    if not candidates:
        candidates = [max(tracks, key=lambda tr: len(tr["frames"]))]
    stats = []
    for tr in candidates:
        arr = np.stack(list(tr["frames"].values()))
        stats.append((tr, float(np.median(lm.face_height(arr))), float(np.std(lm.mouth_opening(arr)))))
    tallest = max(h for _, h, _ in stats)
    big = [(tr, h, m) for tr, h, m in stats if h >= min_relative_height * tallest]
    chosen = max(big, key=lambda x: (x[2], x[1]))[0]
    return [chosen["frames"].get(t) for t in range(num_frames)], len(candidates)


# ---------------------------------------------------------------------------
# cropping
# ---------------------------------------------------------------------------


def crop_face(frames, landmarks, out_size, roi_scale, smoothing):
    """Square crop around the face centre (midpoint of forehead and chin)."""
    from avsr.preprocess.smoothing import smooth_trajectory

    centers = smooth_trajectory((landmarks[:, lm.FOREHEAD] + landmarks[:, lm.CHIN]) / 2.0, smoothing)
    side = float(np.median(lm.face_height(landmarks))) * roi_scale
    out = np.empty((len(frames), out_size, out_size, 3), dtype=np.uint8)
    scale = out_size / side
    for t, frame in enumerate(frames):
        matrix = np.array(
            [[scale, 0.0, out_size / 2.0 - scale * centers[t][0]], [0.0, scale, out_size / 2.0 - scale * centers[t][1]]]
        )
        out[t] = cv2.warpAffine(frame, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def crop_clip(frames, landmarks, options):
    if options.variant == "face":
        return crop_face(frames, landmarks, options.out_size, options.roi_scale, options.smoothing)
    return crop_mouth(
        frames, landmarks, out_size=options.out_size, roi_scale=options.roi_scale,
        smoothing=options.smoothing, align=options.align,
    )


# ---------------------------------------------------------------------------
# one utterance
# ---------------------------------------------------------------------------


class _ClipFrames:
    """Collects the frames of one utterance: a bounded-scale square around the faces.

    ``faces`` of a frame is a list of landmark arrays (full-frame pixels),
    ``[]`` when the detector found nothing, or ``None`` when the frame was
    not given to the detector. The first detection fixes the scale (the
    tallest face is ``keep_face_px`` high) and the side of the stored
    square; every stored frame is that square around the tallest face seen
    last, so ``offsets[i]`` (full-frame origin) and ``scale`` map landmarks
    into frame ``i``.
    """

    def __init__(self, keep_face_px, region_scale=0.0):
        self.keep_face_px = keep_face_px
        self.region_scale = region_scale
        self.scale = None  # set at the first detection
        self.side = None  # stored square side in stored pixels, None = whole frame
        self.centre = None  # full-frame centre of the tallest face seen last
        self.frames = []  # stored frames, all the same size
        self.offsets = []  # full-frame origin of every stored frame
        self.raw = []  # frames seen before the first detection (full size)
        self.faces = []
        self.times = []

    def add(self, t, rgb, faces):
        self.times.append(t)
        self.faces.append(faces)
        if faces:
            tallest = max(faces, key=lambda pts: float(np.linalg.norm(pts[lm.FOREHEAD] - pts[lm.CHIN])))
            self.centre = (tallest[lm.FOREHEAD] + tallest[lm.CHIN]) / 2.0
            if self.scale is None:
                self._set_scale(float(np.linalg.norm(tallest[lm.FOREHEAD] - tallest[lm.CHIN])), rgb.shape)
                for old in self.raw:
                    self._store(old)
                self.raw = []
        if self.scale is None:
            self.raw.append(rgb)
        else:
            self._store(rgb)

    def _set_scale(self, height, shape):
        h, w = shape[:2]
        self.scale = min(1.0, self.keep_face_px / max(height, 1.0))
        if self.region_scale > 0:
            src_side = int(round(min(self.region_scale * height, h, w)))
            if src_side < min(h, w):
                self.side = max(2, int(round(src_side * self.scale)))
                self.src_side = src_side
                self.scale = self.side / float(src_side)  # exact, so landmarks map onto stored pixels

    def _store(self, rgb):
        h, w = rgb.shape[:2]
        if self.side is None:
            region, origin = rgb, (0, 0)
            size = (max(2, int(round(w * self.scale))), max(2, int(round(h * self.scale))))
        else:
            x0 = int(round(min(max(self.centre[0] - self.src_side / 2.0, 0), w - self.src_side)))
            y0 = int(round(min(max(self.centre[1] - self.src_side / 2.0, 0), h - self.src_side)))
            region, origin = rgb[y0 : y0 + self.src_side, x0 : x0 + self.src_side], (x0, y0)
            size = (self.side, self.side)
        self.offsets.append(origin)
        if size == (region.shape[1], region.shape[0]):
            self.frames.append(np.ascontiguousarray(region))
        else:
            self.frames.append(cv2.resize(region, size, interpolation=cv2.INTER_AREA))

    def repeat_last(self):
        """Append the last frame again (the recording ended inside the utterance)."""
        self.times.append(self.times[-1])
        self.faces.append(None)  # not a detection: filled from the last one
        if self.scale is None:
            self.raw.append(self.raw[-1])
        else:
            self.frames.append(self.frames[-1])
            self.offsets.append(self.offsets[-1])

    def stored_landmarks(self, full):
        """Full-frame landmarks [T, N, 2] -> coordinates of the stored frames."""
        return (full - np.array(self.offsets, dtype=np.float32)[:, None, :]) * self.scale

    @property
    def detected_any(self):
        return self.scale is not None


def _frames_around(source, start, end, num_frames, options, centres, src_side, side):
    """Read the utterance's frames again and store the ``src_side`` square around ``centres[t]``.

    Used when the chosen speaker is not the tallest face the square was
    centred on while the frames were first read. Returns (frames [T, side,
    side, 3], offsets [T, 2] in source pixels).
    """
    source.seek(start - 1.0)
    selector = FrameSelector(start, num_frames, options.fps, source.dt)
    frames, offsets = [], []
    for t, frame in source.frames_between(start, end + source.dt):
        count = selector.take(t)
        if count == 0:
            continue
        rgb = frame.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        for _ in range(count):
            c = centres[min(len(frames), num_frames - 1)]
            x0 = int(round(min(max(c[0] - src_side / 2.0, 0), w - src_side)))
            y0 = int(round(min(max(c[1] - src_side / 2.0, 0), h - src_side)))
            region = rgb[y0 : y0 + src_side, x0 : x0 + src_side]
            frames.append(np.ascontiguousarray(region) if side == src_side else cv2.resize(region, (side, side), interpolation=cv2.INTER_AREA))
            offsets.append((x0, y0))
        if selector.done:
            break
    if not frames:
        raise RuntimeError("no frames when reading the utterance again")
    while len(frames) < num_frames:
        frames.append(frames[-1])
        offsets.append(offsets[-1])
    return np.stack(frames), np.array(offsets, dtype=np.float32)


def _inside_square(stored, side, margin):
    mouth = stored[:, lm.MOUTH_OUTER].mean(axis=1)
    return not ((mouth - margin < 0).any() or (mouth + margin > side).any())


def cut_utterance(row, source, detector, samples, options, out_dir, cache_dir):
    """Cut one utterance. Never raises; the result carries the verdict."""
    utt_id = row["utt_id"]
    start, end = float(row["start"]), float(row["end"])
    result = ClipResult(utt_id, "kept")
    num_frames = int(round((end - start) * options.fps))
    result.num_frames = num_frames
    if num_frames < options.min_frames:
        result.verdict = "too_short"
    elif num_frames > options.max_frames:
        result.verdict = "too_long"
    if result.verdict != "kept":
        # Consume the span so the decoder stays in step.
        for _ in source.frames_between(start, end):
            pass
        return result

    try:
        selector = FrameSelector(start, num_frames, options.fps, source.dt)
        clip = _ClipFrames(options.keep_face_px, options.region_scale)
        every = max(1, int(options.detect_every))
        for t, frame in source.frames_between(start, end + source.dt):
            count = selector.take(t)
            if count == 0:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            for _ in range(count):
                faces = detector.detect_all(rgb, t) if len(clip.times) % every == 0 else None
                clip.add(t, rgb, faces)
            if selector.done:
                break
        while len(clip.times) < num_frames and clip.times:
            clip.repeat_last()
        if not clip.times:
            result.verdict = "no_video"
            return result

        # The speaker is chosen among the frames the detector saw; the others
        # are filled by interpolation below.
        sampled = np.array([f is not None for f in clip.faces], dtype=bool)
        chosen_sampled, result.num_faces = select_speaker(
            [f for f in clip.faces if f is not None], options.min_track_share, options.min_relative_height
        )
        chosen = [None] * len(clip.faces)
        for i, c in zip(np.flatnonzero(sampled), chosen_sampled):
            chosen[i] = c
        array, detected = lm.pack_landmarks(chosen)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cache_dir, f"{utt_id}.tmp.npz"),
            landmarks=array, detected=detected, sampled=sampled, times=np.array(clip.times),
        )
        os.replace(os.path.join(cache_dir, f"{utt_id}.tmp.npz"), os.path.join(cache_dir, f"{utt_id}.npz"))

        result.detection_rate = float(detected[sampled].mean()) if sampled.any() else 0.0
        if result.detection_rate < options.min_detection_rate or not clip.detected_any:
            result.verdict = "no_face" if result.detection_rate < 0.5 else "low_detection_rate"
            return result
        full = lm.interpolate_missing(array, detected)
        result.face_px = float(np.median(lm.face_height(full)))
        if result.face_px < options.min_face_px:
            result.verdict = "small_face"
            return result
        result.yaw = float(np.median(lm.yaw_proxy(full)))
        result.mouth_motion = float(np.std(lm.mouth_opening(full)))

        stored = clip.stored_landmarks(full)
        frames = None
        # The crop is a square of roi_scale face heights centred on the mouth.
        margin = 0.55 * options.roi_scale * result.face_px * clip.scale
        if clip.side is not None and not _inside_square(stored, clip.side, margin):
            # The square was centred on the tallest face and the chosen one
            # is elsewhere: read the span again around the chosen face.
            centres = (full[:, lm.FOREHEAD] + full[:, lm.CHIN]) / 2.0
            frames, offsets = _frames_around(source, start, end, num_frames, options, centres, clip.src_side, clip.side)
            stored = (full - offsets[:, None, :]) * clip.scale
            if not _inside_square(stored, clip.side, margin):
                raise RuntimeError("chosen speaker outside the stored region")
        if frames is None:
            frames = np.stack(clip.frames)
        clip.frames = []
        cropped = crop_clip(frames, stored, options)
        audio = audio_slice(samples, start, num_frames)
        os.makedirs(out_dir, exist_ok=True)
        video_io.write_clip(os.path.join(out_dir, f"{utt_id}.mp4"), cropped, audio, fps=int(options.fps), crf=options.crf)
        atomic_write(os.path.join(out_dir, f"{utt_id}.txt"), seg.label_file_text_from_row(row))
    except Exception as e:  # noqa: BLE001 - every failure becomes a verdict
        result.verdict = "error"
        result.error = f"{type(e).__name__}: {e}"
        if os.environ.get("AVSR_DEBUG"):
            traceback.print_exc()
    return result


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------


def read_clip_results(path):
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["utt_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def make_detector(options=None):
    options = options or CutOptions()
    return FaceFinder(
        lm.FaceLandmarker(num_faces=options.num_faces, delegate=options.delegate),
        scan_height=options.scan_height, scan_every=options.scan_every, track_roi=options.track_roi,
    )


def cut_session(session, video, work_dir, dataset_dir, options=None, detector=None, log=print, progress_every=500, retry_errors=False):
    """Cut every kept utterance of a session. Returns the verdict counts.

    ``retry_errors`` cuts the utterances whose verdict is ``error`` again
    (after a fix of the cutter); their new rows replace the old ones.
    """
    options = (options or CutOptions()).validate()
    status = SessionStatus(work_dir)
    opts = {"options": options.identity(), "video_bytes": os.path.getsize(video)}
    results_path = os.path.join(work_dir, "clips.tsv")
    if status.is_done(STAGE, opts) and os.path.isfile(results_path) and not retry_errors:
        log(f"skip {session}: cut")
        return Counter(r["verdict"] for r in read_clip_results(results_path).values())

    segments = [r for r in seg.read_segments(os.path.join(work_dir, "segments.tsv")) if r["verdict"] == "kept"]
    os.makedirs(dataset_dir, exist_ok=True)
    out_dir = os.path.join(dataset_dir, session)
    cache_dir = os.path.join(work_dir, "landmarks")
    # Partial results are only continued when they were made with these
    # options and this video; ``clips.options`` records what clips.tsv holds.
    options_path = os.path.join(work_dir, "clips.options")
    current = options_hash(opts)
    previous = open(options_path, encoding="utf-8").read().strip() if os.path.isfile(options_path) else None
    if os.path.isfile(results_path) and previous != current:
        log(f"{session}: options or video changed, discarding partial results")
        for path in (results_path, options_path):
            if os.path.isfile(path):
                os.remove(path)
        for folder in (cache_dir, out_dir):
            shutil.rmtree(folder, ignore_errors=True)
    atomic_write(options_path, current + "\n")
    done = read_clip_results(results_path)
    for utt_id, r in list(done.items()):
        if r["verdict"] == "kept" and not os.path.isfile(os.path.join(out_dir, f"{utt_id}.mp4")):
            del done[utt_id]  # the file was lost: cut it again
        elif retry_errors and r["verdict"] == "error":
            del done[utt_id]
    todo = [r for r in segments if r["utt_id"] not in done]
    log(f"{session}: {len(segments)} utterances, {len(done)} done, {len(todo)} to cut")

    counts = Counter(r["verdict"] for r in done.values())
    if todo:
        wav = extract_session_audio(video, os.path.join(work_dir, "audio.wav"))
        samples = open_wav(wav)
        detector = detector or make_detector(options)
        source = FrameSource(video)
        if done:
            # Resume: jump to the first utterance left instead of decoding everything before it.
            source.seek(float(todo[0]["start"]) - 1.0)
        new_file = not os.path.isfile(results_path) or os.path.getsize(results_path) == 0
        started = time.time()
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CLIP_FIELDS, delimiter="\t")
            if new_file:
                writer.writeheader()
            for i, row in enumerate(todo, 1):
                while shutil.disk_usage(dataset_dir).free / 1e9 < options.min_free_gb:
                    log(f"{session}: waiting, under {options.min_free_gb} GB free")
                    time.sleep(300)
                result = cut_utterance(row, source, detector, samples, options, out_dir, cache_dir)
                writer.writerow(result.to_row())
                f.flush()
                counts[result.verdict] += 1
                if i % progress_every == 0 or i == len(todo):
                    rate = i / (time.time() - started) * 60
                    log(f"{session}: {i}/{len(todo)} ({rate:.0f} per min), kept {counts['kept']}")
        source.close()
        if hasattr(detector, "close"):
            detector.close()
        del samples
        os.remove(wav)
    # The last row of an utterance counts (a retry appends a new one).
    counts = Counter(r["verdict"] for r in read_clip_results(results_path).values())
    status.mark(STAGE, opts, **{k: int(v) for k, v in counts.items()}, dataset=os.path.abspath(dataset_dir))
    return counts

