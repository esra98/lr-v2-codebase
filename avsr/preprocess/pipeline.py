"""Building one clip of the training dataset.

The dataset has the layout of the source corpus:

    <session>/<clip>.mp4   video (stream 0) and audio (stream 1)
    <session>/<clip>.aac   audio, copied unchanged
    <session>/<clip>.txt   transcript, copied unchanged

The video is the mouth crop, optionally with Partition-Time Masking written
into it. Face landmarks are cached, because detecting them is by far the most
expensive step.
"""

import os
import shutil
import traceback
from dataclasses import asdict, dataclass

import numpy as np

from avsr.preprocess import landmarks as lm
from avsr.preprocess import video_io
from avsr.preprocess.crop import crop_mouth
from avsr.preprocess.ptm_bake import build_mask, clip_rng, mask_frames

SIDE_FILES = (".aac", ".txt")


@dataclass
class BuildOptions:
    """How the clips of a dataset are made."""

    out_size: int = 96
    # 0.63 x face height shows the same area as the 96 px Auto-AVSR crop.
    roi_scale: float = 0.63
    smoothing: str = "kalman"
    align: str = "none"
    crf: int = 18
    ptm: bool = True
    ptm_grid: int = 4
    ptm_alpha: float = 0.2
    ptm_partitions: int = 2
    ptm_fill: str = "zero"
    seed: int = 0
    min_frames: int = 12
    min_detection_rate: float = 0.8

    def validate(self):
        if self.out_size % 2:
            raise ValueError("out_size must be even (H.264 with 4:2:0 chroma)")
        if self.ptm:
            build_mask(self.ptm_grid, self.ptm_alpha, self.ptm_partitions, self.ptm_fill)


@dataclass
class ClipResult:
    session: str
    clip_id: str
    verdict: str
    num_frames: int = 0
    detection_rate: float = 0.0
    yaw: float = 0.0
    mouth_motion: float = 0.0
    error: str = ""

    def to_dict(self):
        return asdict(self)


def _load_or_detect(frames, path, detector):
    if os.path.isfile(path):
        try:
            with np.load(path) as cache:
                array, detected = cache["landmarks"], cache["detected"]
            if len(array) == len(frames) and array.shape[1] == lm.NUM_LANDMARKS:
                return array, detected
        except Exception:
            pass  # unreadable or stale cache: detect again
    array, detected = lm.pack_landmarks(detector.detect_video(frames))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, landmarks=array, detected=detected)
    os.replace(tmp, path)
    return array, detected


def build_clip(raw_root, out_root, cache_root, session, clip_id, options, detector, check=None, overwrite=False):
    """Write the files of one clip, unless the video shows it is unusable.

    Args:
        check: optional function ``(num_frames) -> reason or None`` for tests
            that need the real clip length, such as the speaking rate.

    Never raises; the verdict is "kept", a rejection reason, or "error".
    """
    result = ClipResult(session=session, clip_id=clip_id, verdict="kept")
    src = os.path.join(raw_root, session, clip_id)
    dst = os.path.join(out_root, session, clip_id)
    try:
        frames = video_io.read_video(src + ".mp4")
        result.num_frames = len(frames)
        if abs(video_io.read_video_fps(src + ".mp4") - video_io.VIDEO_FPS) > 0.01:
            result.verdict = "bad_fps"
            return result
        if len(frames) < options.min_frames:
            result.verdict = "too_short"
            return result
        reason = check(len(frames)) if check else None
        if reason:
            result.verdict = reason
            return result

        cache = os.path.join(cache_root, "landmarks", session, f"{clip_id}.npz")
        array, detected = _load_or_detect(frames, cache, detector)
        result.detection_rate = float(detected.mean())
        if result.detection_rate < options.min_detection_rate:
            result.verdict = "no_face" if result.detection_rate < 0.5 else "low_detection_rate"
            return result
        landmarks = lm.interpolate_missing(array, detected)
        result.yaw = float(np.median(lm.yaw_proxy(landmarks)))
        result.mouth_motion = float(np.std(lm.mouth_opening(landmarks)))

        if overwrite or not os.path.isfile(dst + ".mp4"):
            crop = crop_mouth(
                frames,
                landmarks,
                out_size=options.out_size,
                roi_scale=options.roi_scale,
                smoothing=options.smoothing,
                align=options.align,
            )
            if options.ptm:
                mask = build_mask(options.ptm_grid, options.ptm_alpha, options.ptm_partitions, options.ptm_fill)
                crop = mask_frames(crop, mask, clip_rng(options.seed, session, clip_id))
            video_io.write_video_with_audio(dst + ".mp4", crop, src + ".mp4", crf=options.crf)
        for ext in SIDE_FILES:
            if os.path.isfile(src + ext) and (overwrite or not os.path.isfile(dst + ext)):
                shutil.copyfile(src + ext, dst + ext + ".tmp")
                os.replace(dst + ext + ".tmp", dst + ext)
    except Exception as e:
        result.verdict = "error"
        result.error = f"{type(e).__name__}: {e}"
        if os.environ.get("AVSR_DEBUG"):
            traceback.print_exc()
    return result
