"""Mouth region-of-interest extraction from face-crop videos."""

import cv2
import numpy as np

from avsr.preprocess import landmarks as lm
from avsr.preprocess.smoothing import smooth_trajectory


def _crop_matrix(center, side, out_size):
    """Affine matrix mapping a ``side``-sized square around ``center`` to the output."""
    scale = out_size / side
    return np.array(
        [
            [scale, 0.0, out_size / 2.0 - scale * center[0]],
            [0.0, scale, out_size / 2.0 - scale * center[1]],
        ],
        dtype=np.float64,
    )


def _similarity_matrix(points, center, side, out_size):
    """Like ``_crop_matrix`` but also rotates so the eye line is horizontal."""
    right_eye, left_eye = points[0], points[1]
    dx, dy = left_eye - right_eye
    # Near-profile faces give an unreliable eye line; skip the rotation there.
    angle = np.arctan2(dy, dx) if np.hypot(dx, dy) > 1e-3 else 0.0
    scale = out_size / side
    cos, sin = np.cos(-angle) * scale, np.sin(-angle) * scale
    return np.array(
        [
            [cos, -sin, out_size / 2.0 - (cos * center[0] - sin * center[1])],
            [sin, cos, out_size / 2.0 - (sin * center[0] + cos * center[1])],
        ],
        dtype=np.float64,
    )


def crop_mouth(
    frames,
    landmarks,
    out_size=96,
    roi_scale=0.63,
    smoothing="kalman",
    smoothing_kwargs=None,
    align="none",
):
    """Crop a square region centred on the mouth from every frame.

    Args:
        frames: uint8 [T, H, W, 3].
        landmarks: float [T, NUM_LANDMARKS, 2] with no missing frames.
        out_size: output side in pixels.
        roi_scale: ROI side as a fraction of the face height. 0.63 covers
            the same area as the 96 px Auto-AVSR crop; larger values bring in
            cheeks and jaw.
        smoothing: trajectory smoothing method for the crop centre.
        align: "none" crops without rotating; "similarity" additionally
            removes in-plane head roll.

    Returns:
        uint8 [T, out_size, out_size, 3].
    """
    if len(frames) != len(landmarks):
        raise ValueError(f"{len(frames)} frames but {len(landmarks)} landmark sets")
    if align not in ("none", "similarity"):
        raise ValueError(f"Unknown align mode: {align}")

    centers = smooth_trajectory(lm.mouth_center(landmarks), smoothing, **(smoothing_kwargs or {}))
    # One scale per clip: a per-frame scale would make the crop zoom in and
    # out as the jaw opens, since the chin is one end of the face height.
    side = float(np.median(lm.face_height(landmarks))) * roi_scale
    if not np.isfinite(side) or side < 2.0:
        raise ValueError(f"Degenerate face size (ROI side {side:.2f}px)")

    points = None
    if align == "similarity":
        raw = lm.stable_points(landmarks)
        flat = smooth_trajectory(raw.reshape(len(raw), -1), smoothing, **(smoothing_kwargs or {}))
        points = flat.reshape(raw.shape)

    out = np.empty((len(frames), out_size, out_size, 3), dtype=np.uint8)
    for t, frame in enumerate(frames):
        if align == "similarity":
            matrix = _similarity_matrix(points[t], centers[t], side, out_size)
        else:
            matrix = _crop_matrix(centers[t], side, out_size)
        out[t] = cv2.warpAffine(
            frame,
            matrix,
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out
