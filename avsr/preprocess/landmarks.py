"""Face landmark detection with the MediaPipe Tasks API.

The legacy ``mp.solutions`` API used by Auto-AVSR was removed from recent
MediaPipe releases, so detection goes through ``FaceLandmarker`` instead.

Only a small subset of the 478 mesh points is kept, which keeps the landmark
cache for the whole dataset at a few gigabytes.
"""

import os

import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "face_landmarker.task",
)

# MediaPipe face-mesh indices that are kept, in cache order.
_MESH_INDICES = (
    61, 291, 0, 17, 37, 267, 84, 314,  # outer lips
    33, 133,  # subject's right eye (image left): outer, inner corner
    362, 263,  # subject's left eye: inner, outer corner
    1,  # nose tip
    10,  # forehead
    152,  # chin
    13, 14, 78, 308,  # inner lips: upper, lower, right corner, left corner
)
NUM_LANDMARKS = len(_MESH_INDICES)

# Positions inside the reduced layout.
MOUTH_OUTER = slice(0, 8)
MOUTH_CORNERS = (0, 1)
RIGHT_EYE = (8, 9)
LEFT_EYE = (10, 11)
NOSE_TIP = 12
FOREHEAD = 13
CHIN = 14
INNER_LIP_UPPER = 15
INNER_LIP_LOWER = 16


class FaceLandmarker:
    """Per-frame face-mesh detector returning pixel coordinates."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, min_confidence=0.3):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Download face_landmarker.task "
                "from https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            # IMAGE mode is stateless, so one instance can serve many clips.
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """Return [NUM_LANDMARKS, 2] float32 pixel coordinates, or None."""
        mp = self._mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        points = result.face_landmarks[0]
        return np.array(
            [[points[i].x * width, points[i].y * height] for i in _MESH_INDICES],
            dtype=np.float32,
        )

    def detect_video(self, frames):
        """Detect on every frame; entries are None where detection failed."""
        return [self.detect(frame) for frame in frames]

    def close(self):
        self._detector.close()


def pack_landmarks(landmarks):
    """Pack per-frame detections into (array [T, N, 2], detected mask [T])."""
    detected = np.array([lm is not None for lm in landmarks], dtype=bool)
    array = np.zeros((len(landmarks), NUM_LANDMARKS, 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if lm is not None:
            array[i] = lm
    return array, detected


def interpolate_missing(array, detected):
    """Fill missing detections by linear interpolation in time.

    Leading and trailing gaps are filled with the nearest detection. Returns
    a [T, N, 2] array, or None when no frame had a detection.
    """
    valid = np.flatnonzero(detected)
    if len(valid) == 0:
        return None
    out = array.astype(np.float32).copy()
    out[: valid[0]] = out[valid[0]]
    out[valid[-1] + 1 :] = out[valid[-1]]
    for a, b in zip(valid[:-1], valid[1:]):
        for i in range(a + 1, b):
            w = (i - a) / (b - a)
            out[i] = (1 - w) * out[a] + w * out[b]
    return out


def mouth_center(landmarks):
    """Mouth centre per frame, [T, 2], from the outer-lip landmarks."""
    return landmarks[:, MOUTH_OUTER].mean(axis=1)


def face_height(landmarks):
    """Forehead-to-chin distance per frame, [T].

    Used as the scale reference because, unlike the inter-ocular distance, it
    does not collapse when the head turns towards profile.
    """
    return np.linalg.norm(landmarks[:, FOREHEAD] - landmarks[:, CHIN], axis=-1)


def mouth_opening(landmarks):
    """Inner-lip gap relative to face height per frame, [T]."""
    gap = np.linalg.norm(landmarks[:, INNER_LIP_UPPER] - landmarks[:, INNER_LIP_LOWER], axis=-1)
    return gap / np.maximum(face_height(landmarks), 1e-6)


def yaw_proxy(landmarks):
    """Signed head-yaw indicator per frame in [-1, 1], 0 when frontal.

    Compares the horizontal distance from the nose tip to each eye's outer
    corner; in a profile view one of them shrinks towards zero.
    """
    nose = landmarks[:, NOSE_TIP, 0]
    right = np.abs(nose - landmarks[:, RIGHT_EYE[0], 0])
    left = np.abs(landmarks[:, LEFT_EYE[1], 0] - nose)
    return (left - right) / np.maximum(left + right, 1e-6)


def stable_points(landmarks):
    """Right eye, left eye, nose tip and mouth centre per frame, [T, 4, 2]."""
    return np.stack(
        [
            landmarks[:, RIGHT_EYE].mean(axis=1),
            landmarks[:, LEFT_EYE].mean(axis=1),
            landmarks[:, NOSE_TIP],
            landmarks[:, MOUTH_CORNERS].mean(axis=1),
        ],
        axis=1,
    )
