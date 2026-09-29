import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV is only installed where datasets are built")

from avsr.preprocess import landmarks as lm
from avsr.preprocess import video_io
from avsr.preprocess.crop import crop_mouth
from avsr.preprocess.smoothing import kalman_smooth, moving_average, one_euro, smooth_trajectory


def _jittery_track(n=100, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    clean = np.stack([100 + 0.3 * t, 150 + 5 * np.sin(t / 15)], axis=1)
    return clean, clean + rng.normal(0, 1.5, clean.shape)


@pytest.mark.parametrize("method", ["moving_average", "kalman", "one_euro"])
def test_smoothing_reduces_jitter(method):
    clean, noisy = _jittery_track()
    smoothed = smooth_trajectory(noisy, method)
    assert smoothed.shape == noisy.shape
    jitter_before = np.abs(np.diff(noisy - clean, axis=0)).mean()
    jitter_after = np.abs(np.diff(smoothed - clean, axis=0)).mean()
    assert jitter_after < 0.6 * jitter_before


@pytest.mark.parametrize("method", ["moving_average", "kalman"])
def test_offline_smoothing_tracks_without_lag(method):
    clean, noisy = _jittery_track()
    smoothed = smooth_trajectory(noisy, method)
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()


@pytest.mark.parametrize("fn", [moving_average, kalman_smooth, one_euro])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_smoothing_handles_very_short_tracks(fn, n):
    x = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    out = fn(x)
    assert out.shape == x.shape and np.isfinite(out).all()


def test_smoothing_none_is_identity():
    _, noisy = _jittery_track()
    assert np.allclose(smooth_trajectory(noisy, "none"), noisy, atol=1e-4)


def test_smoothing_rejects_unknown_method():
    with pytest.raises(ValueError):
        smooth_trajectory(np.zeros((5, 2)), "median")


def test_interpolate_missing_fills_gaps_and_edges():
    array = np.zeros((6, lm.NUM_LANDMARKS, 2), dtype=np.float32)
    detected = np.array([False, True, False, False, True, False])
    array[1], array[4] = 10.0, 40.0
    out = lm.interpolate_missing(array, detected)
    assert np.allclose(out[:, 0, 0], [10, 10, 20, 30, 40, 40])


def test_interpolate_missing_without_any_detection():
    array = np.zeros((4, lm.NUM_LANDMARKS, 2), dtype=np.float32)
    assert lm.interpolate_missing(array, np.zeros(4, dtype=bool)) is None


def _synthetic_face(n_frames=8, size=224, mouth=(112.0, 160.0)):
    """Frames with a bright square at the mouth plus matching landmarks."""
    frames = np.zeros((n_frames, size, size, 3), dtype=np.uint8)
    landmarks = np.zeros((n_frames, lm.NUM_LANDMARKS, 2), dtype=np.float32)
    for t in range(n_frames):
        cx, cy = mouth[0] + t, mouth[1]
        frames[t, int(cy) - 6 : int(cy) + 6, int(cx) - 6 : int(cx) + 6] = 255
        landmarks[t, lm.MOUTH_OUTER] = (cx, cy)
        landmarks[t, lm.FOREHEAD] = (112, 20)
        landmarks[t, lm.CHIN] = (112, 200)
        landmarks[t, list(lm.RIGHT_EYE)] = (80, 80)
        landmarks[t, list(lm.LEFT_EYE)] = (144, 80)
        landmarks[t, lm.NOSE_TIP] = (112, 120)
    return frames, landmarks


@pytest.mark.parametrize("align", ["none", "similarity"])
def test_crop_mouth_centres_the_mouth(align):
    frames, landmarks = _synthetic_face()
    crop = crop_mouth(frames, landmarks, out_size=96, roi_scale=0.55, smoothing="none", align=align)
    assert crop.shape == (8, 96, 96, 3) and crop.dtype == np.uint8
    for frame in crop:
        ys, xs = np.nonzero(frame[..., 0] > 128)
        assert abs(xs.mean() - 47.5) < 2 and abs(ys.mean() - 47.5) < 2


@pytest.mark.parametrize("out_size", [88, 98, 112])
def test_crop_mouth_output_size(out_size):
    frames, landmarks = _synthetic_face()
    crop = crop_mouth(frames, landmarks, out_size=out_size, smoothing="none")
    assert crop.shape == (8, out_size, out_size, 3)
    ys, xs = np.nonzero(crop[0, ..., 0] > 128)
    centre = (out_size - 1) / 2
    assert abs(xs.mean() - centre) < 2 and abs(ys.mean() - centre) < 2


def test_crop_mouth_larger_roi_shows_more_context():
    frames, landmarks = _synthetic_face()
    tight = crop_mouth(frames, landmarks, roi_scale=0.55, smoothing="none")
    wide = crop_mouth(frames, landmarks, roi_scale=0.80, smoothing="none")
    assert (wide[0, ..., 0] > 128).sum() < (tight[0, ..., 0] > 128).sum()


def test_crop_mouth_near_border_does_not_fail():
    frames, landmarks = _synthetic_face(mouth=(215.0, 218.0))
    crop = crop_mouth(frames, landmarks, smoothing="kalman")
    assert crop.shape == (8, 96, 96, 3)


def test_crop_mouth_rejects_mismatched_lengths():
    frames, landmarks = _synthetic_face()
    with pytest.raises(ValueError):
        crop_mouth(frames[:-1], landmarks)


def test_crop_mouth_rejects_degenerate_face():
    frames, landmarks = _synthetic_face()
    landmarks[:, lm.CHIN] = landmarks[:, lm.FOREHEAD]
    with pytest.raises(ValueError):
        crop_mouth(frames, landmarks)


def test_yaw_proxy_sign():
    _, landmarks = _synthetic_face()
    assert abs(lm.yaw_proxy(landmarks)[0]) < 1e-6
    landmarks[:, lm.NOSE_TIP] = (85, 120)  # nose close to the subject's right eye
    assert lm.yaw_proxy(landmarks)[0] > 0.5


def test_fit_audio_to_video():
    audio = np.ones(1000, dtype=np.float32)
    assert len(video_io.fit_audio_to_video(audio, 1)) == 640
    padded = video_io.fit_audio_to_video(audio, 2)
    assert len(padded) == 1280 and padded[1000:].sum() == 0


def test_video_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    # Smooth content: H.264 is lossy, so compare on something it can represent.
    base = np.linspace(0, 255, 96, dtype=np.float32)
    frames = np.stack([np.tile(np.roll(base, 3 * t), (96, 1)) for t in range(10)])
    frames = np.repeat(frames[..., None], 3, axis=-1).astype(np.uint8)
    path = str(tmp_path / "sub" / "clip.mp4")
    video_io.write_video(path, frames)
    back = video_io.read_video(path)
    assert back.shape == frames.shape
    assert np.abs(back.astype(int) - frames.astype(int)).mean() < 6
    assert video_io.read_video_fps(path) == pytest.approx(25.0)
    assert rng is not None


def test_write_video_rejects_odd_size_and_float(tmp_path):
    with pytest.raises(ValueError):
        video_io.write_video(str(tmp_path / "a.mp4"), np.zeros((2, 95, 96, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        video_io.write_video(str(tmp_path / "b.mp4"), np.zeros((2, 96, 96, 3), dtype=np.float32))


@pytest.mark.parametrize("ext", [".flac", ".wav"])
def test_audio_roundtrip(tmp_path, ext):
    audio = (0.5 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    path = str(tmp_path / f"a{ext}")
    video_io.write_audio(path, audio)
    back = video_io.load_audio_file(path)
    assert back.shape == audio.shape and np.abs(back - audio).max() < 1e-3
