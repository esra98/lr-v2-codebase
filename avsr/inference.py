"""Use a trained run on new video: load the weights, crop the mouth, decode.

The crop repeats what the corpus pipeline (``avsr/corpus/cut.py``) does for
the training clips: faces found with its tiled ``FaceFinder`` on every frame
at 25 fps, the speaker chosen by lip motion (``select_speaker``), the frame
shrunk so the face is at most ``keep_face_px`` high, then a 96x96 square of
0.63 face heights around the Kalman-smoothed mouth centre. The model then
sees the same centre 88x88 grey crop as during validation.
"""

import glob
import os

import av
import numpy as np
import torch

from avsr.config import load_run_config
from avsr.decoding.beam import beam_decode, build_beam_search
from avsr.preprocess import landmarks as lm
from avsr.preprocess.video_io import VIDEO_FPS
from avsr.training.module import AVSRModule
from avsr.training.schedule import average_checkpoints

OUT_SIZE = 96
ROI_SCALE = 0.63
# As CutOptions.keep_face_px: the face height the training crops were made from.
KEEP_FACE_PX = int(np.ceil(OUT_SIZE / ROI_SCALE * 1.05))


def pick_checkpoints(run_dir, average):
    """The ``average`` most recent of the kept epoch checkpoints.

    Training keeps the ``train.save_top_k`` best epochs by validation CER.
    Falls back to last.ckpt when no epoch checkpoint exists.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch*.ckpt")), key=os.path.getmtime)
    if not paths:
        last = os.path.join(run_dir, "checkpoints", "last.ckpt")
        if not os.path.isfile(last):
            raise FileNotFoundError(f"No checkpoints under {run_dir}")
        return [last]
    return paths[-average:]


def load_trained(run_dir, average=1, overrides=()):
    """Build the module of a finished run with its trained weights.

    ``run_dir`` is a training run or a model folder made by
    ``scripts/export_model.py``, which carries its own ``tokenizer.model``.

    Returns (module, checkpoint paths, whether the moving average was used).
    """
    # Weights come from the run's checkpoint, not from the original initialisation.
    overrides = [*overrides, "model.pretrained.path="]
    tokenizer = os.path.join(os.path.abspath(run_dir), "tokenizer.model")
    if os.path.isfile(tokenizer):
        overrides.append(f"tokenizer.model_path={tokenizer}")
    cfg = load_run_config(os.path.join(run_dir, "config.yaml"), overrides)
    module = AVSRModule(cfg)
    checkpoints = pick_checkpoints(run_dir, average)
    state = average_checkpoints(checkpoints)
    # The moving average kept during training is used when a single
    # checkpoint is evaluated; averaging several checkpoints replaces it.
    use_ema = module.ema is not None and len(checkpoints) == 1
    if not use_ema:
        module.ema = None
        state = {k: v for k, v in state.items() if not k.startswith("ema.")}
    module.load_state_dict(state)
    return module, checkpoints, use_ema


def _frames_25fps(path):
    """Yield RGB frames of the first video stream resampled to 25 fps."""
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        index = 0
        previous = None
        for frame in container.decode(stream):
            t = float(frame.time) if frame.time is not None else index / VIDEO_FPS
            rgb = frame.to_ndarray(format="rgb24")
            # Every output instant up to this frame shows the latest frame before it.
            while previous is not None and index / VIDEO_FPS < t - 0.5 / VIDEO_FPS:
                yield previous
                index += 1
            previous = rgb
        if previous is not None:
            yield previous


def mouth_crops(path, detector=None):
    """Mouth crops of the speaker in a video, uint8 [T, 96, 96, 3] at 25 fps.

    Among the faces that stay in the picture, the one whose lips move most
    is taken as the speaker, as for the training clips; the video should
    show one speaker throughout. Returns (crops, share of frames in which the
    speaker's face was found).
    """
    # OpenCV and MediaPipe are only needed here, not for training or evaluation.
    import cv2

    from avsr.corpus.cut import CutOptions, make_detector, select_speaker
    from avsr.preprocess.crop import _crop_matrix
    from avsr.preprocess.smoothing import smooth_trajectory

    own = detector is None
    detector = detector or make_detector(CutOptions(delegate="cpu"))
    try:
        # First pass: landmarks only, so the full frames need not be kept.
        faces = [detector.detect_all(rgb, i / VIDEO_FPS) for i, rgb in enumerate(_frames_25fps(path))]
    finally:
        if own:
            detector.close()
    if not faces:
        raise ValueError(f"{path}: no frames")
    chosen, _ = select_speaker(faces)
    array, detected = lm.pack_landmarks(chosen)
    full = lm.interpolate_missing(array, detected)
    if full is None:
        raise ValueError(f"{path}: no face found")

    face_px = float(np.median(lm.face_height(full)))
    scale = min(1.0, KEEP_FACE_PX / max(face_px, 1.0))
    centres = smooth_trajectory(lm.mouth_center(full) * scale, "kalman")
    side = face_px * scale * ROI_SCALE

    # Second pass: shrink every frame as the corpus pipeline did, then crop.
    crops = np.empty((len(full), OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    for t, rgb in enumerate(_frames_25fps(path)):
        if t >= len(crops):
            break
        if scale < 1.0:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
        matrix = _crop_matrix(centres[t], side, OUT_SIZE)
        crops[t] = cv2.warpAffine(
            rgb, matrix, (OUT_SIZE, OUT_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    return crops, float(detected.mean())


def crops_to_input(crops, crop_size):
    """uint8 RGB [T, 96, 96, 3] -> normalised grey [1, T, 1, crop, crop] as in validation."""
    from avsr.data.transforms import VIDEO_MEAN, VIDEO_STD, CenterCrop, to_grayscale

    video = CenterCrop(crop_size)(to_grayscale(torch.from_numpy(crops)))
    return ((video - VIDEO_MEAN) / VIDEO_STD).unsqueeze(0)


@torch.no_grad()
def transcribe(module, crops, chunk_frames=250, beam_size=None):
    """Decode mouth crops in pieces of at most ``chunk_frames`` frames.

    Returns a list of (start seconds, end seconds, text).
    """
    cfg = module.cfg
    model = module.ema.shadow if module.ema is not None else module.model
    model.eval()
    device = next(model.parameters()).device
    beam = build_beam_search(
        model, module.tokenizer.token_list, beam_size or cfg.decode.beam_size, cfg.decode.ctc_weight,
        cfg.decode.length_bonus,
    ).to(device)
    pieces = []
    for start in range(0, len(crops), chunk_frames):
        part = crops[start : start + chunk_frames]
        if len(part) < 12:  # shorter than the shortest training clip
            continue
        video = crops_to_input(part, cfg.augment.video.crop_size).to(device)
        encoded, _, _ = model.encode(video=video, mode="video")
        text = module.tokenizer.decode(beam_decode(beam, encoded[0]))
        pieces.append((start / VIDEO_FPS, (start + len(part)) / VIDEO_FPS, text))
    return pieces
