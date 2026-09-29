"""Configuration loading.

Layers, later ones winning: ``configs/base.yaml``, any number of override
files, then ``key=value`` command-line overrides. Keys that do not exist in
the base configuration are rejected, so a mistyped option fails loudly instead
of being silently ignored.
"""

import os

from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "base.yaml")


def load_config(files=(), overrides=()):
    cfg = OmegaConf.load(BASE_CONFIG)
    OmegaConf.set_struct(cfg, True)
    for path in files:
        if not os.path.isfile(path):
            candidate = os.path.join(PROJECT_ROOT, path)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"Config file not found: {path}")
            path = candidate
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    validate(cfg)
    return cfg


def load_run_config(path, overrides=()):
    """Configuration of a finished run, for evaluation.

    The saved file is layered over the current defaults, so a run made before
    an option existed can still be loaded; the new option takes its default.
    """
    cfg = OmegaConf.load(BASE_CONFIG)
    OmegaConf.set_struct(cfg, True)
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    except Exception as e:
        raise ValueError(
            f"{path} contains an option that this version of the code does not have: {e}"
        ) from e
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    validate(cfg)
    return cfg


def validate(cfg):
    """Check option combinations that would otherwise fail deep inside a run."""
    if cfg.model.modality not in ("video", "audio", "audiovisual"):
        raise ValueError(f"model.modality: unknown value {cfg.model.modality!r}")
    if cfg.model.freeze not in ("none", "frontend", "encoder"):
        raise ValueError(f"model.freeze: unknown value {cfg.model.freeze!r}")
    if cfg.model.freeze == "encoder" and (list(cfg.model.interctc.layers) and cfg.model.interctc.self_condition):
        raise ValueError("Self-conditioned CTC changes the encoder and cannot be used with model.freeze=encoder")
    if not 0.0 < cfg.train.pretrained_lr_scale <= 1.0:
        raise ValueError("train.pretrained_lr_scale must be in (0, 1]")
    if cfg.train.hold_pretrained_epochs < 0 or cfg.train.hold_pretrained_epochs >= cfg.train.max_epochs > 0:
        raise ValueError("train.hold_pretrained_epochs must be at least 0 and below train.max_epochs")
    if cfg.train.devices != 1:
        raise ValueError(
            "train.devices must be 1: the length-bucketed batch sampler is not split "
            "across devices, so several GPUs would all train on the same batches"
        )
    if cfg.model.adim % cfg.model.aheads:
        raise ValueError("model.adim must be divisible by model.aheads")
    if not 0.0 <= cfg.model.ctc_weight <= 1.0:
        raise ValueError("model.ctc_weight must be in [0, 1]")
    probs = list(cfg.train.input_mode_probs)
    if len(probs) != 3 or min(probs) < 0 or sum(probs) <= 0:
        raise ValueError("train.input_mode_probs needs three non-negative values, not all zero")
    speed = list(cfg.augment.speed_range)
    if speed and (len(speed) != 2 or not 0 < speed[0] <= speed[1]):
        raise ValueError("augment.speed_range must be empty or [low, high] with 0 < low <= high")
    if cfg.decode.input_mode and cfg.model.modality != "audiovisual":
        if cfg.decode.input_mode != cfg.model.modality:
            raise ValueError(
                f"decode.input_mode={cfg.decode.input_mode!r} is not available "
                f"for a {cfg.model.modality} model"
            )
    if cfg.data.max_frames_per_clip > cfg.data.max_frames_per_batch:
        raise ValueError(
            f"data.max_frames_per_clip ({cfg.data.max_frames_per_clip}) is larger than "
            f"data.max_frames_per_batch ({cfg.data.max_frames_per_batch}): a clip of that length "
            "would not fit into any batch. Lower max_frames_per_clip, which leaves longer clips out, "
            "or raise max_frames_per_batch."
        )
    if cfg.data.ptm_in_clips and cfg.augment.video.ptm.enabled:
        raise ValueError(
            "data.ptm_in_clips is true, so the clips already contain Partition-Time Masking; "
            "augment.video.ptm.enabled must be false, or every clip would be masked twice"
        )
    if cfg.augment.video.crop_size < 0:
        raise ValueError("augment.video.crop_size must be 0 (no crop) or a positive size")


def resolve_path(path):
    """Interpret a relative path as relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
