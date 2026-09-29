"""Initialisation from Auto-AVSR checkpoints.

Public Auto-AVSR checkpoints are single-modality models with parameters named
``frontend.*``, ``proj_encoder.*``, ``encoder.*``, ``decoder.*`` and ``ctc.*``.
Their vocabulary is English, so the output layers are never transferred.
"""

import logging

import torch

log = logging.getLogger(__name__)

SCOPES = ("frontend", "encoder", "full")
_VOCAB_DEPENDENT = ("decoder.embed.0.", "decoder.output_layer.", "ctc.ctc_lo.")


def _unwrap(checkpoint):
    """Return the flat state dict inside a checkpoint file."""
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    # Lightning prefixes every parameter with the attribute name of the model.
    if all(k.startswith("model.") for k in checkpoint):
        checkpoint = {k[len("model.") :]: v for k, v in checkpoint.items()}
    return checkpoint


def _rename(key, kind, shared_encoder):
    """Map an Auto-AVSR parameter name to this project's name, or None to skip."""
    if key.startswith(("trunk.", "frontend3D.")):  # front-end-only LRW checkpoints
        return f"streams.{kind}.frontend.{key}"
    if key.startswith("frontend."):
        return f"streams.{kind}.{key}"
    if key.startswith("proj_encoder."):
        return f"streams.{kind}.proj.{key[len('proj_encoder.'):]}"
    if key.startswith("encoder."):
        if shared_encoder:
            return f"shared_encoder.{key[len('encoder.'):]}"
        return f"streams.{kind}.{key}"
    if key.startswith(("decoder.", "ctc.")):
        return key
    return None


def load_pretrained(model, path, kind, scope="encoder"):
    """Copy matching weights from an Auto-AVSR checkpoint into ``model``.

    Args:
        kind: "video" or "audio", the modality of the checkpoint.
        scope: "frontend", "encoder" (front-end, projection and Conformer) or
            "full" (also the decoder, except vocabulary-dependent layers).

    Returns:
        (number of tensors loaded, list of skipped names with reasons)
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if kind not in model.streams:
        raise ValueError(f"The model has no {kind} stream")
    source = _unwrap(torch.load(path, map_location="cpu", weights_only=False))
    target = model.state_dict()
    shared = model.shared_encoder is not None

    loaded, skipped = {}, []
    for key, value in source.items():
        if key.startswith(_VOCAB_DEPENDENT):
            continue
        is_frontend = key.startswith(("frontend.", "trunk.", "frontend3D."))
        is_encoder = key.startswith(("proj_encoder.", "encoder."))
        if scope == "frontend" and not is_frontend:
            continue
        if scope == "encoder" and not (is_frontend or is_encoder):
            continue
        name = _rename(key, kind, shared)
        if name is None or name not in target:
            skipped.append((key, "no matching parameter"))
        elif target[name].shape != value.shape:
            skipped.append((key, f"shape {tuple(value.shape)} vs {tuple(target[name].shape)}"))
        else:
            loaded[name] = value
    if not loaded:
        raise ValueError(
            f"No weights from {path} fit the model. The checkpoint was probably trained "
            "with a different model size (Auto-AVSR checkpoints need configs/model/base.yaml)."
        )
    model.load_state_dict(loaded, strict=False)
    # Remembered so that training can treat pretrained and new parameters differently.
    model.pretrained_names = set(getattr(model, "pretrained_names", set())) | set(loaded)
    log.info("Loaded %d tensors from %s (%d skipped)", len(loaded), path, len(skipped))
    return len(loaded), skipped
