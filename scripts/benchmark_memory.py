#!/usr/bin/env python
"""Measure GPU memory and speed of a training step for a configuration.

Feeds synthetic batches of increasing size through a full forward/backward
pass, so data.max_frames_per_batch can be chosen before a long run.

The batches here all have the same shape. Real batches vary, and training
also holds the averaged weights when train.ema_decay is set, so real use is
higher: in a rehearsal, 4.4 GB measured here became 4.7 GB in training.
Choose a size whose peak stays below about 80 % of the GPU memory.

Usage:
    python scripts/benchmark_memory.py -c configs/model/small.yaml model.modality=video
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Batches differ in length, which fragments the GPU memory pool of PyTorch
# until allocations fail although enough memory is free. Expandable segments
# avoid that. Must be set before PyTorch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from avsr.config import load_config  # noqa: E402
from avsr.models.e2e import AVSRModel, count_parameters  # noqa: E402
from avsr.training.module import is_oom  # noqa: E402


def synthetic_batch(cfg, clips, frames, vocab, device):
    batch = {"utt_ids": [""] * clips}
    lengths = torch.full((clips,), frames, device=device)
    size = cfg.augment.video.crop_size or 96
    if cfg.model.modality in ("video", "audiovisual"):
        batch["video"] = torch.randn(clips, frames, 1, size, size, device=device)
        batch["video_lengths"] = lengths
    if cfg.model.modality in ("audio", "audiovisual"):
        batch["audio"] = torch.randn(clips, frames * 640, 1, device=device)
        batch["audio_lengths"] = lengths * 640
    tokens = max(1, frames // 4)
    batch["targets"] = torch.randint(1, vocab - 1, (clips, tokens), device=device)
    batch["target_lengths"] = torch.full((clips,), tokens, device=device)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("--clip-frames", type=int, default=75, help="Length of every clip in the batch")
    parser.add_argument("--batch-frames", type=int, nargs="+", default=[150, 300, 600, 1200, 1800, 2400])
    parser.add_argument("--vocab", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device available.")
    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda")
    model = AVSRModel(args.vocab, cfg.model).to(device)
    model.freeze(cfg.model.freeze)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    precision = cfg.train.precision
    dtype = {"16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}.get(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "16-mixed")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{cfg.model.modality} model, {count_parameters(model) / 1e6:.1f}M parameters, {precision}, GPU {total:.1f} GB")
    print(f"{'frames/batch':>12} {'clips':>6} {'peak GB':>8} {'s/step':>7}")

    for batch_frames in args.batch_frames:
        clips = max(1, batch_frames // args.clip_frames)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                batch = synthetic_batch(cfg, clips, args.clip_frames, args.vocab, device)
                torch.cuda.synchronize()
                start = time.time()
                with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    loss = model(batch)["loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                times.append(time.time() - start)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"{clips * args.clip_frames:>12} {clips:>6} {peak:>8.2f} {min(times):>7.2f}")
        except RuntimeError as e:
            if not is_oom(e):
                raise
            print(f"{clips * args.clip_frames:>12} {clips:>6} {'OOM':>8}")
            del e
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
