#!/usr/bin/env python
"""Pack a finished run into a model folder that works without the dataset.

Usage:
    python scripts/export_model.py --run exp/vsr_finetune --average 10 --out tr-vsr-model

The folder holds the averaged weights (checkpoints/epoch-avg.ckpt), the run
configuration, the tokenizer and the evaluation results of the run, and is
used like a run directory:

    python scripts/transcribe.py --run tr-vsr-model video.mp4
"""

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from avsr.config import resolve_path  # noqa: E402
from avsr.inference import pick_checkpoints  # noqa: E402
from avsr.training.schedule import average_checkpoints  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoints/")
    parser.add_argument("--average", type=int, default=10, help="Average the weights of the last N checkpoints")
    parser.add_argument("--out", required=True, help="Model folder to create")
    args = parser.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"{args.out} already exists")
    cfg = OmegaConf.load(os.path.join(args.run, "config.yaml"))
    checkpoints = pick_checkpoints(args.run, args.average)
    state = average_checkpoints(checkpoints)
    state = {k: v for k, v in state.items() if not k.startswith("ema.")}

    os.makedirs(os.path.join(args.out, "checkpoints"))
    torch.save({"state_dict": state}, os.path.join(args.out, "checkpoints", "epoch-avg.ckpt"))
    shutil.copy(os.path.join(args.run, "config.yaml"), os.path.join(args.out, "config.yaml"))
    shutil.copy(resolve_path(cfg.tokenizer.model_path), os.path.join(args.out, "tokenizer.model"))
    for path in glob.glob(os.path.join(args.run, "eval", "*")):
        os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
        shutil.copy(path, os.path.join(args.out, "eval"))
    metrics = os.path.join(args.run, "csv", "metrics.csv")
    if os.path.isfile(metrics):
        shutil.copy(metrics, os.path.join(args.out, "training_metrics.csv"))
    info = {
        "run": os.path.abspath(args.run),
        "averaged_checkpoints": [os.path.basename(p) for p in checkpoints],
        "modality": cfg.model.modality,
        "tokenizer": os.path.basename(cfg.tokenizer.model_path),
    }
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
