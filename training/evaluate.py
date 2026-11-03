#!/usr/bin/env python
"""Evaluate a trained model on the test set with beam search.

Usage:
    python training/evaluate.py --run exp/my_run [--average 10] [key=value ...]

Examples:
    # visual-only scoring of an audio-visual model
    python training/evaluate.py --run exp/av decode.input_mode=video
    # audio-visual scoring with babble noise at 0 dB
    python training/evaluate.py --run exp/av decode.snr_target=0
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Batches differ in length, which fragments the GPU memory pool of PyTorch
# until allocations fail although enough memory is free. Expandable segments
# avoid that. Must be set before PyTorch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402

from avsr.inference import load_trained  # noqa: E402
from avsr.training.module import AVSRDataModule  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoints/")
    parser.add_argument("--average", type=int, default=1, help="Average the weights of the last N checkpoints")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    module, checkpoints, use_ema = load_trained(args.run, args.average, args.overrides)
    cfg = module.cfg

    datamodule = AVSRDataModule(cfg, module.tokenizer)
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule._loader("val")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    result = trainer.test(module, dataloaders=loader, verbose=False)[0]

    mode = cfg.decode.input_mode or cfg.model.modality
    tag = f"{args.split}_{mode}" + (f"_snr{cfg.decode.snr_target}" if cfg.decode.snr_target is not None else "")
    out_dir = os.path.join(args.run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.tsv"), "w", encoding="utf-8") as f:
        f.write("utt_id\treference\thypothesis\n")
        for utt_id, ref, hyp in module.test_results:
            f.write(f"{utt_id}\t{ref}\t{hyp}\n")
    summary = {
        "split": args.split,
        "input_mode": mode,
        "snr_target": cfg.decode.snr_target,
        "wer": result["test/wer"],
        "cer": result["test/cer"],
        "utterances": len(module.test_results),
        "checkpoints": [os.path.basename(p) for p in checkpoints],
        "ema": use_ema,
    }
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
