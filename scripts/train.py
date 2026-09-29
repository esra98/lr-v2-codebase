#!/usr/bin/env python
"""Train a model.

Usage:
    python scripts/train.py [-c override.yaml ...] [key=value ...]

Example:
    python scripts/train.py -c configs/model/small.yaml model.modality=audiovisual
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Batches differ in length, which fragments the GPU memory pool of PyTorch
# until allocations fail although enough memory is free. Expandable segments
# avoid that. Must be set before PyTorch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning import Trainer, seed_everything  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger  # noqa: E402

from avsr.config import load_config, resolve_path  # noqa: E402
from avsr.training.module import AVSRDataModule, AVSRModule  # noqa: E402


def build_trainer(cfg, run_dir, enable_progress_bar=True):
    train = cfg.train
    monitor = "val/cer"
    if cfg.model.modality == "audiovisual" and not cfg.decode.input_mode:
        if sum(p > 0 for p in train.input_mode_probs) > 1:
            monitor = "val/cer_video"
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    callbacks = [
        # The best epochs, weights only: they are averaged for evaluation and
        # never resumed from, and the optimizer state would triple their size.
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:03d}",
            auto_insert_metric_name=False,
            monitor=monitor,
            mode="min",
            save_top_k=train.save_top_k,
            save_weights_only=True,
        ),
        # The latest state in full, overwritten every epoch, for resuming.
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="last",
            monitor=None,
            save_top_k=1,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    gpu = torch.cuda.is_available() and train.accelerator in ("auto", "gpu")
    return Trainer(
        default_root_dir=run_dir,
        accelerator="gpu" if gpu else "cpu",
        devices=train.devices if gpu else 1,
        precision=train.precision if gpu else "32-true",
        max_epochs=train.max_epochs,
        max_steps=train.max_steps,
        gradient_clip_val=train.grad_clip,
        accumulate_grad_batches=train.accumulate_grad_batches,
        val_check_interval=train.val_check_interval,
        log_every_n_steps=train.log_every_n_steps,
        callbacks=callbacks,
        logger=[TensorBoardLogger(run_dir, name="", version="tb"), CSVLogger(run_dir, name="", version="csv")],
        use_distributed_sampler=False,
        # Every epoch has its own batches, and their number varies a little.
        # Without reloading, validation is planned from the count of the
        # first epoch and is skipped in epochs that turn out shorter.
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=2,
        enable_progress_bar=enable_progress_bar,
    )


def keep_earlier_metrics(run_dir):
    """Set aside the metrics of an earlier part of the run.

    The CSV logger starts a new file every time, which would erase the
    epochs trained before a resume.
    """
    current = os.path.join(run_dir, "csv", "metrics.csv")
    if os.path.isfile(current):
        part = 1
        while os.path.exists(os.path.join(run_dir, "csv", f"metrics.part{part}.csv")):
            part += 1
        os.replace(current, os.path.join(run_dir, "csv", f"metrics.part{part}.csv"))


def read_metrics(run_dir):
    """Every logged row of a run, across resumes, in the order it was written."""
    folder = os.path.join(run_dir, "csv")
    parts = sorted(
        glob.glob(os.path.join(folder, "metrics.part*.csv")),
        key=lambda p: int(os.path.basename(p)[len("metrics.part") : -len(".csv")]),
    )
    rows = []
    for path in parts + [os.path.join(folder, "metrics.csv")]:
        if os.path.isfile(path):
            with open(path, newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def run(cfg, resume=None, enable_progress_bar=True):
    """Train and return the final validation metrics."""
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    run_dir = os.path.join(resolve_path(cfg.exp_dir), cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    keep_earlier_metrics(run_dir)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    module = AVSRModule(cfg)
    datamodule = AVSRDataModule(cfg, module.tokenizer)
    trainer = build_trainer(cfg, run_dir, enable_progress_bar)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    metrics["oom_batches"] = module.oom_batches
    metrics["steps"] = trainer.global_step
    metrics["train_minutes"] = (time.time() - started) / 60
    if torch.cuda.is_available():
        metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", action="append", default=[], help="Override file (repeatable)")
    parser.add_argument("--resume", help="Checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    metrics = run(cfg, args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
