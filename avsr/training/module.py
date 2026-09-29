"""Lightning modules for training and evaluation."""

import logging
import os
import random

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule

from avsr.config import resolve_path
from avsr.data.dataset import AVDataset, BucketBatchSampler, collate, split_batch
from avsr.data.transforms import AudioTransform, VideoTransform
from avsr.decoding.beam import beam_decode, build_beam_search
from avsr.models.e2e import INPUT_MODES, AVSRModel, count_parameters
from avsr.models.pretrained import load_pretrained
from avsr.text.tokenizer import build_tokenizer
from avsr.training.schedule import EMA, WarmupCosine
from avsr.utils.metrics import ErrorRate

log = logging.getLogger(__name__)


# torch.OutOfMemoryError was added in PyTorch 2.5; before that the class lived in torch.cuda.
_OOM_ERROR = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", MemoryError)


def process_memory_gb():
    """Resident memory of this process in GB, or 0 where it cannot be read."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, IndexError):
        return 0.0


def is_oom(error):
    return isinstance(error, _OOM_ERROR) or "out of memory" in str(error).lower()


class AVSRDataModule(LightningDataModule):
    """Loaders for the three splits.

    The training loader is built anew for every epoch (the trainer is created
    with ``reload_dataloaders_every_n_epochs=1``), each time with the batches
    of that epoch.
    """

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._datasets = {}

    def _dataset(self, subset):
        if subset in self._datasets:
            return self._datasets[subset]
        cfg = self.cfg
        train = subset == "train"
        speed = tuple(cfg.augment.speed_range) if train and list(cfg.augment.speed_range) else None
        limit = {"train": cfg.data.limit_train, "val": cfg.data.limit_val, "test": cfg.data.limit_test}[subset]
        self._datasets[subset] = AVDataset(
            manifest_path=os.path.join(resolve_path(cfg.data.manifests), f"{subset}.tsv"),
            data_root=resolve_path(cfg.data.root),
            modality=cfg.model.modality,
            tokenizer=self.tokenizer,
            video_transform=VideoTransform(subset, cfg.augment.video),
            audio_transform=AudioTransform(
                subset, cfg.augment.audio, snr_target=None if train else cfg.decode.snr_target
            ),
            speed_range=speed,
            max_frames=cfg.data.max_frames_per_clip,
            limit=limit or None,
        )
        return self._datasets[subset]

    def _loader(self, subset):
        cfg = self.cfg
        dataset = self._dataset(subset)
        train = subset == "train"
        epoch = self.trainer.current_epoch if train and self.trainer is not None else 0
        sampler = BucketBatchSampler(
            dataset.lengths,
            max_frames=cfg.data.max_frames_per_batch,
            max_batch_size=cfg.data.max_batch_size,
            num_buckets=cfg.data.num_buckets if train else 1,
            shuffle=train,
            seed=cfg.seed,
            epoch=epoch,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")


class AVSRModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.tokenizer = build_tokenizer(cfg.tokenizer)
        self.model = AVSRModel(self.tokenizer.vocab_size, cfg.model)
        if cfg.model.pretrained.path:
            kind = "audio" if cfg.model.modality == "audio" else "video"
            load_pretrained(self.model, resolve_path(cfg.model.pretrained.path), kind, cfg.model.pretrained.scope)
        self.model.freeze(cfg.model.freeze)
        self.ema = EMA(self.model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
        self.val_metrics = {}
        self.test_metric = ErrorRate()
        self.beam_search = None
        self.oom_batches = 0
        self.skipped_in_a_row = 0
        self.skipped_in_epoch = 0
        self.batches_in_epoch = 0
        self._current_batch = None
        log.info("Model has %.1fM trainable parameters", count_parameters(self.model) / 1e6)

    # -- optimisation -----------------------------------------------------

    def _updates_per_epoch(self, total):
        """Number of optimiser updates in one epoch.

        Counted from the batches of the training set, so that it stays right
        when train.max_steps ends the run early.
        """
        batches = self.trainer.num_training_batches  # known once the total was estimated
        if isinstance(batches, int) and batches > 0:
            return max(1, -(-batches // self.trainer.accumulate_grad_batches))
        return total / max(self.trainer.max_epochs, 1) if self.trainer.max_epochs > 0 else total

    def configure_optimizers(self):
        """AdamW with separate learning rates for pretrained and new parameters.

        Layers that start from random values (the output layers of a new
        vocabulary) need a high learning rate. Applied to pretrained weights,
        the same rate erases what they know before the new layers can use it.
        """
        cfg = self.cfg.train
        pretrained = getattr(self.model, "pretrained_names", set())
        groups = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            old = name in pretrained
            decay = p.ndim > 1  # biases and normalisation weights are not decayed
            groups.setdefault((old, decay), []).append(p)
        total = self.trainer.estimated_stepping_batches
        steps_per_epoch = self._updates_per_epoch(total)
        param_groups, hold = [], []
        for (old, decay), params in sorted(groups.items()):
            param_groups.append(
                {
                    "params": params,
                    "lr": cfg.lr * (cfg.pretrained_lr_scale if old else 1.0),
                    "weight_decay": cfg.weight_decay if decay else 0.0,
                    "name": ("pretrained" if old else "new") + ("" if decay else "_nodecay"),
                }
            )
            hold.append(cfg.hold_pretrained_epochs * steps_per_epoch if old else 0)
        for group in param_groups:
            log.info(
                "%-18s %6.1fM parameters, learning rate %g",
                group["name"], sum(p.numel() for p in group["params"]) / 1e6, group["lr"],
            )
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.98))
        warmup = min(cfg.warmup_epochs * steps_per_epoch, 0.5 * total)
        scheduler = WarmupCosine(optimizer, warmup, total, hold_steps=hold)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _sample_input_mode(self):
        if self.cfg.model.modality != "audiovisual":
            return self.cfg.model.modality
        return random.choices(INPUT_MODES, weights=list(self.cfg.train.input_mode_probs))[0]

    def _skip_batch(self, reason, batch):
        """Record a batch that could not be used, and stop if that becomes the rule.

        Skipping is meant for the occasional oversized batch. When many
        batches are skipped the settings are wrong, and continuing would only
        look like training.
        """
        self.skipped_in_a_row += 1
        self.skipped_in_epoch += 1
        if reason == "out of memory":
            self.oom_batches += 1
            torch.cuda.empty_cache()
        frames = batch.get("video_lengths", batch.get("audio_lengths"))
        log.warning(
            "Skipping a batch of %d clips (longest %d): %s", len(batch["utt_ids"]), int(frames.max()), reason
        )
        limit = self.cfg.train.max_skipped_in_a_row
        if self.skipped_in_a_row > limit:
            raise RuntimeError(
                f"{self.skipped_in_a_row} batches in a row were skipped ({reason}). "
                + (
                    "The batches do not fit the GPU: lower data.max_frames_per_batch "
                    f"(now {self.cfg.data.max_frames_per_batch})."
                    if reason == "out of memory"
                    else "The loss is not a number: lower train.lr or use train.precision=32-true."
                )
            )

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        self.batches_in_epoch += 1
        mode = self._sample_input_mode()
        try:
            out = self.model(batch, mode)
        except RuntimeError as e:
            if not (self.cfg.train.skip_oom_batches and is_oom(e)):
                raise
            self._skip_batch("out of memory", batch)
            return None
        loss = out["loss"]
        if not torch.isfinite(loss):
            self._skip_batch("loss is not finite", batch)
            return None
        self._current_batch = batch
        size = len(batch["utt_ids"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=size)
        self.log("train/loss_ctc", out["loss_ctc"], on_epoch=True, on_step=False, batch_size=size)
        self.log("train/loss_att", out["loss_att"], on_epoch=True, on_step=False, batch_size=size)
        self.log("train/acc", out["acc"], on_epoch=True, on_step=False, prog_bar=True, batch_size=size)
        if self.model.interctc_weight > 0:
            self.log("train/loss_interctc", out["loss_interctc"], on_epoch=True, on_step=False, batch_size=size)
        return loss

    def backward(self, loss, *args, **kwargs):
        """Backward pass that survives running out of memory.

        The gradients of the failed batch are dropped, so the optimizer step
        that follows changes nothing.
        """
        try:
            super().backward(loss, *args, **kwargs)
            self.skipped_in_a_row = 0
        except RuntimeError as e:
            if not (self.cfg.train.skip_oom_batches and is_oom(e)):
                raise
            self.zero_grad(set_to_none=True)
            self._skip_batch("out of memory", self._current_batch)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None and outputs is not None:
            self.ema.update(self.model)
        # With 16-bit precision the gradient scaler halves its scale whenever
        # gradients overflow. A scale that keeps falling means steps are being
        # skipped and the run should switch to bf16-mixed or 32-true.
        if batch_idx % self.cfg.train.log_every_n_steps:
            return
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.log("train/amp_scale", float(scaler.get_scale()))
        # Memory of the training process, without the loader processes. It
        # rises during the first half hour and should then level off.
        self.log("train/ram_gb", process_memory_gb())

    def on_train_epoch_start(self):
        self.skipped_in_epoch = 0
        self.batches_in_epoch = 0

    def on_train_epoch_end(self):
        self.log("train/oom_batches", float(self.oom_batches))
        self.log("train/batches", float(self.batches_in_epoch))
        share = self.skipped_in_epoch / max(self.batches_in_epoch, 1)
        self.log("train/skipped_share", share)
        # A handful of skipped batches is tolerated whatever their share,
        # so that a very short epoch is not ended by a single one.
        if share > self.cfg.train.max_skipped_share and self.skipped_in_epoch >= 5:
            raise RuntimeError(
                f"{self.skipped_in_epoch} of {self.batches_in_epoch} batches of this epoch were skipped "
                f"({share:.0%}). If they ran out of memory, lower data.max_frames_per_batch "
                f"(now {self.cfg.data.max_frames_per_batch}); see the warnings in the log."
            )

    # -- evaluation -------------------------------------------------------

    def _eval_model(self):
        return self.ema.shadow if self.ema is not None else self.model

    def _eval_modes(self):
        """Input modes to report. A unified model is scored in all three."""
        if self.cfg.decode.input_mode:
            return [self.cfg.decode.input_mode]
        unified = (
            self.cfg.model.modality == "audiovisual"
            and sum(p > 0 for p in self.cfg.train.input_mode_probs) > 1
        )
        return list(INPUT_MODES) if unified else [self.cfg.model.modality]

    def _lengths(self, batch):
        if "video_lengths" in batch:
            return batch["video_lengths"]
        return torch.div(batch["audio_lengths"], 640, rounding_mode="floor")

    def on_validation_epoch_start(self):
        self.val_metrics = {mode: ErrorRate() for mode in self._eval_modes()}

    def _in_parts(self, step, batch):
        """Run an evaluation step, on halves of the batch if it does not fit the GPU.

        Evaluation must not skip clips, or the error rates would be those of
        a different set.
        """
        try:
            step(batch)
        except RuntimeError as e:
            if not is_oom(e) or len(batch["utt_ids"]) < 2:
                raise
            torch.cuda.empty_cache()
            for half in split_batch(batch):
                self._in_parts(step, half)

    def validation_step(self, batch, batch_idx):
        if batch is not None:
            self._in_parts(self._validate, batch)

    def _validate(self, batch):
        model = self._eval_model()
        size = len(batch["utt_ids"])
        limit = self.cfg.decode.val_max_utts
        results = []
        for i, mode in enumerate(self._eval_modes()):
            out = model(batch, mode) if i == 0 else None
            pairs = None
            if not (limit and self.val_metrics[mode].utterances >= limit):
                encoded, lengths, _ = model.encode(batch.get("video"), batch.get("audio"), self._lengths(batch), mode)
                pairs = list(zip(model.ctc_greedy(encoded, lengths), batch["targets"]))
            results.append((mode, out, pairs))
        # Recorded only once the whole batch went through, so that a batch
        # that is repeated in halves is not counted twice.
        for mode, out, pairs in results:
            if out is not None:
                self.log("val/loss", out["loss"], batch_size=size, prog_bar=True, sync_dist=True)
                self.log("val/acc", out["acc"], batch_size=size, sync_dist=True)
            for hyp, target in pairs or []:
                self.val_metrics[mode].update(self.tokenizer.decode(target), self.tokenizer.decode(hyp))

    def on_validation_epoch_end(self):
        modes = self._eval_modes()
        for mode, metric in self.val_metrics.items():
            if not metric.utterances:
                continue
            suffix = "" if len(modes) == 1 else f"_{mode}"
            self.log(f"val/cer{suffix}", metric.cer, prog_bar=len(modes) == 1, sync_dist=True)
            self.log(f"val/wer{suffix}", metric.wer, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_metric = ErrorRate()
        self.test_results = []
        decode = self.cfg.decode
        self.beam_search = build_beam_search(
            self._eval_model(), self.tokenizer.token_list, decode.beam_size, decode.ctc_weight, decode.length_bonus
        )

    def test_step(self, batch, batch_idx):
        if batch is not None:
            self._in_parts(self._test, batch)

    def _test(self, batch):
        model = self._eval_model()
        mode = self._eval_modes()[0]
        encoded, lengths, _ = model.encode(batch.get("video"), batch.get("audio"), self._lengths(batch), mode)
        for enc, length, target, utt_id in zip(encoded, lengths, batch["targets"], batch["utt_ids"]):
            hyp = self.tokenizer.decode(beam_decode(self.beam_search, enc[:length]))
            ref = self.tokenizer.decode(target)
            self.test_metric.update(ref, hyp)
            self.test_results.append((utt_id, ref, hyp))

    def on_test_epoch_end(self):
        self.log("test/wer", self.test_metric.wer)
        self.log("test/cer", self.test_metric.cer)
