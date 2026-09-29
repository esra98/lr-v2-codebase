"""End-to-end training smoke tests on the synthetic dataset."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import train as train_script  # noqa: E402

from avsr.training.module import AVSRModule, is_oom  # noqa: E402

pytestmark = pytest.mark.slow


def _run(cfg, **kwargs):
    return train_script.run(cfg, enable_progress_bar=False, **kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.modality=video"],
        ["model.modality=audio"],
        ["model.modality=audiovisual", "model.fusion.type=late_mlp"],
        [
            "model.modality=audiovisual",
            "model.fusion.type=early_shared",
            "train.input_mode_probs=[0.4,0.2,0.4]",
        ],
        ["model.modality=video", "model.interctc.layers=[1,2]", "model.layer_drop=0.1", "train.ema_decay=0.99"],
        ["model.modality=video", "augment.speed_range=[0.9,1.1]", "tokenizer.type=char"],
    ],
    ids=["video", "audio", "av_late", "av_unified", "interctc_ema", "speed_char"],
)
def test_training_runs(toy_cfg, overrides):
    cfg = toy_cfg(*overrides, "train.max_epochs=2", "train.warmup_epochs=1", "exp_name=run")
    metrics = _run(cfg)
    assert metrics["steps"] > 0
    assert metrics["oom_batches"] == 0
    assert torch.isfinite(torch.tensor(metrics["val/loss"]))
    run_dir = os.path.join(cfg.exp_dir, "run")
    assert os.path.isfile(os.path.join(run_dir, "checkpoints", "last.ckpt"))
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "metrics.json")) as f:
        assert json.load(f)["steps"] == metrics["steps"]


def test_pretrained_and_new_parameters_get_their_own_learning_rate(toy_cfg, tmp_path):
    from avsr.models.e2e import AVSRModel

    source = AVSRModel(62, toy_cfg("model.modality=video").model)
    state = {}
    for name, value in source.state_dict().items():
        name = name.replace("streams.video.frontend.", "frontend.").replace("streams.video.proj.", "proj_encoder.")
        state[name.replace("streams.video.encoder.", "encoder.")] = value.clone()
    path = str(tmp_path / "pretrained.pth")
    torch.save(state, path)

    # The run ends inside the hold: pretrained parameters must be untouched.
    cfg = toy_cfg(
        "model.modality=video", f"model.pretrained.path={path}", "model.pretrained.scope=full",
        "train.pretrained_lr_scale=0.01", "train.hold_pretrained_epochs=1", "train.max_epochs=2",
        "train.warmup_epochs=0.5", "train.max_steps=3", "train.log_every_n_steps=1", "exp_name=held",
    )
    module = AVSRModule(cfg)
    names = module.model.pretrained_names
    assert "streams.video.encoder.encoders.0.self_attn.linear_q.weight" in names
    assert "decoder.decoders.0.self_attn.linear_q.weight" in names
    assert "ctc.ctc_lo.weight" not in names and "decoder.output_layer.weight" not in names
    start = {k: v.clone() for k, v in module.model.named_parameters()}

    _run(cfg)
    rows = train_script.read_metrics(os.path.join(cfg.exp_dir, "held"))
    held = [float(r[k]) for r in rows for k in r if k.startswith("lr-") and "pretrained" in k and r[k]]
    fresh = [float(r[k]) for r in rows for k in r if k.startswith("lr-") and "new" in k and r[k]]
    assert held and max(held) == 0.0
    assert fresh and max(fresh) > 0.0

    last = torch.load(os.path.join(cfg.exp_dir, "held", "checkpoints", "last.ckpt"), weights_only=False)
    end = {k[len("model."):]: v for k, v in last["state_dict"].items() if k.startswith("model.")}
    for name, value in start.items():
        if name in names:
            assert torch.equal(end[name], value), name
    assert not torch.equal(end["ctc.ctc_lo.weight"], start["ctc.ctc_lo.weight"])


def test_validation_runs_in_every_epoch(toy_cfg):
    """Epochs differ in their number of batches; none may lose its validation."""
    cfg = toy_cfg(
        "model.modality=video", "train.max_epochs=6", "data.max_frames_per_batch=70", "data.max_frames_per_clip=70",
        "data.num_buckets=1", "train.accumulate_grad_batches=3", "exp_name=everyepoch",
    )
    _run(cfg)
    rows = train_script.read_metrics(os.path.join(cfg.exp_dir, "everyepoch"))
    validated = sorted({int(r["epoch"]) for r in rows if r.get("val/cer")})
    batches = [int(float(r["train/batches"])) for r in rows if r.get("train/batches")]
    assert validated == [0, 1, 2, 3, 4, 5]
    assert len(set(batches)) > 1, "the test needs epochs of different length to mean anything"


def test_unified_model_reports_every_input_mode(toy_cfg):
    cfg = toy_cfg(
        "model.modality=audiovisual",
        "model.fusion.type=early_shared",
        "train.input_mode_probs=[0.4,0.2,0.4]",
        "train.max_epochs=1",
        "exp_name=unified",
    )
    metrics = _run(cfg)
    for mode in ("video", "audio", "audiovisual"):
        assert f"val/cer_{mode}" in metrics


def test_resume_continues_from_the_checkpoint(toy_cfg):
    cfg = toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=resume")
    first = _run(cfg)
    ckpt = os.path.join(cfg.exp_dir, "resume", "checkpoints", "last.ckpt")
    cfg.train.max_epochs = 2
    second = _run(cfg, resume=ckpt)
    assert second["steps"] == 2 * first["steps"]
    # the metrics of the epoch before the resume are still there
    run_dir = os.path.join(cfg.exp_dir, "resume")
    assert os.path.isfile(os.path.join(run_dir, "csv", "metrics.part1.csv"))
    epochs = sorted({int(r["epoch"]) for r in train_script.read_metrics(run_dir) if r.get("val/cer")})
    assert epochs == [0, 1]


def test_training_is_reproducible(toy_cfg):
    a = _run(toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=a"))
    b = _run(toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=b"))
    assert a["val/loss"] == pytest.approx(b["val/loss"], rel=1e-4)


def test_out_of_memory_batch_is_skipped(toy_cfg, monkeypatch):
    cfg = toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=oom")
    calls = {"n": 0}
    original = AVSRModule.training_step

    def flaky(self, batch, batch_idx):
        calls["n"] += 1
        if calls["n"] == 1:
            real_forward = self.model.forward

            def boom(*args, **kwargs):
                self.model.forward = real_forward
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")

            self.model.forward = boom
        return original(self, batch, batch_idx)

    monkeypatch.setattr(AVSRModule, "training_step", flaky)
    metrics = _run(cfg)
    assert metrics["oom_batches"] == 1
    assert metrics["steps"] > 0


def test_training_stops_when_most_batches_do_not_fit(toy_cfg, monkeypatch):
    """A wrong batch size must end the run, not turn it into hours of skipping."""
    from avsr.models.e2e import AVSRModel

    cfg = toy_cfg(
        "model.modality=video", "train.max_epochs=2", "train.max_skipped_in_a_row=3", "data.max_batch_size=2",
        "exp_name=alloom",
    )
    original = AVSRModel.forward

    def training_never_fits(self, batch, mode=None):
        if self.training:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(self, batch, mode)

    monkeypatch.setattr(AVSRModel, "forward", training_never_fits)
    with pytest.raises(RuntimeError, match="lower data.max_frames_per_batch"):
        _run(cfg)


def test_training_stops_when_a_share_of_the_epoch_is_skipped(toy_cfg, monkeypatch):
    cfg = toy_cfg(
        "model.modality=video", "train.max_epochs=2", "train.max_skipped_share=0.2", "data.max_batch_size=1",
        "exp_name=shareoom",
    )
    original = AVSRModule.training_step
    calls = {"n": 0}

    def every_second_fails(self, batch, batch_idx):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            real_forward = self.model.forward

            def boom(*args, **kwargs):
                self.model.forward = real_forward
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

            self.model.forward = boom
        return original(self, batch, batch_idx)

    monkeypatch.setattr(AVSRModule, "training_step", every_second_fails)
    with pytest.raises(RuntimeError, match="batches of this epoch were skipped"):
        _run(cfg)


def test_out_of_memory_in_the_backward_pass_is_survived(toy_cfg, monkeypatch):
    cfg = toy_cfg("model.modality=video", "train.max_epochs=1", "train.max_skipped_share=0.5", "exp_name=backoom")
    original = torch.Tensor.backward
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", flaky)
    metrics = _run(cfg)
    assert metrics["oom_batches"] == 1 and metrics["steps"] > 0


def test_validation_splits_a_batch_that_does_not_fit(toy_cfg, monkeypatch):
    from avsr.models.e2e import AVSRModel

    cfg = toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=valoom")
    original = AVSRModel.encode
    seen = []

    def small_gpu(self, video=None, audio=None, lengths=None, mode=None):
        if not self.training and video is not None and video.size(0) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        if not self.training:
            seen.append(video.size(0))
        return original(self, video, audio, lengths, mode)

    monkeypatch.setattr(AVSRModel, "encode", small_gpu)
    module = AVSRModule(cfg)
    from pytorch_lightning import Trainer

    from avsr.training.module import AVSRDataModule

    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    result = trainer.validate(module, datamodule=AVSRDataModule(cfg, module.tokenizer), verbose=False)[0]
    assert set(seen) == {1}
    # every validation clip was scored exactly once: once for the loss, once for the error rate
    assert module.val_metrics["video"].utterances == 4 and len(seen) == 8
    assert 0.0 <= result["val/cer"]


def test_other_runtime_errors_are_not_swallowed(toy_cfg, monkeypatch):
    cfg = toy_cfg("model.modality=video", "train.max_epochs=1", "exp_name=err")

    def broken(self, batch, mode=None):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr("avsr.models.e2e.AVSRModel.forward", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _run(cfg)


def test_is_oom():
    assert is_oom(torch.OutOfMemoryError("x"))
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_oom(RuntimeError("size mismatch"))


def test_test_loop_writes_wer(toy_cfg):
    from pytorch_lightning import Trainer

    from avsr.training.module import AVSRDataModule

    cfg = toy_cfg("model.modality=video", "decode.beam_size=3")
    module = AVSRModule(cfg)
    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
    result = trainer.test(module, datamodule=AVSRDataModule(cfg, module.tokenizer), verbose=False)[0]
    assert 0.0 <= result["test/cer"] and "test/wer" in result
    assert len(module.test_results) == 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    "overrides",
    [
        ["model.modality=video", "train.ema_decay=0.99", "model.interctc.layers=[1]"],
        ["model.modality=audiovisual", "model.fusion.type=early_shared", "train.input_mode_probs=[0.4,0.2,0.4]"],
        ["model.modality=video", "model.freeze=encoder"],
    ],
    ids=["ema_interctc", "unified", "frozen"],
)
def test_training_on_gpu_with_mixed_precision(toy_cfg, overrides):
    cfg = toy_cfg(*overrides, "train.max_epochs=2", "train.accelerator=gpu", "train.precision=16-mixed", "exp_name=gpu")
    metrics = _run(cfg)
    assert metrics["steps"] > 0 and metrics["gpu_peak_gb"] > 0
    assert torch.isfinite(torch.tensor(metrics["val/loss"]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_evaluate_script(toy_cfg, monkeypatch):
    import evaluate as evaluate_script

    cfg = toy_cfg("model.modality=video", "train.ema_decay=0.9", "train.max_epochs=2", "train.accelerator=gpu",
                  "decode.beam_size=3", "train.save_top_k=2", "exp_name=evalrun")
    _run(cfg)
    run_dir = os.path.join(cfg.exp_dir, "evalrun")
    # One checkpoint uses the moving average; several are averaged instead.
    for average, uses_ema in ((1, True), (2, False)):
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", run_dir, "--average", str(average)])
        evaluate_script.main()
        with open(os.path.join(run_dir, "eval", "test_video.json")) as f:
            summary = json.load(f)
        assert summary["utterances"] == 4 and summary["ema"] is uses_ema
        assert len(summary["checkpoints"]) == average
    assert os.path.isfile(os.path.join(run_dir, "eval", "test_video.tsv"))
