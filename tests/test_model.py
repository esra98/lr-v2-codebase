import pytest
import torch

from avsr.config import load_config, load_run_config
from avsr.decoding.beam import beam_decode, build_beam_search
from avsr.models.e2e import AVSRModel
from avsr.models.encoder import Encoder
from avsr.models.pretrained import load_pretrained
from avsr.training.schedule import EMA, WarmupCosine, average_checkpoints
from avsr.utils.metrics import ErrorRate, edit_distance

ODIM = 40


def make_batch(batch_size=3, frames=(30, 24, 18), with_video=True, with_audio=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(frames[:batch_size])
    longest = int(lengths.max())
    batch = {"utt_ids": [f"u{i}" for i in range(batch_size)]}
    if with_video:
        batch["video"] = torch.randn(batch_size, longest, 1, 88, 88, generator=g)
        batch["video_lengths"] = lengths
    if with_audio:
        batch["audio"] = torch.randn(batch_size, longest * 640, 1, generator=g)
        batch["audio_lengths"] = lengths * 640
    # Padding is zero, as produced by the collate function.
    for i, n in enumerate(lengths):
        if with_video:
            batch["video"][i, n:] = 0
        if with_audio:
            batch["audio"][i, n * 640 :] = 0
    targets = torch.full((batch_size, 6), -1, dtype=torch.long)
    target_lengths = torch.tensor([6, 4, 3][:batch_size])
    for i, n in enumerate(target_lengths):
        targets[i, :n] = torch.randint(1, ODIM - 1, (int(n),), generator=g)
    batch["targets"], batch["target_lengths"] = targets, target_lengths
    return batch


def build(toy_cfg, *overrides):
    torch.manual_seed(0)
    return AVSRModel(ODIM, toy_cfg(*overrides).model)


@pytest.mark.parametrize(
    "modality, fusion",
    [("video", "late_mlp"), ("audio", "late_mlp"), ("audiovisual", "late_mlp"), ("audiovisual", "early_shared")],
)
def test_forward_backward(toy_cfg, modality, fusion):
    model = build(toy_cfg, f"model.modality={modality}", f"model.fusion.type={fusion}")
    out = model(make_batch(with_video=modality != "audio", with_audio=modality != "video"))
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert 0.0 <= out["acc"] <= 1.0
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("fusion", ["late_mlp", "early_shared"])
@pytest.mark.parametrize("mode", ["video", "audio", "audiovisual"])
def test_audiovisual_model_runs_in_every_input_mode(toy_cfg, fusion, mode):
    model = build(toy_cfg, "model.modality=audiovisual", f"model.fusion.type={fusion}")
    out = model(make_batch(), mode)
    assert torch.isfinite(out["loss"])


def test_single_modality_model_rejects_other_modes(toy_cfg):
    model = build(toy_cfg, "model.modality=video")
    assert model.supported_input_modes() == ("video",)
    with pytest.raises(ValueError):
        model(make_batch(), "audiovisual")


def test_missing_input_is_reported(toy_cfg):
    model = build(toy_cfg, "model.modality=audiovisual")
    with pytest.raises(ValueError, match="needs audio"):
        model.encode(video=torch.randn(1, 20, 1, 88, 88), mode="audiovisual")


def _padded_vs_alone(model, modality):
    batch = make_batch(with_video=modality != "audio", with_audio=modality != "video")
    lengths = batch["video_lengths"] if "video_lengths" in batch else batch["audio_lengths"] // 640
    n = int(lengths[2])
    video, audio = batch.get("video"), batch.get("audio")
    with torch.no_grad():
        padded, _, _ = model.encode(video, audio, lengths)
        alone, _, _ = model.encode(
            video[2:3, :n] if video is not None else None,
            audio[2:3, : n * 640] if audio is not None else None,
            lengths[2:3],
        )
    return (padded[2, :n] - alone[0, :n]).abs().max().item()


@pytest.mark.parametrize(
    "modality, fusion",
    [("video", "late_mlp"), ("audio", "late_mlp"), ("audiovisual", "late_mlp"), ("audiovisual", "early_shared")],
)
def test_padding_does_not_change_the_encoding(toy_cfg, modality, fusion):
    """A clip must encode the same alone and inside a padded batch."""
    model = build(toy_cfg, f"model.modality={modality}", f"model.fusion.type={fusion}").eval()
    assert _padded_vs_alone(model, modality) < 1e-4


def test_original_conformer_leaks_padding(toy_cfg):
    """Documents the Auto-AVSR behaviour that mask_conv_padding removes."""
    model = build(toy_cfg, "model.modality=video", "model.mask_conv_padding=false").eval()
    assert _padded_vs_alone(model, "video") > 1e-3


def test_interctc_adds_a_loss_term(toy_cfg):
    base = build(toy_cfg, "model.modality=video")
    inter = build(toy_cfg, "model.modality=video", "model.interctc.layers=[1,2]")
    batch = make_batch(with_audio=False)
    assert base(batch)["loss_interctc"] == 0
    out = inter(batch)
    assert out["loss_interctc"] > 0
    # same parameters: the intermediate loss reuses the CTC projection
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in inter.parameters())


def test_self_conditioned_ctc_feeds_predictions_back(toy_cfg):
    model = build(
        toy_cfg, "model.modality=video", "model.interctc.layers=[1]", "model.interctc.self_condition=true"
    )
    encoder = model.streams["video"].encoder
    assert encoder.conditioning_layer is not None
    out = model(make_batch(with_audio=False))
    out["loss"].backward()
    assert encoder.conditioning_layer.weight.grad.abs().sum() > 0


def test_interctc_layer_must_be_inside_the_encoder():
    with pytest.raises(ValueError):
        Encoder(64, 4, 128, num_blocks=3, interctc_layers=(3,))


def test_layer_drop_only_acts_in_training(toy_cfg):
    model = build(toy_cfg, "model.modality=video", "model.layer_drop=0.9")
    video = torch.randn(1, 20, 1, 88, 88)
    model.eval()
    with torch.no_grad():
        a, _, _ = model.encode(video)
        b, _, _ = model.encode(video)
    assert torch.equal(a, b)
    model.train()
    torch.manual_seed(0)
    outs = {model.encode(video)[0].sum().item() for _ in range(8)}
    assert len(outs) > 1


def test_ctc_loss_is_finite_under_autocast(toy_cfg):
    model = build(toy_cfg, "model.modality=video")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(make_batch(with_audio=False))
    assert out["loss"].dtype == torch.float32 and torch.isfinite(out["loss"])


def test_ctc_greedy_collapses_repeats_and_blanks(toy_cfg):
    model = build(toy_cfg, "model.modality=video")
    path = torch.tensor([[0, 5, 5, 0, 5, 7, 7, 0, 0, 9]])
    model.ctc.argmax = lambda hidden: path
    assert model.ctc_greedy(torch.zeros(1, 10, 96), torch.tensor([10])) == [[5, 5, 7, 9]]
    assert model.ctc_greedy(torch.zeros(1, 10, 96), torch.tensor([6])) == [[5, 5, 7]]


def test_beam_search_returns_valid_tokens(toy_cfg):
    model = build(toy_cfg, "model.modality=video").eval()
    token_list = [f"t{i}" for i in range(ODIM)]
    search = build_beam_search(model, token_list, beam_size=3)
    with torch.no_grad():
        encoded, _, _ = model.encode(torch.randn(1, 25, 1, 88, 88))
    tokens = beam_decode(search, encoded[0])
    assert all(0 < t < ODIM - 1 for t in tokens)


def test_model_can_memorise_a_batch(toy_cfg):
    """End-to-end check that gradients reach every part that matters."""
    model = build(toy_cfg, "model.modality=video", "model.dropout=0.0", "model.label_smoothing=0.0")
    batch = make_batch(with_audio=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        out = model(batch)
        first = first if first is not None else out["loss"].item()
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
    assert out["loss"].item() < 0.2 * first
    assert out["acc"] > 0.9


# ----------------------------------------------------------------- pretrained


def _auto_avsr_style_checkpoint(model, path):
    """Save a video model's weights under Auto-AVSR parameter names."""
    state = {}
    for name, value in model.state_dict().items():
        name = name.replace("streams.video.frontend.", "frontend.")
        name = name.replace("streams.video.proj.", "proj_encoder.")
        name = name.replace("streams.video.encoder.", "encoder.")
        state[name] = value.clone()
    torch.save(state, path)


@pytest.mark.parametrize("scope", ["frontend", "encoder", "full"])
def test_load_pretrained_scopes(toy_cfg, tmp_path, scope):
    source = build(toy_cfg, "model.modality=video")
    path = str(tmp_path / "ckpt.pth")
    _auto_avsr_style_checkpoint(source, path)

    torch.manual_seed(1)
    target = AVSRModel(ODIM + 10, toy_cfg("model.modality=video").model)  # different vocabulary
    before = {k: v.clone() for k, v in target.state_dict().items()}
    loaded, _ = load_pretrained(target, path, "video", scope)
    after, src = target.state_dict(), source.state_dict()
    assert loaded > 0

    def copied(name):
        return torch.equal(after[name], src[name])

    assert copied("streams.video.frontend.frontend3D.0.weight")
    assert copied("streams.video.encoder.encoders.0.self_attn.linear_q.weight") == (scope != "frontend")
    assert copied("decoder.decoders.0.self_attn.linear_q.weight") == (scope == "full")
    # vocabulary-dependent layers are never transferred
    assert torch.equal(after["ctc.ctc_lo.weight"], before["ctc.ctc_lo.weight"])
    assert torch.equal(after["decoder.output_layer.weight"], before["decoder.output_layer.weight"])


def test_load_pretrained_into_audiovisual_and_shared_models(toy_cfg, tmp_path):
    source = build(toy_cfg, "model.modality=video")
    path = str(tmp_path / "ckpt.pth")
    _auto_avsr_style_checkpoint(source, path)
    late = build(toy_cfg, "model.modality=audiovisual", "model.fusion.type=late_mlp")
    load_pretrained(late, path, "video", "encoder")
    assert torch.equal(
        late.state_dict()["streams.video.encoder.after_norm.weight"],
        source.state_dict()["streams.video.encoder.after_norm.weight"],
    )
    shared = build(toy_cfg, "model.modality=audiovisual", "model.fusion.type=early_shared")
    load_pretrained(shared, path, "video", "encoder")
    assert torch.equal(
        shared.state_dict()["shared_encoder.encoders.0.self_attn.linear_q.weight"],
        source.state_dict()["streams.video.encoder.encoders.0.self_attn.linear_q.weight"],
    )


def test_load_pretrained_reports_a_size_mismatch(toy_cfg, tmp_path):
    source = build(toy_cfg, "model.modality=video")
    path = str(tmp_path / "ckpt.pth")
    _auto_avsr_style_checkpoint(source, path)
    bigger = AVSRModel(ODIM, toy_cfg("model.modality=video", "model.adim=128", "model.eunits=256").model)
    loaded, skipped = load_pretrained(bigger, path, "video", "encoder")
    assert loaded > 0 and skipped  # the front-end fits, the Conformer does not
    with pytest.raises(ValueError):
        load_pretrained(source, path, "audio", "encoder")


# ------------------------------------------------------------------ utilities


def test_schedule_holds_a_group_still_and_then_warms_it_up():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [a], "lr": 0.01}, {"params": [b], "lr": 1.0}], lr=1.0)
    schedule = WarmupCosine(optimizer, warmup_steps=10, total_steps=200, hold_steps=[30, 0])
    held, new = [], []
    for _ in range(200):
        held.append(optimizer.param_groups[0]["lr"])
        new.append(optimizer.param_groups[1]["lr"])
        optimizer.step()
        schedule.step()
    assert all(v == 0.0 for v in held[:30]) and held[30] > 0
    assert held[30] < held[35] < held[39]  # its own warm-up starts when the hold ends
    assert new[0] == pytest.approx(0.1) and max(new) == pytest.approx(1.0)
    # after both warm-ups the two follow the same curve, 100 times apart
    assert held[100] == pytest.approx(new[100] * 0.01)
    assert held[-1] < 1e-4 and new[-1] < 1e-2
    with pytest.raises(ValueError):
        WarmupCosine(optimizer, 10, 200, hold_steps=[30])


def test_warmup_cosine_shape():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    schedule = WarmupCosine(optimizer, warmup_steps=10, total_steps=110)
    lrs = []
    for _ in range(110):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule.step()
    assert lrs[0] == pytest.approx(0.1)
    assert max(lrs) == pytest.approx(1.0) and lrs.index(max(lrs)) == 9
    assert lrs[-1] < 0.01
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


def test_ema_tracks_the_model():
    model = torch.nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    before = ema.shadow.weight.clone()
    ema.update(model)
    assert torch.allclose(ema.shadow.weight, before + 0.5)


def test_ema_is_saved_and_stays_in_eval_mode():
    holder = torch.nn.Module()
    holder.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    holder.ema = EMA(holder.model, decay=0.9)
    holder.train()
    assert holder.model.training and not holder.ema.shadow.training
    assert "ema.shadow.0.weight" in holder.state_dict()
    assert not any(p.requires_grad for p in holder.ema.parameters())


def test_average_checkpoints(tmp_path):
    paths = []
    for i, value in enumerate([1.0, 3.0]):
        path = str(tmp_path / f"{i}.ckpt")
        torch.save({"state_dict": {"w": torch.full((2,), value), "n": torch.tensor(i)}}, path)
        paths.append(path)
    avg = average_checkpoints(paths)
    assert torch.allclose(avg["w"], torch.full((2,), 2.0))
    assert avg["n"].item() == 1
    with pytest.raises(ValueError):
        average_checkpoints([])


def test_error_rates():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1
    metric = ErrorRate()
    metric.update("BIR ŞEY SÖYLE", "BIR ŞEY SÖYLE")
    metric.update("TEŞEKKÜR EDERIM", "TEŞEKKÜR EDERİZ")
    assert metric.wer == pytest.approx(1 / 5)
    assert metric.cer == pytest.approx(2 / 25)
    assert ErrorRate().wer != ErrorRate().wer  # NaN before any update


@pytest.mark.parametrize(
    "overrides",
    [
        ["model.modality=lips"],
        ["model.adim=100", "model.aheads=3"],
        ["model.ctc_weight=1.5"],
        ["train.input_mode_probs=[0,0,0]"],
        ["augment.speed_range=[1.1,0.9]"],
        ["model.modality=video", "decode.input_mode=audio"],
        ["model.does_not_exist=1"],
        ["train.devices=2"],
        ["model.freeze=decoder"],
        ["data.max_frames_per_batch=200"],
        ["train.pretrained_lr_scale=0"],
        ["train.hold_pretrained_epochs=80"],
    ],
)
def test_config_rejects_bad_values(overrides):
    with pytest.raises(Exception):
        load_config(overrides=overrides)


# -------------------------------------------------------------------- freezing


@pytest.mark.parametrize("scope", ["frontend", "encoder"])
def test_frozen_parts_do_not_change(toy_cfg, scope):
    model = build(toy_cfg, "model.modality=video")
    model.freeze(scope)
    model.train()
    stream = model.streams["video"]
    assert not stream.frontend.training and model.decoder.training
    assert stream.encoder.training == (scope == "frontend")

    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        loss = model(make_batch(with_audio=False))["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    # includes batch-norm running statistics, which are buffers, not parameters
    assert not any(k.startswith("streams.video.frontend.") for k in changed)
    assert any(k.startswith("streams.video.encoder.") for k in changed) == (scope == "frontend")
    assert any(k.startswith("decoder.") for k in changed)
    assert any(k.startswith("ctc.") for k in changed)


def test_freeze_can_be_undone(toy_cfg):
    model = build(toy_cfg, "model.modality=audiovisual", "model.fusion.type=early_shared")
    total = sum(p.numel() for p in model.parameters())
    model.freeze("encoder")
    assert 0 < sum(p.numel() for p in model.parameters() if p.requires_grad) < total
    assert not model.shared_encoder.training
    model.freeze("none")
    model.train()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == total
    assert model.shared_encoder.training
    with pytest.raises(ValueError):
        model.freeze("decoder")


def test_run_config_from_an_older_version_still_loads(tmp_path):
    from omegaconf import OmegaConf

    saved = OmegaConf.to_container(load_config(overrides=["model.modality=audio", "seed=7"]))
    del saved["data"]["limit_test"]  # an option added after the run was made
    del saved["model"]["freeze"]
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(saved), path)
    cfg = load_run_config(str(path), ["decode.beam_size=5", "model.pretrained.path="])
    assert cfg.model.modality == "audio" and cfg.seed == 7 and cfg.decode.beam_size == 5
    assert cfg.data.limit_test == 0 and cfg.model.freeze == "none"
    assert cfg.model.pretrained.path in ("", None)


def test_run_config_with_an_unknown_option_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  removed_option: 3\n")
    with pytest.raises(ValueError, match="does not have"):
        load_run_config(str(path))
