"""End-to-end VSR / ASR / AVSR model with hybrid CTC/attention training."""

import torch
import torch.nn.functional as F

from avsr.espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from avsr.espnet.nets.pytorch_backend.frontend.resnet import video_resnet
from avsr.espnet.nets.pytorch_backend.frontend.resnet1d import audio_resnet
from avsr.espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from avsr.espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from avsr.espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from avsr.espnet.nets.pytorch_backend.transformer.mask import target_mask
from avsr.espnet.nets.scorers.ctc import CTCPrefixScorer
from avsr.models.encoder import Encoder
from avsr.preprocess.video_io import SAMPLES_PER_FRAME

FRONTEND_DIM = 512
INPUT_MODES = ("video", "audio", "audiovisual")


class CTCHead(torch.nn.Module):
    """CTC projection and loss.

    The loss is always computed in float32: half-precision log-probabilities
    underflow on long sequences and collapse training to all-blank outputs.
    """

    def __init__(self, odim, idim, dropout_rate):
        super().__init__()
        self.ctc_lo = torch.nn.Linear(idim, odim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def loss(self, hidden, hidden_lengths, targets, target_lengths):
        """Sum of per-utterance losses divided by the batch size."""
        logits = self.ctc_lo(self.dropout(hidden))
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probs = logits.float().log_softmax(-1).transpose(0, 1)
            flat_targets = torch.cat([t[:n] for t, n in zip(targets, target_lengths)])
            loss = F.ctc_loss(
                log_probs,
                flat_targets,
                hidden_lengths,
                target_lengths,
                blank=0,
                reduction="sum",
                zero_infinity=True,
            )
        return loss / hidden.size(0)

    def log_softmax(self, hidden):
        return F.log_softmax(self.ctc_lo(hidden).float(), dim=-1)

    def argmax(self, hidden):
        return self.ctc_lo(hidden).argmax(-1)


class Stream(torch.nn.Module):
    """Front-end and projection for one modality, optionally with its own encoder."""

    def __init__(self, kind, cfg, with_encoder, encoder_kwargs):
        super().__init__()
        self.kind = kind
        self.frontend = video_resnet() if kind == "video" else audio_resnet()
        self.proj = torch.nn.Linear(FRONTEND_DIM, cfg.adim)
        self.encoder = Encoder(**encoder_kwargs) if with_encoder else None

    def features(self, x, train_frontend=True, train_proj=True):
        with torch.set_grad_enabled(train_frontend and torch.is_grad_enabled()):
            x = self.frontend(x)
        with torch.set_grad_enabled(train_proj and torch.is_grad_enabled()):
            return self.proj(x)


class AVSRModel(torch.nn.Module):
    """Hybrid CTC/attention model for video, audio or audio-visual input.

    Fusion modes for audio-visual models:

    - ``late_mlp``: one Conformer per modality; the two encoder outputs are
      concatenated and merged by an MLP (Auto-AVSR).
    - ``early_shared``: the projected front-end features are merged before a
      single shared Conformer, which can also be run on either modality alone.
      Training with random input modes yields one model for VSR, ASR and AVSR.
    """

    def __init__(self, odim, cfg):
        super().__init__()
        if cfg.modality not in INPUT_MODES:
            raise ValueError(f"modality must be one of {INPUT_MODES}, got {cfg.modality!r}")
        if cfg.fusion.type not in ("late_mlp", "early_shared"):
            raise ValueError(f"Unknown fusion type: {cfg.fusion.type}")
        self.modality = cfg.modality
        self.fusion_type = cfg.fusion.type if cfg.modality == "audiovisual" else None
        self.odim = odim
        self.sos = self.eos = odim - 1
        self.ignore_id = -1
        self.ctc_weight = cfg.ctc_weight
        self.interctc_weight = cfg.interctc.weight if list(cfg.interctc.layers) else 0.0

        encoder_kwargs = dict(
            attention_dim=cfg.adim,
            attention_heads=cfg.aheads,
            linear_units=cfg.eunits,
            num_blocks=cfg.elayers,
            dropout_rate=cfg.dropout,
            cnn_module_kernel=cfg.cnn_kernel,
            layer_drop=cfg.layer_drop,
            interctc_layers=tuple(cfg.interctc.layers),
            self_condition_dim=odim if cfg.interctc.self_condition else None,
            mask_conv_padding=cfg.mask_conv_padding,
        )
        shared = self.fusion_type == "early_shared"
        kinds = ["video", "audio"] if cfg.modality == "audiovisual" else [cfg.modality]
        self.streams = torch.nn.ModuleDict(
            {kind: Stream(kind, cfg, with_encoder=not shared, encoder_kwargs=encoder_kwargs) for kind in kinds}
        )

        self.shared_encoder = None
        self.fusion = None
        if shared:
            self.fusion = torch.nn.Linear(2 * cfg.adim, cfg.adim)
            self.shared_encoder = Encoder(**encoder_kwargs)
        elif self.fusion_type == "late_mlp":
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(2 * cfg.adim, cfg.fusion.hidden),
                torch.nn.BatchNorm1d(cfg.fusion.hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(cfg.fusion.hidden, cfg.adim),
            )

        self.decoder = TransformerDecoder(
            odim=odim,
            attention_dim=cfg.adim,
            attention_heads=cfg.aheads,
            linear_units=cfg.dunits,
            num_blocks=cfg.dlayers,
            dropout_rate=cfg.dropout,
            positional_dropout_rate=cfg.dropout,
            self_attention_dropout_rate=cfg.dropout,
            src_attention_dropout_rate=cfg.dropout,
        )
        self.ctc = CTCHead(odim, cfg.adim, cfg.dropout)
        self.criterion = LabelSmoothingLoss(odim, self.ignore_id, cfg.label_smoothing, False)
        self.freeze_scope = "none"

    # -- freezing ---------------------------------------------------------

    def _frozen_modules(self):
        modules = []
        if self.freeze_scope in ("frontend", "encoder"):
            modules += [s.frontend for s in self.streams.values()]
        if self.freeze_scope == "encoder":
            modules += [s.proj for s in self.streams.values()]
            modules += [s.encoder for s in self.streams.values() if s.encoder is not None]
            if self.shared_encoder is not None:
                modules.append(self.shared_encoder)
        return modules

    def freeze(self, scope):
        """Stop training the front-ends ("frontend") or everything up to and
        including the Conformer ("encoder"). Frozen parts run without
        gradients and stay in evaluation mode, so their batch-norm statistics
        and dropout do not change.
        """
        if scope not in ("none", "frontend", "encoder"):
            raise ValueError(f"Unknown freeze scope: {scope}")
        for p in self.parameters():
            p.requires_grad_(True)
        self.freeze_scope = scope
        for module in self._frozen_modules():
            for p in module.parameters():
                p.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self._frozen_modules():
            module.eval()
        return self

    # -- encoding ---------------------------------------------------------

    def supported_input_modes(self):
        """Input modes this model can be run with."""
        if self.modality != "audiovisual":
            return (self.modality,)
        return INPUT_MODES

    def _resolve_mode(self, mode):
        mode = mode or self.modality
        if mode not in self.supported_input_modes():
            raise ValueError(f"A {self.modality} model cannot run in {mode!r} mode")
        return mode

    def _apply_fusion_mlp(self, x):
        batch, frames, dim = x.shape
        return self.fusion(x.reshape(batch * frames, dim)).reshape(batch, frames, -1)

    def encode(self, video=None, audio=None, lengths=None, mode=None):
        """Encode a batch.

        Args:
            video: [B, T, 1, H, W] or None.
            audio: [B, T * 640, 1] or None.
            lengths: [B] lengths in video frames, or None for a single
                unpadded sequence.
            mode: which inputs to use; defaults to the model's modality.

        Returns:
            (encoded [B, T', D], output lengths [B] or None, intermediates)
        """
        mode = self._resolve_mode(mode)
        train_frontend = self.freeze_scope == "none"
        train_encoder = self.freeze_scope != "encoder"
        feats = {}
        if mode in ("video", "audiovisual"):
            if video is None:
                raise ValueError(f"mode {mode!r} needs video input")
            feats["video"] = self.streams["video"].features(video, train_frontend, train_encoder)
        if mode in ("audio", "audiovisual"):
            if audio is None:
                raise ValueError(f"mode {mode!r} needs audio input")
            feats["audio"] = self.streams["audio"].features(audio, train_frontend, train_encoder)

        frames = min(f.size(1) for f in feats.values())
        feats = {k: f[:, :frames] for k, f in feats.items()}
        mask = None
        if lengths is not None:
            lengths = lengths.clamp(max=frames)
            mask = make_non_pad_mask(lengths).to(next(iter(feats.values())).device).unsqueeze(-2)

        ctc_lo = self.ctc.ctc_lo
        encoder_grad = torch.set_grad_enabled(train_encoder and torch.is_grad_enabled())
        if self.fusion_type == "early_shared":
            if mode == "audiovisual":
                x = self.fusion(torch.cat([feats["video"], feats["audio"]], dim=-1))
            else:
                x = feats[mode]
            with encoder_grad:
                encoded, _, inter = self.shared_encoder(x, mask, ctc_lo)
        elif self.fusion_type == "late_mlp":
            outs, inter = {}, []
            for kind in ("video", "audio"):
                if kind in feats:
                    with encoder_grad:
                        outs[kind], _, inter_k = self.streams[kind].encoder(feats[kind], mask, ctc_lo)
                    inter.extend(inter_k)
                else:
                    # A missing modality is represented by zeros, which is
                    # also what modality dropout feeds during training.
                    outs[kind] = None
            ref = outs["video"] if outs["video"] is not None else outs["audio"]
            outs = {k: v if v is not None else torch.zeros_like(ref) for k, v in outs.items()}
            encoded = self._apply_fusion_mlp(torch.cat([outs["video"], outs["audio"]], dim=-1))
            # Intermediate outputs of a single stream are only comparable to
            # the fused output when that stream ran alone.
            if mode == "audiovisual":
                inter = []
        else:
            with encoder_grad:
                encoded, _, inter = self.streams[mode].encoder(feats[mode], mask, ctc_lo)
        return encoded, lengths, inter

    # -- training ---------------------------------------------------------

    def forward(self, batch, mode=None):
        """Compute the training loss for a collated batch.

        Returns a dict with "loss", "loss_ctc", "loss_att", "loss_interctc"
        and the teacher-forced decoder accuracy "acc".
        """
        mode = self._resolve_mode(mode)
        video, audio = batch.get("video"), batch.get("audio")
        if "video_lengths" in batch:
            lengths = batch["video_lengths"]
        else:
            lengths = torch.div(batch["audio_lengths"], SAMPLES_PER_FRAME, rounding_mode="floor")
        targets, target_lengths = batch["targets"], batch["target_lengths"]

        encoded, lengths, inter = self.encode(video, audio, lengths, mode)
        mask = make_non_pad_mask(lengths).to(encoded.device).unsqueeze(-2)

        loss_ctc = self.ctc.loss(encoded, lengths, targets, target_lengths)
        loss_inter = encoded.new_zeros(())
        if inter and self.interctc_weight > 0:
            loss_inter = torch.stack(
                [self.ctc.loss(h, lengths, targets, target_lengths) for h in inter]
            ).mean()
            loss_ctc_total = (1 - self.interctc_weight) * loss_ctc + self.interctc_weight * loss_inter
        else:
            loss_ctc_total = loss_ctc

        ys_in, ys_out = add_sos_eos(targets, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in, self.ignore_id)
        pred, _ = self.decoder(ys_in, ys_mask, encoded, mask)
        loss_att = self.criterion(pred.float(), ys_out)

        loss = self.ctc_weight * loss_ctc_total + (1 - self.ctc_weight) * loss_att
        acc = th_accuracy(pred.reshape(-1, self.odim), ys_out, ignore_label=self.ignore_id)
        return {
            "loss": loss,
            "loss_ctc": loss_ctc.detach(),
            "loss_att": loss_att.detach(),
            "loss_interctc": loss_inter.detach(),
            "acc": acc,
        }

    # -- decoding ---------------------------------------------------------

    def scorers(self):
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    @torch.no_grad()
    def ctc_greedy(self, encoded, lengths):
        """Best-path CTC decoding. Returns a list of token-id lists."""
        best = self.ctc.argmax(encoded)
        results = []
        for path, length in zip(best, lengths):
            path = torch.unique_consecutive(path[:length])
            results.append([int(t) for t in path if t != 0])
        return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
