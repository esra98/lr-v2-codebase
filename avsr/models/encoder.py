"""Conformer encoder with stochastic depth and intermediate CTC outputs.

Parameter names match the Auto-AVSR encoder, so its public checkpoints load
without renaming.
"""

import torch

from avsr.espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder


class Encoder(ConformerEncoder):
    """Conformer encoder.

    Args:
        layer_drop: maximum probability of skipping a block during training
            (stochastic depth, Huang et al. 2016). The probability grows
            linearly with depth, so early blocks are almost always kept.
        interctc_layers: 1-based block indices whose outputs are also returned
            for intermediate CTC losses (Lee & Watanabe 2021).
        self_condition_dim: vocabulary size. When set, the intermediate CTC
            posteriors are projected and added back to the hidden states
            (self-conditioned CTC, Nozaki & Komatsu 2021).
        mask_conv_padding: zero padded frames before every convolution
            module, so that a clip is encoded the same alone or in a batch.
            Auto-AVSR does not do this.
    """

    def __init__(
        self,
        attention_dim,
        attention_heads,
        linear_units,
        num_blocks,
        dropout_rate=0.1,
        cnn_module_kernel=31,
        layer_drop=0.0,
        interctc_layers=(),
        self_condition_dim=None,
        mask_conv_padding=True,
    ):
        super().__init__(
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            cnn_module_kernel=cnn_module_kernel,
        )
        invalid = [i for i in interctc_layers if not 1 <= i < num_blocks]
        if invalid:
            raise ValueError(f"interctc_layers must lie in [1, {num_blocks - 1}], got {invalid}")
        for layer in self.encoders:
            layer.mask_conv_padding = mask_conv_padding
        self.num_blocks = num_blocks
        self.layer_drop = layer_drop
        self.interctc_layers = tuple(sorted(set(interctc_layers)))
        self.conditioning_layer = None
        if self_condition_dim and self.interctc_layers:
            self.conditioning_layer = torch.nn.Linear(self_condition_dim, attention_dim)

    def forward(self, xs, masks, ctc_output_layer=None):
        """Encode features.

        Args:
            xs: [B, T, D] input features.
            masks: [B, 1, T] non-padding mask, or None.
            ctc_output_layer: the CTC projection, needed for self-conditioning.

        Returns:
            (encoded [B, T, D], masks, list of intermediate outputs [B, T, D])
        """
        xs = self.embed(xs)
        intermediates = []
        for index, layer in enumerate(self.encoders, start=1):
            if self.training and self.layer_drop > 0:
                drop_prob = self.layer_drop * index / self.num_blocks
                if torch.rand(()).item() < drop_prob:
                    continue
            xs, masks = layer(xs, masks)
            if index in self.interctc_layers:
                hidden, pos_emb = xs
                normed = self.after_norm(hidden)
                intermediates.append(normed)
                if self.conditioning_layer is not None and ctc_output_layer is not None:
                    posteriors = torch.softmax(ctc_output_layer(normed).float(), dim=-1)
                    hidden = hidden + self.conditioning_layer(posteriors.to(hidden.dtype))
                    xs = (hidden, pos_emb)
        xs = self.after_norm(xs[0])
        return xs, masks, intermediates
