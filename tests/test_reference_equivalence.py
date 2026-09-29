"""The model must compute exactly what the original Auto-AVSR code computes.

Needs the reference code and the public checkpoints; skipped when either is
missing (for instance on a machine that only holds the training code).
"""

import importlib
import os
import sys

import pytest
import torch

from avsr.config import PROJECT_ROOT, load_config
from avsr.models.e2e import AVSRModel
from avsr.models.pretrained import load_pretrained

REFERENCE = os.path.join(PROJECT_ROOT, "reference-codebase", "auto_avsr")
CHECKPOINTS = {
    "video": os.path.join(PROJECT_ROOT, "pretrained", "vsr_trlrs2lrs3vox2avsp_base.pth"),
    "audio": os.path.join(PROJECT_ROOT, "pretrained", "asr_trlrs3vox2_base.pth"),
}
VOCAB = 5049
_VOCAB_LAYERS = ("decoder.embed.0.", "decoder.output_layer.", "ctc.ctc_lo.")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def reference_e2e():
    if not os.path.isdir(REFERENCE):
        pytest.skip("reference code not available")
    sys.path.insert(0, REFERENCE)
    try:
        yield importlib.import_module("espnet.nets.pytorch_backend.e2e_asr_conformer").E2E
    finally:
        sys.path.remove(REFERENCE)
        for name in [m for m in sys.modules if m == "espnet" or m.startswith("espnet.")]:
            del sys.modules[name]


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_outputs_match_the_reference_implementation(reference_e2e, kind):
    path = CHECKPOINTS[kind]
    if not os.path.isfile(path):
        pytest.skip(f"{os.path.basename(path)} not downloaded")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    reference = reference_e2e(VOCAB, kind)
    reference.load_state_dict(checkpoint)
    reference.eval()

    cfg = load_config(["configs/model/base.yaml"], [f"model.modality={kind}"])
    model = AVSRModel(VOCAB, cfg.model)
    loaded, skipped = load_pretrained(model, path, kind, "full")
    assert not skipped
    state = model.state_dict()
    state.update({k: v for k, v in checkpoint.items() if k.startswith(_VOCAB_LAYERS)})
    model.load_state_dict(state)
    model.eval()

    torch.manual_seed(0)
    if kind == "video":
        x = torch.randn(1, 40, 1, 88, 88)
        inputs = {"video": x}
    else:
        x = torch.randn(1, 40 * 640, 1)
        inputs = {"audio": x}
    tokens = torch.randint(1, VOCAB - 1, (1, 7))
    causal = torch.tril(torch.ones(7, 7, dtype=torch.bool)).unsqueeze(0)
    with torch.no_grad():
        expected = reference.encoder(reference.proj_encoder(reference.frontend(x)), None)[0]
        encoded, _, _ = model.encode(**inputs)
        assert torch.equal(encoded, expected)
        assert torch.equal(model.ctc.log_softmax(encoded), reference.ctc.log_softmax(expected))
        assert torch.equal(
            model.decoder(tokens, causal, encoded, None)[0],
            reference.decoder(tokens, causal, expected, None)[0],
        )
