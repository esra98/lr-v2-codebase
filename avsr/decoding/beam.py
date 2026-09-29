"""Joint CTC/attention beam search."""

import torch

from avsr.espnet.nets.batch_beam_search import BatchBeamSearch
from avsr.espnet.nets.scorers.length_bonus import LengthBonus


def build_beam_search(model, token_list, beam_size=40, ctc_weight=0.1, length_bonus=0.0):
    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "length_bonus": length_bonus,
    }
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


@torch.no_grad()
def beam_decode(beam_search, encoded):
    """Decode one utterance. ``encoded`` is [T, D]. Returns token ids without sos/eos."""
    hyps = beam_search(encoded)
    if not hyps:
        return []
    tokens = hyps[0].asdict()["yseq"]
    return [int(t) for t in tokens[1:-1]]
