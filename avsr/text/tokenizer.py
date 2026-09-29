"""Tokenizers that follow the ESPnet id layout used by the model.

Id 0 is the CTC blank, the last id is the shared <sos>/<eos>, and -1 is the
padding/ignore id for targets.
"""

import os

import torch

BLANK = "<blank>"
UNK = "<unk>"
EOS = "<eos>"
IGNORE_ID = -1


class _BaseTokenizer:
    token_list: list

    @property
    def vocab_size(self):
        return len(self.token_list)

    @property
    def eos_id(self):
        return len(self.token_list) - 1

    def decode(self, token_ids):
        raise NotImplementedError

    def _clean_ids(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [int(i) for i in token_ids if 0 < int(i) < self.eos_id]


class CharTokenizer(_BaseTokenizer):
    """Character-level tokenizer; space is an explicit token."""

    SPACE = "<space>"

    def __init__(self, alphabet):
        chars = [c for c in dict.fromkeys(alphabet) if c != " "]
        self.token_list = [BLANK, UNK, self.SPACE] + chars + [EOS]
        self._index = {t: i for i, t in enumerate(self.token_list)}

    def encode(self, text):
        unk = self._index[UNK]
        ids = [self._index[self.SPACE] if c == " " else self._index.get(c, unk) for c in text]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids):
        pieces = [self.token_list[i] for i in self._clean_ids(token_ids)]
        text = "".join(" " if p == self.SPACE else "" if p == UNK else p for p in pieces)
        return " ".join(text.split())


class SentencePieceTokenizer(_BaseTokenizer):
    """SentencePiece (unigram/BPE) tokenizer.

    SentencePiece ids are shifted by one so that 0 stays free for the CTC
    blank; the SentencePiece model itself is trained without bos/eos/pad.
    """

    def __init__(self, model_path):
        import sentencepiece

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.spm = sentencepiece.SentencePieceProcessor(model_file=model_path)
        pieces = [self.spm.id_to_piece(i) for i in range(self.spm.get_piece_size())]
        self.token_list = [BLANK] + pieces + [EOS]

    def encode(self, text):
        ids = [i + 1 for i in self.spm.encode(text, out_type=int)]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids):
        ids = [i - 1 for i in self._clean_ids(token_ids)]
        return " ".join(self.spm.decode(ids).split())


def train_sentencepiece(texts, model_prefix, vocab_size, model_type="unigram"):
    """Train a SentencePiece model on an iterable of normalised sentences."""
    import sentencepiece

    os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        # Small corpora cannot always fill the requested vocabulary.
        hard_vocab_limit=False,
    )
    return model_prefix + ".model"


def build_tokenizer(cfg):
    """Build a tokenizer from the ``tokenizer`` config node."""
    if cfg.type == "char":
        from avsr.text.normalize import FOREIGN_LETTERS, TURKISH_ALPHABET

        return CharTokenizer(TURKISH_ALPHABET + FOREIGN_LETTERS)
    if cfg.type in ("unigram", "bpe"):
        return SentencePieceTokenizer(cfg.model_path)
    raise ValueError(f"Unknown tokenizer type: {cfg.type}")
