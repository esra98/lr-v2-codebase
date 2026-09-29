#!/usr/bin/env python
"""Train a SentencePiece tokenizer on the training manifest."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avsr.data.manifest import read_manifest  # noqa: E402
from avsr.text.tokenizer import SentencePieceTokenizer, train_sentencepiece  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/train.tsv")
    parser.add_argument("--out-dir", default="data/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--type", default="unigram", choices=["unigram", "bpe"])
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    texts = [r.text for r in rows]
    prefix = os.path.join(args.out_dir, f"{args.type}{args.vocab_size}")
    model_path = train_sentencepiece(texts, prefix, args.vocab_size, args.type)

    tokenizer = SentencePieceTokenizer(model_path)
    lengths = [len(tokenizer.encode(t)) for t in texts]
    too_long = sum(n > r.num_frames for n, r in zip(lengths, rows))
    print(f"model            : {model_path}")
    print(f"vocabulary       : {tokenizer.vocab_size} (requested {args.vocab_size}; includes <blank> and <eos>)")
    print(f"tokens/utterance : mean {sum(lengths) / len(lengths):.1f}, max {max(lengths)}")
    print(f"tokens/word      : {sum(lengths) / sum(len(t.split()) for t in texts):.2f}")
    print(f"more tokens than frames: {too_long} utterances")


if __name__ == "__main__":
    main()
