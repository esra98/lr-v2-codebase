"""Word and character error rates."""


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


class ErrorRate:
    """Accumulates WER and CER over a set of utterances.

    Rates are micro-averaged: total edits divided by total reference length,
    which is the convention used for LRS2/LRS3 results.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.word_errors = self.words = 0
        self.char_errors = self.chars = 0
        self.utterances = 0

    def update(self, reference, hypothesis):
        ref_words, hyp_words = reference.split(), hypothesis.split()
        self.word_errors += edit_distance(ref_words, hyp_words)
        self.words += len(ref_words)
        # CER ignores spaces so that word-boundary errors are not counted twice.
        ref_chars, hyp_chars = reference.replace(" ", ""), hypothesis.replace(" ", "")
        self.char_errors += edit_distance(ref_chars, hyp_chars)
        self.chars += len(ref_chars)
        self.utterances += 1

    @property
    def wer(self):
        return self.word_errors / self.words if self.words else float("nan")

    @property
    def cer(self):
        return self.char_errors / self.chars if self.chars else float("nan")
