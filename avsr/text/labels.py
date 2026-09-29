"""Parsing and quality scoring of TR-LRS transcript files.

File format::

    Text:  MERSIN MILLETVEKILI ...
    MERSIN 0.0400 0.3400 0.8800
    MILLETVEKILI 0.3400 1.0000 0.9920

i.e. the sentence, then one ``WORD start end confidence`` line per word. The
labels are machine generated, so every utterance gets quality statistics that
the manifest builder uses to reject likely ASR hallucinations.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    text: str
    start: float
    end: float
    conf: float

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class Utterance:
    text: str
    words: List[Word] = field(default_factory=list)

    @property
    def mean_conf(self):
        return sum(w.conf for w in self.words) / len(self.words) if self.words else 0.0

    @property
    def min_conf(self):
        return min((w.conf for w in self.words), default=0.0)

    @property
    def speech_end(self):
        return max((w.end for w in self.words), default=0.0)

    @property
    def speech_start(self):
        return min((w.start for w in self.words), default=0.0)

    def short_word_fraction(self, min_sec_per_char=0.015):
        """Fraction of words that are too short to have been spoken.

        Hallucinated words are typically squeezed into one or two timestamp
        ticks regardless of their length.
        """
        if not self.words:
            return 1.0
        bad = sum(1 for w in self.words if w.duration < min_sec_per_char * max(len(w.text), 1))
        return bad / len(self.words)


def parse_label_file(path):
    """Parse one transcript file. Raises ValueError when it is malformed."""
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    lines = [line for line in lines if line.strip()]
    if not lines or not lines[0].startswith("Text:"):
        raise ValueError(f"{path}: missing 'Text:' header")
    utt = Utterance(text=lines[0][len("Text:"):].strip())
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: malformed word line {line!r}")
        try:
            start, end, conf = (float(p) for p in parts[-3:])
        except ValueError as e:
            raise ValueError(f"{path}: malformed word line {line!r}") from e
        utt.words.append(Word(" ".join(parts[:-3]), start, end, conf))
    return utt
