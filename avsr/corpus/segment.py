"""From a Whisper transcript to the list of utterances to cut.

Input is ``work/<session>/transcript.json`` in the ``whisper_timestamped``
layout: ``segments``, each with ``words`` (``text``, ``start``, ``end``,
``confidence``) and the decoder statistics ``no_speech_prob``,
``avg_logprob`` and ``compression_ratio``.

Output is ``work/<session>/segments.tsv``: one utterance per row with its
time span in the recording, the text in Whisper's own casing, the normalised
training text, the confidence and decoder statistics, and the verdict of the
text-only cleaning rules. Only utterances with verdict ``kept`` are cut, so
the expensive stages never touch a rejected transcript.

Splitting rules, in this order:

1. Whisper segment boundaries are never crossed.
2. A word that ends a sentence (``.``, ``?``, ``!``) ends the utterance.
3. A pause longer than ``pause_split`` seconds ends the utterance.
4. An utterance longer than ``max_duration`` is split at its longest
   internal pause, repeatedly, so no word is cut in the middle.
5. Utterances with fewer than ``min_words`` words or shorter than
   ``min_duration`` are dropped.
"""

import csv
import json
import os
from dataclasses import dataclass, field, replace

from avsr.corpus.status import atomic_write
from avsr.data.manifest import FilterConfig, judge_label
from avsr.data.recurrence import find_recurrent
from avsr.text.labels import Utterance, Word
from avsr.text.normalize import normalize_text

SEGMENT_FIELDS = [
    "utt_id", "start", "end", "duration", "num_words", "verdict",
    "mean_conf", "min_conf", "no_speech_prob", "avg_logprob", "compression_ratio",
    "text", "text_raw", "words",
]

SENTENCE_END = ".?!"


@dataclass
class SegmentConfig:
    pad: float = 0.04
    max_duration: float = 10.0
    min_duration: float = 0.4
    min_words: int = 2
    pause_split: float = 0.4
    # Whisper's own estimate that the window holds no speech. Invented
    # transcripts in this corpus have a median of 0.7 to 0.8, genuine ones
    # 0.03 with a 90th percentile of 0.2 to 0.5.
    max_no_speech_prob: float = 0.6
    fps: float = 25.0
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class Segment:
    """One utterance of the cut plan."""

    utt_id: str
    start: float
    end: float
    words: list  # dicts with text, start, end, confidence (absolute times)
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float
    verdict: str = "kept"

    @property
    def duration(self):
        return self.end - self.start

    @property
    def text_raw(self):
        return " ".join(w["text"] for w in self.words)

    @property
    def text(self):
        return normalize_text(self.text_raw)

    @property
    def mean_conf(self):
        return sum(w["confidence"] for w in self.words) / len(self.words)

    @property
    def min_conf(self):
        return min(w["confidence"] for w in self.words)

    def utterance(self):
        """The clip-relative :class:`Utterance` the cleaning rules work on."""
        words = [
            Word(w["text"], w["start"] - self.start, w["end"] - self.start, w["confidence"]) for w in self.words
        ]
        return Utterance(self.text_raw, words)

    def to_row(self):
        return {
            "utt_id": self.utt_id,
            "start": f"{self.start:.3f}",
            "end": f"{self.end:.3f}",
            "duration": f"{self.duration:.3f}",
            "num_words": len(self.words),
            "verdict": self.verdict,
            "mean_conf": f"{self.mean_conf:.3f}",
            "min_conf": f"{self.min_conf:.3f}",
            "no_speech_prob": f"{self.no_speech_prob:.3f}",
            "avg_logprob": f"{self.avg_logprob:.3f}",
            "compression_ratio": f"{self.compression_ratio:.3f}",
            "text": self.text,
            "text_raw": self.text_raw,
            # Clip-relative word times, so the clip's label file can be
            # written without the transcript: [[text, start, end, confidence], ...]
            "words": json.dumps(
                [[w["text"], round(w["start"] - self.start, 3), round(w["end"] - self.start, 3), round(w["confidence"], 3)] for w in self.words],
                ensure_ascii=False,
            ),
        }


def load_transcript(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _clean_words(segment):
    words = []
    for w in segment.get("words", []):
        text = str(w.get("text") or "").strip()
        start, end = w.get("start"), w.get("end")
        if not text or start is None or end is None or end <= start:
            continue
        words.append({"text": text, "start": float(start), "end": float(end), "confidence": float(w.get("confidence", 1.0))})
    return words


def _ends_sentence(word):
    return word["text"][-1] in SENTENCE_END


def split_words(words, cfg):
    """Split the words of one Whisper segment into utterances (lists of words)."""
    chunks, current = [], []
    for word in words:
        if current and word["start"] - current[-1]["end"] > cfg.pause_split:
            chunks.append(current)
            current = []
        current.append(word)
        if _ends_sentence(word):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    out = []
    for chunk in chunks:
        out.extend(_split_long(chunk, cfg))
    return [c for c in out if len(c) >= cfg.min_words and c[-1]["end"] - c[0]["start"] >= cfg.min_duration]


def _split_long(chunk, cfg):
    if chunk[-1]["end"] - chunk[0]["start"] <= cfg.max_duration or len(chunk) < 2:
        return [chunk]
    gaps = [chunk[i + 1]["start"] - chunk[i]["end"] for i in range(len(chunk) - 1)]
    # The longest pause, preferring one that keeps both halves within the limit.
    cut = max(range(len(gaps)), key=lambda i: gaps[i])
    return _split_long(chunk[: cut + 1], cfg) + _split_long(chunk[cut + 1 :], cfg)


def segment_transcript(transcript, cfg=None):
    """All utterances of a transcript, in time order, before any verdict."""
    cfg = cfg or SegmentConfig()
    utterances = []
    for seg in transcript.get("segments", []):
        words = _clean_words(seg)
        for chunk in split_words(words, cfg):
            utterances.append(
                Segment(
                    utt_id="",
                    start=max(0.0, chunk[0]["start"] - cfg.pad),
                    end=chunk[-1]["end"] + cfg.pad,
                    words=chunk,
                    no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                    avg_logprob=float(seg.get("avg_logprob", 0.0)),
                    compression_ratio=float(seg.get("compression_ratio", 1.0)),
                )
            )
    utterances.sort(key=lambda u: u.start)
    return [replace(u, utt_id=f"{i:05d}") for i, u in enumerate(utterances)]


def judge(utterances, session, cfg=None):
    """Set the verdict of every utterance from the transcript alone.

    The text rules of :mod:`avsr.data.manifest` come first, then Whisper's
    no-speech estimate, then the speaking rate measured on the utterance's
    own span (the video is not read here, so the clip length is the span).
    """
    cfg = cfg or SegmentConfig()
    texts = [(u.utt_id, u.text) for u in utterances]
    recurrent = find_recurrent({session: texts}, cfg.filter.recurrence)
    for u in utterances:
        reason = judge_label(u.utterance(), u.text, recurrent.get((session, u.utt_id)), cfg.filter)
        if reason is None and u.no_speech_prob > cfg.max_no_speech_prob:
            reason = "no_speech"
        if reason is None:
            rate = len(u.text.replace(" ", "")) / u.duration
            if not cfg.filter.min_chars_per_sec <= rate <= cfg.filter.max_chars_per_sec:
                reason = "speaking_rate"
        u.verdict = reason or "kept"
    return utterances


def write_segments(path, utterances):
    lines = ["\t".join(SEGMENT_FIELDS)]
    for u in utterances:
        row = u.to_row()
        lines.append("\t".join(str(row[k]) for k in SEGMENT_FIELDS))
    atomic_write(path, "\n".join(lines) + "\n")


def read_segments(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(SEGMENT_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: segments table is missing columns {sorted(missing)}")
        return list(reader)


def label_file_text(segment):
    """The ``.txt`` of a clip: LRS2 layout, text in Whisper's casing, clip-relative times."""
    lines = [f"Text:  {segment.text_raw}"]
    for w in segment.words:
        start = max(0.0, w["start"] - segment.start)
        end = max(start, w["end"] - segment.start)
        lines.append(f"{w['text']} {start:.3f} {end:.3f} {w['confidence']:.3f}")
    return "\n".join(lines) + "\n"


def label_file_text_from_row(row):
    """The same label file, from a row of ``segments.tsv``."""
    lines = [f"Text:  {row['text_raw']}"]
    for text, start, end, conf in json.loads(row["words"]):
        lines.append(f"{text} {max(0.0, start):.3f} {max(0.0, start, end):.3f} {conf:.3f}")
    return "\n".join(lines) + "\n"


def summarize(utterances):
    """Counts per verdict and kept hours, for logs and reports."""
    counts = {}
    kept_seconds = total_seconds = 0.0
    for u in utterances:
        counts[u.verdict] = counts.get(u.verdict, 0) + 1
        total_seconds += u.duration
        if u.verdict == "kept":
            kept_seconds += u.duration
    return {"counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])), "hours": total_seconds / 3600, "kept_hours": kept_seconds / 3600}


def run(session, work_dir, cfg=None, transcript_name="transcript.json", log=print):
    """Segment one session: ``transcript.json`` -> ``segments.tsv``. Returns the summary."""
    cfg = cfg or SegmentConfig()
    transcript = load_transcript(os.path.join(work_dir, transcript_name))
    utterances = judge(segment_transcript(transcript, cfg), session, cfg)
    write_segments(os.path.join(work_dir, "segments.tsv"), utterances)
    summary = summarize(utterances)
    log(
        f"{session}: {len(utterances)} utterances, kept {summary['counts'].get('kept', 0)} "
        f"({summary['kept_hours']:.2f} of {summary['hours']:.2f} h)"
    )
    return summary
