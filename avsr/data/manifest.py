"""Label cleaning rules, the clip table of a dataset, and train/val/test manifests.

A manifest is a TSV with one utterance per row. It names clips by session and
clip number, so it stays valid when the dataset is moved to another machine.
"""

import csv
import os
from dataclasses import asdict, dataclass, field

from avsr.data.recurrence import RecurrenceConfig
from avsr.text.normalize import fold_dotted_i

MANIFEST_FIELDS = ["utt_id", "session", "clip_id", "num_frames", "text", "mean_conf", "yaw"]

# Sentences Whisper is known to invent on silence or noise in Turkish audio
# (subtitle credits and video sign-offs from its training data), normalised.
# They are compared with İ folded into I, so they match transcripts of either
# alphabet.
HALLUCINATED_PHRASES = (
    "ALTYAZI M K",
    "ALTYAZI",
    "ABONE OL",
    "İZLEDİĞİNİZ İÇİN TEŞEKKÜR",
    "İZLEDİĞİNİZ İÇİN TEŞEKKÜRLER",
    "BİR SONRAKİ VİDEODA GÖRÜŞMEK ÜZERE",
    "SESLENDİREN",
    "ÇEVİRİ VE ALTYAZI",
)


def is_hallucinated_phrase(text, min_share=0.5):
    """Whether a known invented phrase makes up most of the transcript.

    Matching whole words and requiring a share of the transcript keeps genuine
    sentences that merely contain the words, such as a speech about
    subscriptions containing "ABONE OLAN".
    """
    words = fold_dotted_i(text).split()
    for phrase in HALLUCINATED_PHRASES:
        target = fold_dotted_i(phrase).split()
        if len(target) < min_share * len(words):
            continue
        for start in range(len(words) - len(target) + 1):
            if words[start : start + len(target)] == target:
                return True
    return False


@dataclass
class FilterConfig:
    """Thresholds for rejecting utterances with unreliable labels."""

    min_words: int = 1
    min_mean_conf: float = 0.80
    min_word_conf: float = 0.30
    drop_recurrent: bool = True
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    max_short_word_fraction: float = 0.25
    # Letters per second over the utterance's span (avsr.corpus.segment).
    min_chars_per_sec: float = 4.0
    max_chars_per_sec: float = 25.0
    drop_hallucinated_phrases: bool = True


# Columns of the clip table written next to a dataset: one row per source
# clip, with the verdict of the cleaning.
CLIP_FIELDS = [
    "session",
    "clip_id",
    "verdict",
    "split",
    "masked",
    "num_frames",
    "mean_conf",
    "detection_rate",
    "yaw",
    "mouth_motion",
    "text",
    "error",
]


@dataclass
class Row:
    utt_id: str
    session: str
    clip_id: str
    num_frames: int
    text: str
    mean_conf: float
    yaw: float

    def to_dict(self):
        d = asdict(self)
        d["mean_conf"] = f"{self.mean_conf:.4f}"
        d["yaw"] = f"{self.yaw:.4f}"
        return d


def judge_label(utt, text, recurrent_reason, cfg):
    """Reason for rejecting a clip because of its label, or None.

    Uses the label file only, so it can run before any video is opened. The
    first check that fails is reported.
    """
    if utt is None:
        return "bad_label_file"
    if len(text.split()) < cfg.min_words:
        return "empty_text"
    if cfg.drop_hallucinated_phrases and is_hallucinated_phrase(text):
        return "hallucinated_phrase"
    if cfg.drop_recurrent and recurrent_reason:
        return recurrent_reason
    if utt.short_word_fraction() > cfg.max_short_word_fraction:
        return "squeezed_words"
    if utt.mean_conf < cfg.min_mean_conf or utt.min_conf < cfg.min_word_conf:
        return "low_confidence"
    return None


def write_clip_table(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIP_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + ".tmp", path)


def read_clip_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(CLIP_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: clip table is missing columns {sorted(missing)}")
        return list(reader)


def kept_rows(clip_table, split=None):
    """Usable utterances of a clip table, optionally of one split."""
    return [
        Row(
            f"{r['session']}/{r['clip_id']}",
            r["session"],
            r["clip_id"],
            int(r["num_frames"]),
            r["text"],
            float(r["mean_conf"]),
            float(r["yaw"]),
        )
        for r in clip_table
        if r["verdict"] == "kept" and split in (None, r["split"])
    ]


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    os.replace(path + ".tmp", path)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: manifest is missing columns {sorted(missing)}")
        return [
            Row(
                r["utt_id"],
                r["session"],
                r["clip_id"],
                int(r["num_frames"]),
                r["text"],
                float(r["mean_conf"]),
                float(r["yaw"]),
            )
            for r in reader
        ]
