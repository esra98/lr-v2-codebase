"""Label cleaning rules, the clip table of a dataset, and train/val/test manifests.

A manifest is a TSV with one utterance per row. It names clips by session and
clip number, so it stays valid when the dataset is moved to another machine.
"""

import csv
import os
from dataclasses import asdict, dataclass, field

from avsr.data.recurrence import RecurrenceConfig, find_recurrent
from avsr.text.labels import parse_label_file
from avsr.text.normalize import normalize_text

MANIFEST_FIELDS = ["utt_id", "session", "clip_id", "num_frames", "text", "mean_conf", "yaw"]

# Sentences Whisper is known to invent on silence or noise in Turkish audio
# (subtitle credits and video sign-offs from its training data), normalised.
HALLUCINATED_PHRASES = (
    "ALTYAZI M K",
    "ALTYAZI",
    "ABONE OL",
    "IZLEDIĞINIZ IÇIN TEŞEKKÜR",
    "IZLEDIĞINIZ IÇIN TEŞEKKÜRLER",
    "BIR SONRAKI VIDEODA GÖRÜŞMEK ÜZERE",
    "SESLENDIREN",
    "ÇEVIRI VE ALTYAZI",
)


def is_hallucinated_phrase(text, min_share=0.5):
    """Whether a known invented phrase makes up most of the transcript.

    Matching whole words and requiring a share of the transcript keeps genuine
    sentences that merely contain the words, such as a speech about
    subscriptions containing "ABONE OLAN".
    """
    words = text.split()
    for phrase in HALLUCINATED_PHRASES:
        target = phrase.split()
        if len(target) < min_share * len(words):
            continue
        for start in range(len(words) - len(target) + 1):
            if words[start : start + len(target)] == target:
                return True
    return False


@dataclass
class FilterConfig:
    """Thresholds for rejecting utterances with unreliable labels."""

    max_frames: int = 600
    min_words: int = 1
    min_mean_conf: float = 0.80
    min_word_conf: float = 0.30
    drop_recurrent: bool = True
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    max_short_word_fraction: float = 0.25
    min_chars_per_sec: float = 4.0
    max_chars_per_sec: float = 25.0
    # Speech must end near the end of the clip; a large gap means the
    # transcript does not describe this clip.
    max_end_gap_sec: float = 1.0
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


def load_session_labels(raw_root, session):
    """Labels of one session in clip order, for every clip that has a video.

    Returns (clip_id, utterance or None, normalised text) triples. The
    utterance is None when the label file is missing or malformed.
    """
    folder = os.path.join(raw_root, session)
    clips = []
    for name in sorted(os.listdir(folder)):
        clip_id, ext = os.path.splitext(name)
        if ext != ".mp4":
            continue
        try:
            utt = parse_label_file(os.path.join(folder, clip_id + ".txt"))
            text = normalize_text(utt.text)
        except (OSError, ValueError):
            utt, text = None, ""
        clips.append((clip_id, utt, text))
    return clips


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


def judge_duration(num_frames, utt, text, cfg, fps=25.0):
    """Reason for rejecting a clip whose label does not fit its length, or None."""
    if num_frames > cfg.max_frames:
        return "too_long"
    duration = num_frames / fps
    rate = len(text.replace(" ", "")) / duration
    if not cfg.min_chars_per_sec <= rate <= cfg.max_chars_per_sec:
        return "speaking_rate"
    if abs(duration - utt.speech_end) > cfg.max_end_gap_sec:
        return "timing_mismatch"
    return None


def judge_session_labels(session, clips, cfg):
    """Label verdict of every clip of a session: dict clip_id -> reason or None."""
    recurrent = find_recurrent({session: [(clip_id, text) for clip_id, _, text in clips]}, cfg.recurrence)
    return {
        clip_id: judge_label(utt, text, recurrent.get((session, clip_id)), cfg) for clip_id, utt, text in clips
    }


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
