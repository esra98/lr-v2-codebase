"""Detection of recurrent (hallucinated) transcripts.

Speech recognisers that hallucinate tend to get stuck: the same sentence is
emitted for clip after clip, or one word is repeated inside a transcript.
Genuine repeated phrases ("TEŞEKKÜR EDERIM") look different: they are spread
over the recordings instead of bunched together. The rules here rely on that
difference, so frequent genuine phrases are kept.
"""

import zlib
from collections import defaultdict
from dataclasses import dataclass

from avsr.utils.metrics import edit_distance

BURST = "recurrent_burst"
SENTENCE = "recurrent_sentence"
LOOP = "internal_loop"


@dataclass
class RecurrenceConfig:
    # Two clips this many positions apart or closer count as neighbours.
    window: int = 5
    # Word-level similarity from which two transcripts count as the same.
    # Genuine neighbours often differ in a single word ("IKI BIN ON SEKIZ",
    # "IKI BIN ON DOKUZ"), so short transcripts must match exactly.
    similarity: float = 0.8
    fuzzy_min_words: int = 5
    # A sentence is rejected throughout a session once it occurs there at
    # least this often and at least this share of those occurrences lie in
    # bursts. Other sessions are not affected: a phrase that loops in one
    # recording can be genuine in another.
    sentence_min_count: int = 3
    sentence_burst_share: float = 0.5
    # Consecutive repetitions of a group of 1 to 4 words that make a loop.
    loop_repeats: int = 3
    # Text that compresses better than this is almost pure repetition.
    max_compression_ratio: float = 2.4
    compression_min_chars: int = 40


def similar(a, b, cfg):
    """Whether two word lists are the same transcript up to small differences."""
    if a == b:
        return True
    if min(len(a), len(b)) < cfg.fuzzy_min_words:
        return False
    longest = max(len(a), len(b))
    # The distance is at least the length difference: skip hopeless pairs.
    if abs(len(a) - len(b)) > (1 - cfg.similarity) * longest:
        return False
    return 1 - edit_distance(a, b) / longest >= cfg.similarity


def has_internal_loop(words, cfg):
    """Whether a group of up to four words repeats several times in a row."""
    for size in range(1, 5):
        span = size * cfg.loop_repeats
        for start in range(len(words) - span + 1):
            group = words[start : start + size]
            if all(
                words[start + k * size : start + (k + 1) * size] == group for k in range(1, cfg.loop_repeats)
            ):
                return True
    return False


def compression_ratio(text):
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


def find_recurrent(sessions, cfg=None):
    """Flag recurrent transcripts.

    Args:
        sessions: dict mapping a session name to its transcripts in clip
            order, as (clip_id, normalised text) pairs. Every labelled clip
            must be included, also those that will be rejected for other
            reasons, because the rules look at neighbouring clips.

    Returns:
        dict mapping (session, clip_id) to the reason it was flagged.
    """
    cfg = cfg or RecurrenceConfig()
    flags = {}
    in_burst = set()
    occurrences = defaultdict(list)

    for session, clips in sessions.items():
        words = [text.split() for _, text in clips]
        for i, (clip_id, text) in enumerate(clips):
            if not text:
                continue
            occurrences[(session, text)].append((session, clip_id))
            looping = has_internal_loop(words[i], cfg) or (
                len(text) >= cfg.compression_min_chars and compression_ratio(text) > cfg.max_compression_ratio
            )
            if looping:
                flags[(session, clip_id)] = LOOP
            for j in range(i + 1, min(i + cfg.window, len(clips) - 1) + 1):
                if words[j] and similar(words[i], words[j], cfg):
                    in_burst.add((session, clip_id))
                    in_burst.add((session, clips[j][0]))

    for key in in_burst:
        flags.setdefault(key, BURST)

    # A sentence that mostly occurs in bursts is invented; its occurrences
    # outside bursts, in the same session, are rejected too.
    for keys in occurrences.values():
        if len(keys) < cfg.sentence_min_count:
            continue
        share = sum(key in in_burst for key in keys) / len(keys)
        if share >= cfg.sentence_burst_share:
            for key in keys:
                flags.setdefault(key, SENTENCE)
    return flags


def sentence_statistics(sessions, flags):
    """Per-sentence counts for the audit report, most frequent first."""
    stats = defaultdict(lambda: {"count": 0, "sessions": set(), "flagged": 0})
    for session, clips in sessions.items():
        for clip_id, text in clips:
            if not text:
                continue
            entry = stats[text]
            entry["count"] += 1
            entry["sessions"].add(session)
            entry["flagged"] += (session, clip_id) in flags
    rows = [
        {"text": text, "count": e["count"], "sessions": len(e["sessions"]), "flagged": e["flagged"]}
        for text, e in stats.items()
        if e["count"] > 1
    ]
    return sorted(rows, key=lambda r: (-r["count"], r["text"]))
