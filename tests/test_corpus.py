import json
import os
import shutil

import numpy as np
import pytest

from avsr.corpus import segment as seg
from avsr.corpus import sources, transcribe
from avsr.corpus.status import SessionStatus, atomic_write, options_hash
from avsr.preprocess import video_io
from avsr.text.labels import parse_label_file

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is only installed where datasets are built",
)

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_marks_stages_with_option_hashes(tmp_path):
    status = SessionStatus(str(tmp_path))
    assert not status.is_done("segment")
    status.mark("segment", {"a": 1}, kept=3)
    assert SessionStatus(str(tmp_path)).is_done("segment", {"a": 1})
    assert not SessionStatus(str(tmp_path)).is_done("segment", {"a": 2})
    assert SessionStatus(str(tmp_path)).get("segment")["kept"] == 3
    status.clear("segment")
    assert not SessionStatus(str(tmp_path)).is_done("segment")


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "a" / "b.txt"
    atomic_write(str(path), "hello")
    assert path.read_text() == "hello"
    assert os.listdir(tmp_path / "a") == ["b.txt"]


def test_options_hash_is_stable_and_order_independent():
    assert options_hash({"a": 1, "b": 2}) == options_hash({"b": 2, "a": 1})
    assert options_hash({"a": 1}) != options_hash({"a": 2})
    assert options_hash(seg.SegmentConfig()) == options_hash(seg.SegmentConfig())


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _toy_video(path, num_frames=50, size=96):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (num_frames, size, size, 3), dtype=np.uint8)
    audio = rng.normal(0, 0.1, num_frames * 640).astype(np.float32)
    video_io.write_clip(str(path), frames, audio)
    return str(path)


@needs_ffmpeg
def test_probe_and_register(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    _toy_video(videos / "yt-abc123.mp4")
    _toy_video(videos / "2025-2026-birlesim-1.mp4", 75)
    facts = sources.probe(str(videos / "yt-abc123.mp4"))
    assert (facts["width"], facts["height"], facts["fps"]) == (96, 96, 25.0)
    assert facts["duration"] == pytest.approx(2.0, abs=0.1)

    rows = sources.register(str(videos), log=lambda *_: None)
    assert [r["session"] for r in rows] == ["2025-2026-birlesim-1", "yt-abc123"]
    by = {r["session"]: r for r in rows}
    assert by["yt-abc123"]["origin"] == "youtube" and by["yt-abc123"]["url"].endswith("v=abc123")
    assert by["2025-2026-birlesim-1"]["origin"] == "tbmm"
    # a second run keeps the rows without probing again
    again = sources.register(str(videos), log=lambda *_: pytest.fail("probed again"))
    assert again == sources.read_sources(str(videos / "sources.tsv"))


def test_ytdlp_command_asks_for_h264_and_a_js_runtime():
    cmd = sources.ytdlp_command("abc", "out.%(ext)s")
    assert "--js-runtimes" in cmd and "avc1" in cmd[cmd.index("-f") + 1]
    assert cmd[-1].endswith("v=abc")


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


def _words(spec, start=0.0, conf=0.95):
    """'a b c' -> words 0.3 s long with 0.05 s gaps; '|' inserts a 1 s pause."""
    words, t = [], start
    for token in spec.split():
        if token == "|":
            t += 1.0
            continue
        words.append({"text": token, "start": round(t, 3), "end": round(t + 0.3, 3), "confidence": conf})
        t += 0.35
    return words


def _transcript(*segments):
    out = []
    for words, extra in segments:
        s = {"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["text"] for w in words),
             "no_speech_prob": 0.02, "avg_logprob": -0.2, "compression_ratio": 1.5, "words": words}
        s.update(extra)
        out.append(s)
    return {"segments": out, "language": "tr"}


def test_split_on_sentence_end_and_pause():
    cfg = seg.SegmentConfig()
    words = _words("Sayın başkan, değerli milletvekilleri. Teşekkür ederim | Söz veriyorum")
    chunks = seg.split_words(words, cfg)
    assert [[w["text"] for w in c] for c in chunks] == [
        ["Sayın", "başkan,", "değerli", "milletvekilleri."],
        ["Teşekkür", "ederim"],
        ["Söz", "veriyorum"],
    ]


def test_long_utterance_is_split_at_its_longest_pause():
    cfg = seg.SegmentConfig(max_duration=4.0, pause_split=10.0)
    words = _words("a b c d e f g h i j k l m n")  # 14 words, 4.85 s, uniform gaps
    words[6]["end"] -= 0.0  # keep uniform except one slightly longer gap after word 7
    words[7]["start"] += 0.2
    chunks = seg.split_words(words, cfg)
    assert len(chunks) == 2 and [w["text"] for w in chunks[0]][-1] == "g"
    assert all(c[-1]["end"] - c[0]["start"] <= 4.0 for c in chunks)


def test_short_and_single_word_chunks_are_dropped():
    cfg = seg.SegmentConfig()
    words = _words("Evet. | Hayır efendim.")
    chunks = seg.split_words(words, cfg)
    assert [[w["text"] for w in c] for c in chunks] == [["Hayır", "efendim."]]


def test_segment_transcript_keeps_casing_and_normalises_with_dotted_i():
    t = _transcript((_words("İstanbul'da ılık bir gün.", start=1.0), {}))
    utts = seg.segment_transcript(t)
    assert len(utts) == 1 and utts[0].utt_id == "00000"
    u = utts[0]
    assert u.text_raw == "İstanbul'da ılık bir gün."
    assert u.text == "İSTANBULDA ILIK BİR GÜN"
    assert u.start == pytest.approx(0.96) and u.end == pytest.approx(words_end(t) + 0.04)
    label = seg.label_file_text(u)
    parsed = parse_label_file(_write(label))
    assert parsed.text == u.text_raw and parsed.words[0].start == pytest.approx(0.04)


def words_end(t):
    return t["segments"][-1]["words"][-1]["end"]


def _write(text):
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


def test_judge_applies_text_rules_no_speech_and_rate():
    good = (_words("Genel kurulu saygıyla selamlıyorum."), {})
    credit = (_words("Altyazı M.K."), {})
    silent = (_words("Yine bir şey.", start=20.0), {"no_speech_prob": 0.9})
    low = (_words("Bilmem ne dedi.", start=40.0, conf=0.2), {})
    burst = [(_words("Özgür Karabat", start=60.0 + 2 * i), {}) for i in range(4)]
    t = _transcript(good, credit, silent, low, *burst)
    utts = seg.judge(seg.segment_transcript(t), "s1")
    verdicts = {u.text_raw: u.verdict for u in utts}
    assert verdicts["Genel kurulu saygıyla selamlıyorum."] == "kept"
    assert verdicts["Altyazı M.K."] == "hallucinated_phrase"
    assert verdicts["Yine bir şey."] == "no_speech"
    assert verdicts["Bilmem ne dedi."] == "low_confidence"
    assert verdicts["Özgür Karabat"] == "recurrent_burst"
    # a huge pad stretches the first utterance (1.35 s of speech) to 9.35 s: under 4 letters per second
    slow = seg.judge(seg.segment_transcript(t, seg.SegmentConfig(pad=8.0)), "s1")
    assert slow[0].verdict == "speaking_rate"


def test_segments_table_roundtrip_and_run(tmp_path):
    t = _transcript((_words("Sayın başkan teşekkür ederim."), {}), (_words("Evet efendim.", start=5.0), {}))
    work = tmp_path / "s1"
    work.mkdir()
    (work / "transcript.json").write_text(json.dumps(t), encoding="utf-8")
    summary = seg.run("s1", str(work), log=lambda *_: None)
    assert summary["counts"] == {"kept": 2}
    rows = seg.read_segments(str(work / "segments.tsv"))
    assert [r["utt_id"] for r in rows] == ["00000", "00001"]
    assert rows[0]["text"] == "SAYIN BAŞKAN TEŞEKKÜR EDERİM" and rows[0]["text_raw"] == "Sayın başkan teşekkür ederim."
    assert float(rows[1]["start"]) == pytest.approx(4.96)


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_transcribe_chunks_merge_and_resume(tmp_path):
    video = _toy_video(tmp_path / "s1.mp4", num_frames=150)  # 6 s
    calls = []

    def fake_backend(wav_path):
        calls.append(wav_path)
        assert os.path.getsize(wav_path) > 1000
        return [{"start": 0.5, "end": 1.0, "text": "bir iki", "no_speech_prob": 0.1, "avg_logprob": -0.3,
                 "compression_ratio": 1.2, "words": [{"text": "bir", "start": 0.5, "end": 0.7, "confidence": 0.9},
                                                     {"text": "iki", "start": 0.8, "end": 1.0, "confidence": 0.8}]}]

    cfg = transcribe.TranscribeConfig(backend="fake", model="m", chunk_minutes=4 / 60)
    work = str(tmp_path / "work" / "s1")
    result = transcribe.transcribe_session("s1", video, work, cfg, backend=fake_backend, log=lambda *_: None)
    assert len(calls) == 2
    starts = [s["start"] for s in result["segments"]]
    assert starts == pytest.approx([0.5, 4.5])
    assert result["segments"][1]["words"][0]["start"] == pytest.approx(4.5)
    assert sorted(os.listdir(os.path.join(work, "chunks", "fake-m"))) == ["0000.json", "0001.json"]

    # a second run does nothing; after deleting one chunk only that chunk is redone
    transcribe.transcribe_session("s1", video, work, cfg, backend=fake_backend, log=lambda *_: None)
    assert len(calls) == 2
    os.remove(os.path.join(work, "chunks", "fake-m", "0001.json"))
    SessionStatus(work).clear("transcribe")
    transcribe.transcribe_session("s1", video, work, cfg, backend=fake_backend, log=lambda *_: None)
    assert len(calls) == 3
    assert SessionStatus(work).is_done("transcribe")
