import json
import os

import numpy as np
import pytest

from avsr.corpus import segment as seg
from avsr.corpus import sources, transcribe
from avsr.corpus.status import SessionStatus, atomic_write, options_hash
from avsr.preprocess import video_io
from avsr.text.labels import parse_label_file

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


# ---------------------------------------------------------------------------
# cut
# ---------------------------------------------------------------------------

from avsr.corpus import cut as cutter  # noqa: E402
from avsr.preprocess import landmarks as lm  # noqa: E402


def test_frame_selector_maps_30_fps_onto_25_fps():
    sel = cutter.FrameSelector(start=1.0, num_frames=5, fps=25.0, src_dt=1 / 30)
    src_times = [1.0 + i / 30 for i in range(7)]
    picks = [sel.take(t) for t in src_times]
    assert sum(picks) == 5 and sel.done
    # every output frame gets the source frame nearest to its centre time
    targets = cutter.target_times(1.0, 5, 25.0)
    chosen = [t for t, n in zip(src_times, picks) for _ in range(n)]
    for target, t in zip(targets, chosen):
        assert abs(target - t) <= 1 / 60 + 1e-9


def test_frame_source_seek_continues_at_the_right_time(tmp_path):
    video = _toy_video(tmp_path / "v.mp4", num_frames=100)  # 4 s at 25 fps
    src = cutter.FrameSource(video)
    src.seek(2.5)
    times = [t for t, _ in src.frames_between(3.0, 3.2)]
    assert len(times) == 5 and abs(times[0] - 3.0) < 0.001
    src.seek(0.5)  # seeking back works too
    assert abs(next(src.frames_between(1.0, 1.1))[0] - 1.0) < 0.001
    src.close()


def test_frame_source_hands_out_frames_once_and_allows_small_overlaps(tmp_path):
    video = _toy_video(tmp_path / "v.mp4", num_frames=60)  # 25 fps, 2.4 s
    src = cutter.FrameSource(video)
    first = [t for t, _ in src.frames_between(0.0, 1.0)]
    second = [t for t, _ in src.frames_between(0.96, 2.0)]  # overlaps by one frame
    third = [t for t, _ in src.frames_between(2.0, 9.0)]
    assert len(first) == 25 and first[0] == pytest.approx(0.0)
    assert second[0] == pytest.approx(0.96) and len(second) == 26
    assert len(third) == 10 and src.exhausted
    src.close()


def test_wav_memmap_and_audio_slice(tmp_path):
    video = _toy_video(tmp_path / "v.mp4", num_frames=50)
    wav = cutter.extract_session_audio(video, str(tmp_path / "a.wav"))
    samples = cutter.open_wav(wav)
    assert len(samples) == pytest.approx(50 * 640, abs=2500)  # AAC priming adds a few frames
    chunk = cutter.audio_slice(samples, 0.5, 10)
    assert chunk.shape == (6400,) and chunk.dtype == np.float32
    assert np.array_equal(chunk, cutter.audio_slice(samples, 0.5, 10))


def _face(cx, cy, height=40.0, opening=2.0):
    """Synthetic landmarks of a face ``height`` pixels tall centred at (cx, cy)."""
    s = height / 40.0
    pts = np.zeros((lm.NUM_LANDMARKS, 2), dtype=np.float32)
    pts[:, 0], pts[:, 1] = cx, cy
    pts[lm.FOREHEAD] = (cx, cy - 20 * s)
    pts[lm.CHIN] = (cx, cy + 20 * s)
    pts[lm.RIGHT_EYE[0]] = (cx - 10 * s, cy - 8 * s)
    pts[lm.RIGHT_EYE[1]] = (cx - 4 * s, cy - 8 * s)
    pts[lm.LEFT_EYE[0]] = (cx + 4 * s, cy - 8 * s)
    pts[lm.LEFT_EYE[1]] = (cx + 10 * s, cy - 8 * s)
    pts[lm.NOSE_TIP] = (cx, cy)
    for i in range(8):
        pts[i] = (cx - 8 * s + 2 * s * i, cy + 10 * s)
    pts[lm.INNER_LIP_UPPER] = (cx, cy + 9 * s)
    pts[lm.INNER_LIP_LOWER] = (cx, cy + 9 * s + opening)
    return pts


def test_select_speaker_prefers_the_moving_mouth_among_large_faces():
    frames = []
    for t in range(30):
        listener = _face(30, 40, height=44, opening=2.0)  # tallest, still
        speaker = _face(70, 40, height=40, opening=2.0 + 3 * (t % 2))  # talking
        tiny = _face(90, 10, height=8, opening=(t % 5))  # far away, ignored
        flicker = _face(50, 80, height=40, opening=5 * (t % 2)) if t < 5 else None  # short track
        frames.append([f for f in (listener, speaker, tiny, flicker) if f is not None])
    chosen, candidates = cutter.select_speaker(frames)
    assert candidates == 3 and all(c is not None for c in chosen)
    assert all(c[lm.NOSE_TIP][0] == 70 for c in chosen)
    # with no face at all in some frames the track still continues
    frames[10] = []
    chosen, _ = cutter.select_speaker(frames)
    assert chosen[10] is None and chosen[11][lm.NOSE_TIP][0] == 70


class _FakeDetector:
    """A face of fixed size moving slowly across the frame; misses every N-th call."""

    def __init__(self, miss_every=0):
        self.calls = 0
        self.miss_every = miss_every

    def detect_all(self, frame, t=None):
        self.calls += 1
        if self.miss_every and self.calls % self.miss_every == 0:
            return []
        h, w = frame.shape[:2]
        return [_face(w / 2 + (self.calls % 10), h / 2, opening=2 + (self.calls % 3))]


class _SquareOnlyDetector:
    """Finds the faces of a synthetic frame only when given a square image.

    Faces are listed in full-frame coordinates and mapped into the window
    the detector is given; a face is found when its centre is inside the
    window and the window is square (like MediaPipe on a 16:9 frame).
    """

    def __init__(self, faces):
        self.faces = faces  # (x, y, height) in full-frame pixels
        self.calls = []

    def detect_all(self, frame):
        h, w = frame.shape[:2]
        self.calls.append((h, w))
        if abs(h - w) > 2:
            return []
        # The window's place in the frame is written into its first two pixels.
        x0, y0, s = _read_tag(frame)
        out = []
        for x, y, height in self.faces:
            fx, fy = (x - x0 / s) * s, (y - y0 / s) * s
            if 0 <= fx < w and 0 <= fy < h:
                out.append(_face(fx, fy, height=height * s))
        return out


def _tagged_frame(h, w, x0=0, y0=0, s=1.0):
    """A frame whose first two pixels carry its offset and scale (see _SquareOnlyDetector)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[0, 0] = (x0 >> 8, x0 & 255, int(round(s * 100)))
    frame[0, 1] = (y0 >> 8, y0 & 255, 0)
    return frame


def _read_tag(frame):
    p, q = frame[0, 0].astype(int), frame[0, 1].astype(int)
    return p[0] * 256 + p[1], q[0] * 256 + q[1], p[2] / 100.0


class _TaggingFinder(cutter.FaceFinder):
    def scan(self, rgb):
        # The real scan shrinks the frame; the fake detector reads the tag instead.
        h, w = rgb.shape[:2]
        s = min(1.0, self.scan_height / float(h))
        sh, sw = int(round(h * s)), int(round(w * s))
        side = min(sh, sw)
        found = []
        for y in cutter.tile_offsets(sh, side):
            for x in cutter.tile_offsets(sw, side):
                tile = _tagged_frame(side, side, x, y, s)
                for pts in self.detector.detect_all(tile):
                    found.append((pts + np.array([x, y], dtype=np.float32)) / s)
        return found

    def redetect(self, rgb, pts):
        h, w = rgb.shape[:2]
        height = float(np.linalg.norm(pts[lm.FOREHEAD] - pts[lm.CHIN]))
        centre = (pts[lm.FOREHEAD] + pts[lm.CHIN]) / 2.0
        side = int(round(min(max(self.track_roi * height, 64.0), h, w)))
        x0 = int(round(min(max(centre[0] - side / 2.0, 0), w - side)))
        y0 = int(round(min(max(centre[1] - side / 2.0, 0), h - side)))
        window = _tagged_frame(side, side, x0, y0, 1.0)
        found = self.detector.detect_all(window)
        if not found:
            return None
        offset = np.array([x0, y0], dtype=np.float32)
        mouth = pts[lm.MOUTH_OUTER].mean(axis=0)
        best = min(found, key=lambda f: float(np.linalg.norm(f[lm.MOUTH_OUTER].mean(axis=0) + offset - mouth)))
        return best + offset


def test_tile_offsets_cover_the_length_with_half_overlap():
    assert cutter.tile_offsets(540, 540) == [0]
    assert cutter.tile_offsets(960, 540) == [0, 210, 420]
    offsets = cutter.tile_offsets(1920, 1080)
    assert offsets[0] == 0 and offsets[-1] == 840 and all(b - a <= 540 for a, b in zip(offsets, offsets[1:]))


def test_merge_faces_drops_duplicates_and_keeps_distinct_faces():
    a, b, c = _face(100, 100, height=20), _face(104, 101, height=18), _face(400, 100, height=20)
    merged = cutter.merge_faces([a, b, c])
    assert len(merged) == 2
    assert any(np.allclose(m, a) for m in merged) and any(np.allclose(m, c) for m in merged)


def test_face_finder_finds_faces_a_wide_frame_hides_and_rescans_on_a_gap():
    faces = [(960, 400, 250), (300, 500, 120)]
    detector = _SquareOnlyDetector(faces)
    finder = _TaggingFinder(detector, scan_height=540, scan_every=4)
    frame = _tagged_frame(1080, 1920)
    assert detector.detect_all(frame) == []  # the plain detector misses everything
    found = finder.detect_all(frame, t=0.0)
    assert len(found) == 2
    centres = sorted(round(float(f[lm.NOSE_TIP][0])) for f in found)
    assert centres == [300, 960]
    scans = len(detector.calls)
    for i in range(1, 4):  # tracked frames: one window per known face, no scan
        assert len(finder.detect_all(frame, t=i * 0.04)) == 2
    assert len(detector.calls) == scans + 3 * 2
    finder.detect_all(frame, t=4 * 0.04)  # fourth frame: a scan again
    assert len(detector.calls) > scans + 4 * 2
    before = len(detector.calls)
    finder.detect_all(frame, t=60.0)  # a jump in time forces a scan
    assert len(detector.calls) - before > 2



def _trial_session(tmp_path, num_frames=150):
    videos = tmp_path / "videos"
    videos.mkdir()
    video = _toy_video(videos / "s1.mp4", num_frames=num_frames)  # 6 s at 25 fps
    work = tmp_path / "work" / "s1"
    work.mkdir(parents=True)
    t = _transcript(
        (_words("Sayın başkan teşekkür ederim.", start=0.5), {}),
        (_words("Evet.", start=2.0), {}),  # dropped: one word
        (_words("Söz veriyorum efendim.", start=3.0), {}),
        (_words("Son sözüm bu kadar.", start=5.5), {}),  # runs past the end of the video
    )
    (work / "transcript.json").write_text(json.dumps(t), encoding="utf-8")
    seg.run("s1", str(work), log=lambda *_: None)
    return video, str(work)


def test_cut_session_writes_clips_labels_cache_and_resumes(tmp_path):
    video, work = _trial_session(tmp_path)
    dataset = str(tmp_path / "dataset")
    options = cutter.CutOptions(out_size=32, min_face_px=10)
    detector = _FakeDetector()
    counts = cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None)
    assert counts == {"kept": 3}
    clips = sorted(os.listdir(os.path.join(dataset, "s1")))
    assert clips == ["00000.mp4", "00000.txt", "00001.mp4", "00001.txt", "00002.mp4", "00002.txt"]
    frames = video_io.read_video(os.path.join(dataset, "s1", "00000.mp4"))
    rows = cutter.read_clip_results(os.path.join(work, "clips.tsv"))
    assert frames.shape[1:] == (32, 32, 3) and len(frames) == int(rows["00000"]["num_frames"])
    audio = video_io.read_audio(os.path.join(dataset, "s1", "00000.mp4"))
    assert abs(len(audio) - len(frames) * 640) < 2000
    label = parse_label_file(os.path.join(dataset, "s1", "00000.txt"))
    assert label.text == "Sayın başkan teşekkür ederim." and label.words[0].start == pytest.approx(0.04)
    assert set(os.listdir(os.path.join(work, "landmarks"))) == {"00000.npz", "00001.npz", "00002.npz"}
    assert not os.path.exists(os.path.join(work, "audio.wav"))
    # the last utterance ran past the end of the recording and was padded
    assert int(rows["00002"]["num_frames"]) == round((rows_end(work, "00002")) * 25)

    # resume: one clip lost, status cleared -> only that clip is cut again
    calls = detector.calls
    os.remove(os.path.join(dataset, "s1", "00001.mp4"))
    SessionStatus(work).clear("cut")
    counts = cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None)
    assert counts == {"kept": 3}
    assert os.path.isfile(os.path.join(dataset, "s1", "00001.mp4"))
    assert 0 < detector.calls - calls < 60
    assert len(cutter.read_clip_results(os.path.join(work, "clips.tsv"))) == 3
    # a third run with the same options does nothing
    assert cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None) == {"kept": 3}


class _TwoFaceDetector:
    """A tall still face on the left and a smaller talking face far to the right."""

    def __init__(self):
        self.calls = 0

    def detect_all(self, frame, t=None):
        self.calls += 1
        h = frame.shape[0]
        return [_face(60, h / 2, height=40, opening=2), _face(200, h / 2, height=24, opening=2 + (self.calls % 3))]


def test_cut_utterance_reads_the_span_again_when_the_speaker_is_not_the_tallest_face(tmp_path):
    video, work = _trial_session(tmp_path)
    big = _toy_video(tmp_path / "big.mp4", num_frames=150, size=256)
    dataset = str(tmp_path / "dataset")
    options = cutter.CutOptions(out_size=32, min_face_px=10)
    assert options.validate().keep_face_px >= 40  # the square is 3 x 40 px: the right face lies outside it
    counts = cutter.cut_session("s1", big, work, dataset, options, detector=_TwoFaceDetector(), log=lambda *_: None)
    assert counts == {"kept": 3}
    rows = cutter.read_clip_results(os.path.join(work, "clips.tsv"))
    assert all(r["error"] == "" for r in rows.values())
    frames = video_io.read_video(os.path.join(dataset, "s1", "00000.mp4"))
    assert frames.shape[1:] == (32, 32, 3)


def test_cut_session_retries_error_rows_only(tmp_path):
    video, work = _trial_session(tmp_path)
    dataset = str(tmp_path / "dataset")
    options = cutter.CutOptions(out_size=32, min_face_px=10)
    detector = _FakeDetector()
    cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None)
    path = os.path.join(work, "clips.tsv")
    lines = open(path, encoding="utf-8").read().splitlines()
    # pretend the second utterance failed
    cols = lines[2].split("\t")
    cols[1], cols[-1] = "error", "RuntimeError: x"
    lines[2] = "\t".join(cols)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    assert cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None) == {"kept": 2, "error": 1}
    calls = detector.calls
    counts = cutter.cut_session("s1", video, work, dataset, options, detector=detector, log=lambda *_: None, retry_errors=True)
    assert counts == {"kept": 3} and detector.calls > calls
    rows = cutter.read_clip_results(path)
    assert len(rows) == 3 and all(r["verdict"] == "kept" for r in rows.values())


def rows_end(work, utt_id):
    r = {x["utt_id"]: x for x in seg.read_segments(os.path.join(work, "segments.tsv"))}[utt_id]
    return float(r["duration"])


def test_cut_session_verdicts_for_missing_and_small_faces(tmp_path):
    video, work = _trial_session(tmp_path)
    dataset = str(tmp_path / "dataset")
    counts = cutter.cut_session("s1", video, work, dataset, cutter.CutOptions(out_size=32, min_face_px=100),
                                detector=_FakeDetector(), log=lambda *_: None)
    assert counts == {"small_face": 3}
    SessionStatus(work).clear("cut")
    os.remove(os.path.join(work, "clips.tsv"))
    counts = cutter.cut_session("s1", video, work, dataset, cutter.CutOptions(out_size=32, min_face_px=10),
                                detector=_FakeDetector(miss_every=2), log=lambda *_: None)
    assert set(counts) <= {"no_face", "low_detection_rate"} and sum(counts.values()) == 3


# ---------------------------------------------------------------------------
# build and verify
# ---------------------------------------------------------------------------

from avsr.corpus import build as builder  # noqa: E402


def test_build_and_verify_toy_dataset(tmp_path):
    video, work = _trial_session(tmp_path)
    # a second session that is held out as the test set
    work2 = tmp_path / "work" / "s2"
    work2.mkdir()
    import shutil

    shutil.copy(os.path.join(work, "transcript.json"), work2 / "transcript.json")
    seg.run("s2", str(work2), log=lambda *_: None)
    dataset = str(tmp_path / "dataset")
    options = cutter.CutOptions(out_size=32, min_face_px=10)
    for s, w in (("s1", work), ("s2", str(work2))):
        cutter.cut_session(s, video, w, dataset, options, detector=_FakeDetector(), log=lambda *_: None)

    rows = builder.build(str(tmp_path / "work"), dataset, holdout=["s2"], val_fraction=0.34, log=lambda *_: None)
    by = {(r["session"], r["clip_id"]): r for r in rows}
    assert {r["verdict"] for r in rows} == {"kept"}
    assert [by[("s1", c)]["split"] for c in ("00000", "00001", "00002")] == ["train", "train", "val"]
    assert all(r["split"] == "test" for r in rows if r["session"] == "s2")
    assert by[("s1", "00000")]["text"] == "SAYIN BAŞKAN TEŞEKKÜR EDERİM" and by[("s1", "00000")]["masked"] == "no"
    assert os.path.isfile(dataset + ".clips.tsv") and os.path.isfile(dataset + ".settings.json")
    for split in ("train", "val", "test"):
        assert os.path.isfile(os.path.join(dataset + "-manifests", f"{split}.tsv"))

    messages = []
    failed = builder.verify(dataset, sample=5, log=messages.append)
    assert failed == [], "\n".join(messages)

    # a stray file and a missing clip are found
    open(os.path.join(dataset, "s1", "junk.txt"), "w").close()
    os.remove(os.path.join(dataset, "s2", "00000.mp4"))
    failed = builder.verify(dataset, sample=5, log=lambda *_: None)
    assert "no file that is not a kept clip" in failed and "every kept clip has mp4 and txt" in failed
