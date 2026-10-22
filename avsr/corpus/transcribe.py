"""Speech recognition of a full recording, in resumable chunks.

The recording is cut into chunks of ``chunk_minutes`` of audio (16 kHz mono
WAV, extracted with ffmpeg), each chunk is transcribed and saved as
``work/<session>/chunks/<model>/<index>.json``, and the chunks are merged
into ``transcript.json`` with the times shifted into the recording. A run
that is interrupted continues with the first chunk that has no file.

Every backend returns the ``whisper_timestamped`` layout, so the rest of the
pipeline does not depend on which recogniser was used:

    {"segments": [{"start", "end", "text", "no_speech_prob", "avg_logprob",
                   "compression_ratio", "words": [{"text", "start", "end", "confidence"}]}],
     "language": "tr"}
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

from avsr.corpus.sources import probe
from avsr.corpus.status import SessionStatus, atomic_json

STAGE = "transcribe"


@dataclass
class TranscribeConfig:
    backend: str = "faster_whisper"  # faster_whisper | whisper_timestamped
    model: str = "large-v3"
    language: str = "tr"
    chunk_minutes: float = 20.0
    device: str = "auto"
    # int8_float16 keeps large-v3 under 5 GB of GPU memory.
    compute_type: str = "int8_float16"
    vad: bool = True
    beam_size: int = 5

    @property
    def name(self):
        return f"{self.backend}-{self.model}".replace("/", "_")


def extract_audio(video, wav_path, start=0.0, duration=None, sample_rate=16000):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", video]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", wav_path]
    subprocess.run(cmd, check=True)
    return wav_path


# ---------------------------------------------------------------------------
# backends: wav path -> list of segments (times relative to the chunk)
# ---------------------------------------------------------------------------


class _FasterWhisper:
    def __init__(self, cfg):
        try:
            import torch  # noqa: F401 - loads the CUDA and cuDNN libraries that CTranslate2 needs
        except ImportError:
            pass
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)

    def __call__(self, wav_path):
        segments, _ = self.model.transcribe(
            wav_path,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            word_timestamps=True,
            vad_filter=self.cfg.vad,
            condition_on_previous_text=False,
        )
        out = []
        for s in segments:
            words = [
                {"text": w.word.strip(), "start": float(w.start), "end": float(w.end), "confidence": float(w.probability)}
                for w in (s.words or [])
                if w.word.strip()
            ]
            out.append(
                {
                    "start": float(s.start), "end": float(s.end), "text": s.text.strip(),
                    "no_speech_prob": float(s.no_speech_prob), "avg_logprob": float(s.avg_logprob),
                    "compression_ratio": float(s.compression_ratio), "words": words,
                }
            )
        return out


class _WhisperTimestamped:
    def __init__(self, cfg):
        import whisper_timestamped

        self.cfg = cfg
        self.wt = whisper_timestamped
        self.model = whisper_timestamped.load_model(cfg.model, device=None if cfg.device == "auto" else cfg.device)

    def __call__(self, wav_path):
        result = self.wt.transcribe(self.model, wav_path, language=self.cfg.language, beam_size=self.cfg.beam_size)
        out = []
        for s in result["segments"]:
            out.append(
                {
                    "start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip(),
                    "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                    "avg_logprob": float(s.get("avg_logprob", 0.0)),
                    "compression_ratio": float(s.get("compression_ratio", 1.0)),
                    "words": [
                        {"text": w["text"].strip(), "start": float(w["start"]), "end": float(w["end"]), "confidence": float(w.get("confidence", 1.0))}
                        for w in s.get("words", [])
                        if w["text"].strip()
                    ],
                }
            )
        return out


def make_backend(cfg):
    if cfg.backend == "faster_whisper":
        return _FasterWhisper(cfg)
    if cfg.backend == "whisper_timestamped":
        return _WhisperTimestamped(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")


# ---------------------------------------------------------------------------
# chunking and merging
# ---------------------------------------------------------------------------


def chunk_spans(duration, chunk_seconds):
    """(start, length) of every chunk covering ``duration`` seconds."""
    spans, start = [], 0.0
    while start < duration:
        spans.append((start, min(chunk_seconds, duration - start)))
        start += chunk_seconds
    return spans


def shift(segment, offset):
    s = dict(segment)
    s["start"] += offset
    s["end"] += offset
    s["words"] = [dict(w, start=w["start"] + offset, end=w["end"] + offset) for w in segment["words"]]
    return s


def merge_chunks(chunks, language):
    """Chunk results (offset, segments) -> one transcript with recording times."""
    segments = []
    for offset, segs in sorted(chunks, key=lambda c: c[0]):
        segments.extend(shift(s, offset) for s in segs)
    for i, s in enumerate(segments):
        s["id"] = i
    return {"text": " ".join(s["text"] for s in segments), "segments": segments, "language": language}


def transcribe_session(session, video, work_dir, cfg=None, backend=None, out_name="transcript.json", log=print):
    """Transcribe one recording. Returns the merged transcript.

    ``backend`` is a callable ``wav_path -> segments``; it is built from the
    configuration when not given (tests pass a fake).
    """
    cfg = cfg or TranscribeConfig()
    status = SessionStatus(work_dir)
    duration = probe(video)["duration"]
    # Keyed on the duration, not the file: a re-download at a higher
    # resolution carries the same audio and needs no new transcription.
    options = {"cfg": cfg, "duration": round(duration)}
    out_path = os.path.join(work_dir, out_name)
    if status.is_done(STAGE, options) and os.path.isfile(out_path):
        log(f"skip {session}: transcribed with {cfg.name}")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    spans = chunk_spans(duration, cfg.chunk_minutes * 60)
    chunk_dir = os.path.join(work_dir, "chunks", cfg.name)
    os.makedirs(chunk_dir, exist_ok=True)
    results = []
    for index, (start, length) in enumerate(spans):
        chunk_path = os.path.join(chunk_dir, f"{index:04d}.json")
        if os.path.isfile(chunk_path):
            with open(chunk_path, encoding="utf-8") as f:
                results.append((start, json.load(f)["segments"]))
            continue
        if backend is None:
            backend = make_backend(cfg)
        with tempfile.TemporaryDirectory(prefix="avsr-audio-") as tmp:
            wav = extract_audio(video, os.path.join(tmp, "chunk.wav"), start, length)
            segments = backend(wav)
        atomic_json(chunk_path, {"start": start, "length": length, "segments": segments})
        results.append((start, segments))
        log(f"{session}: chunk {index + 1}/{len(spans)} ({start / 3600:.2f} h), {len(segments)} segments")

    transcript = merge_chunks(results, cfg.language)
    transcript.update(backend=cfg.backend, model=cfg.model, chunk_minutes=cfg.chunk_minutes)
    atomic_json(out_path, transcript)
    status.mark(STAGE, options, model=cfg.name, segments=len(transcript["segments"]), chunks=len(spans))
    return transcript
