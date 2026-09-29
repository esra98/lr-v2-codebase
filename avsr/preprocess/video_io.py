"""Video/audio reading and writing built on PyAV."""

import os

import av
import numpy as np

SAMPLE_RATE = 16000
VIDEO_FPS = 25
# Audio samples per video frame; the model relies on this exact ratio.
SAMPLES_PER_FRAME = SAMPLE_RATE // VIDEO_FPS


def read_video(path):
    """Decode every frame of the first video stream as RGB uint8 [T, H, W, 3]."""
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        raise ValueError(f"{path}: video stream has no frames")
    return np.stack(frames)


# Pixel formats whose first plane is the brightness of the picture.
_LUMA_FIRST = ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "gray")
_FULL_RANGE = 2  # value of AVCOL_RANGE_JPEG


def read_video_gray(path):
    """Decode a clip as grey frames, float32 [T, H, W] in [0, 1].

    The brightness plane of the decoded picture is used directly. That gives
    the same image as converting to RGB and then to grey, to within about
    1 of 255, without the colour conversion, which costs most of the loading
    time in recent PyAV versions.
    """
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        context = stream.codec_context
        if context.pix_fmt not in _LUMA_FIRST:
            frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            return (frames.astype(np.float32) @ weights) / 255.0
        full_range = context.pix_fmt.startswith("yuvj") or int(context.color_range or 0) == _FULL_RANGE
        planes = []
        for frame in container.decode(stream):
            plane = frame.planes[0]
            rows = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
            planes.append(rows[:, : plane.width].copy())
    if not planes:
        raise ValueError(f"{path}: video stream has no frames")
    luma = np.stack(planes).astype(np.float32)
    if full_range:
        return luma / 255.0
    # Video normally stores brightness as 16 (black) to 235 (white).
    return np.clip((luma - 16.0) / 219.0, 0.0, 1.0)


def read_video_fps(path):
    with av.open(path) as container:
        rate = container.streams.video[0].average_rate
    return float(rate) if rate else 0.0


def read_audio(path, sample_rate=SAMPLE_RATE):
    """Decode the first audio stream to mono float32 in [-1, 1] at ``sample_rate``."""
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"{path}: no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"{path}: audio stream has no samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def fit_audio_to_video(audio, num_frames):
    """Trim or zero-pad audio to exactly ``num_frames * SAMPLES_PER_FRAME`` samples."""
    target = num_frames * SAMPLES_PER_FRAME
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def write_video(path, frames, fps=VIDEO_FPS, crf=18):
    """Encode RGB or grayscale uint8 frames [T, H, W(, 3)] as H.264."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {frames.dtype}")
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError(f"frame size must be even for yuv420p, got {width}x{height}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.mp4"
    try:
        with av.open(tmp, mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "medium"}
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_clip(path, frames, audio, fps=VIDEO_FPS, crf=18, sample_rate=SAMPLE_RATE):
    """Write frames and a mono float waveform into one mp4 (H.264 and AAC).

    For synthetic clips; real clips keep their original audio through
    ``write_video_with_audio``.
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError("frames must be uint8 [T, H, W, 3]")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.mp4"
    try:
        with av.open(tmp, mode="w") as container:
            video = container.add_stream("libx264", rate=fps)
            video.width, video.height = frames.shape[2], frames.shape[1]
            video.pix_fmt = "yuv420p"
            video.options = {"crf": str(crf)}
            sound = container.add_stream("aac", rate=sample_rate)
            sound.layout = "mono"
            for frame in frames:
                for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0).reshape(1, -1)
            block = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            block.sample_rate = sample_rate
            for packet in sound.encode(block):
                container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_video_with_audio(path, frames, audio_source, fps=VIDEO_FPS, crf=18):
    """Encode frames as H.264 and copy the audio stream of ``audio_source``.

    The result has the layout of the source clips: video as stream 0 and the
    original audio, bit for bit, as stream 1. Returns False when the source
    has no audio stream, in which case only video is written.
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError("frames must be uint8 [T, H, W, 3]")
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError(f"frame size must be even for yuv420p, got {width}x{height}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.mp4"
    try:
        with av.open(audio_source) as source, av.open(tmp, mode="w") as container:
            video = container.add_stream("libx264", rate=fps)
            video.width, video.height = width, height
            video.pix_fmt = "yuv420p"
            video.options = {"crf": str(crf), "preset": "medium"}
            source_audio = source.streams.audio[0] if source.streams.audio else None
            audio = container.add_stream_from_template(source_audio) if source_audio else None
            for frame in frames:
                for packet in video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            if audio is not None:
                for packet in source.demux(source_audio):
                    if packet.dts is None:  # flush marker
                        continue
                    packet.stream = audio
                    container.mux(packet)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return audio is not None


def write_audio(path, audio, sample_rate=SAMPLE_RATE):
    """Write mono float audio as 16-bit audio; the extension picks the format.

    FLAC is lossless and about half the size of wav.
    """
    import soundfile

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    try:
        soundfile.write(tmp, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_audio_file(path):
    """Read an audio file (flac or wav) as float32 [T]."""
    import soundfile

    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio
