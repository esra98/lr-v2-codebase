"""Training-time augmentation and input normalisation.

Video tensors are [T, C, H, W] and audio tensors are [S, 1]. Every random
decision is drawn once per clip, so all frames of a clip stay consistent.
"""

import math
import os
import random

import torch
import torch.nn.functional as F

from avsr.preprocess.video_io import SAMPLES_PER_FRAME

# Grey-level statistics of the Auto-AVSR training crops. Keeping them makes
# inputs compatible with the public pretrained front-ends.
VIDEO_MEAN = 0.421
VIDEO_STD = 0.165

DEFAULT_NOISE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "babble_noise.wav",
)


def _fill_value(x, fill):
    if fill == "zero":
        return torch.zeros((), dtype=x.dtype)
    if fill == "mean":
        return x.mean()
    raise ValueError(f"Unknown fill mode: {fill}")


class AdaptiveTimeMask:
    """Time masking with a number of masks proportional to the clip length.

    One mask per ``stride`` samples, each covering a uniformly drawn length of
    at most ``max_len`` samples (Ma et al. 2022: one mask per second, at most
    0.4 s each).
    """

    def __init__(self, max_len, stride, fill="zero"):
        self.max_len = max_len
        self.stride = stride
        self.fill = fill

    def __call__(self, x):
        length = x.size(0)
        if length < 2:
            return x
        x = x.clone()
        value = _fill_value(x, self.fill)
        for _ in range(math.ceil(length / self.stride)):
            # Never mask more than half of a clip with a single mask.
            span = random.randint(0, min(self.max_len, length // 2))
            if span == 0:
                continue
            start = random.randint(0, length - span)
            x[start : start + span] = value
        return x


class PartitionTimeMask:
    """Partition-Time Masking (PTM).

    Picks one temporal span covering ``alpha`` of the clip. Inside that span
    every frame is divided into a ``grid`` x ``grid`` layout and
    ``num_partitions`` randomly chosen cells are masked, with a new choice of
    cells for each frame.
    """

    def __init__(self, grid=4, alpha=0.2, num_partitions=2, fill="zero", p=1.0):
        if not 0 < num_partitions <= grid * grid:
            raise ValueError("num_partitions must be between 1 and grid*grid")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.grid = grid
        self.alpha = alpha
        self.num_partitions = num_partitions
        self.fill = fill
        self.p = p

    def __call__(self, x, rng=None):
        """Mask ``x`` of shape [T, C, H, W].

        ``rng`` is a ``random.Random``; pass one to make the mask reproducible,
        as when the masks are written into a stored dataset.
        """
        rng = rng or random
        length, _, height, width = x.shape
        if length == 0 or rng.random() > self.p:
            return x
        x = x.clone()
        value = _fill_value(x, self.fill)
        span = min(length, max(1, round(self.alpha * length)))
        start = rng.randint(0, length - span)
        # Cell edges via linspace so grids that do not divide the frame size
        # still cover every pixel.
        ys = torch.linspace(0, height, self.grid + 1).round().long().tolist()
        xs = torch.linspace(0, width, self.grid + 1).round().long().tolist()
        for t in range(start, start + span):
            for cell in rng.sample(range(self.grid * self.grid), self.num_partitions):
                row, col = divmod(cell, self.grid)
                x[t, :, ys[row] : ys[row + 1], xs[col] : xs[col + 1]] = value
        return x


def _check_crop(size, height, width):
    if height < size or width < size:
        raise ValueError(
            f"augment.video.crop_size={size} is larger than the stored {width}x{height} crops; "
            "lower it, or set it to 0 to use the stored frames as they are"
        )


class RandomCropFlip:
    """Random crop and horizontal flip, identical for all frames of a clip.

    A size of 0 keeps the whole frame, so only the flip is applied.
    """

    def __init__(self, size, flip_p=0.5):
        self.size = size
        self.flip_p = flip_p

    def __call__(self, x):
        if self.size:
            height, width = x.shape[-2:]
            _check_crop(self.size, height, width)
            top = random.randint(0, height - self.size)
            left = random.randint(0, width - self.size)
            x = x[..., top : top + self.size, left : left + self.size]
        if random.random() < self.flip_p:
            x = x.flip(-1)
        return x


class CenterCrop:
    """Centre crop; a size of 0 keeps the whole frame."""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        if not self.size:
            return x
        height, width = x.shape[-2:]
        _check_crop(self.size, height, width)
        top, left = (height - self.size) // 2, (width - self.size) // 2
        return x[..., top : top + self.size, left : left + self.size]


class AddNoise:
    """Mix a random segment of a noise recording into the speech at a random SNR."""

    def __init__(self, noise_path=DEFAULT_NOISE_PATH, snr_levels=(-5, 0, 5, 10, 15, 20, None)):
        from avsr.preprocess.video_io import load_audio_file

        self.noise = torch.from_numpy(load_audio_file(noise_path))
        self.snr_levels = list(snr_levels)

    def __call__(self, speech):
        snr = random.choice(self.snr_levels)
        if snr is None:  # clean
            return speech
        length = speech.size(0)
        noise = self.noise
        if noise.numel() < length:
            noise = noise.repeat(math.ceil(length / noise.numel()))
        start = random.randint(0, noise.numel() - length)
        noise = noise[start : start + length].unsqueeze(1)
        speech_power = speech.pow(2).mean()
        noise_power = noise.pow(2).mean()
        if speech_power <= 0 or noise_power <= 0:
            return speech
        scale = torch.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
        return speech + scale * noise


def speed_perturb(video, audio, factor):
    """Play a clip ``factor`` times faster, keeping audio and video in sync.

    Either input may be None. The new clip has ``round(T / factor)`` frames
    and exactly SAMPLES_PER_FRAME audio samples per frame.
    """
    if factor == 1.0:
        return video, audio
    if video is not None:
        old_frames = video.size(0)
    else:
        old_frames = audio.size(0) // SAMPLES_PER_FRAME
    new_frames = max(1, round(old_frames / factor))
    if video is not None:
        # [T, C, H, W] -> interpolate along time only.
        v = video.permute(1, 0, 2, 3).unsqueeze(0)
        v = F.interpolate(v, size=(new_frames,) + tuple(video.shape[-2:]), mode="trilinear", align_corners=False)
        video = v.squeeze(0).permute(1, 0, 2, 3)
    if audio is not None:
        a = audio.t().unsqueeze(0)
        a = F.interpolate(a, size=new_frames * SAMPLES_PER_FRAME, mode="linear", align_corners=False)
        audio = a.squeeze(0).t()
    return video, audio


def to_grayscale(video):
    """uint8 RGB [T, H, W, 3] -> float grey [T, 1, H, W] in [0, 1]."""
    video = video.float() / 255.0
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=video.dtype)
    return (video * weights).sum(-1).unsqueeze(1)


class VideoTransform:
    """Full video pipeline for one subset ("train", "val" or "test")."""

    def __init__(self, subset, cfg):
        self.train = subset == "train"
        self.crop = RandomCropFlip(cfg.crop_size, cfg.flip_p) if self.train else CenterCrop(cfg.crop_size)
        self.time_mask = None
        self.ptm = None
        if self.train and cfg.time_mask.enabled:
            self.time_mask = AdaptiveTimeMask(cfg.time_mask.max_frames, cfg.time_mask.stride_frames, cfg.time_mask.fill)
        if self.train and cfg.ptm.enabled:
            self.ptm = PartitionTimeMask(cfg.ptm.grid, cfg.ptm.alpha, cfg.ptm.num_partitions, cfg.ptm.fill, cfg.ptm.p)

    def __call__(self, video):
        """``video`` is float grey [T, 1, H, W] in [0, 1]."""
        video = self.crop(video)
        if self.ptm is not None:
            video = self.ptm(video)
        if self.time_mask is not None:
            video = self.time_mask(video)
        return (video - VIDEO_MEAN) / VIDEO_STD


class AudioTransform:
    """Full audio pipeline for one subset."""

    def __init__(self, subset, cfg, snr_target=None):
        self.train = subset == "train"
        self.time_mask = None
        self.noise = None
        if self.train and cfg.time_mask.enabled:
            self.time_mask = AdaptiveTimeMask(cfg.time_mask.max_samples, cfg.time_mask.stride_samples, "zero")
        if self.train and cfg.noise.enabled:
            self.noise = AddNoise(cfg.noise.path or DEFAULT_NOISE_PATH, cfg.noise.snr_levels)
        elif snr_target is not None:
            self.noise = AddNoise(cfg.noise.path or DEFAULT_NOISE_PATH, [snr_target])

    def __call__(self, audio):
        """``audio`` is float [S, 1]."""
        if self.time_mask is not None:
            audio = self.time_mask(audio)
        if self.noise is not None:
            audio = self.noise(audio)
        # Per-utterance normalisation to zero mean and unit variance.
        return (audio - audio.mean()) / audio.std().clamp_min(1e-8)
