"""Dataset, collation and length-bucketed batching."""

import logging
import os
import random

import torch

from avsr.data.manifest import read_manifest
from avsr.data.transforms import speed_perturb
from avsr.preprocess import video_io
from avsr.preprocess.video_io import SAMPLES_PER_FRAME
from avsr.text.tokenizer import IGNORE_ID

log = logging.getLogger(__name__)

MODALITIES = ("video", "audio", "audiovisual")


class AVDataset(torch.utils.data.Dataset):
    """Loads the clips listed in a manifest from a dataset folder.

    The folder has the layout of the source corpus: ``<session>/<clip>.mp4``
    with the video and the audio in the same file.

    Items are dicts with "video" [T, 1, H, W], "audio" [T * 640, 1] (whichever
    the modality needs), "target" [L] and "utt_id". A clip that cannot be
    loaded yields None, which the collate function drops, so one corrupt file
    cannot stop a training run.
    """

    def __init__(
        self,
        manifest_path,
        data_root,
        modality,
        tokenizer,
        video_transform=None,
        audio_transform=None,
        speed_range=None,
        max_frames=None,
        limit=None,
    ):
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self.data_root = data_root
        self.modality = modality
        self.tokenizer = tokenizer
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.speed_range = speed_range

        rows = read_manifest(manifest_path)
        if max_frames:
            rows = [r for r in rows if r.num_frames <= max_frames]
        if limit:
            rows = rows[:limit]

        self.rows, self.targets, skipped = [], [], 0
        for row in rows:
            target = tokenizer.encode(row.text)
            # CTC needs at least one frame per token, plus one per repeated token.
            repeats = int((target[1:] == target[:-1]).sum()) if len(target) > 1 else 0
            shortest = row.num_frames
            if speed_range:
                shortest = int(row.num_frames / max(speed_range))
            if len(target) == 0 or len(target) + repeats > shortest:
                skipped += 1
                continue
            self.rows.append(row)
            self.targets.append(target)
        if skipped:
            log.warning("%s: skipped %d utterances with more tokens than frames", manifest_path, skipped)
        if not self.rows:
            raise ValueError(f"{manifest_path}: no usable utterances")

    @property
    def lengths(self):
        return [r.num_frames for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def clip_path(self, row):
        """The mp4 of a clip; it holds both the video and the audio."""
        return os.path.join(self.data_root, row.session, f"{row.clip_id}.mp4")

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self.clip_path(row)
        try:
            video = audio = None
            if self.modality in ("video", "audiovisual"):
                video = torch.from_numpy(video_io.read_video_gray(path)).unsqueeze(1)
            if self.modality in ("audio", "audiovisual"):
                audio = torch.from_numpy(video_io.read_audio(path)).unsqueeze(1)
            if video is not None and audio is not None:
                # Guard against off-by-one differences between the streams.
                frames = min(video.size(0), audio.size(0) // SAMPLES_PER_FRAME)
                video, audio = video[:frames], audio[: frames * SAMPLES_PER_FRAME]
            elif audio is not None:
                # The audio stream is a few milliseconds longer than the video.
                frames = min(row.num_frames, audio.size(0) // SAMPLES_PER_FRAME)
                audio = audio[: frames * SAMPLES_PER_FRAME]
            if self.speed_range:
                video, audio = speed_perturb(video, audio, random.uniform(*self.speed_range))
            item = {"target": self.targets[idx], "utt_id": row.utt_id}
            if video is not None:
                item["video"] = self.video_transform(video) if self.video_transform else video
            if audio is not None:
                item["audio"] = self.audio_transform(audio) if self.audio_transform else audio
            return item
        except Exception as e:
            log.warning("Could not load %s: %s: %s", row.utt_id, type(e).__name__, e)
            return None


def _pad(tensors, pad_value):
    lengths = torch.tensor([t.size(0) for t in tensors])
    out = tensors[0].new_full((len(tensors), int(lengths.max())) + tuple(tensors[0].shape[1:]), pad_value)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out, lengths


def collate(batch):
    """Pad a list of items into a batch. Returns None if every item failed to load."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {"utt_ids": [b["utt_id"] for b in batch]}
    out["targets"], out["target_lengths"] = _pad([b["target"] for b in batch], IGNORE_ID)
    if "video" in batch[0]:
        out["video"], out["video_lengths"] = _pad([b["video"] for b in batch], 0.0)
    if "audio" in batch[0]:
        out["audio"], out["audio_lengths"] = _pad([b["audio"] for b in batch], 0.0)
    return out


def split_batch(batch):
    """Split a collated batch into two halves, each trimmed to its own longest item."""
    size = len(batch["utt_ids"])
    if size < 2:
        raise ValueError("a batch of one item cannot be split")
    halves = []
    for part in (slice(0, size // 2), slice(size // 2, size)):
        half = {"utt_ids": batch["utt_ids"][part]}
        for key in ("targets", "video", "audio"):
            if key not in batch:
                continue
            lengths = batch[f"{key[:-1] if key == 'targets' else key}_lengths"][part]
            half[f"{key[:-1] if key == 'targets' else key}_lengths"] = lengths
            half[key] = batch[key][part][:, : int(lengths.max())]
        halves.append(half)
    return halves


class BucketBatchSampler(torch.utils.data.Sampler):
    """Groups clips of similar length into batches capped by total frames.

    Batches are limited by the padded size (longest clip times batch size),
    which is what actually determines GPU memory.

    The batches of an epoch are fixed when the sampler is created, from
    ``seed`` and ``epoch``, so ``len()`` is exactly the number of batches that
    iterating yields. That matters because the number differs a little from
    epoch to epoch, and a trainer that plans its validation from a stale
    count skips it. Create a new sampler, or call ``set_epoch``, for every
    epoch.
    """

    def __init__(self, lengths, max_frames, max_batch_size=None, num_buckets=50, shuffle=True, seed=0, epoch=0):
        self.lengths = list(lengths)
        if not self.lengths:
            raise ValueError("lengths is empty")
        if max(self.lengths) > max_frames:
            raise ValueError(
                f"Longest clip has {max(self.lengths)} frames but max_frames is {max_frames}; "
                "raise max_frames or filter long clips from the dataset."
            )
        self.max_frames = max_frames
        self.max_batch_size = max_batch_size
        self.num_buckets = max(1, num_buckets)
        self.shuffle = shuffle
        self.seed = seed
        self.set_epoch(epoch)

    def _make_batches(self, rng):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        size = max(1, len(order) // self.num_buckets)
        buckets = [order[i : i + size] for i in range(0, len(order), size)]
        batches = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            current, longest = [], 0
            for idx in bucket:
                new_longest = max(longest, self.lengths[idx])
                too_many_frames = new_longest * (len(current) + 1) > self.max_frames
                too_many_items = self.max_batch_size and len(current) >= self.max_batch_size
                if current and (too_many_frames or too_many_items):
                    batches.append(current)
                    current, new_longest = [], self.lengths[idx]
                current.append(idx)
                longest = new_longest
            if current:
                batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._make_batches(random.Random(self.seed * 100003 + epoch))

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)
