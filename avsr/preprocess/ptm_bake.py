"""Writing Partition-Time Masking into stored clips.

Masking during training is the default in this project. This module exists
for datasets that must carry the masks themselves. The mask of a clip is
derived from its name and a seed, so rebuilding the dataset reproduces it.
"""

import random
import zlib

import numpy as np
import torch

from avsr.data.transforms import PartitionTimeMask


def clip_rng(seed, session, clip_id):
    """Random generator that depends only on the seed and the clip name."""
    return random.Random(zlib.crc32(f"{seed}/{session}/{clip_id}".encode("utf-8")))


def mask_frames(frames, mask, rng):
    """Apply ``mask`` to uint8 frames [T, H, W, 3] and return uint8 frames."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = mask(x, rng)
    return (x * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def build_mask(grid=4, alpha=0.2, num_partitions=2, fill="zero"):
    return PartitionTimeMask(grid=grid, alpha=alpha, num_partitions=num_partitions, fill=fill, p=1.0)
