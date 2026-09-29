import os

import numpy as np
import pytest

from avsr.config import PROJECT_ROOT, load_config
from avsr.data.manifest import Row, write_manifest
from avsr.preprocess import video_io
from avsr.text.tokenizer import train_sentencepiece

SENTENCES = [
    "SAYIN BAŞKAN DEĞERLI MILLETVEKILLERI",
    "BIR ŞEY SÖYLEMEK ISTIYORUM",
    "TEŞEKKÜR EDERIM",
    "BU KANUN TEKLIFI ÇOK ÖNEMLI",
    "GENEL KURULU SAYGIYLA SELAMLIYORUM",
    "BÜTÇE GÖRÜŞMELERI DEVAM EDIYOR",
]


def _moving_pattern(num_frames, seed):
    """Smooth synthetic clip whose motion depends on the seed."""
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 220, 96, dtype=np.float32)
    speed = int(rng.integers(1, 5))
    frames = np.stack([np.tile(np.roll(base, speed * t), (96, 1)) for t in range(num_frames)])
    return np.repeat(frames[..., None], 3, axis=-1).astype(np.uint8)


@pytest.fixture(scope="session")
def toy_data(tmp_path_factory):
    """A tiny preprocessed dataset with manifests and a tokenizer."""
    root = tmp_path_factory.mktemp("toy")
    processed = root / "dataset"
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        num_frames = int(rng.integers(30, 70))
        clip_id = f"{i:05d}"
        audio = rng.normal(0, 0.1, num_frames * 640).astype(np.float32)
        # Same layout as the real dataset: video and audio in one mp4.
        video_io.write_clip(str(processed / "s1" / f"{clip_id}.mp4"), _moving_pattern(num_frames, i), audio)
        rows.append(Row(f"s1/{clip_id}", "s1", clip_id, num_frames, SENTENCES[i % len(SENTENCES)], 0.95, 0.0))
    manifests = root / "manifests"
    write_manifest(str(manifests / "train.tsv"), rows[:16])
    write_manifest(str(manifests / "val.tsv"), rows[16:20])
    write_manifest(str(manifests / "test.tsv"), rows[20:])
    spm = train_sentencepiece(SENTENCES * 20, str(root / "tokenizer" / "unigram"), vocab_size=60)
    return {"root": str(root), "processed": str(processed), "manifests": str(manifests), "spm": spm}


@pytest.fixture
def toy_cfg(toy_data, tmp_path):
    def make(*overrides, files=()):
        base = [
            f"data.root={toy_data['processed']}",
            f"data.manifests={toy_data['manifests']}",
            f"tokenizer.model_path={toy_data['spm']}",
            "tokenizer.vocab_size=60",
            f"exp_dir={tmp_path}",
            "data.num_workers=0",
            "data.max_frames_per_batch=200",
            "data.max_frames_per_clip=200",
            "data.num_buckets=2",
            "train.accelerator=cpu",
            "train.precision=32-true",
            "augment.audio.noise.enabled=false",
            "train.save_top_k=1",
        ]
        files = [os.path.join(PROJECT_ROOT, "configs", "model", "tiny.yaml"), *files]
        return load_config(files, base + list(overrides))

    return make
