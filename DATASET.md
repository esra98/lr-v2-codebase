# The dataset this code trains on: TR-LRS v3

Naming note: this repository is the *second codebase* of the project
(`lr-v2-codebase`), but the dataset it uses is the *third* corpus,
**TR-LRS v3** (`dataset/tr-lrs-v3`, built 2026-10-22). The first corpus
(`tr-lrs-clean-ptm`, September 2026, "v2 dataset") is obsolete: its
transcripts had merged the dotted and dotless I, half of its video was
upscaled 360p, and its test session carried the transcript of another day.
Its construction is kept in `docs/archive/` for the record only.

## Where the dataset is

| Copy | Location |
|---|---|
| Working copy | `dataset/tr-lrs-v3/` on the build machine (14 GB, 451,954 files) |
| Archive | `tr-lrs-v3.tar.gz` (12.9 GB) in the Google Drive folder `16tybAQsivB4GPGB2piDDpuoSJeWToBCS`, with `tr-lrs-v3.tar.gz.sha256` beside it |
| Format sample | `samples/` in this repository: 50 clips with labels and their clip-table rows |

```
sha256  143122937a84a16748a8b22b415d3d49b993c522f44a2b8842eb46d1dda55b8e  tr-lrs-v3.tar.gz
```

The archive unpacks into `dataset/` and contains everything the code needs:

```
dataset/tr-lrs-v3/<session>/<utt>.mp4 + .txt   225,977 clips, 39 sessions
dataset/tr-lrs-v3-manifests/{train,val,test}.tsv
dataset/tr-lrs-v3-tokenizer/unigram1000.{model,vocab}, unigram500.{model,vocab}
dataset/tr-lrs-v3.clips.tsv                     all 272,258 utterances with verdict and split
dataset/tr-lrs-v3.settings.json                 stage options and split
```

```bash
sha256sum -c tr-lrs-v3.tar.gz.sha256
tar -xzf tr-lrs-v3.tar.gz -C <project folder>      # creates <project folder>/dataset/
```

## Compatibility with this codebase

`configs/base.yaml` and the fine-tuning configs already point at
`dataset/tr-lrs-v3`, its manifests and `unigram1000.model`; no override is
needed when the archive is unpacked inside the project folder. Partition-Time
Masking is applied during training (`augment.video.ptm.enabled: true`), not
written into the clips.

Checked on 2026-10-22 with `scripts/preflight.py -c configs/finetune/vsr_base.yaml -c configs/finetune/local_6gb.yaml`
on the build machine (RTX 2060):

| Check | Result |
|---|---|
| Every train, val and test clip on disk | 208,811 / 10,993 / 6,173 clips |
| 600 sampled clips decode and match the manifest | pass, 94 clips/s |
| Stored frames 96x96, training crop 88 | pass |
| Tokenizer loads; all training text covered; text survives round trip | 1,002 units, 0 unknown, 0 changed |
| Pretrained Auto-AVSR weights load | pass |
| Trial training run, base model | 20 steps, no batch out of memory |
| `python -m pytest` | 254 passed |

On a large GPU use `configs/finetune/vsr_base.yaml` alone
(`data.max_frames_per_batch: 1800`); the `local_6gb.yaml` overlay is only
for trying the run on a 6 GB card. See `docs/deployment.md` for the server
steps and `docs/dataset_construction.md` for how the dataset was built.

| | Train | Val | Test | Total |
|---|---|---|---|---|
| Clips | 208,811 | 10,993 | 6,173 | 225,977 |
| Hours | 207.8 | 10.5 | 6.4 | 224.6 |
| Sessions | 37 | 37 | 2 (`2025-2026-birlesim-3`, `yt-NPJohShU_iw`) | 39 |
