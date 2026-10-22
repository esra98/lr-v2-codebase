# Deployment: full fine-tuning on a remote GPU server

How to put the project on a rented GPU server, where the dataset goes, and
how to start, watch, resume and evaluate the full fine-tuning run.

What was rehearsed locally and what was not is stated in section 10.

**Note (2026-10-22).** Written for the first corpus. The current dataset is
`dataset/tr-lrs-v3` with `dataset/tr-lrs-v3-manifests/`,
`dataset/tr-lrs-v3-tokenizer/`, `dataset/tr-lrs-v3.clips.tsv` and
`dataset/tr-lrs-v3.settings.json` next to it (14 GB, 451,954 files, no
`.aac` side files); `configs/base.yaml` already points there, so the
`data.root` and `data.manifests` overrides below are only needed when the
dataset is unpacked elsewhere. Read `data/` below as `dataset/`.

## 1. Overview

| Step | Where | Time |
|---|---|---|
| 1. Pack the dataset and the support files | local machine | 10 min |
| 2. Get the code onto the server | server | 1 min |
| 3. Upload and unpack the data | both | depends on the connection, 12 GB |
| 4. Install the software | server | 5 min |
| 5. Preflight check | server | 1 min |
| 6. Measure the batch size the GPU holds | server | 5 min |
| 7. Start the fine-tuning | server | days |
| 8. Evaluate | server | under an hour |

## 2. Folder layout on the server

Everything lives in one project folder. The code comes from git; the data,
the pretrained weights and two binary assets are not in git and are uploaded.

```
~/lr-v2-codebase/                     project folder (git clone)
  avsr/ scripts/ configs/ tests/ docs/     code, from git
  requirements.txt
  assets/                             from tr-lrs-support.tar
    babble_noise.wav                    noise for audio augmentation
    face_landmarker.task                only needed to build datasets
  data/
    tr-lrs-clean-ptm/                 from tr-lrs-clean-ptm.tar
      2024-2025-birlesim-72/
        00000.mp4  00000.aac  00000.txt
        ...
      video-1/
      ...                               33 session folders
    manifests/                        from tr-lrs-support.tar
      train.tsv  val.tsv  test.tsv
    tokenizer/                        from tr-lrs-support.tar
      unigram1000.model  unigram1000.vocab
    tr-lrs-clean-ptm.clips.tsv        from tr-lrs-support.tar (record of the build)
    tr-lrs-clean-ptm.settings.json    from tr-lrs-support.tar (record of the build)
  pretrained/
    vsr_trlrs2lrs3vox2avsp_base.pth   uploaded
  exp/                                created by training
```

The paths in the configuration are relative to the project folder, so the
project can sit anywhere on the server. If the dataset has to live on
another disk, leave it there and pass its location:
`data.root=/mnt/disk/tr-lrs-clean-ptm`.

Not needed on the server: `data/raw/`, `data/cache/`, `data/sample*`,
`papers/`, `reference-codebase/`, `exp/` and `logs/` of the local machine.

## 3. What is uploaded

| File | Content | Size |
|---|---|---|
| `upload/tr-lrs-clean-ptm.tar` | the dataset folder, 462,573 files | 11 GB |
| `upload/tr-lrs-support.tar` | `assets/`, `data/manifests/`, `data/tokenizer/`, the clip table and the build settings | 56 MB |
| `upload/vsr_trlrs2lrs3vox2avsp_base.pth` | Auto-AVSR lip-reading weights | 1 GB |
| `upload/SHA256SUMS` | checksums of the three files | |

The dataset is packed into one file because uploading 462,573 small files
one by one is slow with every tool. It is not compressed: video and audio
are compressed already.

The files are made on the local machine, in the project folder, with:

```bash
mkdir -p upload
tar -cf upload/tr-lrs-support.tar assets data/manifests data/tokenizer \
    data/tr-lrs-clean-ptm.clips.tsv data/tr-lrs-clean-ptm.settings.json
tar -cf upload/tr-lrs-clean-ptm.tar -C data tr-lrs-clean-ptm
cp pretrained/vsr_trlrs2lrs3vox2avsp_base.pth upload/
(cd upload && sha256sum *.tar *.pth > SHA256SUMS)
```

`upload/` is ignored by git and can be deleted after the upload.

For audio or audio-visual fine-tuning, `pretrained/asr_trlrs3vox2_base.pth`
is uploaded the same way.

## 4. Code on the server

```bash
git clone <repository URL> ~/lr-v2-codebase
cd ~/lr-v2-codebase
```

## 5. Data on the server

Upload the content of `upload/` into `~/lr-v2-codebase/upload/`, either
directly

```bash
# on the local machine
rsync -av --progress upload/ <user>@<server>:~/lr-v2-codebase/upload/
```

or through a drive: upload the four files to the drive from the local
machine and download them on the server into `~/lr-v2-codebase/upload/`
(for example with `rclone copy`).

Then, on the server, in the project folder:

```bash
cd upload && sha256sum -c SHA256SUMS && cd ..      # all three must say OK
tar -xf upload/tr-lrs-support.tar                  # assets/, data/manifests/, data/tokenizer/
tar -xf upload/tr-lrs-clean-ptm.tar -C data        # data/tr-lrs-clean-ptm/
mkdir -p pretrained && mv upload/vsr_trlrs2lrs3vox2avsp_base.pth pretrained/
rm upload/*.tar                                    # frees 11 GB
```

Check:

```bash
ls data/tr-lrs-clean-ptm | wc -l                   # 33
find data/tr-lrs-clean-ptm -type f | wc -l         # 462573
```

## 6. Software

```bash
python3 -m venv .venv && source .venv/bin/activate
nvidia-smi                       # note the CUDA version the driver supports
```

If the server image already has a working PyTorch, keep it and create the
environment with `--system-site-packages`. Otherwise install PyTorch first,
choosing a build the driver supports, and then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Plain `pip install torch` installs the newest CUDA build, which needs a
recent driver. In the rehearsal it installed a CUDA 13.0 build on a machine
whose driver supports CUDA 12.8, and PyTorch then refused to use the GPU.

Versions the code was run with:

| | Development machine | Clean environment |
|---|---|---|
| Python | 3.13.3 | 3.10.18 |
| PyTorch | 2.6.0 | 2.14.0 |
| PyTorch Lightning | 2.6.1 | 2.6.6 |
| PyAV | 16.1.0 | 17.1.0 |
| NumPy | 2.2.6 | 2.2.6 |

Optional, about 10 minutes: `python -m pytest`. Tests that need the
reference code or the dataset-building packages are skipped.

## 7. Check the server

```bash
python scripts/preflight.py -c configs/finetune/vsr_base.yaml
```

It checks the software, the GPU, that every clip of the manifests is on
disk, that clips decode and decode fast enough, the tokenizer, the
pretrained weights, the free disk space, and then trains for 20 steps on
real data. It ends with `READY TO TRAIN` or a list of failed checks.

Then find the largest batch that fits the GPU:

```bash
python scripts/benchmark_memory.py -c configs/finetune/vsr_base.yaml \
    --batch-frames 600 900 1200 1800 2400
```

Use the largest value whose peak stays below about 80 % of the GPU memory.
The benchmark uses batches of one shape; real batches vary, so training
needs somewhat more (4.4 GB in the benchmark became 4.7 GB in a rehearsal).

Auto-AVSR used 1800 frames per batch. When fewer fit, keep
(frames per batch) x (accumulated batches) at 1800:

| Fits | Settings |
|---|---|
| 1800 | `data.max_frames_per_batch=1800 train.accumulate_grad_batches=1` |
| 900 | `data.max_frames_per_batch=900 train.accumulate_grad_batches=2` |
| 600 | `data.max_frames_per_batch=600 train.accumulate_grad_batches=3` |

`data.max_frames_per_batch` must not be below `data.max_frames_per_clip`
(300); the configuration is rejected otherwise.

## 8. Fine-tune

```bash
nohup python scripts/train.py -c configs/finetune/vsr_base.yaml \
    exp_name=vsr_finetune data.max_frames_per_batch=1800 data.num_workers=12 \
    > vsr_finetune.log 2>&1 &
```

`nohup` keeps the run alive when the connection to the server closes.
`tmux` or `screen` do the same.

### The recipe

| Setting | Value | Source |
|---|---|---|
| Model | 3D convolution and ResNet-18, 12 Conformer blocks of width 768, 6 decoder blocks, 241M parameters | Auto-AVSR |
| Starting weights | `vsr_trlrs2lrs3vox2avsp_base.pth`: front-end, Conformer and decoder blocks, 762 tensors | Auto-AVSR, trained on 3,291 h |
| New layers | token embedding, decoder output, CTC output, 2.3M parameters | the vocabulary is Turkish |
| Trained parts | everything | |
| Learning rate, new layers | 0.001 | Auto-AVSR |
| Learning rate, pretrained parts | 0.00002 (`train.pretrained_lr_scale=0.02`) | this project, section 9 of `docs/project_report.md` |
| Pretrained parts held still | first epoch (`train.hold_pretrained_epochs=1`) | this project |
| Optimiser | AdamW, betas 0.9 and 0.98, weight decay 0.03 | Auto-AVSR |
| Schedule | 5 warm-up epochs, cosine decay, 75 epochs | Auto-AVSR |
| Frames per batch | 1800 | Auto-AVSR |
| Loss | 0.1 CTC + 0.9 attention, label smoothing 0.1 | Auto-AVSR |
| Vocabulary | 1000 SentencePiece unigram units | |
| Video during training | random 88x88 crop of the stored 96x96 frames, adaptive time masking; Partition-Time Masking is already in the clips | |

The two learning rates are what makes fine-tuning work. With 0.001 for
everything the model stopped using its input and the character error rate
stayed above 0.93; see `docs/findings.md`.

The schedule is fixed by `train.max_epochs` at the start, so the number of
epochs has to be chosen before the run begins, not during it.

### Optional: a short pilot before the full run

The learning rate of the pretrained parts was chosen in short runs with
small batches on the development GPU. It has not been compared at 1800
frames per batch. A pilot of an hour or two settles it before days of GPU
time are spent:

```bash
for scale in 0.02 0.1; do
  python scripts/train.py -c configs/finetune/vsr_base.yaml \
      exp_name=pilot_scale_$scale train.pretrained_lr_scale=$scale \
      data.limit_train=20000 data.limit_val=1000 \
      train.max_epochs=4 train.warmup_epochs=0.5 train.hold_pretrained_epochs=0.25
done
grep -H "val/cer" exp/pilot_scale_*/metrics.json
```

Use the value with the lower validation character error rate.

### What to expect

| Moment | Expected |
|---|---|
| Start of the log | `Loaded 762 tensors`, then four lines with the parameter groups and their learning rates |
| First epoch | only the new layers change; the loss falls slowly |
| `val/cer` | falls from epoch to epoch. In the local trial it was 0.73 after 2,373 updates on 8,000 clips |
| `val/cer` stays above 0.9 for 5 epochs | the run is not learning: stop it |

How long an epoch takes on the server is not known. The development GPU
(RTX 2060) processes about 1,000 frames per second with this model, which
would be 2.4 hours per epoch of the full training set (8.6 million frames).

### If the run stops

```bash
python scripts/train.py -c configs/finetune/vsr_base.yaml <same options> \
    --resume exp/vsr_finetune/checkpoints/last.ckpt
```

`last.ckpt` is written after every epoch and holds the full training state.
The metrics logged before the resume are kept as `csv/metrics.part1.csv`,
`metrics.part2.csv` and so on.

### What to watch

Progress is in `exp/vsr_finetune/csv/metrics.csv`, and in TensorBoard with
`tensorboard --logdir exp/vsr_finetune/tb`.

| Signal | Where | Meaning |
|---|---|---|
| `val/cer`, `val/wer` | metrics | error rates on the validation set with greedy decoding |
| `lr-AdamW/new`, `lr-AdamW/pretrained` | metrics | the two learning rates |
| `train/amp_scale` keeps falling | metrics | gradients overflow in 16-bit precision; restart with `train.precision=bf16-mixed` |
| `train/oom_batches` above 0 | metrics, log | some batches did not fit and were skipped; lower `data.max_frames_per_batch` |
| `train/ram_gb` still rising after the first hours | metrics | memory of the training process keeps growing; see section 10 |
| `Could not load ...` | log | a clip could not be read and was left out of its batch |
| Warning `lr_scheduler.step() before optimizer.step()` | log, once | the first update was skipped because of a gradient overflow; harmless |

If the batch is too large for the GPU, training does not limp along: it
stops after 10 skipped batches in a row, or when more than 5 % of an epoch
was skipped, and says which setting to lower.

### Other models

| To train | Command |
|---|---|
| Speech recognition from audio | `-c configs/finetune/asr_base.yaml` (needs `pretrained/asr_trlrs3vox2_base.pth`) |
| Smaller models from scratch | `-c configs/model/small.yaml model.modality=video` |
| Audio-visual, as in Auto-AVSR | `model.modality=audiovisual model.fusion.type=late_mlp` |
| One model for all three inputs | `model.modality=audiovisual model.fusion.type=early_shared train.input_mode_probs=[0.4,0.2,0.4]` |

## 9. Evaluate

```bash
python scripts/evaluate.py --run exp/vsr_finetune --average 10
```

This averages the 10 kept checkpoints, decodes the test set (6,094 clips of
two sessions that are not in the training set) with beam search, and writes
the error rates and every reference and hypothesis to
`exp/vsr_finetune/eval/`.

Copy `exp/vsr_finetune/` without `checkpoints/last.ckpt` back to the local
machine before the server is released: it holds the configuration, the
metrics, the kept checkpoints and the evaluation.

## 10. Limits and what was not rehearsed

Rehearsed locally, in a clean environment with only the uploaded files:
installation, test suite, preflight, training, resume, evaluation. With the
fine-tuning configuration of this document: preflight, and one epoch of
training on 8,000 clips with 252 frames per batch.

| Not rehearsed | Consequence |
|---|---|
| The 241M-parameter model at 1800 frames per batch | memory and speed on the server are unmeasured; preflight and the memory benchmark measure them |
| Runs of many hours | the longest local run took 26 minutes |
| The fine-tuning recipe on the full training set | the local trial used 8,000 of 140,823 clips |

- **One GPU.** Training on several GPUs is not supported and is rejected by
  the configuration. A server with one large GPU is the right choice.
- **Disk.** A kept epoch of the base model is 1 GB, `last.ckpt` 3 GB; 10
  kept epochs and `last.ckpt` need about 13 GB, the dataset 11 GB, and the
  tar file another 11 GB until it is deleted. 50 GB are enough.
- **Memory over long runs is not settled.** In a 26-minute rehearsal the
  training process grew from 2.4 to 2.9 GB, and the growth slowed from
  1.6 GB per hour at the start to 0.2 GB per hour at the end, without
  becoming flat. Choose a server with at least 32 GB of RAM and look at
  `train/ram_gb` after the first hours. The loader processes use about
  0.5 GB each and are started anew for every epoch.
