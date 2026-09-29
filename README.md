# Turkish visual and audio-visual speech recognition

Sentence-level speech recognition from video, with or without audio, for the
TR-LRS corpus (Turkish Grand National Assembly recordings). The starting point
is Pingchuan Ma's Auto-AVSR; the goal is to improve on it, with every change
behind a switch so it can be A/B tested against the original behaviour.

One setting selects the mode:

| `model.modality` | Input | Task |
|---|---|---|
| `video` | mouth crops | VSR (lip reading) |
| `audio` | waveform | ASR |
| `audiovisual` | both | AVSR |

An audio-visual model can also be *evaluated* on video only, audio only or
both (`decode.input_mode`), which is how a single unified model serves all
three tasks.

## Layout

```
avsr/
  text/         Turkish normalisation, label parsing, tokenizers
  preprocess/   landmarks, trajectory smoothing, mouth cropping, masking, audio/video I/O
  data/         label cleaning, manifests, augmentation, batching
  models/       model, Conformer encoder extensions, pretrained-weight loading
  decoding/     joint CTC/attention beam search
  training/     Lightning modules, schedule, weight averaging
  espnet/       Transformer/Conformer building blocks from Auto-AVSR (Apache 2.0)
configs/
  base.yaml     every option, with defaults
  model/        size presets: tiny, small, base
  exp/          A/B experiment definitions
scripts/        command-line entry points (see below)
tests/          unit and end-to-end tests
docs/           literature notes and findings
```

`reference-codebase/` and `papers/` are reference material and are never
imported by the training code. The original dataset folder is only ever read.

## Setup

```bash
python3 -m venv --system-site-packages .venv      # reuses the system PyTorch
.venv/bin/pip install -r requirements-build.txt    # requirements.txt is enough for training only
.venv/bin/python -m pytest                          # about 10 minutes
```

`assets/face_landmarker.task` (MediaPipe, for building datasets) and
`assets/babble_noise.wav` (for audio augmentation) are not in git; they
travel with the data (`docs/deployment.md`). The landmark model is
downloaded from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`.

## Pipeline

```bash
# 1. Copy sessions out of the read-only source folder
python scripts/extract_raw.py --src "/path/to/tr-lrs" --dest data/raw

# 2. Audit the transcripts (labels only, about a minute)
python scripts/audit_labels.py

# 3. Build the dataset: clean, crop, mask
python scripts/build_dataset.py --out data/tr-lrs-clean-ptm \
    --holdout-sessions 2025-2026-birlesim-3 video-10 --workers 12

# 4. Check the dataset against the source
python scripts/build_manifest.py --dataset data/tr-lrs-clean-ptm
python scripts/verify_dataset.py --dataset data/tr-lrs-clean-ptm --manifests data/manifests

# 5. Tokenizer
python scripts/train_tokenizer.py --vocab-size 1000

# 6. Choose a batch size that fits the GPU
python scripts/benchmark_memory.py -c configs/model/small.yaml model.modality=video

# 7. Train
python scripts/train.py -c configs/model/small.yaml model.modality=video exp_name=vsr_small

# 8. Evaluate with beam search, averaging the kept checkpoints
python scripts/evaluate.py --run exp/vsr_small --average 10
```

Every step can be interrupted and re-run; finished work is skipped.

Any option in `configs/base.yaml` can be overridden with `key=value` on the
command line or collected in a YAML file passed with `-c`. Unknown keys are
rejected.

## The dataset

The dataset has the layout of the source corpus, so it can be read by
anything that reads the source:

```
data/tr-lrs-clean-ptm/
  2024-2025-birlesim-72/
    00000.mp4     96x96 mouth crop (stream 0) and the original audio (stream 1)
    00000.aac     original audio, copied unchanged
    00000.txt     original transcript with word times, copied unchanged
    00003.mp4     clips keep their numbers; left-out clips leave gaps
    ...
  video-1/
  ...
```

It differs from the source in three ways:

| | Source | Dataset |
|---|---|---|
| Clips | all | those that pass the cleaning (see below) |
| Video | 224x224 face | 96x96 mouth crop |
| Masking | none | Partition-Time Masking in the training clips |

Two files are written *next to* the folder, not into it:

| File | Content |
|---|---|
| `data/tr-lrs-clean-ptm.clips.tsv` | every source clip with its verdict, split, frame count and normalised text |
| `data/tr-lrs-clean-ptm.settings.json` | crop, masking and split settings the dataset was built with |

The training code reads the clips from the dataset folder and takes the
split, the clip lengths and the normalised text from `data/manifests/`.

### Design choices

| Choice | Decision | Reason |
|---|---|---|
| Audio format | original AAC, inside the mp4 | re-encoding to FLAC doubles the size and cannot add quality; the stream is copied bit for bit |
| Manifests | kept, outside the dataset | training needs the split, clip lengths for batching and normalised text; Auto-AVSR needs the same in its label files |
| Which clips are masked | training clips only | error rates measured on masked video would not describe accuracy on real video |
| When the split is made | while building | it decides which clips are masked, so it cannot change afterwards |
| Test set | whole sessions | neighbouring clips come from the same speech; a random split would leak |
| Validation set | last 5 % of each other session | a contiguous block, for the same reason |

### Mouth crop

| Setting | Default | Meaning |
|---|---|---|
| `--roi-scale` | 0.63 | side of the crop as a fraction of face height; 0.63 shows the area of the 96 px Auto-AVSR crop (measured on 567 clips) |
| `--smoothing` | `kalman` | smoothing of the crop position: `kalman` (with backward pass), `moving_average` (Auto-AVSR), `one_euro` (causal), `none` |
| `--out-size` | 96 | side of the stored frames |

- **Scale** comes from the forehead-to-chin distance, which survives head
  turns; the distance between the eyes collapses in profile views. One scale
  is used per clip so the crop does not zoom as the jaw moves.
- **Landmarks are cached** in `data/cache/`, so building another dataset from
  the same clips takes minutes instead of hours.
- CLAHE and Procrustes alignment are deliberately left out.

During training a random 88x88 crop is taken from the stored frames (a centre
crop in evaluation), as in Auto-AVSR. `augment.video.crop_size=0` uses the
stored frames unchanged.

### Partition-Time Masking

One span covering 20 % of the clip; inside it, 2 cells of a 4x4 grid are
blacked out in every frame, with a new choice of cells per frame. The mask of
a clip is derived from its name and `--seed`, so rebuilding reproduces it.

Because the masks are in the clips, `data.ptm_in_clips` is true and masking
during training is off. Setting both on is rejected, as every clip would be
masked twice. For a dataset built with `--no-ptm`, set
`data.ptm_in_clips=false augment.video.ptm.enabled=true` to mask during
training instead, with a new mask in every epoch.

## Label filtering

The transcripts are machine generated and noisy. `scripts/audit_labels.py`
audits them (results in `docs/label_audit.md`), and the dataset builder
leaves a clip out when any of these holds:

| Check | Catches |
|---|---|
| Same transcript as a clip at most 5 positions away | invented sentences, which come in runs |
| Sentence that mostly occurs in runs within its session | the stray occurrences of such a sentence |
| Word or short phrase looping inside the transcript | `DA DA DA DA ...` |
| Known invented phrase making up half of the transcript or more | subtitle credits such as `ALTYAZI M.K.` |
| Words too short for their length, in over 25 % of words | words squeezed into one timestamp tick |
| Mean / minimum word confidence below 0.80 / 0.30 | uncertain transcripts |
| Speaking rate outside 4 to 25 characters per second | text that does not fit the clip |
| Speech end more than 1 s from clip end | transcript of a different segment |
| Face found in under 80 % of frames | occluded or missing faces |

Genuine phrases that are merely frequent, such as `TEŞEKKÜR EDERIM`, are kept:
they are spread over the recordings instead of bunched together.

The clip table next to the dataset lists every source clip with its verdict.

Text is normalised to upper-case Turkish letters and spaces: punctuation is
removed, digits are spelled out (`1921'DE` becomes `BIN DOKUZ YÜZ YIRMI BIRDE`)
and circumflexes are dropped.

## Model options

| Option | Values | Meaning |
|---|---|---|
| `model.fusion.type` | `late_mlp` | one Conformer per modality, outputs merged by an MLP (Auto-AVSR) |
| | `early_shared` | features merged before one shared Conformer; usable on either modality alone |
| `train.input_mode_probs` | `[video, audio, audiovisual]` | chance of training a batch on each input mode |
| `model.interctc.layers` | e.g. `[3, 6, 9]` | intermediate CTC losses |
| `model.interctc.self_condition` | bool | feed intermediate predictions back into the encoder |
| `model.layer_drop` | 0 to 1 | stochastic depth |
| `model.mask_conv_padding` | bool | keep padded frames out of the Conformer convolutions |
| `model.pretrained.path` / `.scope` | `frontend`, `encoder`, `full` | initialise from an Auto-AVSR checkpoint |
| `model.freeze` | `none`, `frontend`, `encoder` | parts that are not trained |
| `train.pretrained_lr_scale` | 0 to 1 | learning rate of pretrained parts as a fraction of `train.lr` |
| `train.hold_pretrained_epochs` | e.g. 1 | epochs during which pretrained parts are not changed |
| `augment.video.ptm.*` | | Partition-Time Masking |
| `augment.video.time_mask.*` | | adaptive time masking |
| `augment.speed_range` | e.g. `[0.9, 1.1]` | playback-speed perturbation, audio and video in sync |
| `train.ema_decay` | e.g. 0.999 | evaluate an exponential moving average of the weights |
| `tokenizer.type` | `unigram`, `bpe`, `char` | output units |

With the base preset and `model.mask_conv_padding=false` the model computes
exactly what the original Auto-AVSR code computes from the same weights
(`tests/test_reference_equivalence.py`).

## A/B experiments

An experiment file lists arms (sets of overrides), a control arm and seeds:

```bash
python scripts/ab_test.py configs/exp/ptm.yaml --dry-run   # validate only
python scripts/ab_test.py configs/exp/ptm.yaml             # run and report
python scripts/ab_test.py configs/exp/ptm.yaml --report-only
```

Each run is its own process. Runs that already finished are skipped, so an
interrupted experiment continues where it stopped. The report compares every
arm with the control **paired by seed** and gives a 95 % confidence interval
for the difference; a verdict is only stated when that interval excludes zero.

## Robustness

- A batch that runs out of GPU memory is skipped and counted
  (`train/oom_batches`) instead of ending the run.
- A clip that cannot be read is dropped from its batch with a warning.
- The CTC loss is always computed in 32-bit, which avoids the all-blank
  collapse seen with half precision.
- Utterances with more tokens than frames are excluded, since CTC cannot
  align them.
- `train/amp_scale` is logged; a value that keeps falling means half-precision
  overflow, and `train.precision=bf16-mixed` or `32-true` should be used.
- Training resumes from `checkpoints/last.ckpt` with `--resume`.

## Fine-tuning from the Auto-AVSR weights

```bash
python scripts/train.py -c configs/finetune/vsr_base.yaml exp_name=vsr_finetune
```

Pretrained parts are trained with a much lower learning rate than the new
output layers (`train.pretrained_lr_scale`) and are held still at the start
(`train.hold_pretrained_epochs`). With one learning rate for everything the
model does not learn; see `docs/findings.md`.

## Moving to a cloud GPU

See `docs/deployment.md`. In short: clone the code, unpack the dataset into
`data/`, the support files into the project folder and the weights into
`pretrained/`, install `requirements.txt`, and run

```bash
python scripts/preflight.py -c configs/finetune/vsr_base.yaml
```

which checks the server and ends with `READY TO TRAIN` or a list of what is
wrong.

## Further reading

- `docs/dataset_construction.md`: how the dataset was cleaned and built, for the methods section of a paper.
- `docs/project_report.md`: what was built, from which sources, and what was measured.
- `docs/deployment.md`: the remote server and the full fine-tuning, step by step.
- `docs/findings.md`: what has been measured so far.
- `docs/label_audit.md`: audit of every transcript in the corpus.
- `docs/roadmap.md`: notes on dataset cleaning and low-latency recognition.
- `docs/literature.md`: the papers behind each option.
