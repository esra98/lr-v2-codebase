# Codebase reference

The technical reference of the project: how the corpus pipeline works, which
utterances are kept, every model option, A/B experiments and robustness. For
downloading the dataset and training, start with the [README](../README.md).

## Overview

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

The project has three parts: the code that builds the dataset, the code that
trains on it, and the dataset itself. Both code parts use the shared `avsr/`
library. Run every command from this folder (the project root).

```
dataset-creation/              recordings -> dataset
  corpus.py                    pipeline: ingest, download, transcribe, segment, cut, build, verify
  download_youtube_1080p.sh    re-download the YouTube sessions at 1080p
  samples/                     50 clips of the dataset with their labels, to see the format
training/                      dataset -> model
  train.py evaluate.py         training and evaluation
  transcribe.py export_model.py  inference on any video, model packaging
  preflight.py benchmark_memory.py ab_test.py train_tokenizer.py
  configs/                     base.yaml (every option), data/, model/ (tiny, small, base), exp/ (A/B), finetune/
  vast/                        rented-GPU (vast.ai) setup, training and packaging scripts
dataset/
  tr-lrs-v3.tar.gz (+.sha256)  the dataset; unpacking it here creates the folders below
  tr-lrs-v3/<session>/         the clips: <utt>.mp4 (96x96 mouth + 16 kHz audio), <utt>.txt
  tr-lrs-v3.clips.tsv          every utterance with its verdict and split
  tr-lrs-v3-manifests/         train / val / test manifests
  tr-lrs-v3-tokenizer/         SentencePiece models
avsr/                          shared library
  corpus/       ingest, transcribe, segment, cut, build (used by dataset-creation)
  text/         Turkish normalisation, label parsing, tokenizers
  preprocess/   landmarks, trajectory smoothing, mouth cropping, audio/video I/O
  data/         label rules, clip table, manifests, augmentation, batching
  models/       model, Conformer encoder extensions, pretrained-weight loading
  decoding/     joint CTC/attention beam search
  training/     Lightning modules, schedule, weight averaging
  espnet/       Transformer/Conformer building blocks from Auto-AVSR (Apache 2.0)
docs/          how the corpus was built (dataset_construction.md, ham_videolar.md),
               literature notes, findings, project report
tests/         unit and end-to-end tests
reference/      original Auto-AVSR and VSR-for-multiple-languages code (git submodules, pinned)
pretrained/    Auto-AVSR starting weights;  exp/  training runs;  logs/
videos/, work/ only while building the dataset: raw recordings and per-session state
```

Unpack the dataset before training:

```bash
(cd dataset && sha256sum -c tr-lrs-v3.tar.gz.sha256)
tar -xzf dataset/tr-lrs-v3.tar.gz -C .
```

`videos/`, `work/`, `dataset/`, `exp/` and `pretrained/` are not in git.

The current dataset is `dataset/tr-lrs-v3` (October 2026): 39 recordings,
225,977 clips, 224.6 hours, built as described in
`dataset_construction.md`. `DATASET.md` says where the archive is,
how to unpack it and that this codebase was checked against it. The first corpus (September 2026, built
from 224x224 face clips with the dotted and dotless I merged) and the
scripts that built it are tagged `v2-dataset-2026-09-28` and described in
`archive/`.

## Setup

```bash
python3 -m venv --system-site-packages .venv      # reuses the system PyTorch
.venv/bin/pip install -r requirements-build.txt    # requirements.txt is enough for training only
.venv/bin/python -m pytest                          # about 10 minutes
```

`assets/face_landmarker.task` (MediaPipe) and `assets/babble_noise.wav`
(audio augmentation) are not in git. The landmark model comes from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`.
Building the corpus also needs `ffmpeg`, `yt-dlp` (2026.08 or later, with
`node` as its JavaScript runtime) and `faster-whisper`.

## From recordings to a dataset

A session is one recording: an assembly session (`2024-2025-birlesim-72.mp4`)
or a YouTube upload (`yt-<id>.mp4`). Everything is driven by one script:

```bash
# 1. Register the recordings: resolution, length, checksum -> videos/sources.tsv
python dataset-creation/corpus.py ingest

# 2. Fetch YouTube sessions at up to 1080p (resumable, replaces smaller files)
python dataset-creation/corpus.py download --ids fP99G1PONhQ ...

# 3. Transcribe, segment and cut, session by session (NVIDIA GPU for Whisper; the
#    face detector runs on the Intel GPU for --gpu-workers of the --workers, CPU for the rest)
python dataset-creation/corpus.py run --dataset dataset/tr-lrs-v3 --workers 7 --gpu-workers 4

# 4. Clip table, split and manifests; then check the result
python dataset-creation/corpus.py build --dataset dataset/tr-lrs-v3 --holdout 2025-2026-birlesim-3 yt-NPJohShU_iw
python dataset-creation/corpus.py verify --dataset dataset/tr-lrs-v3

# 5. Tokenizer, batch size, training, evaluation
python training/train_tokenizer.py --manifest dataset/tr-lrs-v3-manifests/train.tsv --out-dir dataset/tr-lrs-v3-tokenizer
python training/benchmark_memory.py -c training/configs/model/small.yaml model.modality=video
python training/train.py -c training/configs/model/small.yaml model.modality=video exp_name=vsr_small
python training/evaluate.py --run exp/vsr_small --average 10
```

`run` is `transcribe`, `segment` and `cut`, which can also be called one at a
time; `status` lists what is done per session. Every stage is resumable:
transcription per 20-minute chunk, cutting per utterance, and a stage runs
again only when its inputs or options changed (`work/<session>/status.json`).
After a download, run `ingest` again before `run`: a session whose
`sources.tsv` entry is still below 720p is transcribed but not cut.
`cut --retry-errors` cuts only the utterances whose verdict is `error`.

| Stage | Input | Output | What it does |
|---|---|---|---|
| transcribe | `videos/<s>.mp4` | `work/<s>/transcript.json` | `faster-whisper large-v3` with word times and confidences, in Whisper's own casing; `no_speech_prob`, `avg_logprob`, `compression_ratio` per segment |
| segment | transcript | `work/<s>/segments.tsv` | utterances: cut at sentence ends and pauses over 0.4 s, at most 10 s (split at the longest pause), at least 2 words; the text rules below give each one a verdict |
| cut | video + segments | `dataset/<name>/<s>/`, `work/<s>/clips.tsv`, `work/<s>/landmarks/` | one sequential decode; frames nearest the 25 fps grid; up to 4 faces found by a tiled search (MediaPipe alone misses 200 to 300 px faces in a 1080p frame), tracked through the clip, and the **speaker chosen by lip motion**; 96x96 mouth crop with the utterance's 16 kHz mono audio in one mp4 |
| build | all `clips.tsv` | clip table, manifests | whole held-out sessions as the test set, the last 5 % of every other session as validation |

### The clips

```
dataset/tr-lrs-v3/
  2024-2025-birlesim-72/
    00000.mp4     96x96 mouth crop at 25 fps (stream 0), 16 kHz mono AAC (stream 1)
    00000.txt     Text: line in Whisper's casing, then WORD start end confidence per word
    00003.mp4     utterance numbers are those of segments.tsv; rejected ones leave gaps
```

The training target is the normalised text of the clip table: upper case
with Turkish casing (`i` -> `İ`, `ı` -> `I`, 29 letters plus Q, W, X),
digits spelled out, punctuation removed. Partition-Time Masking is applied
during training (`augment.video.ptm`), not written into the clips.
`training/configs/base.yaml` points at this dataset and its tokenizer.

### Mouth crop

| Setting | Default | Meaning |
|---|---|---|
| `--roi-scale` | 0.63 | side of the crop as a fraction of face height; 0.63 shows the area of the 96 px Auto-AVSR crop |
| `--smoothing` | `kalman` | smoothing of the crop position: `kalman` (with backward pass), `moving_average` (Auto-AVSR), `one_euro` (causal), `none` |
| `--out-size` | 96 | side of the stored frames |
| `--variant` | `mouth` | `face` writes a square face crop instead (`--out-size 224 --roi-scale 1.3`) |

Scale comes from the forehead-to-chin distance, one value per clip. The
speaker is the tracked face whose lip opening varies most among the faces at
least half as tall as the tallest; close-ups in the chamber regularly show a
listener beside the speaker. During training a random 88x88 crop is taken
from the stored frames (a centre crop in evaluation).

## Which utterances are kept

| Check | Catches |
|---|---|
| Same transcript as an utterance at most 5 positions away | invented sentences, which come in runs |
| Sentence that mostly occurs in runs within its session | the stray occurrences of such a sentence |
| Word or short phrase looping inside the transcript | `DA DA DA DA ...` |
| Known invented phrase making up half of the transcript or more | subtitle credits such as `ALTYAZI M.K.` |
| Words too short for their length, in over 25 % of words | words squeezed into one timestamp tick |
| Mean / minimum word confidence below 0.80 / 0.30 | uncertain transcripts |
| Whisper's `no_speech_prob` above 0.6 | text invented over silence or noise |
| Speaking rate outside 4 to 25 letters per second | text that does not fit the span |
| Shorter than 12 frames or longer than 600 | |
| Face found in under 80 % of frames, or face under 50 px tall | wide shots of the chamber, occlusions |

Genuine phrases that are merely frequent, such as `TEŞEKKÜR EDERİM`, are kept:
they are spread over the recordings instead of bunched together. The clip
table lists every utterance with its verdict. On the 39 recordings the text
rules remove 4.0 % of the utterances and the video checks 13.0 %, nearly all
of them shots without a visible face.

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
(`tests/test_reference_equivalence.py`; it needs the original code in the
`reference/auto_avsr` submodule, `git submodule update --init`, or at
`AUTO_AVSR_REFERENCE`, and skips when it is absent).

## A/B experiments

An experiment file lists arms (sets of overrides), a control arm and seeds:

```bash
python training/ab_test.py training/configs/exp/ptm.yaml --dry-run   # validate only
python training/ab_test.py training/configs/exp/ptm.yaml             # run and report
python training/ab_test.py training/configs/exp/ptm.yaml --report-only
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
python training/train.py -c training/configs/finetune/vsr_base.yaml exp_name=vsr_finetune
```

Pretrained parts are trained with a much lower learning rate than the new
output layers (`train.pretrained_lr_scale`) and are held still at the start
(`train.hold_pretrained_epochs`). With one learning rate for everything the
model does not learn; see `findings.md`.

## Further reading

- `dataset_construction.md`: how the dataset was transcribed, cut, cleaned and verified, for the methods section of a paper.
- `v3_degisiklik_raporu.md` (Turkish): what was wrong with the first corpus, how it was fixed, and the accounting of hours from recordings to dataset.
- `audit_and_roadmap_2026-10-17.md`: the audit that led to the rebuild, the decisions, and the log of the rebuild.
- `project_report.md`: what was built, from which sources, and what was measured (September 2026, first corpus).
- `deployment.md`: the remote server and the full fine-tuning, step by step.
- `findings.md`: what has been measured so far (first corpus).
- `roadmap.md`: notes on dataset cleaning and low-latency recognition.
- `literature.md`: the papers behind each option.
- `archive/`: the construction and label audit of the first corpus.
