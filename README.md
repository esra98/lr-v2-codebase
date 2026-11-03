# tr-lipreading: Turkish lip reading

Turkish sentence-level lip reading (visual speech recognition): the model sees
only the speaker's mouth and writes what was said. This repository holds the
three parts of the project:

| Part | Where | What it is |
|---|---|---|
| **TR-LRS v3** dataset | Google Drive, unpacked into `dataset/` | 225,977 Turkish sentence clips, 224.6 hours |
| Dataset construction | `dataset-creation/` | turns full-length recordings into the dataset |
| Training | `training/` | fine-tunes Auto-AVSR on the dataset, evaluates it, runs it on any video |

The model is Auto-AVSR (Ma et al., ICASSP 2023), fine-tuned from its public
English checkpoint. The original Auto-AVSR code is kept in `reference/` for
comparison.

---

## 1. The dataset: TR-LRS v3

| | |
|---|---|
| Source | TBMM TV (Meclis TV) broadcasts of the plenary sessions of the Grand National Assembly of Türkiye: 39 recordings, all downloaded from YouTube uploads of the broadcasts |
| Clips | 225,977 sentences |
| Length | 224.6 hours. Clips last 0.5 to 10 s, typically 3 s |
| Video | 96x96 mouth crop at 25 fps, with the 16 kHz mono audio in the same mp4 |
| Text | one `.txt` per clip: the Whisper large-v3 transcript with per-word timings |
| Split | train 208,811, validation 10,993, test 6,173 clips. The test set is two whole recordings that are not in training |
| Size | 12.9 GB packed, 14 GB unpacked |

### Download

The archive is on Google Drive and anyone with the link can download it:
[**tr-lrs-v3.tar.gz**](https://drive.google.com/file/d/15dwNvrNDL_YJbRYSsC7T_Vnk2Ng25jwC/view)
(in [this folder](https://drive.google.com/drive/folders/16tybAQsivB4GPGB2piDDpuoSJeWToBCS)).
From the command line, run these in the repository folder:

```bash
pip install gdown
mkdir -p dataset
gdown 15dwNvrNDL_YJbRYSsC7T_Vnk2Ng25jwC -O dataset/tr-lrs-v3.tar.gz
echo "143122937a84a16748a8b22b415d3d49b993c522f44a2b8842eb46d1dda55b8e  dataset/tr-lrs-v3.tar.gz" | sha256sum -c
tar -xzf dataset/tr-lrs-v3.tar.gz -C .
```

The last command creates:

```
dataset/
  tr-lrs-v3/<recording>/<clip>.mp4 + <clip>.txt   the clips
  tr-lrs-v3-manifests/train.tsv val.tsv test.tsv   which clip is in which split, with its normalised text
  tr-lrs-v3-tokenizer/unigram1000.model            SentencePiece tokenizer used by training
  tr-lrs-v3.clips.tsv                              all 272,258 candidate sentences, kept or rejected and why
```

To see the format without downloading anything, look at the 50 clips in
`dataset-creation/samples/`. The training target is upper-case Turkish without
punctuation, with numbers spelled out and İ and I kept apart
(`1921` → `BİN DOKUZ YÜZ YİRMİ BİR`).

### How it was made

1. **Transcription.** Every recording is transcribed with Whisper large-v3,
   with a timestamp for each word.
2. **Segmentation.** Each transcript is split into sentences of at most
   10 seconds. Sentences whose text is unreliable are rejected: repeated or
   invented text, low confidence, text over silence, implausible speaking rate.
3. **Cutting.** In each sentence the faces are tracked and the one whose lips
   move is taken as the speaker. Their mouth is cut out at 96x96. Clips are
   rejected when no face is visible or the face is too small.
4. **Split.** The test set is two whole recordings. Validation is the last 5 %
   of every other recording.

All of it is run with `dataset-creation/corpus.py`. The full method is in
[`docs/dataset_construction.md`](docs/dataset_construction.md). Where each
recording came from (links, sizes, checksums) is in
[`docs/ham_videolar.md`](docs/ham_videolar.md).

---

## 2. Training on your own machine

### Setup

You need Python 3.10 to 3.13 and an NVIDIA GPU. Install the PyTorch build that
matches your driver first (see pytorch.org), then:

```bash
git clone --recurse-submodules https://github.com/esra98/tr-lipreading.git
cd tr-lipreading
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset as in section 1, then download the pretrained Auto-AVSR
weights (1.0 GB), which are the starting point of training:

```bash
mkdir -p pretrained
gdown 162axOfnnZxVfhpbm9XK4SKAo_l3Qdpf_ -O pretrained/vsr_trlrs2lrs3vox2avsp_base.pth
echo "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb  pretrained/vsr_trlrs2lrs3vox2avsp_base.pth" | sha256sum -c
```

Check that everything is in place. Run every command from the repository folder:

```bash
python training/preflight.py -c training/configs/finetune/vsr_base.yaml
```

It reads a sample of clips, loads the weights and the tokenizer, and runs
20 real training steps. The last line should be `READY TO TRAIN`.

### A short trial run (6 GB GPU)

This run uses 8,000 clips and 3 epochs. It shows that fine-tuning starts and
that the model learns, not a usable model:

```bash
python training/train.py -c training/configs/finetune/vsr_base.yaml \
    -c training/configs/finetune/local_6gb.yaml exp_name=trial
```

On an RTX 2060 (6 GB) it takes about 75 minutes. The validation character
error rate goes from about 0.68 after the first epoch to about 0.64 after the
third.

### Full training

```bash
python training/train.py -c training/configs/finetune/vsr_base.yaml exp_name=vsr_finetune
```

The recipe is the one of Auto-AVSR:

- 75 epochs, AdamW, 5 warm-up epochs, then cosine decay
- batches of up to 1,800 frames
- 0.1 CTC + 0.9 attention loss
- Partition-Time Masking as augmentation

The new Turkish output layers learn at 1e-3 and the pretrained parts at 2e-5,
held still for the first epoch. With one learning rate for everything the
model does not learn ([`docs/findings.md`](docs/findings.md)).

A batch of 1,800 frames needs about 48 GB of GPU memory. On a smaller GPU,
halve the batch and accumulate two of them, so that the product stays 1,800:

```bash
python training/benchmark_memory.py -c training/configs/finetune/vsr_base.yaml --batch-frames 900 1800
python training/train.py -c training/configs/finetune/vsr_base.yaml exp_name=vsr_finetune \
    data.max_frames_per_batch=900 train.accumulate_grad_batches=2
```

The run is written to `exp/vsr_finetune/`: checkpoints, `config.yaml`,
per-epoch metrics in `csv/`, and TensorBoard logs in `tb/`
(`tensorboard --logdir exp`). Watch `val/cer`. After the first epoch it should
be below 0.9 and then keep falling. If it stays above 0.9 for five epochs, the
model is not learning. A stopped run continues with
`--resume exp/vsr_finetune/checkpoints/last.ckpt`.

### Evaluate, package, use

```bash
# Word and character error rate on the test set, with the last 10 checkpoints averaged
python training/evaluate.py --run exp/vsr_finetune --average 10

# A self-contained model folder: averaged weights, config, tokenizer, results
python training/export_model.py --run exp/vsr_finetune --average 10 --out tr-vsr-model

# Read the lips in any video
pip install mediapipe opencv-python-headless
wget -P assets https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
python training/transcribe.py --run tr-vsr-model konusma.mp4
```

`transcribe.py` finds the speaker's face, cuts the mouth as in the dataset and
prints the text, one line per 10 seconds of video. Add
`--save-crop crop.mp4` to see the mouth crop the model sees.

---

## 3. Training on a rented GPU (vast.ai)

The full training needs a large GPU for several days. Renting one on
[vast.ai](https://vast.ai) costs much less than buying one. The scripts in
[`training/vast/`](training/vast) automate everything below.
[`training/vast/REMOTE_RUN.md`](training/vast/REMOTE_RUN.md) is the short
guide to them (in Turkish).

### Choosing the server

| Setting | Value | Why |
|---|---|---|
| GPU | one GPU with 48 GB or more (RTX A6000, L40S, A100 80GB, H100) | holds the 1,800-frame batch. A 24 GB GPU works with the halved batch above. The code uses one GPU, so more than one is wasted |
| CUDA (driver) | 12.8 or newer | for current PyTorch builds |
| Disk | 100 GB | dataset 13 GB packed + 14 GB unpacked, libraries ~11 GB, checkpoints ~15 GB |
| CPU / RAM | 16 cores / 32 GB | the clips are decoded on the CPU. With fewer cores the GPU waits |
| Image | Ubuntu 22.04 with SSH | |

### The run, step by step

1. **Rent and connect.** The vast.ai console gives an SSH command
   (`ssh -p <port> root@<address>`). Everything below happens on the server.
2. **Bring the code and the data.** Clone this repository. Download the
   dataset and the pretrained weights from Drive with `gdown`, as in sections 1
   and 2. Downloading straight from Drive on the server is much faster than
   uploading from a laptop. Check both sha256 sums and unpack the dataset.
3. **Install.** Install PyTorch for the server's CUDA version, then
   `requirements.txt`.
4. **Check before paying for days.** Run `pytest`, then
   `training/preflight.py`, which must end with `READY TO TRAIN`. Then run
   `training/benchmark_memory.py` to see whether 1,800 frames fit in GPU memory
   or the batch must be halved. This takes about 15 minutes and catches a
   missing file or a too-small GPU before the expensive part.
5. **Start the training in the background**, with `nohup` or inside `tmux`,
   so that it keeps running when the SSH connection closes. Write the output to
   `logs/`.
6. **Watch it.** Follow the log and `exp/vsr_finetune/csv/` for `val/cer` at
   the end of each epoch, as in section 2. The first epoch also tells you how
   long the whole run will take (about 75 times as long). If the server or the
   connection drops, restart with `--resume` from `last.ckpt`. Nothing is lost
   but the current epoch.
7. **Finish on the server.** When training ends, run `evaluate.py` and
   `export_model.py`. This gives a `tr-vsr-model/` folder of about 1 GB with
   the averaged model and its test results. Pack it and note its sha256.
8. **Download, then destroy the server.** Copy the model to your machine with
   `scp` and check the sha256. Then **Destroy** the instance in the vast.ai
   console. Stop keeps the disk and keeps billing.

The scripts in `training/vast/` do steps 2 to 7 in four commands:
`setup.sh`, `check.sh`, `train.sh full` and `finish.sh`. They are resumable
and verify every download. They come as one bundle built by `make_bundle.sh`,
which also pins the Python and library versions.

---

## Repository layout

```
dataset-creation/   corpus.py (the dataset pipeline), YouTube download script, 50 sample clips
training/           train.py, evaluate.py, export_model.py, transcribe.py, preflight.py, ...
  configs/          base.yaml (every option), finetune/ (the recipes above), model/, data/, exp/
  vast/             rented-GPU scripts and their guides
dataset/            the downloaded archive and the unpacked dataset (not in git)
avsr/               the library both parts use: model, data loading, mouth cropping, text, decoding
reference/          original Auto-AVSR code (git submodules)
docs/               method and results documents
tests/              unit and end-to-end tests (python -m pytest)
pretrained/ exp/    starting weights and training runs (not in git)
```

## Further reading

| Document | Contents |
|---|---|
| [`DATASET.md`](DATASET.md) | the dataset files, checks run on them, statistics |
| [`docs/dataset_construction.md`](docs/dataset_construction.md) | how the dataset was built, in method-section detail |
| [`docs/ham_videolar.md`](docs/ham_videolar.md) | the 39 source recordings: links, sizes, checksums (Turkish) |
| [`docs/codebase.md`](docs/codebase.md) | technical reference: pipeline stages, filtering rules, every model option, A/B experiments |
| [`docs/findings.md`](docs/findings.md) | what has been measured so far |
| [`docs/v3_degisiklik_raporu.md`](docs/v3_degisiklik_raporu.md) | what changed from the first corpus to v3 (Turkish) |
| [`docs/literature.md`](docs/literature.md) | the papers behind each option |

## Credits

The model architecture, the pretrained weights and the Transformer/Conformer
building blocks in `avsr/espnet/` (Apache 2.0) come from Auto-AVSR:
P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, M. Pantic,
*Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels*,
ICASSP 2023 ([github.com/mpc001/auto_avsr](https://github.com/mpc001/auto_avsr)).
The source recordings are TBMM TV (Meclis TV) broadcasts of the plenary
sessions of the Grand National Assembly of Türkiye. Transcripts were made
with Whisper large-v3 through faster-whisper.
