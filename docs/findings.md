# Findings

Everything here was measured on this machine (RTX 2060, 6 GB) between
2026-09-27 and 2026-09-29. No A/B experiment has been run yet, so there is no
result on which of the optional techniques helps.

## Dataset

| Fact | Value |
|---|---|
| Tarballs / sessions | 33 / 33, no two tarballs share a folder name |
| Clips | 218,091 |
| Tarballs 1 to 14 | 86,185 clips, the earlier 94-hour set |
| Clip format | 224x224 face crop, H.264, 25 fps, 16 kHz mono audio |
| Clip length (one session) | median 1.9 s, maximum 10 s |
| Audio and video | both start at 0; audio is a few milliseconds longer and is trimmed |

Landmark detection finds a face in 98.2 % of frames (sample of 2967 frames).
Missing frames are interpolated.

## Labels

`docs/label_audit.md` has the audit of all 218,091 transcripts: 23.1 % fail a
check, 10.6 % because the transcript is recurrent or invented.

## Full dataset

`data/tr-lrs-clean-ptm/`, built on 2026-09-28 from all 33 sessions.

| | Clips | Hours | Sessions |
|---|---|---|---|
| Source | 218,091 | | 33 |
| Dataset | 154,191 (70.7 %) | 105.4 | 33 |
| Training, masked | 140,823 | 96.0 | 31 |
| Validation, unmasked | 7,274 | 5.0 | 31 |
| Test, unmasked | 6,094 | 4.4 | 2 |

Size on disk: 11 GB. No clip failed with an error.

| Reason a clip was left out | Clips | Share of source |
|---|---|---|
| Low recogniser confidence | 24,998 | 11.5 % |
| Same transcript as a neighbouring clip | 21,400 | 9.8 % |
| Implausible speaking rate | 5,408 | 2.5 % |
| Speech end far from clip end | 3,533 | 1.6 % |
| Shorter than 12 frames | 2,764 | 1.3 % |
| Words too short to have been spoken | 2,025 | 0.9 % |
| Face found in under 80 % of frames | 1,351 | 0.6 % |
| Known invented phrase | 1,072 | 0.5 % |
| No face | 807 | 0.4 % |
| Word looping inside the transcript | 441 | 0.2 % |
| Bursty sentence met outside a burst | 101 | 0.0 % |

A clip is counted under the first check it fails.

`scripts/verify_dataset.py` passes every check on the full dataset, with the
checks that decode video run on 900 clips: audio identical to the source in
all of them, video within 2.0 of 255 of the crop rebuilt from raw (worst clip
2.5), masks in training clips only.

A run of 300 steps on the full dataset (small audio-visual model, shared
encoder) and beam-search evaluation in all three input modes completed
without errors; GPU peak 3.4 GB.

One observation from that run: the gradient scale of 16-bit precision fell
from 65536 to 2048 within 300 steps, which means gradients overflowed several
times during warm-up. `train/amp_scale` should be watched at the start of a
long run; if it keeps falling, use `train.precision=bf16-mixed`.

## Rehearsal for the remote server

Done on 2026-09-28 in a clean environment: a new Python 3.10 with only
`requirements.txt` installed (PyTorch 2.14, Lightning 2.6.6, PyAV 17.1) and
only the files that would be uploaded.

| Step | Result |
|---|---|
| Test suite | 187 passed, 4 skipped (they need dataset-building packages or the reference code) |
| Preflight | all checks passed |
| Training, 3 epochs on 6,000 clips, small video model, pretrained front-end, intermediate CTC, weight averaging | 1,095 steps in 14.3 min, no batch skipped, GPU peak 4.7 GB |
| Resume from `last.ckpt` for a 4th epoch | continued at step 1,095, ended at 1,460 |
| Evaluation, 3 checkpoints averaged, and a single checkpoint | both ran |

Problems found by the rehearsal, all fixed:

| Problem | Effect | Fix |
|---|---|---|
| `pip install torch` installed a CUDA 13.0 build; the driver supports 12.8 | PyTorch could not use the GPU | documented; the preflight reports it with the command to run |
| Converting frames to RGB is 24 times slower in PyAV 17 than in 16 (41 ms against 1.7 ms per clip) | loading fell from about 120 to 19 clips per second | the brightness plane is read directly: 150 to 166 clips per second in both versions, image equal to within 1.3 of 255 |
| Out-of-memory batches were skipped without limit | 349 of 365 batches of an epoch were skipped while the run went on | the run stops after 10 skipped batches in a row, or 5 % of an epoch |
| Out of memory during validation was not handled | the run ended with an error | the batch is split in halves and repeated |
| Out of memory in the backward pass was not handled | would have ended the run | the batch is dropped and counted |
| GPU memory fragmented by batches of varying shape | out of memory at a batch size that fits | `expandable_segments` is set by default |
| Metrics file overwritten on resume | numbers of earlier epochs lost | earlier parts are kept as `metrics.partN.csv` |
| Kept checkpoints held the optimizer state | three times the disk space | kept epochs hold weights only |
| `torch.OutOfMemoryError` does not exist before PyTorch 2.5 | the handler itself would fail | falls back to the older name |
| MediaPipe and OpenCV in `requirements.txt` | not needed for training | moved to `requirements-build.txt` |

### Memory of the training process

Small audio-visual model, 600 frames per batch, 8 loader processes, the full
list of 140,823 training clips, 4,500 steps in 26 minutes.

| Minute | Training process | 8 loader processes |
|---|---|---|
| 0 | 2.43 GB | 2.14 GB |
| 6 | 2.69 GB | 3.64 GB |
| 12 | 2.81 GB | 3.73 GB |
| 18 | 2.87 GB | 3.87 GB |
| 24 | 2.89 GB | 3.85 GB |

| Period | Growth of the training process |
|---|---|
| minutes 2 to 8 | 1.59 GB per hour |
| minutes 8 to 16 | 1.01 GB per hour |
| minutes 16 to 26 | 0.23 GB per hour |

The growth slows but was not flat when the run ended. Memory held by Python
objects did not change between step 150 and step 900 (113 MB both times),
while the process grew by 0.33 GB, so the growth is in native libraries.
Pinned memory accounts for about 1 GB of the total but not for the growth,
which was the same with it switched off. The final level is not known.

## Sample dataset

`data/sample/` was built from the first 400 clips of three sessions to prove
the chain from raw clips to a trained model.

| Step | Result |
|---|---|
| Build | 736 of 1200 clips kept; 17 s with cached landmarks |
| Verification (`scripts/verify_dataset.py`) | all checks pass |
| Audio in the mp4 | sample-identical to the source in all 139 clips checked |
| Transcript and aac files | byte-identical to the source |
| Video against the crop rebuilt from raw | mean difference 2.0 of 255 (H.264 loss) |
| Masks | in training clips only; 20 % of the frames; cells black (0.3 of 255) |
| Training, 3 epochs, video / audio / audio-visual | all ran; GPU peak 3.4 / 2.1 / 3.4 GB; no out-of-memory |
| Evaluation with beam search | ran in video and audio-visual mode |

Error rates in these runs are near 100 %. That is the amount of data
(15 minutes) and training (3 epochs), and says nothing about the model.

## Crop geometry

The 96 px Auto-AVSR crop covers 0.63 x face height (median over 567 clips;
10th to 90th percentile 0.605 to 0.667). Frontal and turned heads give almost
the same value (0.627 and 0.641). The project default was 0.55 and is now 0.63.

## Fidelity to the reference implementation

With the public checkpoints `vsr_trlrs2lrs3vox2avsp_base.pth` and
`asr_trlrs3vox2_base.pth`, encoder output, CTC log-probabilities and decoder
logits are identical to those of the original code (maximum difference 0).

One behaviour of the original was changed and is switchable
(`model.mask_conv_padding`): padded frames reached real frames through the
Conformer convolutions, so a clip was encoded differently depending on the
other clips in its batch, and differently in training (batched) than at test
time (one clip at a time).

## GPU memory (16-bit mixed precision, 5.6 GB usable)

| Model | Trainable parameters | Largest batch that fits |
|---|---|---|
| small, video | 53 M | 1200 frames (4.5 GB) |
| small, audio | 46 M | 2400 frames (4.6 GB) |
| small, audio-visual, late fusion | 91 M | 600 frames (4.3 GB) |
| base, video | 241 M | 300 frames (5.3 GB) |
| base, video, front-end frozen | 230 M | 600 frames (4.7 GB) |
| base, video, encoder frozen | 59 M | 2400 frames (3.1 GB) |

## Pilot runs

Both used only the first session: 846 training utterances, 0.60 h.

| Run | Epochs | Best validation CER | Validation loss |
|---|---|---|---|
| base video model, Auto-AVSR encoder frozen, decoder and CTC trained | 7 | 0.94 | lowest at epoch 2, then rising |
| small audio model from scratch | 16 | 0.99 | lowest at epoch 7, then rising |

Neither learned to recognise speech; rising validation loss with improving
training accuracy is overfitting. Half an hour of data is too little even for
audio, so these runs say nothing about the model, only about the data size.

## Fine-tuning from the Auto-AVSR weights

Base model (241M parameters), vocabulary of 1000 units, 8,000 training
clips, validation on 200 to 400 clips with greedy decoding.

### One learning rate does not work

| Run | Updates | Validation CER | WER |
|---|---|---|---|
| Video, encoder transferred, learning rate 0.001, 1800 frames per update | 1,696 | 0.96 | 1.00 |
| Audio, same recipe | | 0.93 to 0.97 | |
| Audio, encoder transferred, 0.0001, 252 frames per update | 600 | 0.97 | |
| Audio, everything transferred, 0.0001 | 600 | 0.97 | |
| Audio, everything transferred, 0.0003 | 600 | 0.97 | |
| Audio, front-end frozen | 600 | 0.97 | |

In all of them the CTC loss stays at about 6.4 per token; a model that knows
nothing has ln(1002) = 6.9. The hypotheses are nearly the same for every
clip.

### The pretrained features are usable

Audio encoder of Auto-AVSR frozen, a small network trained on its output,
4,000 clips (2.83 h):

| Trained on top | Validation CER |
|---|---|
| Linear layer on the last block | 1.09 |
| Linear layer on block 9 | 0.89 |
| Two-layer network on block 9 | 0.39 |

### Two learning rates work

| Run | Updates | Validation CER | WER |
|---|---|---|---|
| Audio, new layers 0.001, pretrained 0.00001 | 1,500 | 0.50 | 0.85 |
| Audio, new layers 0.001, pretrained 0.00002, held for 500 updates | 1,500 | 0.46 | 0.86 |
| Audio, the same with intermediate CTC at blocks 6 and 9 | 1,500 | 0.48 | 0.87 |
| Video, training script with `configs/finetune/vsr_base.yaml` and `local_6gb.yaml` | 2,373 | 0.73 | 0.93 |

The video run was stopped after its first epoch. The loss was still falling
in every run of this table when it ended.

In the video run the gradient scale of 16-bit precision fell from 65536 to
256 over 3,800 updates: about one update in 500 was skipped because of an
overflow.

## Earlier result on this corpus

`tr-test/av-asr/FINAL/scripts/eval.log` from the earlier attempt records a WER
of 0.974. That run used the Auto-AVSR code, 96 px crops, the 250M-parameter
model, the visual front-end of `vsr_trlrs3vox2_base.pth`, and 75 epochs on the
94-hour set with unfiltered labels.
