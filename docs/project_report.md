# Project report: Turkish visual and audio-visual speech recognition

What was built, from which sources, how the dataset was cleaned, what was
measured, and how the full fine-tuning is prepared. Written on 2026-09-29.

**Note (2026-10-22).** This report describes the first corpus
(`tr-lrs-clean-ptm`, 154,191 clips from 224x224 face clips). The corpus has
since been rebuilt from the full recordings as `dataset/tr-lrs-v3`
(225,977 clips, 224.6 hours, dotted and dotless I distinct); see
`docs/dataset_construction.md` and `docs/audit_and_roadmap_2026-10-17.md`.
Section 4 and the numbers in it are kept as a record of the first corpus.

Detailed companions:

| Document | Content |
|---|---|
| `docs/dataset_construction.md` | the current dataset (v3), stage by stage, for the methods section of a paper |
| `docs/archive/dataset_construction_v2_2026-09-28.md` | the first corpus, rule by rule |
| `docs/archive/label_audit_v2_2026-09-28.md` | audit of all 218,091 transcripts of the first corpus |
| `docs/findings.md` | every measurement made so far |
| `docs/literature.md` | literature notes with reported results |
| `docs/deployment.md` | the server, step by step |
| `docs/roadmap.md` | further cleaning and low-latency recognition, not started |

## 1. Goal

A sentence-level recogniser of Turkish speech from video, with or without
audio, that improves on Auto-AVSR (Ma et al.). One setting selects
lip reading, audio speech recognition or audio-visual recognition.

Conditions set at the start of the project:

| Condition | How it is met |
|---|---|
| Partition-Time Masking is used | written into the training clips of the dataset |
| CLAHE and Procrustes alignment are not used | not implemented |
| The source corpus is never changed | it is only read; all work happens on copies in `data/` |
| Sentence level, not word level | no word-level dataset or model is used |
| Transcripts are not re-transcribed | cleaning is analytical, from the existing labels |
| Every change to the baseline can be compared with the baseline | each one is a configuration switch whose default is the Auto-AVSR behaviour |
| The code runs locally before a server is rented | 241 tests, a clean-room rehearsal, a preflight script |

## 2. Sources

### 2.1 Papers in `papers/`

| File | Paper | Used for |
|---|---|---|
| `2202.13084v2.pdf` | Ma, Petridis, Pantic, "Visual speech recognition for multiple languages in the wild", Nature Machine Intelligence 2022 | model, preprocessing, hybrid loss |
| `2303.14307v3.pdf` | Ma et al., "Auto-AVSR: Audio-visual speech recognition with automatic labels", ICASSP 2023 | model, training recipe, pretrained weights, audio-visual fusion |
| `2205.10839v2.pdf` | Sheng et al., "Deep learning for visual speech analysis: a survey", IEEE TPAMI | overview of regions of interest, back-ends and losses |
| `1-s2.0-S2590123025032268-main.pdf` | Rezaee, Yeganeh, "Automatic visual lip reading: a comparative review of machine-learning approaches", Results in Engineering 2025 | overview; reported results of Partition-Time Masking |

### 2.2 Code in `reference-codebase/`

| Folder | Use |
|---|---|
| `auto_avsr` | the baseline. Its Transformer and Conformer building blocks are copied into `avsr/espnet/` (Apache 2.0, licence file included); everything else was written for this project |
| `Visual_Speech_Recognition_for_Multiple_Languages` | reference for the mouth crop and for decoding |

Neither folder is imported by the project. They are needed only by
`tests/test_reference_equivalence.py`, which is skipped without them.

### 2.3 Further literature

Found by a literature search and confirmed online; the reported numbers are
in `docs/literature.md`. The bibliographic details should be checked against
the publications before they are cited.

| Part of the project | Reference | State |
|---|---|---|
| Conformer encoder | Gulati et al., "Conformer", Interspeech 2020 | baseline |
| Hybrid CTC/attention training and decoding | Watanabe et al., IEEE JSTSP 2017 | baseline |
| Subword units | Kudo, Richardson, "SentencePiece", EMNLP 2018 | baseline |
| Partition-Time Masking | "A novel approach for visual speech recognition using the partition-time masking", Sensors 2025 | in the dataset |
| Face landmarks | Lugaresi et al., "MediaPipe", 2019; Kartynnik et al., 2019 | in the dataset |
| Smoothing of the crop position | Kalman 1960; Rauch, Tung, Striebel 1965 | in the dataset |
| Compression-ratio test for looping transcripts | Radford et al., "Whisper", ICML 2023 | in the cleaning |
| Intermediate CTC | Lee, Watanabe, ICASSP 2021; in lip reading: Burchi, Timofte, WACV 2023 | implemented, off by default |
| Self-conditioned CTC | Nozaki, Komatsu, Interspeech 2021 | implemented, off by default |
| Stochastic depth | Huang et al., ECCV 2016; Fan et al., ICLR 2020 | implemented, off by default |
| One model for video, audio and both | Haliassos et al., "USR", NeurIPS 2024; Shi et al., Interspeech 2022 | implemented, off by default |
| Speed perturbation | CNVSRC 2023 system, arXiv 2406.10313 | implemented, off by default |
| Averaging of weights during training | exponential moving average | implemented, off by default |
| Lower learning rate for pretrained layers | Howard, Ruder, "ULMFiT", ACL 2018 | used in fine-tuning |
| Pretrained layers held still at the start | Baevski et al., "wav2vec 2.0", NeurIPS 2020 | used in fine-tuning |
| Reference point for Turkish lip reading | Prajwal, Hegde, Zisserman, "MultiVSR", ICASSP 2025: fine-tuned on about 100 h of Turkish, WER near 53 % (read off a figure) | comparison |

Considered and not implemented, with reasons in `docs/literature.md`:
E-Branchformer and rotary position embeddings (not compatible with the
pretrained Conformer weights), CR-CTC (doubles memory), decoders based on
large language models (memory), SyncVSR (needs an audio tokeniser).

No sentence-level Turkish audio-visual recogniser with a modern architecture
was found in the literature.

## 3. The codebase

### 3.1 Structure

| Folder | Content | Lines |
|---|---|---|
| `avsr/text/` | Turkish normalisation, transcript parsing, tokenizers | |
| `avsr/preprocess/` | video and audio reading and writing, landmarks, smoothing, mouth crop, masking | |
| `avsr/data/` | cleaning rules, clip table and manifests, augmentation, batching | |
| `avsr/models/` | model, encoder extensions, loading of pretrained weights | |
| `avsr/decoding/` | joint CTC/attention beam search | |
| `avsr/training/` | training and data modules, schedule, weight averaging | |
| `avsr/` without `espnet/` | | 3,100 |
| `avsr/espnet/` | building blocks copied from Auto-AVSR | 4,200 |
| `scripts/` | command-line programs | 1,700 |
| `tests/` | 241 tests | 2,200 |
| `configs/` | `base.yaml` with every option; model sizes; experiments; fine-tuning | |

### 3.2 Model

| Part | Design | Source |
|---|---|---|
| Visual front-end | 3D convolution over 5 frames, ResNet-18, one vector per frame | Auto-AVSR |
| Audio front-end | 1D ResNet-18, output at 25 vectors per second | Auto-AVSR |
| Encoder | 12 Conformer blocks, width 768, relative-position attention, convolution kernel 31 | Auto-AVSR |
| Decoder | 6 Transformer blocks | Auto-AVSR |
| Loss | 0.1 CTC + 0.9 attention, label smoothing 0.1 | Auto-AVSR |
| Audio-visual fusion | one encoder per input, merged by an MLP | Auto-AVSR |
| | or: inputs merged before one shared encoder | this project |

With the public checkpoints, encoder output, CTC probabilities and decoder
output are identical to those of the original code (maximum difference 0,
`tests/test_reference_equivalence.py`). Ma's English audio model, run
through this project's loader and beam search on an English clip,
transcribes it without error.

### 3.3 Changes to the baseline

All are switches. The first is on by default, the others off.

| Change | Setting | Reason |
|---|---|---|
| Padded frames kept out of the Conformer convolutions | `model.mask_conv_padding` | in the original a clip is encoded differently depending on the other clips of its batch |
| Intermediate CTC, self-conditioning | `model.interctc.*` | section 2.3 |
| Stochastic depth | `model.layer_drop` | |
| Shared encoder, input-mode sampling | `model.fusion.type=early_shared`, `train.input_mode_probs` | one model for all three tasks |
| Speed perturbation | `augment.speed_range` | |
| Averaged weights | `train.ema_decay` | |
| Separate learning rate and initial hold for pretrained parts | `train.pretrained_lr_scale`, `train.hold_pretrained_epochs` | section 6 |

### 3.4 Robustness

| Problem | Handling |
|---|---|
| Batch does not fit the GPU | skipped and counted; the run stops after 10 in a row or 5 % of an epoch |
| Out of memory in validation | the batch is split in halves and repeated |
| Unreadable clip | left out of its batch, with a warning |
| CTC in 16-bit precision collapses to blanks | CTC is always computed in 32 bit |
| More tokens than frames | the utterance is excluded |
| Interrupted run | resumes from `last.ckpt`; earlier metrics are kept |
| Invalid configuration | rejected before anything is loaded; unknown keys are errors |
| Server differs from the development machine | `scripts/preflight.py` checks software, GPU, data and a trial run |

### 3.5 A/B experiments

`scripts/ab_test.py` runs the arms of an experiment file with several seeds,
each run in its own process, and compares every arm with the control paired
by seed, with a 95 % confidence interval. Experiments are defined for
intermediate CTC, input modality, padding mask, Partition-Time Masking, crop
size and smoothing (`configs/exp/`). None has been run yet.

## 4. The dataset

Described in full in `docs/dataset_construction.md`. In short:

### 4.1 Source

TR-LRS: speeches in the Grand National Assembly of Türkiye, 33 sessions,
218,091 clips; 224x224 face video at 25 frames per second, 16 kHz audio, and
a transcript per clip made with Whisper, with a time and a confidence for
every word. No transcript was checked by a person.

### 4.2 Cleaning

A clip is removed when it fails one of these checks. Each removed clip is
counted under the first check it fails.

| Rule | Removes a clip when | Removed | Share |
|---|---|---|---|
| R6 Low confidence | the mean word confidence is below 0.80 or one word is below 0.30 | 24,998 | 11.5 % |
| R3 Burst | a clip at most 5 positions away has the same transcript (for 5 words or more: 80 % of the words) | 21,400 | 9.8 % |
| R7 Speaking rate | it has fewer than 4 or more than 25 letters per second | 5,408 | 2.5 % |
| R8 Timing | the last word ends more than 1 s from the end of the video | 3,533 | 1.6 % |
| Length | it has fewer than 12 frames | 2,764 | 1.3 % |
| R5 Squeezed words | over 25 % of its words last under 15 ms per letter | 2,025 | 0.9 % |
| Face | a face is found in under 80 % of the frames | 1,351 | 0.6 % |
| R1 Invented phrase | a subtitle credit such as `ALTYAZI M K` makes up half of the words | 1,072 | 0.5 % |
| No face | no face is found | 807 | 0.4 % |
| R2 Internal loop | 1 to 4 words are repeated 3 times in a row, or the text compresses more than 2.4 times | 441 | 0.2 % |
| R4 Bursty sentence | its sentence occurs mostly in bursts in that session | 101 | 0.0 % |

Why recurrence is a sign of a wrong transcript: invented sentences occurred
in single sessions and in runs of neighbouring clips, while genuine formulae
such as `TEŞEKKÜR EDERIM` occurred in every session and spread out. All
clips of a run are removed, since nothing shows which of them is right.

Tested and not used: agreement between lip opening and loudness. Clips with
invented transcripts scored as high as good clips (median correlation 0.29
against 0.20), because people do speak in them; only the text is wrong.

### 4.3 Text

Upper-case Turkish letters and spaces. Punctuation and apostrophes are
removed, digits are spelled out, circumflexes are dropped. The source text
lost the difference between dotted and dotless I (871,668 `I` against 5,372
`İ`), so both are one letter in this dataset.

### 4.4 Video

| Step | Method |
|---|---|
| Landmarks | MediaPipe FaceLandmarker, 19 points, cached |
| Position | centre of the mouth, smoothed with a constant-velocity Kalman filter and a backward pass |
| Scale | 0.63 x forehead-to-chin height, one value per clip |
| Rotation | none |
| Output | 96x96, H.264, with the source audio stream copied bit for bit |
| Masking | Partition-Time Masking in training clips: one span of 20 % of the clip; 2 cells of a 4x4 grid black in each of its frames, new cells per frame |

The scale 0.63 was measured: it is the area the Auto-AVSR crop shows
(median over 567 clips). The mask of a clip is derived from its name and the
seed, so a rebuild reproduces it.

### 4.5 Result

| | Clips | Hours | Sessions | Masked |
|---|---|---|---|---|
| Training | 140,823 | 96.0 | 31 | yes |
| Validation, the last 5 % of each training session | 7,274 | 5.0 | 31 | no |
| Test, sessions `2025-2026-birlesim-3` and `video-10` | 6,094 | 4.4 | 2 | no |
| Total | 154,191 (70.7 % of the source) | 105.4 | 33 | |

The layout is that of the source: one folder per session with `NNNNN.mp4`,
`NNNNN.aac` and `NNNNN.txt`; clips keep their numbers. Size 11 GB.

### 4.6 Verification

`scripts/verify_dataset.py` passes all 21 checks: every clip of the
manifests is on disk, transcripts and `.aac` files are byte-identical to the
source, the audio in the mp4 is sample-identical, the video is within 2.0 of
255 of the crop rebuilt from the raw clip, masks are in training clips only
and cover 20 % of their frames, no test session is in the training set.

## 5. What was tested locally

| Test | Result |
|---|---|
| Test suite | 241 tests pass (4 of them need a free GPU) |
| Clean environment: new Python, only `requirements.txt`, only the uploaded files | installation, tests, preflight, 3 epochs of training, resume, evaluation all ran |
| Small dataset of 736 clips from raw to trained model | every step ran, for video, audio and audio-visual |
| Preflight with the fine-tuning configuration | `READY TO TRAIN` |

Ten problems found by the rehearsal were fixed; they are listed in
`docs/findings.md`.

## 6. Fine-tuning from the Auto-AVSR weights

### 6.1 Background

Training from scratch on this corpus did not work, neither in the earlier
attempt nor in the literature (MultiVSR reports 97 to 105 % WER for
low-resource languages from scratch). The earlier fine-tuning attempt with
the original Auto-AVSR code reached a WER of 97.4 %.

### 6.2 First attempt in this project

Auto-AVSR recipe: front-end and encoder from the checkpoint (604 tensors),
decoder and output layers new, learning rate 0.001 for everything.

| Run | Data | Validation CER | Validation WER |
|---|---|---|---|
| Video, 4 epochs | 8,000 clips | 0.96 | 1.00 |
| Audio, same recipe | 8,000 clips | 0.93 to 0.97 | |

The training loss fell, but every clip received nearly the same hypothesis,
and audio behaved like video. The model had learned which text is likely
and ignored its input.

### 6.3 Diagnosis

Audio was used for the diagnosis because a working recipe shows on audio
within minutes.

| Question | Test | Answer |
|---|---|---|
| Are the weights loaded? | count and compare tensors | yes, 604 tensors; outputs equal to the original code |
| Is the input path right? | Ma's English model on an English clip through this loader | transcribed without error |
| Can the model learn at all? | memorise one batch | yes |
| Is the schedule right? | logged learning rates | yes |
| Do the clips show speech? | frames inspected, including the stillest 2 % | yes |
| Does it depend on what is transferred or frozen? | 4 runs of 600 updates: encoder or all transferred, 0.0001 or 0.0003, front-end frozen or not | no: all four stay at a CTC loss of 6.4 per token and a CER of 0.97. A model that knows nothing has ln(1002) = 6.9 |
| Do Ma's features contain Turkish speech? | encoder frozen, small network trained on the output of block 9, 4,000 clips | yes: CER 0.39 |

The last two rows together are the diagnosis: the pretrained features are
good enough for a CER of 0.39, and training everything at one learning rate
destroys them. The new output layers start from random values and send
large, meaningless gradients into the encoder; at a learning rate that
suits the new layers, the encoder has lost its knowledge before the new
layers can use it.

### 6.4 The recipe that works

| Run, audio, 1,500 updates of 252 frames | Validation CER | WER |
|---|---|---|
| One learning rate (0.0001 or 0.0003), four variants | 0.93 to 0.97 | |
| New layers 0.001, pretrained 0.00001 | 0.50 | 0.85 |
| New layers 0.001, pretrained 0.00002, held still for 500 updates | **0.46** | 0.86 |
| The same with intermediate CTC at blocks 6 and 9 | 0.48 | 0.87 |

Video, with the training script and the configuration of the full run, on
8,000 clips:

| Recipe | Updates | Validation CER | WER |
|---|---|---|---|
| One learning rate 0.001 | 1,696 of 1,800 frames | 0.96 | 1.00 |
| Two learning rates, hold | 2,373 of 252 frames | **0.73** | 0.93 |

The run with two learning rates was stopped after its first epoch, as soon
as it was clear that it learns. These are short runs on 6 % of the data;
they show that the recipe learns, not what the final error rate will be.

### 6.5 Differences from the earlier fine-tuning code (`trainft.py`)

| | Earlier code | This project |
|---|---|---|
| Transferred | every tensor of matching shape | the same, 762 tensors; output layers by name |
| Checkpoint | `vsr_trlrs3vox2_base.pth` | `vsr_trlrs2lrs3vox2avsp_base.pth` (more training data, lower WER) |
| Learning rate | 0.0003 for everything | 0.001 new layers, 0.00002 pretrained |
| Protection of pretrained parts | front-end frozen for 3 epochs | all pretrained parts held for 1 epoch, then the lower rate |
| Vocabulary | 5,089 units | 1,000 units |
| Epochs, warm-up | 150, 10 | 75, 5 |
| Data | 94 h, unfiltered transcripts, 86,185 clips | 96 h after cleaning, 140,823 clips |

A vocabulary of 5,089 units makes the new layers five times larger and
leaves few examples per unit on under 100 hours of speech.

## 7. The environment for the full fine-tuning

`docs/deployment.md` has the commands. The plan:

| | |
|---|---|
| Server | one GPU; 24 GB or more; 32 GB RAM; 50 GB disk |
| Code | cloned from the git repository |
| Data | three files from `upload/`: dataset (11 GB), support files (56 MB), pretrained weights (1 GB), with checksums |
| Software | Python 3.10 or newer, a PyTorch build that fits the driver, `requirements.txt` |
| Before training | `scripts/preflight.py`, then `scripts/benchmark_memory.py` to choose the batch size |
| Training | `configs/finetune/vsr_base.yaml`, 75 epochs, 1800 frames per update |
| Evaluation | average of the 10 kept checkpoints, beam search, test set of two unseen sessions |

Not in git and not uploaded: the raw corpus copy, the landmark cache, the
papers, the reference code, local runs.

## 8. Open points

| Point | Consequence |
|---|---|
| The learning rate of the pretrained parts was chosen with small batches | a pilot with two values on the server is described in `docs/deployment.md` |
| The 241M-parameter model was never run at 1800 frames per batch | memory and speed are measured on the server before the run |
| No A/B experiment has been run | which of the optional techniques help is not known |
| The test set was not checked by a person | every reported number depends on machine-made transcripts |
| Whether low confidence means a wrong transcript was not verified | R6 is the largest cause of removal |
| Dotted and dotless I are one letter | the output cannot distinguish them |
| Details of how the raw corpus was made | listed in section 14 of `docs/dataset_construction.md`, to be added by the authors |
| Low-latency recognition | audited in `docs/roadmap.md`, not started |
