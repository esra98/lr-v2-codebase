# Notes for later work

Dataset cleaning, part of which is done, and low-latency (streaming)
recognition, which is not started. Recorded here so the decisions and the
reasoning are not lost.

## 1. Dataset cleaning

### Decisions (2026-09-27)

- The transcripts were produced with Whisper. They will **not** be
  re-transcribed with Whisper: a second pass of the same model is expected to
  repeat most of the same mistakes.
- Cleaning is analytical. Recurrent transcripts are detected and removed now;
  further checks are listed below.

### What a rebuild needs

The dataset contains only the clips that pass the cleaning, so a change of
the rules means building it again. That is cheap as long as the following are
kept on the local machine: with cached landmarks a rebuild of the whole corpus
takes about an hour and a half instead of most of a day.

| Kept | Where | Used for |
|---|---|---|
| Original label files with word times and confidences | `data/raw/<session>/*.txt` | every text-based check |
| Landmark tracks | `data/cache/landmarks/` | re-cropping without detecting faces again |
| Clip table | `data/tr-lrs-clean-ptm.clips.tsv` | verdict, split, detection rate and head turn of every clip |

### Implemented

Results for the whole corpus are in `docs/label_audit.md`, produced by
`scripts/audit_labels.py`. The same rules are applied by
`scripts/build_dataset.py`, which records the verdict for every clip in the
clip table next to the dataset.

| Rule | How it works |
|---|---|
| Burst | a clip whose transcript equals that of a clip at most 5 positions away in the same session. Transcripts of 5 or more words also match when 80 % of the words agree |
| Bursty sentence | a sentence that occurs at least 3 times in a session, mostly in bursts, is rejected throughout that session |
| Internal loop | a group of 1 to 4 words repeated 3 or more times in a row, or text that compresses by more than 2.4 times |
| Invented phrase | subtitle credits and sign-offs, when they make up at least half of the transcript |
| Squeezed words, speaking rate, confidence, timing | thresholds in `FilterConfig` |

How the thresholds were chosen:

- In five sessions, invented sentences occurred in a single session with all
  occurrences in bursts, while genuine formulae such as `TEŞEKKÜR EDERIM`
  occurred in every session with none in bursts.
- Genuine neighbouring sentences that differ in one word were all short
  (`IKI BIN ON SEKIZ`, `IKI BIN ON DOKUZ`), and true near-duplicates were
  longer (`... GIRESUNUZ`, `... GIRESUNUMUZ`, 8 words). Hence the 5-word
  minimum for approximate matching.

Known cost: genuine formulae said in quick succession are rejected, for
example `KABUL EDENLER` during a series of votes (31 of its 90 occurrences).
These phrases remain plentiful, so the loss does not matter.

### Tested and rejected

**Lip movement against audio energy.** Idea: if the visible person is not
the speaker, lip opening should not follow the loudness of the audio.
Measured on the five local sessions, as the correlation between the two over
the clip, maximised over a lag of up to 4 frames:

| Clips | Number | Median correlation | Median lip motion |
|---|---|---|---|
| Kept by the filter | 578 | 0.20 | 0.0192 |
| Recurrent transcript | 532 | 0.24 | 0.0207 |
| Invented phrase | 363 | 0.29 | 0.0195 |
| Internal loop | 55 | 0.13 | 0.0172 |

Clips with invented transcripts show as much lip movement, and as much
agreement between lips and audio, as good clips. People are speaking in them;
the text is simply wrong. This measure does not separate good from bad clips
and is not used. A learned audio-visual synchronisation model might do
better, but that is a separate project.

### Candidate checks not yet done

| # | Check | Catches | Needs |
|---|---|---|---|
| 1 | Restoring dotted and dotless I from a Turkish lexicon and vowel harmony | the casing loss: both letters are `I` in the labels | a lexicon or morphological analyser |
| 2 | Loss-based filtering with a first trained audio model | labels that disagree with the audio | one trained model |
| 3 | Duplicated footage across sessions, by label sequences or audio fingerprints | the same recording under two names | labels, audio |
| 4 | Lip opening inside word boundaries | transcript of another segment | labels, landmarks |

Check 2 is not circular in the way re-transcription is: it uses a model
trained on this data to find labels that are inconsistent with the rest.

### Open items

- Tarballs 15 to 33 contain folders named `video-1` to `video-19`, which do
  not match the tarball numbers (`video-15.tar.gz` holds `video-1`). Session
  identity comes from the folder name. No two tarballs share a folder name.
- The test set is two whole sessions with few recurrent transcripts
  (`2025-2026-birlesim-3`, `video-10`). It deserves a manual check of a few
  hundred clips, because every reported number depends on it.
- The confidence threshold rejects 11.3 % of clips and is the largest single
  cause of rejection. Whether low confidence really means a wrong transcript
  has not been verified.

## 2. Low-latency recognition

### Audit of the current model

The model is an offline recogniser. Each part below looks at future frames:

| Component | Future context | Latency |
|---|---|---|
| Visual front-end, 3D convolution (5 frames) | 2 frames | 80 ms |
| Conformer convolution, kernel 31, in each of 12 blocks | 15 frames per block, 180 in total | 7.2 s |
| Conformer self-attention | whole utterance | unbounded |
| Attention decoder and beam search | whole utterance | unbounded |
| Crop smoothing, Kalman with backward pass | whole clip | unbounded |
| Crop scale, median face height of the clip | whole clip | unbounded |

Not an obstacle: batch normalisation (fixed statistics at inference), the CTC
output (frame-synchronous) and landmark detection (about 84 frames per second
on one CPU core here, against 25 needed).

### Ways to get there

| Option | Change | Cost | Comment |
|---|---|---|---|
| A. Chunked attention with limited-context convolutions | attention mask restricted to the current and past chunks; smaller or causal kernels; training with random chunk sizes so that one model serves both modes | medium | keeps this codebase and its pretrained weights; full-context accuracy is retained as the upper bound |
| B. Two-pass decoding | streaming CTC output, rescored by the attention decoder at the end of the utterance | low, on top of A | first words appear immediately, final text is more accurate |
| C. Transducer with a streaming encoder | Emformer encoder and RNN-T loss, as in the torchaudio real-time AV-ASR recipe by the Auto-AVSR authors | high | a different model family; public Auto-AVSR weights do not transfer |

Recommendation: option A with B, after offline accuracy is satisfactory. It
is an added attention mask and a training flag rather than a rewrite, and the
same checkpoint can be evaluated in both modes, which gives a direct
measurement of what low latency costs in accuracy.

### What this means for the dataset

Only one thing. At inference a streaming system must cut crops with a
**causal** smoother, so the crops will jitter slightly more than crops made
with the backward pass. Two ways to handle that, neither needed now:

- build the dataset with `--smoothing one_euro` (causal), or
- keep the Kalman-smoothed crops and rely on the small random shifts that
  the 88 px random crop already provides.

The landmark cache makes producing another variant cheap, so this does not
have to be decided before the dataset is created.

### Latency budget to decide on

Before implementing, a target has to be chosen, because it fixes the chunk
size. As an illustration, a chunk of 16 frames (640 ms) with 4 frames of
look-ahead gives about 0.8 s algorithmic delay; shorter chunks cost accuracy.
