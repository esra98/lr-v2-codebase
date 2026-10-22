# Notes for later work

Dataset cleaning, most of which is done, and low-latency (streaming)
recognition, which is not started. Recorded here so the decisions and the
reasoning are not lost.

## 1. Dataset cleaning

### Decisions

- 2026-09-27: the transcripts were produced with Whisper `small` and would
  not be re-transcribed with the same model. Cleaning is analytical.
- 2026-10-17: the corpus is rebuilt from the full recordings
  (`docs/audit_and_roadmap_2026-10-17.md`), re-transcribed with
  `large-v3` and cut into utterances at sentence ends and pauses. This
  recovered the dotted and dotless I and cut the share of recurrent or
  invented transcripts from 10.6 % to 0.8 %.

### What a rebuild needs

The dataset contains only the utterances that pass the cleaning, so a change
of the rules means building it again. With the per-session state kept, that
is cheap: a rule change in `segment` reruns the text rules in seconds, and a
crop change in `cut` reuses the landmarks.

| Kept | Where | Used for |
|---|---|---|
| Transcripts with word times, confidences and decoder statistics | `work/<session>/transcript.json` | every text-based check |
| Landmark tracks of the chosen speaker | `work/<session>/landmarks/` | re-cropping without detecting faces again |
| Clip table | `dataset/tr-lrs-v3.clips.tsv` | verdict, split, detection rate and head turn of every utterance |

### Implemented

The rules are applied by `scripts/corpus.py segment` (text) and `cut`
(video); `build` records the verdict of every utterance in the clip table
next to the dataset, and `docs/dataset_construction.md` has the counts.

| Rule | How it works |
|---|---|
| Burst | a clip whose transcript equals that of a clip at most 5 positions away in the same session. Transcripts of 5 or more words also match when 80 % of the words agree |
| Bursty sentence | a sentence that occurs at least 3 times in a session, mostly in bursts, is rejected throughout that session |
| Internal loop | a group of 1 to 4 words repeated 3 or more times in a row, or text that compresses by more than 2.4 times |
| Invented phrase | subtitle credits and sign-offs, when they make up at least half of the transcript |
| Squeezed words, speaking rate, confidence | thresholds in `FilterConfig` |
| No speech | Whisper's own `no_speech_prob` above 0.6 (`SegmentConfig`) |

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
| 1 | Loss-based filtering with a first trained audio model | labels that disagree with the audio | one trained model |
| 2 | Duplicated footage across recordings, by label sequences or audio fingerprints | the same session under two names | labels, audio |
| 3 | Lip opening inside word boundaries | transcript of another segment | labels, landmarks |
| 4 | Audio-visual synchronisation score of the chosen face | the listener chosen instead of the speaker | a synchronisation model |

Check 1 is not circular in the way re-transcription is: it uses a model
trained on this data to find labels that are inconsistent with the rest.
The dotted-I restoration that was on this list is no longer needed: the
new transcripts carry Whisper's casing.

### Open items

- The test set is two whole recordings with few recurrent transcripts
  (`2025-2026-birlesim-3`, `yt-NPJohShU_iw`). It deserves a manual check of
  a few hundred clips, because every reported number depends on it.
- The confidence threshold rejects 2.3 % of utterances and is the largest
  text-based cause of rejection. Whether low confidence really means a
  wrong transcript has not been verified on the new transcripts.
- 13 % of the utterances have no usable face (wide shots, the chair while a
  member speaks from the floor). Nothing can be done about them without
  other footage.
- The speaker choice by lip motion was checked on a few dozen clips, not
  measured. A sample of the clips with two or more candidate faces (12 % of
  the dataset) should be inspected.

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
