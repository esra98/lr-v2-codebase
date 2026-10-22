# Construction of the TR-LRS corpus (version 3, October 2026)

How the dataset `dataset/tr-lrs-v3` was made from the full-length recordings
of the Grand National Assembly of Türkiye: how the recordings were
transcribed, cut into utterances, cleaned, cropped to the mouth region, split
and verified. It is written to support the methods section of a paper. Every
number was measured on the dataset built on 2026-10-22.

The first corpus (September 2026) was built from 224x224 face clips whose
transcripts had lost the distinction between dotted and dotless I; its
construction is recorded in `docs/archive/dataset_construction_v2_2026-09-28.md`.
Version 3 starts again from the recordings, so every step is in this
repository and reproducible with `scripts/corpus.py`.

## 1. Summary

| | Utterances | Hours |
|---|---|---|
| Recordings | 39 | 332.4 (video) |
| Utterances found by the transcription | 272,258 | 271.0 (speech spans) |
| Removed by the transcript rules (section 4) | 10,963 (4.0 %) | |
| Removed by the video checks (section 5) | 35,318 (13.0 %) | |
| **Dataset** | **225,977 (83.0 %)** | **224.6** |

| Split | Clips | Hours | Sessions |
|---|---|---|---|
| Training | 208,811 | 207.8 | 37 |
| Validation | 10,993 | 10.5 | 37 |
| Test | 6,173 | 6.4 | 2 |

The clips have the LRS2 layout, `<session>/<utterance>.mp4` with a
`<utterance>.txt` next to it: a 96x96 colour mouth crop at 25 frames per
second with the 16 kHz mono audio of the utterance in the same file, and a
label file with the sentence in Whisper's own casing and one line per word
with start time, end time and confidence.

## 2. Recordings

| Property | Value |
|---|---|
| Content | plenary sessions of the Grand National Assembly of Türkiye |
| Recordings | 39: 19 from the assembly's own archive (`<term>-birlesim-<n>`), 20 from the assembly's YouTube channel (`yt-<video id>`) |
| Length | 332.4 hours; 4.4 to 11.6 hours per recording |
| Video | 1920x1080, 30 frames per second, H.264 |
| Audio | AAC, 44.1 kHz stereo (downloads) or as broadcast; converted to 16 kHz mono |
| Size | 180 GB |

YouTube recordings were fetched with `yt-dlp` at the largest H.264 format up
to 1080p (`scripts/corpus.py download`). Two recordings that were only
available at 360p were replaced by their 1080p versions before cutting; no
recording below 720p was cut. `videos/sources.tsv` lists every recording with
its resolution, length, size and SHA-256 checksum.

The recordings show the rostrum, the chair and the chamber from a few fixed
cameras, with cuts between them every few seconds. Close-ups of the speaker
regularly show one or two listeners behind or beside them.

## 3. Procedure

Each recording is processed on its own, in four stages that are each
resumable and are only repeated when their inputs or options change
(`work/<session>/status.json` records a hash of both).

| Stage | Input | Output | Decides |
|---|---|---|---|
| 1. Transcribe | recording | `transcript.json` | the words, their times and confidences |
| 2. Segment | transcript | `segments.tsv` | utterance boundaries; the transcript verdicts (section 4) |
| 3. Cut | recording, segments | clips, `clips.tsv`, `landmarks/` | the speaker; the video verdicts (section 5); the crop (section 6) |
| 4. Build | all `clips.tsv` | clip table, manifests | the split (section 7) |

Stage 3 only reads the utterances that passed stage 2, so face detection,
by far the most expensive step, never runs on a rejected transcript. An
utterance is removed at the first check it fails, so every removed utterance
is counted under exactly one reason.

### 3.1 Transcription

| Setting | Value |
|---|---|
| Recogniser | Whisper `large-v3` through `faster-whisper` 1.2.1 (CTranslate2 4.8.3) |
| Precision | `int8_float16` (the model then fits a 6 GB GPU) |
| Language | Turkish, fixed |
| Decoding | beam search, 5 beams; word timestamps; Silero voice-activity filter; no conditioning on the previous window |
| Chunks | 20 minutes of 16 kHz mono audio extracted with `ffmpeg`; 1,017 chunks in all |

Each chunk is transcribed independently and saved; the chunks are merged
with their times shifted into the recording. The transcript keeps the text
in Whisper's casing and, for every Whisper segment, its `no_speech_prob`,
`avg_logprob` and `compression_ratio`. No transcript was checked by a
person.

### 3.2 Segmentation

The words of each Whisper segment are grouped into utterances:

1. Whisper segment boundaries are never crossed.
2. A word ending in `.`, `?` or `!` ends the utterance.
3. A pause longer than 0.4 s between two words ends the utterance.
4. An utterance longer than 10 s is split at its longest internal pause,
   repeatedly, so no word is cut in the middle.
5. Utterances with fewer than 2 words or shorter than 0.4 s are dropped.

The span of an utterance runs from 0.04 s before its first word to 0.04 s
after its last word. Utterances are numbered in time order within the
recording (`00000`, `00001`, ...); rejected ones keep their number and leave
a gap in the dataset folder.

## 4. Transcript cleaning

### 4.1 Normalisation

Transcripts are compared, and used as training targets, in a normalised
form. The label files keep Whisper's text unchanged.

| Rule | Example |
|---|---|
| Unicode NFC | |
| Circumflex removed | `hâlâ` becomes `HALA` |
| Upper case with Turkish casing | `i` becomes `İ`, `ı` becomes `I` |
| Digits spelled out in Turkish | `1921` becomes `BİN DOKUZ YÜZ YİRMİ BİR` |
| A suffix after an apostrophe joins the number | `1921'de` becomes `... YİRMİ BİRDE` |
| Thousands dots, decimal commas, percent | `1.250.000`; `3,5` becomes `ÜÇ VİRGÜL BEŞ`; `%50` becomes `YÜZDE ELLİ` |
| Apostrophes removed, joining the suffix | `Türkiye'nin` becomes `TÜRKİYENİN` |
| Everything that is not a letter becomes a space | punctuation, hyphens |

The alphabet has 32 letters: the 29 letters of Turkish and `Q`, `W`, `X`
for foreign words (which occur 10, 172 and 59 times in the dataset). The
dataset contains 899,103 `İ` and 450,382 `I`, in line with the frequencies
of the two letters in written Turkish.

### 4.2 Rules

Let a recording contain utterances u₁ … uₙ in time order, with normalised
transcripts t₁ … tₙ, each a sequence of words.

**R1. Known invented phrase.** Whisper emits subtitle credits and sign-offs
from its training data on silence or noise. An utterance is removed if one of
the phrases below occurs in it as a sequence of whole words and makes up at
least half of its words.

`ALTYAZI M K`, `ALTYAZI`, `ABONE OL`, `İZLEDİĞİNİZ İÇİN TEŞEKKÜR`,
`İZLEDİĞİNİZ İÇİN TEŞEKKÜRLER`, `BİR SONRAKİ VİDEODA GÖRÜŞMEK ÜZERE`,
`SESLENDİREN`, `ÇEVİRİ VE ALTYAZI`

**R2. Internal loop.** An utterance is removed if a group of 1 to 4 words is
repeated at least 3 times in a row, or if its transcript has at least 40
characters and compresses by a factor above 2.4 with zlib (the test Whisper
uses itself to detect failed decoding).

**R3. Burst.** Two transcripts are *the same* if they are identical, or if
both have at least 5 words and 1 − d(tᵢ, tⱼ) / max(|tᵢ|, |tⱼ|) ≥ 0.8, where
d is the edit distance in words. Utterance uᵢ is removed if some uⱼ of the
same recording with 0 < |i − j| ≤ 5 has the same transcript. All members of
a burst are removed, because nothing indicates which of them is correct.

**R4. Bursty sentence.** If a sentence occurs at least 3 times in a
recording and at least half of those occurrences were removed by R3, its
remaining occurrences in that recording are removed too.

**R5. Squeezed words.** A word is squeezed if it lasts less than 15 ms per
character. An utterance is removed if more than 25 % of its words are
squeezed.

**R6. Low confidence.** An utterance is removed if the mean word confidence
is below 0.80 or any word's confidence is below 0.30.

**R7. No speech.** An utterance is removed if Whisper's `no_speech_prob` for
its window is above 0.6. On this corpus invented transcripts have a median
of 0.7 to 0.8 and genuine ones 0.03.

**R8. Speaking rate.** An utterance is removed if it has fewer than 4 or
more than 25 letters per second of its span.

The thresholds of R1 to R6 are those of the first corpus, where they were
derived from the data and examined (see the archived document, section
4.4). R7 is new: the first corpus had no access to Whisper's own estimate.
The timing rule of the first corpus (speech must end near the end of the
clip) is no longer needed, because the clips are cut around the words.

### 4.3 Result

| Rule | Utterances removed | Share |
|---|---|---|
| R1 known invented phrase | 11 | 0.0 % |
| R2 internal loop | 142 | 0.1 % |
| R3 burst | 1,867 | 0.7 % |
| R4 bursty sentence | 151 | 0.1 % |
| R5 squeezed words | 166 | 0.1 % |
| R6 low confidence | 6,191 | 2.3 % |
| R7 no speech | 2,062 | 0.8 % |
| R8 speaking rate | 373 | 0.1 % |
| **Transcript, total** | **10,963** | **4.0 %** |

The first corpus lost 27 % of its clips to the transcript rules, 10.6 % to
recurrent or invented transcripts alone. With `large-v3`, the voice-activity
filter and no conditioning on the previous window, the recurrent and
invented transcripts (R1 to R4) fall to 0.8 %, and the confidence rule to
2.3 %. The median word confidence of the kept utterances is 0.984; the 5th
percentile is 0.90.

## 5. Cutting and video checks

The recording is decoded once from start to end. For each kept utterance,
the frames nearest to a 25 frames-per-second grid over its span are taken
(the recordings run at 30 frames per second; a frame is repeated if the
decoder skips one), and the audio of the span is extracted as 16 kHz mono.

### 5.1 Face detection

Landmarks come from the MediaPipe Face Landmarker (`face_landmarker.task`,
float16, 478 points per face), run on every second frame of the utterance
with detection and presence thresholds of 0.3 and up to 4 faces per frame;
landmarks of the frames in between are interpolated linearly in time.

The detector shrinks its input to 192x192 pixels, so a face of 200 to 300
pixels in a 1920x1080 frame is often missed when the whole frame is given to
it. Faces are therefore searched in square tiles: a copy of the frame shrunk
to 540 rows is scanned tile by tile (tiles of 540 pixels with half overlap)
every 16 detections, and every face found is then detected again at full
resolution inside a square of 2.5 face heights around its last position.
Faces found twice are merged. On the first recording this raised the number
of kept utterances from 124 to 276 of 301.

### 5.2 The speaker

Close-ups in the chamber regularly show a listener next to the speaker, and
the listener is often the larger face. Among the faces present in at least
half of the sampled frames and at least half as tall as the tallest, the
speaker is the one whose lip opening (distance between the inner lips,
divided by the face height) varies most over the utterance. 87.8 % of the
kept clips have one candidate face; 7.3 % have two, 3.4 % three, and 1.4 %
four to six.

### 5.3 Checks

| Check | Condition for removal | Utterances removed | Share |
|---|---|---|---|
| Too short | fewer than 12 frames (0.48 s) | 0 | |
| Too long | more than 600 frames (24 s) | 0 | |
| No face | a face is found in fewer than 50 % of the sampled frames | 23,225 | 8.5 % |
| Low detection rate | a face is found in 50 % to 80 % of the sampled frames | 11,862 | 4.4 % |
| Small face | median face height below 50 pixels | 11 | 0.0 % |
| Mouth at the frame edge | the chosen speaker's mouth region leaves the frame | 220 | 0.1 % |
| **Video, total** | | **35,318** | **13.0 %** |

Utterances without a face are mostly wide shots of the chamber and cuts to
the chair while a member speaks from the floor. A face is found in 99.5 % of
the sampled frames of the kept clips, and in every sampled frame of 94.4 %
of them.

## 6. Mouth region

### 6.1 Landmarks used

Of the 478 points, 19 are used:

| Points | MediaPipe indices |
|---|---|
| Outer lips | 61, 291, 0, 17, 37, 267, 84, 314 |
| Eye corners | 33, 133, 362, 263 |
| Nose tip | 1 |
| Forehead, chin | 10, 152 |
| Inner lips | 13, 14, 78, 308 |

The landmarks of every kept utterance are stored in
`work/<session>/landmarks/<utterance>.npz`, so the crop can be redone with
other settings without detecting faces again.

### 6.2 Position

The centre of the crop in frame t is the mean of the 8 outer-lip points,
smoothed over time with a Kalman filter (constant-velocity model, the two
coordinates independent) followed by a Rauch-Tung-Striebel backward pass.
With state (position, velocity) and one frame as the unit of time:

  F = [[1, 1], [0, 1]], H = [1, 0], Q = q · [[1/4, 1/2], [1/2, 1]], R = r,

with q = 1.0 px² per frame⁴ and r = 4.0 px². Detections jitter by a pixel
or two from frame to frame; cropping at the raw positions would turn that
jitter into apparent motion of the whole mouth region.

### 6.3 Scale

The side of the cropped square is 0.63 times the face height, the distance
between the forehead and chin points, taken as the median over the frames of
the utterance. One value per utterance is used because the chin is one end
of the face height: a value per frame would make the crop zoom as the jaw
opens. The factor 0.63 reproduces the area of the 96-pixel crop of Auto-AVSR
(measured on 567 clips of the first corpus: median 0.630, 10th to 90th
percentile 0.605 to 0.667), so weights trained on Auto-AVSR crops remain
applicable.

In the kept clips the face is 167 to 346 pixels tall (5th to 95th
percentile; median 201), so the mouth region is cut from 105 to 218 source
pixels and shrunk to 96.

### 6.4 Cropping and encoding

The square is mapped to 96x96 pixels by an affine transform with bilinear
interpolation; pixels outside the frame repeat the border. The crop is not
rotated. Colour is kept (the loader converts to grey). The video is encoded
with libx264, constant rate factor 18, yuv420p, 25 frames per second; the
audio as AAC, 16 kHz mono. Partition-Time Masking is not written into the
clips; it is applied during training with a new mask every epoch.

### 6.5 Differences from the Auto-AVSR preprocessing

| | Auto-AVSR | This dataset |
|---|---|---|
| Landmark detector | RetinaFace with FAN, or MediaPipe face detection | MediaPipe Face Landmarker, tiled search, 4 faces |
| Speaker | the one detected face | lip-motion choice among the tracked faces |
| Alignment | similarity transform to a mean face, per frame | none; translation and one scale per utterance |
| Smoothing | moving average over 12 frames | Kalman filter with backward pass |
| Scale reference | eyes, nose and mouth | forehead-to-chin distance |
| Crop | 96x96, grey | 96x96, colour (grey when loaded) |

## 7. Split

| Split | Definition |
|---|---|
| Test | all clips of the recordings `2025-2026-birlesim-3` (archive) and `yt-NPJohShU_iw` (YouTube) |
| Validation | the last 5 % of the kept clips of every other recording |
| Training | the remaining clips |

Test recordings are whole recordings and validation clips are a contiguous
block because neighbouring clips come from the same speech: a random split
would place parts of one speech on both sides. The two test recordings have
few recurrent transcripts (0.5 % and 0.8 %) and one comes from each source.

## 8. Resulting dataset

| | Training | Validation | Test | Total |
|---|---|---|---|---|
| Clips | 208,811 | 10,993 | 6,173 | 225,977 |
| Hours | 207.8 | 10.5 | 6.4 | 224.6 |
| Recordings | 37 | 37 | 2 | 39 |
| Words | 1,408,581 | 71,398 | 43,491 | 1,523,470 |
| Distinct words | 105,994 | 17,591 | 12,888 | 110,591 |
| Words not seen in training | | 4.3 % | 4.5 % | |

| Per clip | Minimum | 5th percentile | Median | Mean | 95th percentile | Maximum |
|---|---|---|---|---|---|---|
| Frames | 12 | 25 | 78 | 89.5 | 222 | 252 |
| Seconds | 0.48 | 1.0 | 3.12 | 3.58 | 8.88 | 10.08 |
| Words | 2 | 2 | 6 | 6.74 | 16 | 35 |

Of the 110,591 distinct words, 55,434 occur once. The dataset contains
214,831 distinct sentences. 17.3 % of the kept clips show a strongly turned
head (yaw proxy above 0.5). Size on disk: 14 GB, 451,954 files.

Compared with the first corpus (154,191 clips, 105.4 hours, 4.9 words per
clip) the dataset has 2.1 times the hours and longer utterances, because the
utterances are cut at sentence ends and pauses rather than at a fixed length.

### By recording

Utterances are those found by the transcription. "Recurrent" is the share
removed by R1 to R4, "no face" the share removed by the video checks.

| Recording | Source | Video h | Utterances | Kept | Kept | Hours | Recurrent | Low conf. | No face | Split |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024-2025-birlesim-72 | archive | 9.2 | 8,092 | 6,905 | 85.3 % | 6.95 | 0.6 % | 2.0 % | 11.6 % | train, val |
| 2024-2025-birlesim-73 | archive | 8.7 | 5,938 | 4,762 | 80.2 % | 5.32 | 0.8 % | 2.1 % | 15.3 % | train, val |
| 2024-2025-birlesim-74 | archive | 8.2 | 6,619 | 5,551 | 83.9 % | 5.70 | 0.3 % | 2.6 % | 12.4 % | train, val |
| 2024-2025-birlesim-76 | archive | 7.7 | 6,610 | 5,416 | 81.9 % | 5.39 | 0.4 % | 1.8 % | 14.6 % | train, val |
| 2024-2025-birlesim-80 | archive | 4.4 | 3,546 | 2,912 | 82.1 % | 2.93 | 0.8 % | 2.6 % | 12.3 % | train, val |
| 2024-2025-birlesim-82 | archive | 9.3 | 8,081 | 6,669 | 82.5 % | 6.35 | 0.7 % | 2.3 % | 13.5 % | train, val |
| 2024-2025-birlesim-84 | archive | 8.8 | 7,076 | 6,121 | 86.5 % | 6.60 | 0.4 % | 1.8 % | 10.0 % | train, val |
| 2024-2025-birlesim-94 | archive | 11.3 | 6,948 | 5,204 | 74.9 % | 5.15 | 0.9 % | 2.2 % | 18.8 % | train, val |
| 2024-2025-birlesim-109 | archive | 11.2 | 8,441 | 6,781 | 80.3 % | 6.88 | 1.2 % | 2.6 % | 13.9 % | train, val |
| 2024-2025-birlesim-110 | archive | 11.6 | 8,297 | 6,514 | 78.5 % | 6.30 | 0.8 % | 2.6 % | 16.9 % | train, val |
| 2025-2026-birlesim-1 | archive | 8.5 | 6,415 | 5,099 | 79.5 % | 5.30 | 0.7 % | 2.4 % | 15.5 % | train, val |
| 2025-2026-birlesim-2 | archive | 6.7 | 5,721 | 4,952 | 86.6 % | 4.90 | 0.7 % | 2.6 % | 9.4 % | train, val |
| 2025-2026-birlesim-3 | archive | 4.8 | 3,848 | 3,129 | 81.3 % | 3.31 | 0.5 % | 2.5 % | 15.0 % | test |
| 2025-2026-birlesim-5 | archive | 8.1 | 7,182 | 6,104 | 85.0 % | 5.87 | 0.7 % | 2.9 % | 10.1 % | train, val |
| 2025-2026-birlesim-7 | archive | 7.9 | 6,402 | 5,493 | 85.8 % | 5.43 | 1.0 % | 1.7 % | 11.0 % | train, val |
| 2025-2026-birlesim-8 | archive | 8.0 | 6,269 | 5,087 | 81.1 % | 5.08 | 0.6 % | 2.4 % | 14.6 % | train, val |
| 2025-2026-birlesim-9 | archive | 8.0 | 6,177 | 5,321 | 86.1 % | 5.87 | 0.5 % | 2.0 % | 10.6 % | train, val |
| 2025-2026-birlesim-10 | archive | 10.4 | 7,688 | 5,900 | 76.7 % | 6.07 | 0.6 % | 3.1 % | 17.3 % | train, val |
| 2025-2026-birlesim-12 | archive | 7.2 | 5,313 | 4,072 | 76.6 % | 4.35 | 0.5 % | 2.8 % | 19.0 % | train, val |
| yt-1DIP6Ruvtgg | YouTube | 10.4 | 8,262 | 6,751 | 81.7 % | 6.23 | 0.6 % | 2.0 % | 14.6 % | train, val |
| yt-45SXDKapxh8 | YouTube | 6.8 | 6,048 | 5,241 | 86.7 % | 4.96 | 0.5 % | 1.8 % | 9.9 % | train, val |
| yt-Fpah9dtIsFo | YouTube | 7.2 | 5,905 | 4,992 | 84.5 % | 5.16 | 0.4 % | 1.9 % | 12.9 % | train, val |
| yt-HTCukfIyxoM | YouTube | 7.9 | 7,210 | 6,128 | 85.0 % | 5.63 | 0.7 % | 2.2 % | 11.8 % | train, val |
| yt-MZtO3ofW7eA | YouTube | 8.6 | 6,775 | 5,796 | 85.5 % | 5.84 | 0.5 % | 1.9 % | 10.9 % | train, val |
| yt-NPJohShU_iw | YouTube | 4.5 | 3,772 | 3,044 | 80.7 % | 3.08 | 0.8 % | 2.4 % | 15.2 % | test |
| yt-OPGJ5-ARIC4 | YouTube | 8.2 | 7,189 | 6,231 | 86.7 % | 6.04 | 0.4 % | 1.9 % | 10.2 % | train, val |
| yt-Pw91kFFU44o | YouTube | 7.2 | 5,582 | 4,458 | 79.9 % | 4.83 | 0.4 % | 3.1 % | 15.4 % | train, val |
| yt-QEdmcWbgxrg | YouTube | 6.8 | 5,227 | 4,135 | 79.1 % | 4.05 | 0.9 % | 2.6 % | 15.8 % | train, val |
| yt-S_sCbEZeqCc | YouTube | 11.3 | 11,064 | 9,111 | 82.3 % | 8.25 | 4.4 % | 2.3 % | 10.4 % | train, val |
| yt-UoHW7olV1LU | YouTube | 11.1 | 9,810 | 7,999 | 81.5 % | 7.68 | 1.0 % | 1.9 % | 15.1 % | train, val |
| yt-_3u6pzqTuoo | YouTube | 11.3 | 10,070 | 8,970 | 89.1 % | 8.81 | 0.3 % | 2.2 % | 7.9 % | train, val |
| yt-_8zUPRDIwKM | YouTube | 8.7 | 6,574 | 5,658 | 86.1 % | 6.07 | 0.2 % | 2.3 % | 10.6 % | train, val |
| yt-fP99G1PONhQ | YouTube | 6.5 | 5,489 | 4,654 | 84.8 % | 4.66 | 0.4 % | 2.1 % | 11.8 % | train, val |
| yt-hkcAzBtqACE | YouTube | 8.6 | 7,276 | 6,176 | 84.9 % | 5.96 | 0.5 % | 1.9 % | 11.6 % | train, val |
| yt-io-FZa_APv4 | YouTube | 11.0 | 9,699 | 8,173 | 84.3 % | 7.88 | 0.8 % | 2.7 % | 11.4 % | train, val |
| yt-l419Asc8yqE | YouTube | 11.4 | 9,960 | 8,479 | 85.1 % | 8.19 | 0.6 % | 2.0 % | 11.9 % | train, val |
| yt-r7XQxD_fBz4 | YouTube | 10.4 | 8,942 | 7,379 | 82.5 % | 7.29 | 1.1 % | 1.9 % | 13.8 % | train, val |
| yt-sLfJGTUSnjw | YouTube | 5.8 | 5,251 | 4,200 | 80.0 % | 4.08 | 0.7 % | 2.5 % | 16.1 % | train, val |
| yt-xZkvTV0J1PI | YouTube | 8.9 | 7,490 | 6,410 | 85.6 % | 6.19 | 0.5 % | 2.8 % | 10.3 % | train, val |

The kept share is between 75 % and 89 % in every recording. In the first
corpus it ranged from 15 % to 83 %, because two recordings were dominated by
invented transcripts (`ÖZGÜR KARABAT` 6,146 times in one of them); with the
new transcription the worst recording has 4.4 % recurrent transcripts.

## 9. Verification

`scripts/corpus.py verify` checks the built dataset. Checks that decode
clips were run on 300 random clips of each split, 900 in all; the others
cover every clip. All passed on 2026-10-22.

| Check | Scope |
|---|---|
| Every kept clip has its mp4 and txt; no other file is in the session folders | all |
| Every kept clip has a split; test clips are exactly the clips of the two test recordings | all |
| Each manifest lists exactly the kept clips of its split | all |
| No clip is masked | all |
| Kept texts use the alphabet only; both `İ` and `I` occur | all |
| No kept text is a known invented phrase or an internal loop | all |
| The mp4 has H.264 square video at 25 frames per second as stream 0 and 16 kHz AAC as stream 1 | 900 |
| The number of decoded frames equals the clip table | 900 |
| The decoded audio is as long as the video (up to the AAC block padding) | 900 |
| The label file's text normalises to the clip table's text | 900 |
| Every word time of the label file lies within the clip | 900 |

## 10. Limitations

**Transcripts.**

- The transcripts, including those of the test set, are the output of a
  speech recogniser. Error rates measured on this test set are error rates
  against that recogniser, not against what was said. A manually corrected
  test set would be needed to state absolute accuracy.
- The rules remove transcripts that are implausible. They cannot find a
  transcript that is fluent but wrong.
- The thresholds of R5 and R8 were chosen by judgment and not examined.

**Split.**

- The same members of parliament speak in many recordings, so the test set
  is not independent of the training set in terms of speakers, and speaker
  identities are not annotated.
- Validation clips come from the same recordings as training clips.

**Video.**

- The speaker is chosen by lip motion. When a listener moves their mouth
  more than the speaker over an utterance, the wrong face is cropped; this
  was not measured.
- The crop is not rotated, so head roll remains in the data.
- Landmarks were not checked by hand. A detection that is present but wrong
  is not noticed.

**Domain.** All clips show formal speech in one room, recorded by the same
cameras.

## 11. Reproduction

```bash
python scripts/corpus.py ingest                      # videos/*.mp4 -> videos/sources.tsv
python scripts/corpus.py download                    # YouTube sessions below 720p -> 1080p
python scripts/corpus.py run --dataset dataset/tr-lrs-v3 --workers 7 --gpu-workers 4
python scripts/corpus.py build --dataset dataset/tr-lrs-v3 --holdout 2025-2026-birlesim-3 yt-NPJohShU_iw
python scripts/corpus.py verify --dataset dataset/tr-lrs-v3
python scripts/train_tokenizer.py --manifest dataset/tr-lrs-v3-manifests/train.tsv --out-dir dataset/tr-lrs-v3-tokenizer
```

| File | Content |
|---|---|
| `videos/sources.tsv` | every recording: origin, resolution, length, size, checksum |
| `work/<session>/transcript.json` | the Whisper transcript with word times, confidences and decoder statistics |
| `work/<session>/segments.tsv` | every utterance with its span, texts, statistics and transcript verdict |
| `work/<session>/clips.tsv` | the cutter's verdict, frame count, detection rate, faces, face height, yaw and lip motion per utterance |
| `work/<session>/landmarks/` | the chosen face's landmarks per utterance, for re-cropping |
| `dataset/tr-lrs-v3.clips.tsv` | every utterance with its verdict and split |
| `dataset/tr-lrs-v3.settings.json` | the options of each stage and the split |
| `avsr/corpus/` | the four stages; `avsr/data/{manifest,recurrence}.py` the rules; `avsr/preprocess/` landmarks, smoothing, cropping |

| Software | Version |
|---|---|
| Python | 3.13.3 |
| faster-whisper / CTranslate2 | 1.2.1 / 4.8.3 |
| MediaPipe | 0.10.35; Face Landmarker float16, MD5 `b0e7274907a1644404fef66b28dd6d85` |
| PyAV | 16.1.0 (libavcodec 62.11.100) |
| OpenCV | 4.13.0 |
| ffmpeg, yt-dlp | 7.1.1, 2026.08.19 |

Machine: 16 threads, 15 GB of memory, an RTX 2060 (6 GB) for Whisper and
an Intel UHD graphics unit for four of the seven face-detection workers.
Transcription took 2026-10-17 to 2026-10-20 (about 1.5 hours of GPU time
per 10 hours of recording); cutting ran alongside it at about 80 utterances
per minute and finished on 2026-10-21, including a second pass that recut
the utterances whose speaker had been found outside the stored region. The
landmark cache is 2.2 GB; with it, re-cropping the whole corpus with other
settings needs no face detection.

## 12. References

| Used for | Reference |
|---|---|
| Baseline preprocessing and model | P. Ma, S. Petridis, M. Pantic, "Visual speech recognition for multiple languages in the wild", Nature Machine Intelligence, 2022 |
| Baseline preprocessing and model | P. Ma et al., "Auto-AVSR: Audio-visual speech recognition with automatic labels", ICASSP 2023 |
| Transcripts; compression-ratio test of R2 | A. Radford et al., "Robust speech recognition via large-scale weak supervision", ICML 2023 |
| Transcription engine | faster-whisper (SYSTRAN), CTranslate2 |
| Voice activity filter | Silero VAD |
| Landmark detection | C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines", 2019; Y. Kartynnik et al., "Real-time facial surface geometry from monocular video on mobile GPUs", 2019 |
| Kalman filter and backward pass | R. E. Kalman, 1960; H. E. Rauch, F. Tung, C. T. Striebel, 1965 |
| Partition-Time Masking (applied during training) | "A novel approach for visual speech recognition using the partition-time masking", Sensors, 2025 |
