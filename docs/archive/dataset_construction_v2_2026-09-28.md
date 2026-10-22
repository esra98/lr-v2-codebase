# Construction and cleaning of the TR-LRS training dataset

This document describes how the training dataset `tr-lrs-clean-ptm` was
derived from the raw TR-LRS corpus: which clips were removed and why, how the
mouth region was extracted, how Partition-Time Masking was applied, and how
the result was verified. It is written to support the methods section of a
paper. Every number was measured on the dataset built on 2026-09-28.

Items marked **[to be completed by the authors]** concern the creation of the
raw corpus, which happened before this work and is not recorded in the
project.

## 1. Summary

| | Clips | Hours |
|---|---|---|
| Raw corpus | 218,091 | 144.6 ¹ |
| Removed by the transcript rules (section 4) | 58,978 | |
| Removed by the video checks (section 5) | 4,922 | |
| **Dataset** | **154,191 (70.7 %)** | **105.4** |

¹ Measured to the end of the last word of each transcript, since the clip
lengths of removed clips were not all read.

The dataset keeps the file layout of the raw corpus. It differs from it in
three ways: unreliable clips are removed, the video shows the 96x96 mouth
region instead of the 224x224 face, and Partition-Time Masking is written
into the video of the training clips.

## 2. Raw corpus

| Property | Value |
|---|---|
| Content | speeches in the Grand National Assembly of Türkiye |
| Sessions | 33, one folder each |
| Clips | 218,091 |
| Video | 224x224 face crop, H.264 (High profile), yuv420p, 25 frames per second |
| Audio | AAC-LC, 16 kHz, mono, inside the mp4 and again as a separate `.aac` file |
| Transcript | one `.txt` per clip: the sentence, then one line per word with start time, end time and confidence |
| Transcript source | automatic speech recognition (Whisper) |

**[to be completed by the authors]**: the Whisper model and version, the
decoding settings, how the recordings were cut into clips, and how the faces
were detected and cropped to 224x224.

Two properties of the transcripts matter for everything below.

**They are machine generated.** No transcript was checked by a person. The
corpus therefore contains the typical errors of the recogniser, described in
section 4.

**Dotted and dotless I cannot be told apart.** The text is upper case, and
the upper-casing did not follow Turkish rules, so both `i` and `ı` became
`I`. The raw transcripts contain 871,668 `I` and 5,372 `İ`; in correctly
cased Turkish the two are about equally frequent. The distinction cannot be
recovered from the transcripts.

## 3. Procedure

Each session is processed on its own, in clip order.

| Step | Input | Decides |
|---|---|---|
| 1. Parse and normalise the transcript | `.txt` | malformed or empty transcripts |
| 2. Transcript checks | normalised text, word times, confidences | invented, recurrent, implausible or uncertain transcripts |
| 3. Assign the split | clips that passed step 2 | training, validation or test |
| 4. Read the video | `.mp4` | wrong frame rate, too short, too long, text that does not fit the length |
| 5. Detect face landmarks | frames | clips without a reliably detected face |
| 6. Crop the mouth region | frames, landmarks | |
| 7. Apply Partition-Time Masking | training clips only | |
| 8. Write the clip | | |

A clip is removed at the first check it fails, so every removed clip is
counted under exactly one reason. Steps 1 to 3 need no video, which is why
face detection, by far the most expensive step, runs only on clips whose
transcript is usable.

## 4. Transcript cleaning

### 4.1 Normalisation

Transcripts are compared, and later used as training targets, in a normalised
form. The `.txt` files in the dataset are not changed.

| Rule | Example |
|---|---|
| Unicode NFC | |
| Circumflex removed | `HÂLÂ` becomes `HALA` |
| Digits spelled out in Turkish | `1921` becomes `BIN DOKUZ YÜZ YIRMI BIR` |
| A suffix after an apostrophe joins the number | `1921'DE` becomes `... YIRMI BIRDE` |
| Thousands dots and decimal commas | `1.250.000`, `3,5` becomes `ÜÇ VIRGÜL BEŞ` |
| Percent sign | `%50` becomes `YÜZDE ELLI` |
| Apostrophes removed, joining the suffix | `TÜRKIYE'NIN` becomes `TÜRKIYENIN` |
| `İ` merged into `I` | see section 2 |
| Everything that is not a letter becomes a space | punctuation, hyphens |

The resulting alphabet has 31 letters: the 28 Turkish letters that remain
after merging `İ` into `I`, and `Q`, `W`, `X` for foreign words.

### 4.2 Rules

Let a session contain clips c₁ … cₙ in recording order, with normalised
transcripts t₁ … tₙ, each a sequence of words.

**R1. Known invented phrase.** Speech recognisers trained on subtitled video
emit subtitle credits and sign-offs on silence or noise. A clip is removed if
one of the phrases below occurs in its transcript as a sequence of whole
words and makes up at least half of its words.

`ALTYAZI M K`, `ALTYAZI`, `ABONE OL`, `IZLEDIĞINIZ IÇIN TEŞEKKÜR`,
`IZLEDIĞINIZ IÇIN TEŞEKKÜRLER`, `BIR SONRAKI VIDEODA GÖRÜŞMEK ÜZERE`,
`SESLENDIREN`, `ÇEVIRI VE ALTYAZI`

The half-of-the-words condition keeps genuine sentences that merely contain
the words, for example a speech about subscriptions.

**R2. Internal loop.** A clip is removed if a group of 1 to 4 words is
repeated at least 3 times in a row (`DA DA DA`, `VE BU VE BU VE BU`), or if
its transcript has at least 40 characters and compresses by a factor above
2.4 with zlib. The compression-ratio test and its threshold are those Whisper
uses itself to detect failed decoding.

**R3. Burst.** Two transcripts are *the same* if they are identical, or if
both have at least 5 words and

  1 − d(tᵢ, tⱼ) / max(|tᵢ|, |tⱼ|) ≥ 0.8,

where d is the edit distance counted in words. Clip cᵢ is removed if some
clip cⱼ of the same session with 0 < |i − j| ≤ 5 has the same transcript.
All clips of a burst are removed, not all but one, because nothing indicates
which of them, if any, is correct.

**R4. Bursty sentence.** If a sentence occurs at least 3 times in a session
and at least half of those occurrences were removed by R3, its remaining
occurrences in that session are removed too. Other sessions are not
affected.

**R5. Squeezed words.** A word is *squeezed* if its duration is below 15 ms
per character. A clip is removed if more than 25 % of its words are squeezed.
Invented words are typically assigned one or two timestamp steps regardless
of their length.

**R6. Low confidence.** A clip is removed if the mean confidence of its words
is below 0.80, or if any word has a confidence below 0.30.

**R7. Speaking rate.** A clip is removed if it has fewer than 4 or more than
25 letters per second of video.

**R8. Timing.** A clip is removed if the end of its last word and the end of
the video differ by more than 1 second.

R7 and R8 need the length of the video and are therefore applied in step 4.

### 4.3 Result

| Rule | Clips removed | Share of raw corpus |
|---|---|---|
| R1 known invented phrase | 1,072 | 0.5 % |
| R2 internal loop | 441 | 0.2 % |
| R3 burst | 21,400 | 9.8 % |
| R4 bursty sentence | 101 | 0.0 % |
| R5 squeezed words | 2,025 | 0.9 % |
| R6 low confidence | 24,998 | 11.5 % |
| R7 speaking rate | 5,408 | 2.5 % |
| R8 timing | 3,533 | 1.6 % |
| Malformed or empty transcript | 0 | 0.0 % |
| **Transcript, total** | **58,978** | **27.0 %** |

R1 to R4 together, the recurrent and invented transcripts, remove 23,014
clips (10.6 %).

The amount of recurrence differs greatly between sessions (section 9). In
two sessions most transcripts are invented: in `2024-2025-birlesim-94` the
sentence `ÖZGÜR KARABAT` occurs 6,146 times, in `video-15` the sentence
`UCUN BIR` occurs 4,314 times. In nine sessions fewer than 1 % of the clips
are affected.

### 4.4 How the thresholds were chosen

The thresholds of R3 and R4 were derived from the data, those of R6 were
chosen by judgment and examined afterwards, and those of R5, R7 and R8 were
chosen by judgment and not examined.

**Bursts (R3, R4).** In five sessions (15,082 clips), every sentence that
occurred more than once was characterised by the number of sessions it
occurred in and by the share of its occurrences that had another occurrence
within 5 clips.

| Sentence | Occurrences | Sessions | Share in bursts |
|---|---|---|---|
| `ABONE OL` | 407 | 1 | 100 % |
| `AKLINIZDA GELEBILECEK HER TÜRLÜ` | 306 | 1 | 100 % |
| `BIR NISAN BIN DOKUZ YÜZ YIRMI BIRDE GIRESUNUZ` | 217 | 1 | 100 % |
| `TEŞEKKÜR EDERIM` | 19 | 5 | 0 % |
| `SAYIN BAŞKAN` | 13 | 5 | 0 % |
| `GENEL KURULU SAYGIYLA SELAMLIYORUM` | 12 | 5 | 0 % |

Invented sentences occur in one session and in bursts. Genuine formulae
occur in every session and not in bursts. A rule based on bursts therefore
separates the two, where a rule based on frequency alone would remove the
formulae as well.

**Approximate matching (R3).** Among neighbouring transcripts that were
similar but not identical, the genuine pairs were short and differed in one
word (`IKI BIN ON SEKIZ` and `IKI BIN ON DOKUZ`; `IÇINDE INSAN YOK` and
`IÇINDE TARIH YOK`), while variants of one invented sentence were longer
(`… BIRDE GIRESUNUZ` and `… BIRDE GIRESUNUMUZ`, 8 words). Requiring 5 words
and 80 % agreement keeps the former and removes the latter.

**Confidence (R6).** Two observations support that the confidence carries
information about errors.

Misrecognised words are often not words, and such a word tends to occur only
once in the corpus. Among 1,017,449 words:

| Word confidence | Share of all words | Occur only once in the corpus |
|---|---|---|
| below 0.30 | 0.5 % | 24.2 % |
| 0.30 to 0.50 | 2.3 % | 26.4 % |
| 0.50 to 0.70 | 6.9 % | 18.6 % |
| 0.70 to 0.80 | 7.3 % | 12.5 % |
| 0.80 to 0.90 | 13.0 % | 8.9 % |
| 0.90 to 0.95 | 12.9 % | 5.9 % |
| 0.95 to 1.00 | 57.0 % | 2.5 % |

Reading 8 random clips per group gave this impression:

| Group | Examples | Impression |
|---|---|---|
| Mean below 0.60, removed | `ISTIZYÜN ALTMIŞ IKI`, `YARGI MAANSIZMIŞ` | garbled |
| Mean 0.70 to 0.80, removed | `GEOGRAFAYA ADI GENIŞ BIR GEOGRAFAYA`; also `GENEL KURULU SAYGIYLA SELAMLIYORUM` | mixed |
| Mean 0.80 to 0.85, kept | `BAZEN SOPA ILE YAPIYORLAR`; also `ÇARESIZLIĞE SÜLÜKLEMIŞTIR` | mostly correct |
| Mean above 0.97, kept | `KELIME DEĞIŞIKLIĞI YAPILIYOR` | correct |
| Removed only for one word below 0.30, mean above 0.85 | `BAŞTA EN AZ DÖRT BEŞ TANE UZMANIN OLMASI GEREKIYOR` | correct |

Share of the clips reaching R6 (193,052) that other thresholds would remove:

| Mean threshold | No word rule | Word below 0.30 | Word below 0.50 |
|---|---|---|---|
| 0.60 | 1.1 % | 3.0 % | 12.2 % |
| 0.70 | 3.6 % | 5.0 % | 12.9 % |
| **0.80** | 12.5 % | **12.9 %** | 17.5 % |
| 0.85 | 23.1 % | 23.3 % | 25.4 % |
| 0.90 | 41.8 % | 41.8 % | 42.1 % |

Of the 24,998 clips removed by R6, 20,372 fail only the mean, 937 only the
single-word condition, and 3,689 both.

### 4.5 Known costs of the rules

| Rule | Cost |
|---|---|
| R3 | genuine formulae spoken in quick succession are removed, for example 31 of the 90 occurrences of `KABUL EDENLER`, which the chair repeats during a series of votes |
| R6, mean | correct transcripts with a mean between 0.70 and 0.80 are removed along with garbled ones |
| R6, single word | the 937 clips removed by this condition alone appear to be correct; many contain numbers |
| All | the rules detect implausible transcripts; a plausible but wrong transcript passes |

### 4.6 A check that was tested and not used

It was tested whether the video itself reveals wrong transcripts, on the
assumption that invented transcripts arise where the visible person is not
speaking. For each clip the opening of the lips (distance between the inner
lips, divided by face height) was correlated with the logarithm of the audio
energy per video frame, taking the maximum over a lag of −4 to +4 frames.

| Clips | Number | Median correlation | Median lip motion ² |
|---|---|---|---|
| Kept | 578 | 0.20 | 0.0192 |
| Removed by R3 | 532 | 0.24 | 0.0207 |
| Removed by R1 | 363 | 0.29 | 0.0195 |
| Removed by R2 | 55 | 0.13 | 0.0172 |

² Standard deviation of the lip opening over the clip.

Clips with invented transcripts show as much lip motion, and as much
agreement between lips and audio, as kept clips. The speakers in them are
speaking; the transcript is wrong. The measure does not separate the groups
and was not used.

## 5. Video checks

| Check | Condition for removal | Clips removed |
|---|---|---|
| Frame rate | differs from 25 frames per second | 0 |
| Too short | fewer than 12 frames (0.48 s) | 2,764 |
| Too long | more than 600 frames (24 s) | 0 |
| No face | a face is found in fewer than 50 % of the frames | 807 |
| Low detection rate | a face is found in 50 % to 80 % of the frames | 1,351 |

Of the 168,054 clips whose transcript passed R1 to R6, 156,349 reached face
detection, and 154,191 passed it.

## 6. Mouth region

### 6.1 Landmarks

Landmarks are detected in every frame with the MediaPipe Face Landmarker
(`face_landmarker.task`, float16, 478 points), in single-image mode with
detection and presence thresholds of 0.3. Of the 478 points, 19 are used:

| Points | MediaPipe indices |
|---|---|
| Outer lips | 61, 291, 0, 17, 37, 267, 84, 314 |
| Eye corners | 33, 133, 362, 263 |
| Nose tip | 1 |
| Forehead, chin | 10, 152 |
| Inner lips | 13, 14, 78, 308 |

In the kept clips a face is found in 99.8 % of the frames, and in all frames
of 96.9 % of the clips. Frames without a detection take landmarks
interpolated linearly in time between the nearest detections; at the start
and end of a clip the nearest detection is repeated.

### 6.2 Position

The centre of the crop in frame t is the mean of the 8 outer-lip points,
smoothed over time. Detections jitter by a pixel or two from frame to frame,
and cropping at the raw positions would turn that jitter into apparent
motion of the whole mouth region.

Smoothing uses a Kalman filter with a constant-velocity model, applied to
the two coordinates independently, followed by a Rauch-Tung-Striebel
backward pass. With state (position, velocity) and one frame as the unit of
time:

  F = [[1, 1], [0, 1]], H = [1, 0], Q = q · [[1/4, 1/2], [1/2, 1]], R = r,

with q = 1.0 px² per frame⁴ and r = 4.0 px². The backward pass uses the
whole clip, so the smoothed position does not lag behind the mouth as it
would with a filter that only looks back.

### 6.3 Scale

The side of the cropped square is 0.63 times the face height, where the face
height is the distance between the forehead and chin points. One value is
used for the whole clip, the median over its frames.

- The face height is used rather than the distance between the eyes because
  the latter shrinks as the head turns. Of the kept clips, 14.8 % show a
  strongly turned head.
- One value per clip is used because the chin is one end of the face height:
  a value per frame would make the crop zoom as the jaw opens.

The factor 0.63 reproduces the area of the 96-pixel crop of Auto-AVSR, so
that weights trained on Auto-AVSR crops remain applicable. It was measured on
567 clips: for each, the similarity transform of Auto-AVSR (right eye, left
eye, nose tip and mouth centre mapped onto its mean face) was estimated, and
the side of its 96-pixel crop was expressed in source pixels and divided by
the face height.

| Clips | Number | Median | 10th to 90th percentile |
|---|---|---|---|
| All | 567 | 0.630 | 0.605 to 0.667 |
| Near frontal | 268 | 0.627 | 0.604 to 0.661 |
| Strongly turned | 116 | 0.641 | 0.614 to 0.677 |

### 6.4 Cropping and encoding

The square is mapped to 96x96 pixels by an affine transform with bilinear
interpolation; pixels outside the frame repeat the border. The crop is not
rotated. Colour is kept. The video is encoded once, with libx264, constant
rate factor 18, preset medium, yuv420p, 25 frames per second.

The audio stream of the source mp4 is copied into the new mp4 without
re-encoding. The `.aac` and `.txt` files are copied unchanged.

### 6.5 Differences from the Auto-AVSR preprocessing

| | Auto-AVSR | This dataset |
|---|---|---|
| Landmark detector | RetinaFace with FAN, or MediaPipe face detection | MediaPipe Face Landmarker |
| Alignment | similarity transform to a mean face, per frame | none; translation and one scale per clip |
| Smoothing | moving average over 12 frames | Kalman filter with backward pass |
| Scale reference | eyes, nose and mouth | forehead-to-chin distance |
| Crop | 96x96, grey | 96x96, colour (converted to grey when loaded) |

Alignment by similarity transform and contrast-limited adaptive histogram
equalisation were left out on the basis of earlier experiments by the
authors **[to be completed by the authors: reference to those experiments]**.

## 7. Partition-Time Masking

Masking is applied to the cropped frames of the training clips.

| Parameter | Value |
|---|---|
| Grid | 4x4, cells of 24x24 pixels |
| Masked span | one span of round(0.2 · T) consecutive frames, at least 1, for a clip of T frames |
| Position of the span | uniform over the possible positions |
| Masked cells | 2 of the 16 cells, chosen anew for every frame of the span |
| Fill | black |

The random choices of a clip are drawn from a generator seeded with the
CRC-32 of the string `seed/session/clip`, with seed 0. Building the dataset
again reproduces every mask.

Validation and test clips are not masked, so that error rates describe the
recognition of unaltered video.

Two consequences of writing the masks into the clips should be stated when
the dataset is described:

- Every training clip has one mask, the same in every epoch. Masking during
  training would draw a new mask each time.
- The masks pass through lossy compression. Inside masked cells the mean
  level is 0.3 of 255.

## 8. Split

| Split | Definition |
|---|---|
| Test | all clips of the sessions `2025-2026-birlesim-3` and `video-10` |
| Validation | the last 5 % of the clips of every other session |
| Training | the remaining clips |

The split is made among the clips that passed the transcript checks, before
the video is read, because it decides which clips are masked.

Test sessions are whole sessions and validation clips are a contiguous block
because neighbouring clips come from the same speech: a random split would
place parts of one sentence on both sides. The two test sessions were chosen
among those with few recurrent transcripts (0.2 % and 0.5 %).

## 9. Resulting dataset

| | Training | Validation | Test | Total |
|---|---|---|---|---|
| Clips | 140,823 | 7,274 | 6,094 | 154,191 |
| Hours | 96.0 | 5.0 | 4.4 | 105.4 |
| Sessions | 31 | 31 | 2 | 33 |
| Words | 691,911 | 35,556 | 30,934 | 758,401 |
| Distinct words | 82,018 | 11,516 | 10,411 | 86,861 |
| Words not seen in training | | 7.8 % | 8.3 % | |
| Masked | yes | no | no | |

| Per clip | Minimum | 5th percentile | Median | Mean | 95th percentile | Maximum |
|---|---|---|---|---|---|---|
| Frames | 12 | 22 | 55 | 61.5 | 117 | 252 |
| Seconds | 0.48 | 0.88 | 2.20 | 2.46 | 4.68 | 10.08 |
| Words | | | 4 | 4.92 | 9 | 30 |

Of the 86,861 distinct words, 50,169 occur once. The dataset contains
146,999 distinct sentences. Size on disk: 11 GB.

### By session

| Session | Raw clips | Kept | Kept | Hours | Recurrent or invented | Split |
|---|---|---|---|---|---|---|
| 2024-2025-birlesim-72 | 9,247 | 7,704 | 83.3 % | 5.56 | 0.5 % | train, val |
| 2024-2025-birlesim-73 | 2,370 | 992 | 41.9 % | 0.75 | 37.2 % | train, val |
| 2024-2025-birlesim-74 | 7,346 | 5,868 | 79.9 % | 4.22 | 0.6 % | train, val |
| 2024-2025-birlesim-76 | 7,115 | 5,244 | 73.7 % | 3.65 | 8.0 % | train, val |
| 2024-2025-birlesim-80 | 2,570 | 1,815 | 70.6 % | 1.44 | 6.1 % | train, val |
| 2024-2025-birlesim-82 | 8,995 | 7,059 | 78.5 % | 4.51 | 1.4 % | train, val |
| 2024-2025-birlesim-84 | 6,550 | 5,357 | 81.8 % | 3.80 | 0.6 % | train, val |
| 2024-2025-birlesim-94 | 8,831 | 1,916 | 21.7 % | 1.08 | 70.0 % | train, val |
| 2024-2025-birlesim-109 | 7,595 | 5,610 | 73.9 % | 4.09 | 6.9 % | train, val |
| 2024-2025-birlesim-110 | 7,833 | 5,202 | 66.4 % | 3.23 | 13.9 % | train, val |
| 2025-2026-birlesim-1 | 4,446 | 3,394 | 76.3 % | 2.82 | 2.2 % | train, val |
| 2025-2026-birlesim-2 | 5,740 | 4,635 | 80.7 % | 3.27 | 0.5 % | train, val |
| 2025-2026-birlesim-3 | 3,859 | 3,094 | 80.2 % | 2.20 | 0.2 % | test |
| 2025-2026-birlesim-5 | 3,288 | 2,664 | 81.0 % | 1.84 | 0.5 % | train, val |
| video-1 | 8,300 | 6,450 | 77.7 % | 4.22 | 4.9 % | train, val |
| video-2 | 8,088 | 5,589 | 69.1 % | 4.00 | 10.0 % | train, val |
| video-3 | 8,060 | 5,160 | 64.0 % | 2.70 | 10.8 % | train, val |
| video-4 | 4,933 | 4,037 | 81.8 % | 2.91 | 0.6 % | train, val |
| video-5 | 9,312 | 7,558 | 81.2 % | 5.26 | 0.6 % | train, val |
| video-6 | 2,995 | 1,712 | 57.2 % | 1.21 | 16.4 % | train, val |
| video-7 | 9,206 | 6,677 | 72.5 % | 4.40 | 8.0 % | train, val |
| video-8 | 6,521 | 4,405 | 67.6 % | 2.65 | 7.5 % | train, val |
| video-9 | 7,388 | 5,900 | 79.9 % | 3.80 | 1.2 % | train, val |
| video-10 | 3,726 | 3,000 | 80.5 % | 2.17 | 0.5 % | test |
| video-11 | 6,900 | 5,065 | 73.4 % | 3.49 | 5.5 % | train, val |
| video-12 | 10,236 | 7,970 | 77.9 % | 5.34 | 1.5 % | train, val |
| video-13 | 7,535 | 4,404 | 58.4 % | 3.18 | 25.2 % | train, val |
| video-14 | 7,376 | 5,791 | 78.5 % | 3.86 | 2.6 % | train, val |
| video-15 | 6,071 | 903 | 14.9 % | 0.59 | 79.5 % | train, val |
| video-16 | 5,947 | 4,711 | 79.2 % | 3.71 | 1.6 % | train, val |
| video-17 | 7,738 | 5,061 | 65.4 % | 3.26 | 15.6 % | train, val |
| video-18 | 5,518 | 4,303 | 78.0 % | 2.95 | 4.0 % | train, val |
| video-19 | 6,456 | 4,941 | 76.5 % | 3.19 | 3.5 % | train, val |

The folders named `video-1` to `video-19` are sessions whose folder does not
carry a session number **[to be completed by the authors: the sessions these
correspond to]**.

## 10. Verification

`scripts/verify_dataset.py` compares the dataset with the raw corpus. Checks
on file content that require decoding were run on 300 random clips of each
split, 900 in total; all other checks cover every clip.

| Check | Scope | Result |
|---|---|---|
| The dataset folder holds session folders with mp4, aac and txt files and nothing else | all | passed |
| Every kept clip is present, no removed clip is present | all | passed |
| Every clip has its three files | all | passed |
| Transcript files are byte-identical to the raw corpus | 154,191 | passed |
| Audio files are byte-identical to the raw corpus | 154,191 | passed |
| No kept transcript matches R1 or R2; no clip flagged by R3 or R4 is present | all | passed |
| Test clips are exactly the clips of the two test sessions | all | passed |
| Only training clips are marked as masked | all | passed |
| The mp4 has H.264 96x96 video at 25 frames per second as stream 0 and the source audio format as stream 1 | 900 | passed |
| The number of frames equals that of the raw clip | 900 | passed |
| The decoded audio is identical, sample for sample, to that of the raw clip | 900 | passed |
| The video equals the crop recomputed from the raw clip | 900 | mean absolute difference 1.96 of 255, worst clip 2.52 |
| Masked cells are black and at the positions given by the seed | training clips among the 900 | passed |
| The mask covers round(0.2 · T) frames | training clips among the 900 | passed |
| Manifests agree with the clip table; test sessions do not occur in training or validation | all | passed |

The difference between the stored video and the recomputed crop is the loss
of H.264 compression.

## 11. Limitations

**Transcripts.**

- The transcripts, including those of the test set, are the output of a
  speech recogniser. Error rates measured on this test set are error rates
  against that recogniser's output, not against what was said. A manually
  corrected test set would be needed to state absolute accuracy.
- The rules remove transcripts that are implausible. They cannot find a
  transcript that is fluent but wrong.
- `i` and `ı` share one symbol, so a model trained on this dataset cannot
  distinguish them, and error rates do not count their confusion.
- The thresholds of R5, R7 and R8 were not examined, and the examination of
  R6 rests on a proxy and on reading a small number of clips.

**Split.**

- The test sessions are separate recordings, but the same members of
  parliament speak in many sessions. The test set is not independent of the
  training set in terms of speakers, and speaker identities are not
  annotated.
- Validation clips come from the same sessions as training clips, and
  therefore from the same speakers and partly the same speeches.

**Video.**

- Partition-Time Masking is fixed per clip (section 7).
- The crop is not rotated, so head roll remains in the data.
- Landmarks were not checked by hand. A detection that is present but wrong
  is not noticed.

**Domain.** All clips show formal speech in one room, recorded by the same
cameras.

## 12. Reproduction

```bash
python scripts/extract_raw.py --src <folder with video-*.tar.gz> --dest data/raw
python scripts/audit_labels.py
python scripts/build_dataset.py --out data/tr-lrs-clean-ptm \
    --holdout-sessions 2025-2026-birlesim-3 video-10 --workers 12
python scripts/build_manifest.py --dataset data/tr-lrs-clean-ptm
python scripts/verify_dataset.py --dataset data/tr-lrs-clean-ptm --manifests data/manifests --sample 300
```

| File | Content |
|---|---|
| `data/tr-lrs-clean-ptm.settings.json` | crop, masking and split settings |
| `data/tr-lrs-clean-ptm.clips.tsv` | every raw clip with verdict, split, frames, confidence, detection rate and normalised text |
| `docs/label_audit.md`, `data/label_audit.tsv` | audit of the transcripts alone, with the most frequent removed and kept sentences |
| `avsr/data/manifest.py`, `avsr/data/recurrence.py` | the rules and their thresholds |
| `avsr/preprocess/` | landmarks, smoothing, cropping, masking |

| Software | Version |
|---|---|
| Python | 3.13.3 |
| MediaPipe | 0.10.35 |
| Face Landmarker model | float16, MD5 `b0e7274907a1644404fef66b28dd6d85` |
| PyAV | 16.1.0 (libavcodec 62.11.100) |
| OpenCV | 4.13.0 |
| PyTorch | 2.6.0 |

Building took 8.9 hours on a 16-thread CPU with 12 worker processes; face
landmarks of 33,464 clips were already cached.
With the landmarks cached, a rebuild takes about an hour and a half.

## 13. References

The bibliographic details below should be checked against the publications
before they are cited.

| Used for | Reference |
|---|---|
| Baseline preprocessing and model | P. Ma, S. Petridis, M. Pantic, "Visual speech recognition for multiple languages in the wild", Nature Machine Intelligence, 2022 |
| Baseline preprocessing and model | P. Ma et al., "Auto-AVSR: Audio-visual speech recognition with automatic labels", ICASSP 2023 |
| Transcripts; compression-ratio test of R2 | A. Radford et al., "Robust speech recognition via large-scale weak supervision", ICML 2023 |
| Partition-Time Masking | "A novel approach for visual speech recognition using the partition-time masking", Sensors, 2025 (as named in the project notes; authors and volume to be added) |
| Landmark detection | C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines", 2019; Y. Kartynnik et al., "Real-time facial surface geometry from monocular video on mobile GPUs", 2019 |
| Kalman filter | R. E. Kalman, "A new approach to linear filtering and prediction problems", 1960 |
| Backward pass | H. E. Rauch, F. Tung, C. T. Striebel, "Maximum likelihood estimates of linear dynamic systems", 1965 |

## 14. To be completed by the authors

| Item | Section |
|---|---|
| Whisper model, version and decoding settings | 2 |
| How the recordings were segmented into clips | 2 |
| How faces were detected and cropped to 224x224 | 2 |
| Which sessions the folders `video-1` to `video-19` are | 9 |
| Reference to the earlier experiments on alignment and histogram equalisation | 6.5 |
| Authors and volume of the Partition-Time Masking paper | 13 |
