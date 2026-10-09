# Audit of the clip extraction code and the old model's outputs

Notes from 2026-10-09. The raw TR-LRS clips were produced by the extraction
code in [esra98/t_lips](https://github.com/esra98/t_lips) (commit `a8f2c6a`),
and the predictions of the model trained before this work are published at
[esra98/TR-LR-web](https://github.com/esra98/TR-LR-web) (commit `b50f0c7`).
This document records what the two show about the dataset and what
`tr-lrs-clean-ptm` does and does not fix.

## 1. The old model repeats invented sentences

`data.json` of TR-LR-web holds 1,001 predictions of the old model
(`model_avg_10.pth`, beam size 40). They contain only 442 distinct sentences,
mean WER is 100.8 % and 9 predictions are exact.

| Prediction | Times predicted | Occurrences in the raw labels | Sessions | Removed by the cleaning |
|---|---|---|---|---|
| `ÖZGÜR KARABAT` | 194 | 6,146 | 1 | 6,146 |
| `TÜRKIYE BÜYÜK MILLET MECLISI` | 50 | 527 | 23 | 477 |
| `KENDI ÜLKEMIZDE NASIL` | 43 | 71 | 1 | 71 |
| `KÜRT HALKLI` | 39 | 129 | 1 | 129 |
| `KÜRSÜYE GELEN KÜRSÜYE GELEN` | 22 | 93 | 1 | 93 |
| `VE ISTIGRAR ÇIVASINDAYIZ` | 11 | 71 | 1 | 71 |

Label counts are from `docs/label_audit.md`. These six sentences make up 359
of the 1,001 predictions (36 %). Of the 100 clips shown on the web page, 17
are predicted as `ÖZGÜR KARABAT`.

When the model cannot read a clip it falls back to the sentences most
frequent in its training labels. All six were Whisper output repeated in
bursts, and `tr-lrs-clean-ptm` removes them (rules R3 and R4 in
`docs/dataset_construction.md`). The model will still have a prior; after
the cleaning it should fall on genuine frequent phrases instead.

Proposed metric next to WER: the share of predictions equal to the single
most frequent prediction. Old model: 19.4 % (194 of 1,001).

Unresolved: `BIR TIP` (15), `ÇÜNKÜ ZEYTIN YAŞAMDIR` (12) and
`YALNIZ ZEMIN YANLIŞ FIILEN YAPILMAYAN ETUT` (10) do not appear in the lists
of `docs/label_audit.md`; their counts in `data/tr-lrs-clean-ptm.clips.tsv`
are still to be checked.

## 2. Origin of `ÖZGÜR KARABAT`

Özgür Karabat is a member of parliament, so the name itself is real. It was
not said 6,146 times: in `2024-2025-birlesim-94` it is the label of 6,146 of
8,831 clips (70 %), and of no burst in any other session.

The extraction code explains how this happens.

- `utils/generate_timestamp.py` runs `whisper_timestamped` with the `small`
  model on the whole session file, with default options. These include
  `condition_on_previous_text=True` and no voice activity detection, the
  setting in which Whisper is known to repeat the previous segment over
  long recordings.
- `utils/lrs_cropper.py` turns every Whisper segment into a clip and keeps
  segments of 2 words or more (`MIN_SEGMENT_WORDS = 2`). Each repetition of
  the two-word name became a clip.

The cutting does not create the error but does nothing to stop it. Section
4.6 of `docs/dataset_construction.md` agrees: clips removed as bursts show as
much lip motion and lip–audio agreement as kept clips, so the speakers are
speaking and the label is wrong.

## 3. Dropped frames in the extraction code

`utils/lrs_cropper.py`, `_crop_frames`:

```python
if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
    continue
```

A frame in which no face, or two faces, are found is dropped. The remaining
frames are written back to back, and `_mux` encodes with `-shortest`, which
cuts the audio of the mp4 to the length of the shortened video. The separate
`.aac` is written before muxing and keeps the full length. As a result:

- the clip is shorter than its transcript;
- frames dropped in the middle shift everything after them, so lips no
  longer line up with the audio or the word times;
- the audio inside the mp4 is truncated. In `00951` the `.aac` lasts
  3.04 s, the mp4 audio 0.06 s, and the video has 3 frames.

### Measurement

On the 100 clips in `TR-LR-web/test-videos` and the sample clip `02369` of
t_lips, the number of video frames was compared with the number expected from
the transcript. The extraction adds 0.04 s at each end, so the expected
count is round((end of last word + 0.04) · 25).

| Missing frames | Clips (of 101) |
|---|---|
| at least 1 | 37 |
| at least 5 | 23 |
| at least 13 (about 0.5 s) | 15 |
| at least 25 (1 s) | 11 |

The worst are `00951` (3 of 76 frames), `00978` (16 of 85) and `00058`
(11 of 68). `00058` and `00978` are among the clips predicted as
`ÖZGÜR KARABAT`. Of the 17 clips with that prediction, 18 % keep fewer than
half of their frames, against 7 % of the other 83. The median clip with that
prediction keeps 97 % of its frames, so dropped frames explain some of these
predictions, not most.

### What the current cleaning catches

| Check in `build_dataset.py` | Effect on dropped frames |
|---|---|
| Too short, fewer than 12 frames | removes clips that lost almost everything |
| R8 timing, end of last word more than 1 s from end of video | removes clips that lost more than about 25 frames in total |
| Face detection rate | cannot see the problem: frames without a face were already deleted by the extraction |

In the sample, every clip missing at least 25 frames is removed. About a
quarter of the clips miss 1 to 24 frames and pass. When those frames were
dropped in the middle, the clips are out of sync.

`build_dataset.py` copies the audio stream of the raw mp4, so the truncated
audio reaches the dataset. This matters for audio and audio-visual training.
The `.aac` files have the full audio but do not line up with the shortened
video either.

## 4. To do

**In this repository, before training.**

- [ ] Add a frame-count check: remove a clip if it has more than 2 frames
      fewer than round((end of last word + 0.04) · 25). The frame count and
      word times are already read in step 4 of the build.
- [ ] Measure the check on the whole corpus and add the result to section 5
      of `docs/dataset_construction.md`.
- [ ] Check the three unresolved sentences of section 1 in the clip table.

**In t_lips, if clips are extracted again.**

- [ ] `_crop_frames`: keep frames without a single detected face, using the
      last valid box, instead of `continue`.
- [ ] `_mux`: remove `-shortest`, so the audio is never cut.
- [ ] `generate_timestamp.py`: `condition_on_previous_text=False`,
      `vad=True`, and a larger model (`medium` or `large-v3`).
- [ ] Read frames in order instead of calling `cap.set(CAP_PROP_POS_FRAMES)`
      for every frame; it is faster and avoids inexact seeking in H.264.
