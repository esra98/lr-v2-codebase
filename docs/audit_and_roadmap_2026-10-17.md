# Audit and roadmap: rebuilding the TR-LRS pipeline

Written 2026-10-17 after reading every project folder on this machine:
`lr_codes/{dlp,whisper,t_lips,frequency,lr-web}`, `lr-v2-codebase`,
`reference videos/`. Numbers were measured, not copied from older docs.

## 1. What exists today

### 1.1 Source videos: 41 sessions, 340 hours, stored up to three times

| Group | Sessions | Source | Resolution | Hours | Where |
|---|---|---|---|---|---|
| YouTube | `video-1` … `video-20` | 20 links in `lr_codes/dlp/links.txt` | **640x360**, 30 fps | ~165 | `dlp/*.mp4` (24 GB) **and** `t_lips/videos/video-{1..20}.mp4` (25 GB) |
| TBMM downloads | `video-21,23..41` | 19 assembly recordings | 1920x1080, 30 fps | ~160 | `reference videos/raw/*.mp4` (92 GB) **and** `t_lips/videos/video-{21..41}.mp4` (byte-identical copies) |
| (resolved) | `video-34` transcript | is the transcript of `2025-2026-birlesim-3`; `video-35_ts.json` was a second transcription of `birlesim-5` | | 4.8 | see section 8 |

The `video-N` names of `t_lips` map onto the assembly names by file size
(`video-21` = `2024-2025-birlesim-72`, … , `video-41` = `2025-2026-birlesim-12`;
full table in section 6). The two naming schemes coexist in the current
corpus, which is why `docs/dataset_construction.md` could not say what the
`video-*` sessions were.

**Finding: the YouTube videos were fetched at 360p although 1080p exists.**
`yt-dlp -t mp4` picked the muxed 360p stream; `yt-dlp -F` on the first link
lists 1080p (`137`/`399`). At 360p a face is roughly 80–120 px wide, so the
96x96 mouth crop is an *upscaled* image. Half of the corpus, including the
test session `video-10`, is in this group.

### 1.2 Transcripts: Whisper `small`, original casing preserved in JSON

`t_lips/timestamps/<video>_ts.json` (40 files, 459 MB) holds the raw
`whisper_timestamped` output of the **`small`** model, with word times,
word confidences, and segment-level `no_speech_prob`, `avg_logprob`,
`compression_ratio`.

**Finding: the I/ı problem is fully recoverable.** The JSON text is in
Whisper's own casing (`İngilizce`, `Müzik`, `Fırat`). In `video-1` the words
contain 28,555 `i`, 13,417 `ı`, 511 `İ`, 180 `I`. The loss happened in one
line of the cropper, `sentence.upper()`, which maps `i` → `I` under the
default locale. 54 % of all words contain an `i` or `ı`, so the fix affects
most of the vocabulary. No lexicon or morphological analyser is needed:
the dataset must simply be cut again from the JSON, or the existing clips
re-labelled by aligning their word sequences to the JSON.

**Finding: `small` hallucinates massively on some sessions.** In
`video-29` (= `2024-2025-birlesim-94`), 14,190 of 18,238 segments are the
text "Özgür Karabat"; 11,074 segments have `no_speech_prob` > 0.5. These
signals exist in the JSON but were discarded when the `.txt` files were
written, which is why the cleaning rules had to rediscover the problem from
text statistics alone (R1–R4 in `dataset_construction.md`).

### 1.3 Clips: two different cropper runs, neither complete

| Run | Code | Date | Sessions | Clips | Where |
|---|---|---|---|---|---|
| A | older `lrs_cropper.py` (the `.bak`) | Apr 2026 | 33 (video-1..19 + 14 assembly) | 218,091 | tarballs `reference videos/tr-lrs/*.tar.gz` (17 GB) → extracted `lr-v2-codebase/data/raw` (19 GB) |
| B | current `lrs_cropper.py` (strict tracking) | May–Jun 2026 | 39 (adds video-20, 37–41; video-19 produced 0) | 218,603 | `t_lips/videos/dataset/lrs2/pretrain` (19 GB), never ingested |

Run B used a stricter rule (reject a clip if the face is lost in *any*
frame or the box jumps) and logged 127,707 "Track broken" rejections; it
gives fewer clips per session than run A for the same video (video-1: 6,210
vs 8,300; video-4: 1,191 vs 4,933). `lr-v2-codebase` was built on run A.
Six sessions of run B (≈33,600 clips, ~47 hours of source) exist nowhere in
`lr-v2-codebase`. Run B's `pretrain.txt` lists only 95,826 of its clips.

Both runs share the defects below, all inherited from the cropper:

- `.upper()` destroys i/ı (above).
- Word times inside a clip are **re-derived differently** in the two runs
  (same clip `video-1/00000`: run A `İNGILIZCE 0.04–2.16`, run B `0.04–1.76`).
- The clip is cut on sentence punctuation produced by Whisper, then capped at
  10 s; there is no pause-based splitting, so long run-on segments get cut
  mid-word by the duration cap.
- Video is written twice (OpenCV `mp4v`, then re-encoded with libx264 crf 23),
  and frames are resampled 30 → 25 fps by nearest-neighbour on timestamps.
- Resume is "next free index": a run interrupted mid-session restarts from
  the beginning of the session's segment list and appends *duplicates* under
  new indices, because `seen_chunks` is in memory only.
- Audio is extracted with a separate `ffmpeg` process per clip and the full
  video is seeked per clip with OpenCV, which is why one session took 8–20
  hours.

### 1.4 `lr-v2-codebase`: the second-stage pipeline is sound, the first stage is missing

The cleaning rules (`avsr/data/{manifest,recurrence}.py`), the mouth-crop
pipeline (`avsr/preprocess/`), the clip table, verification and the
training/model code are well made and tested (241 tests). What it lacks is
everything *before* the 224x224 clips: it treats the clips as an immutable
external corpus and tries to repair their labels from the outside. That is
the root of the I/ı problem and of the lost Whisper signals.

Dead weight in and around the repo:

| Item | Size | Verdict |
|---|---|---|
| `upload/` (dataset tar + copy of a `.pth`) | 11 GB | delete, reproducible |
| `exp/` local rehearsal runs | 13 GB | delete after keeping any checkpoint you want |
| `data/sample*`, `data/sample-manifests`, `data/sample-tokenizer` | 52 MB | delete |
| `logs/` | 420 KB | delete |
| `scripts/extract_raw.py` | | delete once clips are cut from full videos |
| `prompt.text`, `.pytest_cache` | | delete |
| `reference-codebase/`, `papers/` | 236 MB | move out of the repo (already git-ignored) |
| `docs/deployment.md`, `docs/project_report.md` | | keep, mark as describing the v2 dataset |
| `lr_codes/whisper`, `frequency`, `lr-web`, `Visual_Speech_Recognition_for_Multiple_Languages` | | archive; nothing in them is needed |
| `lr_codes/t_lips` | | absorb `split_by_timestamp_rules` and the LRS2 txt format into the new code, then archive |
| `reference videos/av-asr.tar` | 1.3 GB | delete, duplicate of `av-asr/` |

### 1.5 Machine

RTX 2060 6 GB, 16 threads, 15 GB RAM, **26 GB free** of 465 GB.
Duplicate videos alone account for about 117 GB. Nothing can be re-cut
before disk is reclaimed.

Python: `.venv` (3.13, mediapipe 0.10.35, PyAV, torch 2.6) for the v2 code;
conda `whisper`/`tlips` (3.10, whisper-timestamped 1.15.8, mediapipe 0.10.21)
for the old pipeline.

## 2. Decisions to take (recommendation first)

1. **Re-cut everything from the full videos and the Whisper JSON.**
   Recommended. It fixes I/ı exactly, restores `no_speech_prob` and the
   other Whisper signals for cleaning, unifies the two runs and the two
   naming schemes, and adds the six never-ingested sessions. The user has
   already agreed to re-cropping.
2. **Re-download the 20 YouTube sessions at 1080p** (~40 GB, 2 GB each).
   Recommended before cutting; otherwise half the dataset stays upscaled.
   Needs ~2 h of download at 10 MB/s. Both 360p copies are deleted after.
3. **Re-transcribe with a larger model, or keep `small`?**
   `small` is the weakest practical Whisper model for Turkish. With
   `faster-whisper large-v3` in `int8_float16` the 6 GB GPU suffices, and
   340 h of audio takes roughly 2–3 days of unattended GPU time with
   per-chunk resume. Recommendation: build the pipeline so the transcript is
   a replaceable input, run it first with the existing `small` JSON so the
   whole chain is validated, then re-transcribe in the background and
   rebuild (the rebuild is cheap once landmarks are cached). The 2026-09-27
   decision "do not re-transcribe" was about re-running the *same* model;
   a larger model is a different case.
4. **Session naming.** Use the assembly name (`2024-2025-birlesim-72`) for
   the 19 TBMM files, and `yt-<videoid>` for the YouTube sessions, with a
   `sources.tsv` that records origin, URL, resolution and checksum. The
   `video-N` names disappear.
5. **Test set.** Keep whole-session held-out tests, but choose sessions
   that are 1080p. A manually corrected subset of a few hundred clips
   remains the only way to get an absolute error rate.

## 3. Target structure

```
lr-v2-codebase/
  videos/                         full-length source videos, one per session
    sources.tsv                     session, origin, url, sha256, width, height, fps, hours
    2024-2025-birlesim-72.mp4
    yt-fP99G1PONhQ.mp4
  work/<session>/                 per-session pipeline state, all resumable
    transcript.json                 Whisper output, original casing (stage 2)
    segments.tsv                    cut plan: utt, start, end, text, verdict (stage 3)
    landmarks/<utt>.npz             cached (stage 4)
    clips.tsv                       verdict per utterance after video checks (stage 4)
    status.json                     which stages are complete, with code/config hash
  dataset/<name>/<session>/<utt>.{mp4,txt}   final clips (stage 5)
  dataset/<name>.clips.tsv, .settings.json, manifests/
  avsr/
    corpus/                         NEW: the first stage
      sources.py                    register/probe/checksum videos, yt-dlp at 1080p
      transcribe.py                 whisper backend (whisper_timestamped | faster-whisper), chunked, resumable
      segment.py                    JSON -> utterances (punctuation + pause + duration rules), true-case text
      cut.py                        sequential decode of a session, landmarks, face-224 and/or mouth-96 writer
      status.py                     stage markers, atomic writes, resume
    text/ data/ preprocess/ models/ decoding/ training/ espnet/ utils/   as now
  scripts/
    corpus.py                       one CLI: ingest | transcribe | segment | cut | build | verify  [--sessions]
    train.py evaluate.py ab_test.py preflight.py train_tokenizer.py benchmark_memory.py  as now
  configs/  tests/  docs/
```

Rules of the structure:

- Every stage reads from `videos/` or `work/` and writes only to `work/` or
  `dataset/`; nothing is ever written next to a source video.
- Every output is written to `<path>.tmp` and renamed, and every per-session
  stage ends by writing `status.json` with a hash of the code version and
  options used. A stage re-runs when its inputs or options changed, and
  skips otherwise. Interrupting at any point loses at most the clip in
  progress.
- Per-clip resume is by utterance id in `segments.tsv`, not "next free
  index", so re-running never duplicates clips.
- The `.txt` of a clip keeps the LRS2 layout but the `Text:` line is in
  **true case**; the upper-cased training target is produced by the
  normaliser with Turkish casing (`i`→`İ`, `ı`→`I`). The alphabet becomes
  29 Turkish letters + QWX.

## 4. Roadmap

Each item lists what it produces and what can be verified when it is done.

### Phase 0: reclaim disk and freeze the old state (1 day)

1. Delete the duplicates listed in 1.4 and 1.5: `t_lips/videos/video-{21..41}.mp4`
   (keep `reference videos/raw`), `dlp/*.mp4` (keep `t_lips/videos/video-{1..20}` until 1080p copies exist),
   `upload/`, `av-asr.tar`. Gain ≈ 128 GB.
2. Move `t_lips/timestamps/` into `work/<session>/transcript.json` under the
   new names (copy, verify, then delete). This is the only irreplaceable
   artefact of the old pipeline.
3. Tag the current repo state (`git tag v2-dataset-2026-09-28`) so the paper
   numbers in `docs/dataset_construction.md` stay reproducible.
4. Write `videos/sources.tsv` from the size-based mapping in section 6.

### Phase 1: `avsr/corpus` skeleton with resume (2–3 days)

5. `status.py`: stage markers, atomic write helper, options hash. Tests:
   interrupt-and-resume on a fake session.
6. `sources.py`: probe with ffprobe, sha256, yt-dlp download at
   `bestvideo[height<=1080]+bestaudio`. Re-download the 20 YouTube sessions.
7. `segment.py`: port `split_by_timestamp_rules`, add a pause rule (split on
   a gap > 0.4 s between words), a duration cap that splits at the longest
   pause instead of mid-sentence, true-case text, and carry over
   `no_speech_prob`, `avg_logprob`, `compression_ratio` per utterance.
   Apply the text-only cleaning rules (R1–R6 of the current code) *here*,
   so that rejected utterances are never cut. Output `segments.tsv`.
8. Move the I/İ merge out of `normalize_text` (default
   `merge_dotted_i=False`), update tokenizer alphabet and tests.

### Phase 2: cutter (3–4 days, then ~2 days of compute)

9. `cut.py`: decode each session **once**, sequentially with PyAV, keeping a
   sliding buffer; for each accepted utterance run the MediaPipe landmarker
   on its frames, cache landmarks, write the face-224 clip (archival, like
   LRS2) with the audio segment, and the clip's `.txt`. One worker per
   session, sessions in parallel, so seeks disappear.
10. Frame-rate handling: 30 → 25 fps by selecting frames at the target
    timestamps (as now) but documented and tested against audio length.
11. The existing `avsr/preprocess` mouth-crop then runs on the face-224 clips
    unchanged, or is called directly from `cut.py` to write 96x96 in the
    same pass when only the mouth variant is wanted.
12. Run on two sessions, compare clip counts, durations, and detection rate
    with run A and run B; adjust the track-break rule (run B's all-frames
    rule is too strict given 99.8 % detection in kept clips).
13. Cut all 40 sessions.

### Phase 3: dataset build and verification (1–2 days)

14. `build` = current `build_dataset.py` logic (split, PTM, clip table) reading
    `work/*/clips.tsv`; `verify` = current `verify_dataset.py` plus: every
    `Text:` line contains both `İ` and `I` across the corpus, word times are
    within the clip, audio and video lengths agree.
15. Regenerate `docs/label_audit.md` and `docs/dataset_construction.md`
    numbers; fill the "to be completed by the authors" items (model, decoding,
    segmentation, cropping are now all in code).

### Phase 4: optional, in the background

16. Re-transcribe with `faster-whisper large-v3` (chunked, resumable,
    `work/<session>/transcript.large-v3.json`), then re-run `segment`, `cut`,
    `build`. Compare retained hours and hallucination counts per session with
    the `small` transcripts.
17. Add the missing source for `video-34` if it can be identified; add further
    TBMM sessions with the same `ingest` command.

### Phase 5: repository cleanup

18. Delete the items in 1.4, archive `lr_codes/*` into one `archive/` folder
    outside the repo, drop `reference videos/tr-lrs` tarballs and
    `data/raw` once the new dataset verifies, drop `t_lips/videos/dataset`.

## 5. Risks

- **Disk**: full videos (92 + 40 GB) + face-224 clips (~20 GB) + mouth-96
  (~12 GB) + landmarks (2 GB) fit only after Phase 0.
- **MediaPipe versions**: the v2 landmarker (0.10.35, Tasks API) and the old
  FaceMesh (0.10.21) give slightly different points; the new cutter uses the
  v2 landmarker only.
- **Changing the alphabet** invalidates the current tokenizer and any
  fine-tuned checkpoint's output layer; pretrained encoder weights still load.
- **Re-transcription changes the test set**, so results before and after are
  not comparable; keep both clip tables.

## 6. Session name mapping (by file size, all exact matches)

| t_lips name | Assembly session | Hours |
|---|---|---|
| video-21 | 2024-2025-birlesim-72 | 9.2 |
| video-23 | 2024-2025-birlesim-73 | 8.7 |
| video-24 | 2024-2025-birlesim-74 | 8.2 |
| video-25 | 2024-2025-birlesim-76 | 7.7 |
| video-26 | 2024-2025-birlesim-80 | 4.4 |
| video-27 | 2024-2025-birlesim-82 | 9.3 |
| video-28 | 2024-2025-birlesim-84 | 8.8 |
| video-29 | 2024-2025-birlesim-94 | 11.3 |
| video-30 | 2024-2025-birlesim-109 | 11.2 |
| video-31 | 2024-2025-birlesim-110 | 11.6 |
| video-32 | 2025-2026-birlesim-1 | 8.5 |
| video-33 | 2025-2026-birlesim-2 | 6.7 |
| video-35 | 2025-2026-birlesim-3 | 8.1 |
| video-36 | 2025-2026-birlesim-5 | 8.1 |
| video-37 | 2025-2026-birlesim-7 | 7.9 |
| video-38 | 2025-2026-birlesim-8 | 8.0 |
| video-39 | 2025-2026-birlesim-9 | 8.0 |
| video-40 | 2025-2026-birlesim-10 | 10.4 |
| video-41 | 2025-2026-birlesim-12 | 7.2 |

The tarball-to-folder mapping of the current corpus is in
`data/raw/.extracted/owners.json` (tarballs 1–14 hold the assembly-named
sessions, tarballs 15–33 hold `video-1` … `video-19`). `video-20` and
`video-37` … `video-41` were cut in run B only.

## 7. Phase 0 log (2026-10-17)

Done:

- All 39 full videos verified byte-identical across copies and moved to
  `videos/`, named `<assembly session>.mp4` or `yt-<youtube id>.mp4`.
  `videos/sources.tsv` records origin, URL, resolution, fps, hours, bytes;
  `videos/SHA256SUMS` is written by a background job.
- Duplicate copies in `lr_codes/t_lips/videos`, `lr_codes/dlp`, `upload/`
  and `av-asr.tar` deleted. **They were btrfs reflinks of each other, so
  this freed only 11 GB, not 128 GB.** Real free space is 37 GB; the big
  consumers left are `~/.local/share/Trash` (61 GB), `~/.cache/pip`
  (21 GB), `exp/` (12 GB), `data/raw` (18 GB, an exact extraction of the
  `reference videos/tr-lrs` tarballs), `lr_codes/t_lips/videos/dataset`
  (18 GB, run B).
- Whisper JSON copied to `work/<session>/transcript.json` for all 40
  transcripts; `video-34` is under `work/unidentified-video-34/`.
- Repo tagged `v2-dataset-2026-09-28`; `videos/`, `work/`, `dataset/`
  git-ignored.
- `scripts/download_youtube_1080p.sh` re-downloads the 20 YouTube sessions
  (H.264 1080p, resumable, replaces the 360p file once verified, pauses
  under 15 GB free). It needs `.venv/bin/yt-dlp` 2026.08+ with
  `--js-runtimes node`; older builds and builds without a JavaScript runtime
  get HTTP 403 after the first bytes. Files are 4–6 GB each, about 80 GB
  in total, so space must be freed while it runs.

| t_lips name | New name |
|---|---|
| video-1 … video-20 | yt-_3u6pzqTuoo, yt-xZkvTV0J1PI, yt-UoHW7olV1LU, yt-sLfJGTUSnjw, yt-S_sCbEZeqCc, yt-QEdmcWbgxrg, yt-r7XQxD_fBz4, yt-Pw91kFFU44o, yt-OPGJ5-ARIC4, yt-NPJohShU_iw, yt-MZtO3ofW7eA, yt-l419Asc8yqE, yt-io-FZa_APv4, yt-HTCukfIyxoM, yt-hkcAzBtqACE, yt-_8zUPRDIwKM, yt-1DIP6Ruvtgg, yt-45SXDKapxh8, yt-fP99G1PONhQ, yt-Fpah9dtIsFo |
| video-21 … video-41 | see section 6 |

## 8. Phase 1 and 2 log (2026-10-17)

Code, all under `avsr/corpus/` with `scripts/corpus.py` as the single entry
point and `tests/test_corpus.py` (19 tests):

| Module | Does |
|---|---|
| `status.py` | atomic writes, option hashes, per-session stage markers (`work/<session>/status.json`) |
| `sources.py` | ffprobe facts and checksums into `videos/sources.tsv`; YouTube download (H.264 ≤1080p, needs `yt-dlp` 2026.08+ and `--js-runtimes node`) |
| `transcribe.py` | `faster_whisper` (default `large-v3`, int8_float16, VAD) or `whisper_timestamped`; 20-minute chunks saved under `work/<session>/chunks/<model>/`, merged into `transcript.json`; resumable per chunk |
| `segment.py` | transcript → `segments.tsv`: sentence end, pause > 0.4 s, 10 s cap split at the longest pause, 2-word minimum; true-case text and word times (`words` column); text verdicts (R1–R6 of the old pipeline plus `no_speech` when Whisper's `no_speech_prob` > 0.6) |
| `cut.py` | one sequential decode per session; frames picked nearest to the 25 fps grid; up to 4 faces per frame tracked through the clip and the **speaker chosen by lip motion** (with a size floor); frames shrunk so the face is ≤160 px; mouth 96x96 (or face) + 16 kHz mono AAC in one mp4, LRS2-style `.txt`; `work/<session>/clips.tsv` appended per clip for resume; landmarks cached; pauses under 5 GB free |
| `build.py` | clip table in the old `CLIP_FIELDS` layout, whole-session test split, last-5 % validation block, manifests; `verify` with 11 checks |
| `scripts/corpus.py run` | transcribes one session at a time on the GPU and cuts finished sessions in 4 worker processes meanwhile |

Findings while testing on real data:

- **Speaker selection was necessary.** With one face per frame MediaPipe
  returned the listener behind the speaker in a close-up; the mouth crop was
  of the wrong person. Tracking several faces and choosing the one whose lip
  opening varies most fixed it (checked visually on a 3-minute trial).
- **One transcript was mis-mapped.** `video-35_ts.json` (8.07 h, "beşinci
  birleşim") was a duplicate transcription of `birlesim-5`; the real
  transcript of `2025-2026-birlesim-3` (4.81 h, "üçüncü birleşim") was
  `video-34_ts.json`. Every other session's transcript length matches its
  video to within 0.02 h. Both cropper runs used `video-35_ts.json` for
  `birlesim-3`, which was the v2 **test session**: its labels were from
  another day. The v2 numbers measured on that test set are therefore not
  meaningful.
- **The old transcripts are not time-shifted.** Word 3-grams of large-v3
  windows at 0.5, 2.5 and 4 hours match the old transcript at the same time
  (within the 20 s bucket of the check) on two sessions.
- **The old transcript of `yt-fP99G1PONhQ` is garbage** (every segment is
  "HECK" or "DEN"); its audio is normal. It is re-transcribed like the rest.
- **large-v3 is clearly better than small** on the same minute of audio
  (readable, correctly segmented sentences against garbled text), and runs
  at about 10x real time on the RTX 2060 with int8_float16, so the whole
  corpus takes about 35 hours of GPU time. Decision: the dataset is built
  from large-v3 transcripts; the old ones stay as
  `work/<session>/transcript.small.json`.
- **Cutting speed**: 41 utterances per minute per worker on 1080p, about
  3.5 h per session; 4 workers run while the next session is transcribed.
- **Partition-Time Masking** is no longer written into the clips; it is
  applied during training (`augment.video.ptm.enabled: true`), one new mask
  per epoch, which removes the fixed-mask limitation of the v2 dataset.
- **No separate `.aac` file**: the mp4 carries the 16 kHz mono audio.

Pipeline started 2026-10-17 with `scripts/corpus.py run --dataset dataset/tr-lrs-v3`
(`logs/run_v3.log`); the 1080p re-download continues alongside
(`logs/download_1080p.log`); sessions still at 360p are transcribed but
cut only after their 1080p file has arrived (`scripts/corpus.py cut` again).

### Finding on the first cut session (2026-10-17, afternoon)

Of the first 300 utterances of `2024-2025-birlesim-109`, 137 were rejected as
`no_face` and 30 as `low_detection_rate`, although the frames show the
speaker's face sharp and frontal, 200 to 300 pixels tall. Reproduced on
single frames: MediaPipe's FaceLandmarker finds faces on a 192x192
letterboxed copy of its input, so a 1920x1080 frame is seen at 192x108 and
a 250-pixel face is about 25 pixels there, at the detector's limit. The
same frame given as a square centre crop is detected every time. The trial
on `2025-2026-birlesim-3` had not shown this because its shot was wider.

Fix (`avsr.corpus.cut.FaceFinder`): every 8th frame, and whenever no face
is known, the frame is shrunk to 540 rows and scanned in overlapping square
tiles; in between, each known face is re-detected at full resolution in a
square window of 2.5 face heights around its last position. Landmarks
therefore always come from the full frame. Scanning every frame would cost
three to four times the present detection time (the per-face landmark
stage dominates), the tracked scheme costs about the same as before.

`clips.tsv` now carries a sidecar `clips.options` with the hash of the cut
options and the video size; a partial result made with other options is
discarded instead of being continued. The pipeline was restarted and the
partial results of `2024-2025-birlesim-109` were cut again.

### Cutting throughput (2026-10-17, evening)

Profiling one worker showed MediaPipe inference at two thirds of the time
(about 35 ms per call under load, 1.5 calls per frame) and the cv2 shrink of
the whole 1080p frame at a further seventh. Changes, all in
`avsr/corpus/cut.py` unless noted, each measured on the same speech of
`2024-2025-birlesim-109`:

- `detect_every = 2`: landmarks on every second output frame, the others
  interpolated (the crop track is Kalman-smoothed anyway); the speaker is
  chosen among the detected frames. `region_scale = 3`: only a square of
  three face heights around the tallest face is stored, at the bounded
  scale, so a worker holds a few hundred MB instead of gigabytes.
  9.5 -> 18.4 utterances per minute per worker, same verdicts, crops
  checked visually.
- `scan_every = 16` (scans were a fifth of the remaining time).
- `delegate = "gpu"` (`avsr/preprocess/landmarks.py`): MediaPipe runs its
  models through OpenGL ES on the integrated Intel GPU, which is otherwise
  idle. 53 ms per call with four processes on it against 96 ms on the loaded
  CPU, and the CPU is freed for the transcriber. The NVIDIA card is not
  usable for this (no GL ES service through EGL). The delegate is a runtime
  choice and is left out of the options hash, so a change of it does not
  discard partial results.
- Four processes saturate the integrated GPU (six gave no more than four),
  so `scripts/corpus.py run --workers 7 --gpu-workers 4` runs four workers
  on the GPU delegate and three on the CPU: 79 utterances per minute in
  total with the transcriber alongside, against 56 before the evening.
- Resume now seeks to the first utterance left (`FrameSource.seek`)
  instead of decoding everything before it, which took ten minutes per
  worker at every restart.

With about 275,000 kept utterances in the corpus this is about 2.3 days of
cutting; transcription (about 10x real time) stays ahead of it.

### Error verdicts of the first full cut (2026-10-20)

The cut of 37 sessions finished on 2026-10-20 with 250,487 utterances
processed, 77.1% kept. 6.2% (16,154) had the verdict `error`, all with the
same message, "chosen speaker outside the stored region": the stored square
is centred on the tallest face while the frames are read, and the
lip-motion rule then chose a different face (a smaller speaker beside a
larger listener, typical of the close-ups of seated members); the bounds
check also demanded a full crop width on each side of the mouth where half
is enough, which rejected large faces near the frame edge. Fix: the check
uses half the crop plus 10%, and when the chosen face still lies outside,
the utterance's frames are read again (a seek, cheap) with the square
centred on the chosen face. On 16 real error utterances of
`2024-2025-birlesim-109`, 14 became `kept`; the two left have the mouth too
close to the frame edge for a full crop. `scripts/corpus.py cut
--retry-errors` cuts only the error rows again; their new rows replace the
old ones in `clips.tsv` (the last row of an utterance counts).

Two sessions (`yt-NPJohShU_iw`, `yt-xZkvTV0J1PI`) were skipped by `run`
because `videos/sources.tsv` still listed them at 360p: their 1080p files
arrived in the download retry after the last `ingest`. `ingest` was run
again and they are cut separately.

## 9. Phase 3 and 5 log (2026-10-22)

The retry pass and the two remaining sessions finished on 2026-10-21
without failures. Final cut verdicts over the 261,295 utterances that
passed the text rules: kept 225,977 (86.5 %), no_face 8.9 %,
low_detection_rate 4.5 %, error 220 (0.1 %, the mouth region leaves the
frame), small_face 11. The retry recovered 15,934 of the 16,154 error rows.

Phase 3: `build` with holdout `2025-2026-birlesim-3` and `yt-NPJohShU_iw`,
`verify` (all checks passed on 900 decoded clips), tokenizers of 1,000 and
500 units in `dataset/tr-lrs-v3-tokenizer/`. Train 208,811 clips (207.8 h,
37 sessions), validation 10,993 (10.5 h), test 6,173 (6.4 h, 2 sessions).
The full numbers are in `docs/dataset_construction.md`, which was rewritten
for v3; the v2 document and label audit moved to `docs/archive/`.

Phase 5: removed `scripts/{extract_raw,build_dataset,audit_labels,
verify_dataset,build_manifest}.py`, `tests/test_scripts.py`,
`avsr/preprocess/{pipeline,ptm_bake}.py` and the raw-corpus helpers of
`avsr/data/manifest.py` (all in git history, tag `v2-dataset-2026-09-28`).
`configs/base.yaml` and the fine-tuning configs point at
`dataset/tr-lrs-v3`; the A/B files for ROI and smoothing name the `cut`
commands that make their variant datasets. Moved out of the repository into
`~/Desktop/archive/`: the Auto-AVSR reference code
(`tests/test_reference_equivalence.py` reads its location from
`AUTO_AVSR_REFERENCE`), the papers, the first-corpus logs and sample data,
and the old `lr_codes` folders (`t_lips`, `dlp`, `whisper`, `frequency`,
`lr-web`, `Visual_Speech_Recognition_for_Multiple_Languages`).

Not deleted, because they are the only copies of the first corpus and
its sources: `~/Desktop/reference videos/tr-lrs` (the 33 tarballs, 17 GB),
`lr-v2-codebase/data/` (the v2 dataset, cache, manifests and tokenizer,
13 GB), `~/Desktop/archive/lr_codes/t_lips/videos/dataset` (run B, 19 GB)
and `.../Visual_Speech_Recognition_for_Multiple_Languages/{data,tr-test}`
(12 GB). They can go once nothing more is wanted from v2.
