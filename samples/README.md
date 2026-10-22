# Sample clips of TR-LRS v3

50 clips of `dataset/tr-lrs-v3`, chosen at random with a fixed seed: 36
from the training split, 7 from validation and 7 from the test split,
spread over 38 of the 39 recordings. Each clip is `<session>/<utt>.mp4`
(96x96 mouth crop at 25 fps as stream 0, 16 kHz mono AAC as stream 1) with
`<utt>.txt` next to it: a `Text:` line in Whisper's casing, then one line
per word with start time, end time and confidence, all relative to the
clip.

`clips.tsv` holds these clips' rows of the clip table: verdict, split,
number of frames, mean word confidence, face detection rate, yaw proxy,
lip motion and the normalised training text.

The full dataset (225,977 clips, 224.6 hours, 14 GB) is described in
`docs/dataset_construction.md`.
