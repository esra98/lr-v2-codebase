# Live demo: lip reading from the webcam

A small desktop app (PyQt6, one file) that lets you try the trained model on
yourself. Press **Başlat**, speak after the 3-2-1 countdown, press **Durdur**.
The app then shows, side by side:

- **Dudak okuma** – the text the lip-reading model reads from the mouth crops
  alone (the sound is not used),
- **Whisper large-v3** – the text Whisper hears in the sound, as the reference,

with the words aligned and coloured (white: matched, red: misread, grey:
missed), WER and CER, the share of frames in which the face was found, and the
mouth crops the model saw. A video file can be read instead of the webcam
(**Video dosyası seç…**). Every take is kept under `~/Videos/dudak_okuma/`
and can be reopened or deleted from the history list.

## Run

```bash
pip install PyQt6 faster-whisper mediapipe opencv-python   # plus the training requirements
python live-demo/dudak_okuma.py --run tr-vsr-model          # a folder from training/export_model.py
```

Options: `--seconds` (longest take: 10, 20 or 30), `--camera` (V4L2 index),
`--beam` (default 40), `--whisper` (default `large-v3`). Needs `ffmpeg` with
PulseAudio input for the microphone and a CUDA GPU; both models together take
about 3.5 GB of GPU memory. Tested on Linux (Fedora, KDE) with an RTX 2060.

The camera's automatic exposure priority is switched off at start, so the
camera keeps 30 fps in dim rooms instead of dropping to 10 fps.

## What to expect

The model was trained only on TBMM TV broadcasts (test WER 34%). Away from
that setting it does much worse, and the reasons are useful to know:

- **Words.** It can only write words it saw in training. Everyday phrases
  such as *"onu çok seviyorum"* are rare or absent in parliamentary speech
  (SEVİYORUM appears in 15 of 225,977 clips, BAYILIYORUM in none).
- **Language prior.** When the lips give little evidence, the decoder falls
  back to the most frequent parliamentary phrases, e.g. *"TÜRKİYE BÜYÜK MİLLET
  MECLİSİ"* (964 clips) or a member's name, even for a closed mouth.
- **Picture.** Broadcast speakers are filmed from the front, at eye level, in
  studio light. A laptop camera below the face, dim or coloured light,
  reflections on the lips or a hand in front of the mouth all take the input
  away from what the model knows.

For the best result: camera at eye level, light from the front, face filling
the dashed oval, and parliament-style sentences (*"Sayın Başkan, değerli
milletvekilleri, …"*).
