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

## Run on a Mac

```bash
brew install ffmpeg python@3.12
git clone https://github.com/esra98/tr-lipreading && cd tr-lipreading
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch -r requirements.txt PyQt6 faster-whisper mediapipe opencv-python
curl -L -o assets/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# the trained model: tr-vsr-model.tar, unpacked into the repository folder
tar -xf ~/Downloads/tr-vsr-model.tar
python live-demo/dudak_okuma.py --run tr-vsr-model
```

On the first start macOS asks for camera and microphone access for the
Terminal; allow both and start the app again. The lip-reading model runs on the
Apple GPU (MPS), Whisper on the CPU (`large-v3-turbo` by default); a 10-second
take takes about half a minute to a minute. If the wrong microphone records,
list the devices with `ffmpeg -f avfoundation -list_devices true -i ""` and
pass its number with `--mic`.

## Run on Linux

```bash
pip install PyQt6 faster-whisper mediapipe opencv-python   # plus the training requirements
python live-demo/dudak_okuma.py --run tr-vsr-model          # a folder from training/export_model.py
```

Needs `ffmpeg` with PulseAudio input and, for speed, an NVIDIA GPU (both models
together take about 3.5 GB of GPU memory; without one everything runs on the
CPU). Tested on Fedora (KDE) with an RTX 2060: a 10-second take takes about
10 seconds. The camera's automatic exposure priority is switched off at start,
so the camera keeps 30 fps in dim rooms instead of dropping to 10 fps.

Options: `--seconds` (longest take: 10, 20 or 30), `--camera`, `--mic` (Mac),
`--beam` (default 40), `--whisper` (default `large-v3`, on a Mac
`large-v3-turbo`).

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
