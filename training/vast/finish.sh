#!/usr/bin/env bash
# Eğitim bittikten sonra: test seti değerlendirmesi ve indirilecek model paketi.
#
#   bash finish.sh                         en son başlatılan eğitim (train.sh full)
#   MODE=audiovisual bash finish.sh        görüntü+ses eğitimi (EXP=avsr_finetune)
#   EXP=vsr_finetune bash finish.sh        adıyla seçilen eğitim
#   SNR="0 -5" bash finish.sh              ses kullanan modelde ek olarak gürültülü
#                                          (babble, 0 ve -5 dB) test; her biri ~test süresi
#
# Sonuç: görüntü modelinde $WORK/tr-vsr-model.tar, görüntü+ses modelinde
# $WORK/tr-avsr-model.tar (+ .sha256). İçinde ortalanmış model, config,
# tokenizer, test sonuçları, eğitim metrikleri, kod ve MODEL_KULLANIMI.md var.
# Bu tek dosyayı kendi bilgisayarına indirdikten sonra sunucu kapatılabilir.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/tr-lipreading
cd "$PROJ"
source .venv/bin/activate

if [ -z "${MODE:-}${EXP:-}" ] && [ -f logs/last_exp ]; then
    EXP=$(cat logs/last_exp)
fi
case "${MODE:-video}" in
video)       EXP=${EXP:-vsr_finetune};;
audiovisual) EXP=${EXP:-avsr_finetune};;
*) echo "MODE video ya da audiovisual olmalı (verilen: $MODE)" >&2; exit 1;;
esac
AVG=${AVG:-10}
SNR=${SNR:-}
[ -f "exp/$EXP/config.yaml" ] || { echo "exp/$EXP/config.yaml yok: eğitim adı yanlış mı? (EXP=...)" >&2; exit 1; }

if pgrep -f "^python training/train.py" >/dev/null; then
    echo "Eğitim hâlâ çalışıyor; bitmesini bekle (bash train.sh status)." >&2; exit 1
fi

# Modelin modu ve test edilecek girdi modları (modelin eğitimde gördükleri)
read -r MODALITY EVAL_MODES < <(python - "exp/$EXP/config.yaml" <<'EOF'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
modality = cfg.model.modality
modes = [modality]
if modality == "audiovisual":
    probs = list(cfg.train.get("input_mode_probs", [0, 0, 1]))
    modes = [m for m, p in zip(("audiovisual", "video", "audio"), (probs[2], probs[0], probs[1])) if p > 0]
print(modality, ",".join(modes))
EOF
)
[ -n "${MODALITY:-}" ] || { echo "exp/$EXP/config.yaml okunamadı" >&2; exit 1; }
NAME=$([ "$MODALITY" = video ] && echo tr-vsr-model || echo tr-avsr-model)
OUT=$WORK/out/$NAME

echo "=== 1/3 Test seti (6,173 klip, beam search, son $AVG checkpoint ortalaması): $EXP, $MODALITY"
: > "logs/$EXP.eval.log"
for mode in ${EVAL_MODES//,/ }; do
    echo "--- girdi: $mode"
    python training/evaluate.py --run "exp/$EXP" --average "$AVG" decode.input_mode=$mode 2>&1 | tee -a "logs/$EXP.eval.log"
    if [ "$mode" != video ]; then
        for snr in $SNR; do
            echo "--- girdi: $mode, babble gürültüsü $snr dB"
            python training/evaluate.py --run "exp/$EXP" --average "$AVG" decode.input_mode=$mode decode.snr_target=$snr \
                2>&1 | tee -a "logs/$EXP.eval.log"
        done
    fi
done

echo; echo "=== 2/3 Model klasörü"
rm -rf "$OUT" && mkdir -p "$(dirname "$OUT")"
python training/export_model.py --run "exp/$EXP" --average "$AVG" --out "$OUT"
cp "$HERE/MODEL_KULLANIMI.md" "$OUT/"
cp "$HERE/code.tar" "$OUT/code.tar"
cp -r assets "$OUT/assets"   # face_landmarker.task: transcribe.py'nin yüz bulucusu (git'te yok)
mkdir -p "$OUT/logs" && cp logs/"$EXP".*log "$OUT/logs/"   # eğitim (devam ettirildiyse parçaları) ve test logları

echo; echo "=== 3/3 Deneme: model klasöründen test klibi okuma"
clip=$(sed -n 2p dataset/tr-lrs-v3-manifests/test.tsv | cut -f1)
python - "$OUT" "dataset/tr-lrs-v3/$clip.mp4" <<'EOF'
import sys, torch
from avsr.inference import load_trained, sound_track, transcribe
from avsr.preprocess.video_io import read_video
module, _, _ = load_trained(sys.argv[1])
module.to("cuda").eval()
crops = read_video(sys.argv[2])  # dataset klipleri zaten 96x96 ağız kırpıntısı
audio = sound_track(sys.argv[2]) if module.cfg.model.modality != "video" else None
print("model klasörü çalışıyor; örnek çıktı:", transcribe(module, crops, audio=audio)[0][2])
EOF

tar -cf "$WORK/$NAME.tar" -C "$(dirname "$OUT")" "$NAME"
(cd "$WORK" && sha256sum "$NAME.tar" > "$NAME.tar.sha256")
ls -lh "$WORK/$NAME.tar"
for f in "exp/$EXP/eval/"test_*.json; do echo "$f"; cat "$f"; echo; done
echo
echo "İNDİR (kendi bilgisayarında):"
echo "  scp -P <port> 'root@<sunucu>:$WORK/$NAME.tar*' ."
echo "  sha256sum -c $NAME.tar.sha256"
