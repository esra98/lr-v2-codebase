#!/usr/bin/env bash
# Eğitim bittikten sonra: test seti değerlendirmesi ve indirilecek model paketi.
#
#   bash finish.sh            (EXP=vsr_finetune AVG=10)
#
# Sonuç: $WORK/tr-vsr-model.tar (+ .sha256). İçinde ortalanmış model, config,
# tokenizer, test sonuçları, eğitim metrikleri, kod ve MODEL_KULLANIMI.md var.
# Bu tek dosyayı kendi bilgisayarına indirdikten sonra sunucu kapatılabilir.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/lr-v2-codebase
cd "$PROJ"
source .venv/bin/activate

EXP=${EXP:-vsr_finetune}
AVG=${AVG:-10}
OUT=$WORK/out/tr-vsr-model

if pgrep -f "^python scripts/train.py" >/dev/null; then
    echo "Eğitim hâlâ çalışıyor; bitmesini bekle (bash train.sh status)." >&2; exit 1
fi

echo "=== 1/3 Test seti (6,173 klip, beam search, son $AVG checkpoint ortalaması)"
python scripts/evaluate.py --run "exp/$EXP" --average "$AVG" 2>&1 | tee "logs/$EXP.eval.log"

echo; echo "=== 2/3 Model klasörü"
rm -rf "$OUT" && mkdir -p "$(dirname "$OUT")"
python scripts/export_model.py --run "exp/$EXP" --average "$AVG" --out "$OUT"
cp "$HERE/MODEL_KULLANIMI.md" "$OUT/"
cp "$HERE/code.tar" "$OUT/code.tar"
cp -r assets "$OUT/assets"   # face_landmarker.task: transcribe.py'nin yüz bulucusu (git'te yok)
mkdir -p "$OUT/logs" && cp logs/"$EXP".*log "$OUT/logs/"   # eğitim (devam ettirildiyse parçaları) ve test logları

echo; echo "=== 3/3 Deneme: model klasöründen test klibi okuma"
clip=$(sed -n 2p dataset/tr-lrs-v3-manifests/test.tsv | cut -f1)
python - "$OUT" "dataset/tr-lrs-v3/$clip.mp4" <<'EOF'
import sys, torch
from avsr.inference import load_trained, transcribe
from avsr.preprocess.video_io import read_video
module, _, _ = load_trained(sys.argv[1])
module.to("cuda").eval()
crops = read_video(sys.argv[2])  # dataset klipleri zaten 96x96 ağız kırpıntısı
print("model klasörü çalışıyor; örnek çıktı:", transcribe(module, crops)[0][2])
EOF

tar -cf "$WORK/tr-vsr-model.tar" -C "$(dirname "$OUT")" tr-vsr-model
(cd "$WORK" && sha256sum tr-vsr-model.tar > tr-vsr-model.tar.sha256)
ls -lh "$WORK/tr-vsr-model.tar"
cat "exp/$EXP/eval/"test_video.json
echo
echo "İNDİR (kendi bilgisayarında):"
echo "  scp -P <port> root@<sunucu>:$WORK/tr-vsr-model.tar* ."
echo "  sha256sum -c tr-vsr-model.tar.sha256"
