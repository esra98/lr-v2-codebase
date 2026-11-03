#!/usr/bin/env bash
# Eğitimden önceki testler. setup.sh'den sonra çalıştır:
#
#   bash check.sh                      görüntü (dudak okuma) modeli için
#   MODE=audiovisual bash check.sh     görüntü+ses modeli için
#
#   1. Kod testleri (pytest, ~5-10 dk)
#   2. Preflight: her klip diskte mi, örnek klipler açılıyor mu, tokenizer,
#      pretrained ağırlıklar, disk, gerçek veriyle 20 adımlık deneme eğitimi
#   3. GPU bellek ölçümü: kaç frame'lik batch sığıyor
# Çıktı ayrıca logs/check.log (görüntü+ses için logs/check_audiovisual.log) dosyasına yazılır.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/tr-lipreading
cd "$PROJ"
source .venv/bin/activate
mkdir -p logs
NW=$(( $(nproc) - 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2

MODE=${MODE:-video}
case "$MODE" in
video)       CONFIG=training/configs/finetune/vsr_base.yaml;  FRAMES_LIST="600 900 1800 2700 3600"; LOG=logs/check.log;   NAME="vsr_base, 241M parametre";;
audiovisual) CONFIG=training/configs/finetune/avsr_base.yaml; FRAMES_LIST="600 900 1200 1800"; LOG=logs/check_audiovisual.log; NAME="avsr_base, 435M parametre";;
*) echo "MODE video ya da audiovisual olmalı (verilen: $MODE)" >&2; exit 1;;
esac
PREFIX=""; [ "$MODE" = audiovisual ] && PREFIX="MODE=audiovisual "

{
echo "=== 1/3 pytest"
python -m pytest -q -p no:cacheprovider

echo; echo "=== 2/3 preflight ($MODE)"
python training/preflight.py -c $CONFIG --sample 500 data.num_workers=$NW

echo; echo "=== 3/3 GPU bellek ölçümü ($NAME)"
python training/benchmark_memory.py -c $CONFIG --batch-frames $FRAMES_LIST

echo
echo "TESTLER TAMAM ($MODE)."
echo "1800 satırının 'peak GB' değeri GPU belleğinin ~%80'inden azsa:  ${PREFIX}bash $HERE/train.sh full"
echo "Değilse 900 satırına bak; o sığıyorsa:  ${PREFIX}FRAMES=900 ACCUM=2 bash $HERE/train.sh full"
echo "Yalnız 600 sığıyorsa:  ${PREFIX}FRAMES=600 ACCUM=3 bash $HERE/train.sh full"
} 2>&1 | tee $LOG
