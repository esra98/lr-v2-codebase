#!/usr/bin/env bash
# Eğitimden önceki testler. setup.sh'den sonra çalıştır:  bash check.sh
#
#   1. Kod testleri (pytest, ~5-10 dk)
#   2. Preflight: her klip diskte mi, örnek klipler açılıyor mu, tokenizer,
#      pretrained ağırlıklar, disk, gerçek veriyle 20 adımlık deneme eğitimi
#   3. GPU bellek ölçümü: kaç frame'lik batch sığıyor
# Çıktı ayrıca logs/check.log dosyasına yazılır.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/tr-lipreading
cd "$PROJ"
source .venv/bin/activate
mkdir -p logs
NW=$(( $(nproc) - 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2

{
echo "=== 1/3 pytest"
python -m pytest -q -p no:cacheprovider

echo; echo "=== 2/3 preflight"
python training/preflight.py -c training/configs/finetune/vsr_base.yaml --sample 500 data.num_workers=$NW

echo; echo "=== 3/3 GPU bellek ölçümü (vsr_base, 241M parametre)"
python training/benchmark_memory.py -c training/configs/finetune/vsr_base.yaml --batch-frames 900 1800 2700 3600

echo
echo "TESTLER TAMAM."
echo "1800 satırının 'peak GB' değeri GPU belleğinin ~%80'inden azsa:  bash $HERE/train.sh full"
echo "Değilse (ör. 24 GB kart):  FRAMES=900 ACCUM=2 bash $HERE/train.sh full"
} 2>&1 | tee logs/check.log
