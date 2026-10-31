#!/usr/bin/env bash
# Fine-tune'u başlat, izle, devam ettir.
#
#   bash train.sh full        tam eğitim (75 epoch, arka planda; SSH kapansa da sürer)
#   bash train.sh status      son epoch'ların val/cer değerleri, GPU durumu
#   bash train.sh resume      yarıda kalan eğitimi last.ckpt'den sürdür
#   bash train.sh pilot       (isteğe bağlı) pretrained lr ölçeği 0.02 ve 0.1 karşılaştırması, ~2 saat
#
# Ayarlar (ortam değişkeni):  EXP=vsr_finetune  FRAMES=1800  ACCUM=1
# FRAMES x ACCUM = 1800 olmalı (Auto-AVSR). Ek ayarlar komutun sonuna key=value olarak eklenebilir.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/lr-v2-codebase
cd "$PROJ"
source .venv/bin/activate
mkdir -p logs

EXP=${EXP:-vsr_finetune}
FRAMES=${FRAMES:-1800}
ACCUM=${ACCUM:-1}
NW=$(( $(nproc) - 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
cmd=${1:-}; shift || true
OPTS=(-c configs/finetune/vsr_base.yaml exp_name=$EXP data.max_frames_per_batch=$FRAMES
      train.accumulate_grad_batches=$ACCUM data.num_workers=$NW "$@")

case "$cmd" in
full)
    if [ -d "exp/$EXP/checkpoints" ]; then
        echo "exp/$EXP zaten var. Sürdürmek için: bash train.sh resume  (ya da EXP=<yeni ad>)" >&2; exit 1
    fi
    printf '%q ' "${OPTS[@]}" > "logs/$EXP.opts"
    nohup python scripts/train.py "${OPTS[@]}" > "logs/$EXP.log" 2>&1 &
    echo "Başladı (pid $!). Log: $PROJ/logs/$EXP.log"
    echo "İzlemek için: bash $HERE/train.sh status"
    ;;
resume)
    [ -f "exp/$EXP/checkpoints/last.ckpt" ] || { echo "exp/$EXP/checkpoints/last.ckpt yok" >&2; exit 1; }
    eval "SAVED=($(cat "logs/$EXP.opts"))"
    mv "logs/$EXP.log" "logs/$EXP.$(date +%s).log" 2>/dev/null || true
    nohup python scripts/train.py "${SAVED[@]}" --resume "exp/$EXP/checkpoints/last.ckpt" > "logs/$EXP.log" 2>&1 &
    echo "Sürdürülüyor (pid $!). Log: $PROJ/logs/$EXP.log"
    ;;
status)
    pgrep -f "scripts/train.py" >/dev/null && echo "eğitim ÇALIŞIYOR" || echo "eğitim çalışmıyor"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    python - "$EXP" <<'EOF'
import csv, glob, os, sys
exp = sys.argv[1]
rows = []
for path in sorted(glob.glob(f"exp/{exp}/csv/metrics*.csv")):
    rows += list(csv.DictReader(open(path)))
val = [r for r in rows if r.get("val/cer")]
last = next((r for r in reversed(rows) if r.get("train/loss_step")), None)
print(f"{'epoch':>5} {'val/cer':>8} {'val/wer':>8}")
for r in val[-10:]:
    print(f"{r['epoch']:>5} {float(r['val/cer']):8.3f} {float(r['val/wer']):8.3f}")
if last:
    print(f"son adım {last['step']}, loss {float(last['train/loss_step']):.1f}, "
          f"amp_scale {last.get('train/amp_scale') or '-'}, RAM {float(last['train/ram_gb'] or 0):.1f} GB")
EOF
    grep -E "Traceback|Error|OOM|out of memory|Could not load" "logs/$EXP.log" | tail -5 || true
    ;;
pilot)
    for scale in 0.02 0.1; do
        python scripts/train.py "${OPTS[@]}" exp_name=pilot_scale_$scale train.pretrained_lr_scale=$scale \
            data.limit_train=20000 data.limit_val=1000 \
            train.max_epochs=4 train.warmup_epochs=0.5 train.hold_pretrained_epochs=0.25 \
            > "logs/pilot_scale_$scale.log" 2>&1
    done
    for scale in 0.02 0.1; do
        echo "scale $scale: son val/cer $(grep -h -o '"val/cer": [0-9.]*' exp/pilot_scale_$scale/metrics.json 2>/dev/null | tail -1)"
    done
    echo "Düşük val/cer veren ölçeği tam eğitimde kullan:  bash train.sh full train.pretrained_lr_scale=<ölçek>"
    ;;
*)
    sed -n 2,11p "$0"; exit 1;;
esac
