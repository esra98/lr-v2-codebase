#!/usr/bin/env bash
# Fine-tune'u başlat, izle, devam ettir.
#
#   bash train.sh full        tam eğitim (75 epoch, arka planda; SSH kapansa da sürer)
#   bash train.sh status      son epoch'ların val/cer değerleri, GPU durumu
#   bash train.sh resume      yarıda kalan eğitimi last.ckpt'den sürdür
#   bash train.sh auto        yarıda kalmış bir eğitim varsa sürdür, yoksa hiçbir şey yapma
#                             (spot sunucu yeniden başlayınca vast "onstart" komutu olarak)
#   bash train.sh pilot       (isteğe bağlı) pretrained lr ölçeği 0.02 ve 0.1 karşılaştırması, ~2 saat
#
# Ayarlar (ortam değişkeni):  MODE=video  EXP=<MODE'a göre>  FRAMES=1800  ACCUM=1
#   MODE=video         görüntü (dudak okuma): training/configs/finetune/vsr_base.yaml, EXP=vsr_finetune
#   MODE=audiovisual   görüntü+ses: training/configs/finetune/avsr_base.yaml, EXP=avsr_finetune
# status, resume ve auto, MODE ya da EXP verilmezse en son başlatılan eğitime bakar.
# last.ckpt epoch sonunda ve eğitim sırasında her 20 dakikada bir yazılır
# (train.checkpoint_every_minutes); resume epoch'un ortasından devam eder.
# FRAMES x ACCUM = 1800 olmalı (Auto-AVSR). Ek ayarlar komutun sonuna key=value olarak eklenebilir.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/tr-lipreading
cd "$PROJ"
source .venv/bin/activate
mkdir -p logs

cmd=${1:-}; shift || true
# En son başlatılan eğitimin adı (full ve resume yazar)
if [ -z "${MODE:-}${EXP:-}" ] && [ "$cmd" != full ] && [ "$cmd" != pilot ] && [ -f logs/last_exp ]; then
    EXP=$(cat logs/last_exp)
fi
MODE=${MODE:-video}
case "$MODE" in
video)       CONFIG=training/configs/finetune/vsr_base.yaml;  EXP=${EXP:-vsr_finetune};;
audiovisual) CONFIG=training/configs/finetune/avsr_base.yaml; EXP=${EXP:-avsr_finetune};;
*) echo "MODE video ya da audiovisual olmalı (verilen: $MODE)" >&2; exit 1;;
esac
FRAMES=${FRAMES:-1800}
ACCUM=${ACCUM:-1}
NW=$(( $(nproc) - 2 )); [ "$NW" -gt 16 ] && NW=16; [ "$NW" -lt 2 ] && NW=2
OPTS=(-c $CONFIG exp_name=$EXP data.max_frames_per_batch=$FRAMES
      train.accumulate_grad_batches=$ACCUM data.num_workers=$NW "$@")

running() { pgrep -f "^python training/train.py" >/dev/null; }
# rclone'da tezdrive tanımlıysa Drive yedeği (drive_backup.sh) eğitimle birlikte çalışır
start_drive_backup() {
    command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q "^tezdrive:" || return 0
    pgrep -f "bash $HERE/drive_backup.sh" >/dev/null && return 0
    nohup bash "$HERE/drive_backup.sh" >> /root/drive_backup.log 2>&1 < /dev/null &
    echo "Drive yedeği başladı (log: /root/drive_backup.log)"
}
refuse_if_running() {
    if running; then
        echo "Bir eğitim zaten çalışıyor (tek GPU): $(cat logs/last_exp 2>/dev/null). Durumu: bash train.sh status" >&2; exit 1
    fi
}

case "$cmd" in
full)
    refuse_if_running
    if [ -d "exp/$EXP/checkpoints" ]; then
        echo "exp/$EXP zaten var. Sürdürmek için: EXP=$EXP bash train.sh resume  (ya da EXP=<yeni ad>)" >&2; exit 1
    fi
    printf '%q ' "${OPTS[@]}" > "logs/$EXP.opts"
    echo "$EXP" > logs/last_exp
    nohup python training/train.py "${OPTS[@]}" > "logs/$EXP.log" 2>&1 &
    echo "Başladı ($MODE, $EXP, pid $!). Log: $PROJ/logs/$EXP.log"
    start_drive_backup
    echo "İzlemek için: bash $HERE/train.sh status"
    ;;
resume)
    refuse_if_running
    [ -f "exp/$EXP/checkpoints/last.ckpt" ] || { echo "exp/$EXP/checkpoints/last.ckpt yok" >&2; exit 1; }
    eval "SAVED=($(cat "logs/$EXP.opts"))"
    mv "logs/$EXP.log" "logs/$EXP.$(date +%s).log" 2>/dev/null || true
    echo "$EXP" > logs/last_exp
    # Başka bir sunucuda sürdürülüyorsa onun çekirdek sayısı; komutun sonundaki ayarlar da eklenir
    nohup python training/train.py "${SAVED[@]}" data.num_workers=$NW "$@" --resume "exp/$EXP/checkpoints/last.ckpt" \
        > "logs/$EXP.log" 2>&1 &
    echo "Sürdürülüyor ($EXP, pid $!). Log: $PROJ/logs/$EXP.log"
    start_drive_backup
    ;;
auto)
    # Bitmiş eğitim metrics.json yazar; ondan yeni bir last.ckpt varsa eğitim yarıda kalmıştır.
    if running; then echo "eğitim zaten çalışıyor"; exit 0; fi
    last=exp/$EXP/checkpoints/last.ckpt
    if [ ! -f "$last" ]; then echo "sürdürülecek eğitim yok ($last)"; exit 0; fi
    if [ -f "exp/$EXP/metrics.json" ] && [ "exp/$EXP/metrics.json" -nt "$last" ]; then
        echo "$EXP bitmiş; sürdürülmedi"; exit 0
    fi
    exec bash "$0" resume "$@"
    ;;
status)
    running && echo "eğitim ÇALIŞIYOR" || echo "eğitim çalışmıyor"
    echo "deney: $EXP"
    [ -f "exp/$EXP/checkpoints/last.ckpt" ] && echo "son checkpoint: $(date -r "exp/$EXP/checkpoints/last.ckpt" '+%F %T')"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    python - "$EXP" <<'EOF'
import csv, glob, os, sys
exp = sys.argv[1]
rows = []
# Earlier parts of a resumed run first (metrics.part1.csv, part2, ...), then the current one.
parts = sorted(glob.glob(f"exp/{exp}/csv/metrics.part*.csv"), key=lambda p: int(p.rsplit("part", 1)[1][:-4]))
for path in parts + glob.glob(f"exp/{exp}/csv/metrics.csv"):
    rows += list(csv.DictReader(open(path)))
# val/cer and val/wer, or val/cer_<mode> for a model validated in several input modes
keys = sorted({k for r in rows for k in r if k.startswith(("val/cer", "val/wer"))}, key=lambda k: (k[4:7] != "cer", k))
val = [r for r in rows if any(r.get(k) for k in keys)]
last = next((r for r in reversed(rows) if r.get("train/loss_step")), None)
width = {k: max(8, len(k)) for k in keys}
print(f"{'epoch':>5} " + " ".join(f"{k:>{width[k]}}" for k in keys))
for r in val[-10:]:
    print(f"{r['epoch']:>5} " + " ".join(f"{float(r[k]):{width[k]}.3f}" if r.get(k) else "-".rjust(width[k]) for k in keys))
if last:
    print(f"son adım {last['step']}, loss {float(last['train/loss_step']):.1f}, "
          f"amp_scale {last.get('train/amp_scale') or '-'}, RAM {float(last['train/ram_gb'] or 0):.1f} GB")
EOF
    grep -E "Traceback|Error|OOM|out of memory|Could not load" "logs/$EXP.log" | tail -5 || true
    ;;
pilot)
    refuse_if_running
    for scale in 0.02 0.1; do
        python training/train.py "${OPTS[@]}" exp_name=pilot_${EXP}_$scale train.pretrained_lr_scale=$scale \
            data.limit_train=20000 data.limit_val=1000 \
            train.max_epochs=4 train.warmup_epochs=0.5 train.hold_pretrained_epochs=0.25 \
            > "logs/pilot_${EXP}_$scale.log" 2>&1
    done
    for scale in 0.02 0.1; do
        echo "scale $scale: son val/cer $(grep -h -o '"val/cer[a-z_]*": [0-9.]*' exp/pilot_${EXP}_$scale/metrics.json 2>/dev/null | tail -1)"
    done
    echo "Düşük val/cer veren ölçeği tam eğitimde kullan:  MODE=$MODE bash train.sh full train.pretrained_lr_scale=<ölçek>"
    ;;
*)
    sed -n 2,17p "$0"; exit 1;;
esac
