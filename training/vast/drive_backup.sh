#!/usr/bin/env bash
# Sunucuda çalışır: eğitimin yedeğini rclone ile Google Drive'a yükler.
#
#   nohup bash drive_backup.sh > /root/drive_backup.log 2>&1 &     (train.sh full/resume kendisi başlatır)
#
# Drive'da (REMOTE, varsayılan tezdrive:EsraThesis):
#   runs/<deney>/config.yaml, csv/, tb/, logs/, metrics.json, eval/   her turda (küçük)
#   her EVERY epoch'ta bir (5, 10, 15, ...) ve eğitim bitince:
#     runs/<deney>/checkpoints/epochNNN.ckpt   o epoch'un ağırlıkları, kalıcı
#     runs/<deney>/checkpoints/last.ckpt       sürdürmek için tam durum, yenisiyle değiştirilir
# Büyük dosyalar geçici bir adla yüklenip yeniden adlandırılır: Drive aynı dosyanın üzerine
# yazınca her seferinde eski sürümü (2,9 GB) saklıyordu.
# Ayarlar: REMOTE=tezdrive:EsraThesis  EVERY=5  POLL=300 (sn)
set -uo pipefail

REMOTE=${REMOTE:-tezdrive:EsraThesis}
EVERY=${EVERY:-5}
POLL=${POLL:-300}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/.work"
PROJ=$WORK/tr-lipreading
STATE=$WORK/.drive-backup
mkdir -p "$STATE"
export RCLONE_LOG_LEVEL=ERROR
# csv ve loglar eğitim sürerken büyür; o anki halleri yüklenir (checkpoint'ler tek hamlede yazılır)
export RCLONE_LOCAL_NO_CHECK_UPDATED=true
log() { echo "$(date '+%F %T')  $*"; }

# Büyük dosya: önce sabit bir kopya (last.ckpt eğitim sürerken her 20 dk yeniden yazılır),
# süre sınırlı yükleme geçici ada, sonra eskisini silip yeniden adlandırma (sunucu tarafında, anında).
put() {
    local src=$1 dst=$2 tmp=$STATE/upload.tmp
    cp "$src" "$tmp" || return 1
    timeout 60m rclone copyto "$tmp" "$dst.uploading" --drive-chunk-size 128M --timeout 10m --retries 3 --low-level-retries 20
    local rc=$?
    rm -f "$tmp"
    [ $rc -eq 0 ] || { rclone deletefile "$dst.uploading" 2>/dev/null; return $rc; }
    rclone deletefile "$dst" --drive-use-trash=false 2>/dev/null
    rclone moveto "$dst.uploading" "$dst"
}

# Bitmiş epoch sayısı: val/cer yazılmış satırlar (devam ettirilen eğitimde parçalar dahil)
epochs_done() {
    python3 - "$1" <<'EOF'
import csv, glob, sys
rows = []
for p in sorted(glob.glob(f"{sys.argv[1]}/csv/metrics*.csv")):
    rows += list(csv.DictReader(open(p)))
print(len({r["epoch"] for r in rows if any(k.startswith("val/cer") and r[k] for k in r)}))
EOF
}

while true; do
    exp=$(cat "$PROJ/logs/last_exp" 2>/dev/null || true)
    run=$PROJ/exp/$exp
    if [ -n "$exp" ] && [ -d "$run" ]; then
        dst=$REMOTE/runs/$exp
        timeout 10m rclone copy "$run" "$dst" --include "config.yaml" --include "metrics.json" --include "csv/**" --include "tb/**" --include "eval/**"
        timeout 10m rclone copy "$PROJ/logs" "$dst/logs" --include "$exp.*" --include "check*.log"
        done_n=$(epochs_done "$run")
        finished=0; [ -f "$run/metrics.json" ] && [ "$run/metrics.json" -nt "$run/checkpoints/last.ckpt" ] && finished=1
        # Son yedeklenen dönüm noktası (5, 10, ...); eğitim bitince son epoch da dönüm noktasıdır
        milestone=$(( done_n - done_n % EVERY ))
        [ "$finished" = 1 ] && milestone=$done_n
        prev=$(cat "$STATE/$exp.milestone" 2>/dev/null || echo 0)
        if [ "$milestone" -gt "$prev" ]; then
            ckpt=$(printf "%s/checkpoints/epoch%03d.ckpt" "$run" "$((milestone - 1))")
            ok=1
            if [ -f "$ckpt" ]; then
                put "$ckpt" "$dst/checkpoints/$(basename "$ckpt")" && log "$exp: $(basename "$ckpt") yüklendi (epoch $milestone)" || ok=0
            else
                log "$exp: epoch $milestone checkpoint'i saklanan en iyiler arasında değil"
            fi
            if [ -f "$run/checkpoints/last.ckpt" ] && put "$run/checkpoints/last.ckpt" "$dst/checkpoints/last.ckpt"; then
                log "$exp: last.ckpt yüklendi (epoch $milestone sonrası)"
            else
                ok=0
            fi
            if [ "$ok" = 1 ]; then
                echo "$milestone" > "$STATE/$exp.milestone"
                [ "$finished" = 1 ] && log "$exp: eğitim bitti, son yedek tamam"
            else
                log "UYARI: $exp epoch $milestone yedeği tamamlanamadı; sonraki turda tekrar"
            fi
        fi
    fi
    sleep "$POLL"
done
