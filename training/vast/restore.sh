#!/usr/bin/env bash
# Laptopta çalışır: backup.sh'nin aldığı en son sağlam yedeği yeni bir sunucuya gönderir ve
# eğitimi orada kaldığı yerden sürdürür (epoch'un ortasından da).
#
#   bash restore.sh <instance id>
#   bash restore.sh <ssh adresi> <port>
#
# Önce yeni sunucuda setup.sh bitmiş olmalı (KURULUM TAMAM); WORK aynı kalmalı (varsayılan
# /workspace), checkpoint'lerin tuttuğu yollar ona göre. Aynı GPU türü önerilir: FRAMES ve
# ACCUM eski sunucudaki gibi kalır (logs/<deney>.opts). Gönderilen: BACKUP_DIR/current
# içindeki exp/ ve logs/ (görüntü modelinde ~15 GB, görüntü+ses ~25 GB; laptopun yükleme
# hızına göre sürer). Ayarlar: BACKUP_DIR=~/lr-v2-yedek  NO_RESUME=1 (yalnızca gönder)
set -euo pipefail

BACKUP_DIR=${BACKUP_DIR:-$HOME/lr-v2-yedek}
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 1; }
SRC=$BACKUP_DIR/current
[ -d "$SRC/exp" ] || { echo "$SRC yok: önce backup.sh ile yedek alınmış olmalı" >&2; exit 1; }
EXP=$(cat "$SRC/logs/last_exp")

if [ $# -eq 1 ]; then
    key=$(tr -d '[:space:]' < ~/.config/vastai/vast_api_key)
    read -r HOST PORT < <(curl -s --max-time 30 -H "Authorization: Bearer $key" \
        "https://console.vast.ai/api/v0/instances/$1/" | python3 -c '
import json, sys
d = json.load(sys.stdin)
i = d.get("instances") or {}
i = (i[0] if i else {}) if isinstance(i, list) else i
print(i.get("ssh_host") or "-", i.get("ssh_port") or "-")')
    [ "$HOST" != "-" ] || { echo "instance $1 için SSH adresi alınamadı" >&2; exit 1; }
else
    HOST=$1; PORT=$2
fi
SSH=("${SSH_BIN:-ssh}" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30
     -o StrictHostKeyChecking=accept-new)
remote() { "${SSH[@]}" "root@$HOST" "$@"; }

# Sunucuda kurulumun bittiği yer: setup.sh'nin yazdığı .work
BUNDLE=$(remote 'for d in /root/lr-v2-vast /workspace/lr-v2-vast; do [ -f $d/.work ] && { echo $d; break; }; done; true')
[ -n "$BUNDLE" ] || { echo "sunucuda kurulum bulunamadı: önce setup.sh (KURULUM.md 3)" >&2; exit 1; }
PROJ=$(remote "source $BUNDLE/.work && echo \$WORK/tr-lipreading")
if remote "pgrep -f '^python training/train.py'" >/dev/null; then
    echo "sunucuda bir eğitim çalışıyor; durdurmadan geri yüklenmez" >&2; exit 1
fi
if [ -z "${FORCE:-}" ] && remote "[ -f $PROJ/exp/$EXP/checkpoints/last.ckpt ]"; then
    echo "sunucuda zaten $EXP checkpoint'i var; üzerine yazmak için FORCE=1" >&2; exit 1
fi

echo "gönderiliyor: $SRC ($EXP, last.ckpt $(date -r "$SRC/exp/$EXP/checkpoints/last.ckpt" '+%F %T'), $(du -shL "$SRC" | cut -f1)) -> $HOST:$PROJ"
rsync -a -W --partial-dir=.rsync-partial --info=progress2 -e "${SSH[*]}" "$SRC/exp" "$SRC/logs" "root@$HOST:$PROJ/"
remote "cd $PROJ/exp/$EXP/checkpoints && ls -l"

if [ -n "${NO_RESUME:-}" ]; then
    echo "gönderildi; sürdürmek için sunucuda: bash $BUNDLE/train.sh resume"; exit 0
fi
remote "EXP=$EXP bash $BUNDLE/train.sh resume"
echo "İzlemek için sunucuda: bash $BUNDLE/train.sh status"
