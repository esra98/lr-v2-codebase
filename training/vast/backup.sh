#!/usr/bin/env bash
# Laptopta çalışır: sunucudaki eğitimin yedeğini her INTERVAL dakikada bir bu bilgisayara çeker.
#
#   bash backup.sh <instance id>            vast instance ID'si; adres her turda API'den bulunur
#   bash backup.sh <ssh adresi> <port>      ör. ssh5.vast.ai 12345 (panelde "Connect")
#
# Ayarlar (ortam değişkeni):
#   INTERVAL=120          dakika. Her tur değişen checkpoint'leri indirir (last.ckpt görüntü
#                         modelinde ~3 GB, görüntü+ses ~5 GB); sunucunun çıkış trafiği ücretlidir
#   BACKUP_DIR=~/lr-v2-yedek
#   KEEP=3                tutulan anlık kopya sayısı
#   ONCE=1                tek tur ve çık
#
# Çekilen: sunucudaki tr-lipreading/exp/ (checkpoint'ler, metrikler, config) ve logs/.
# Her tur BACKUP_DIR/incoming'e çekilir (değişmeyen dosyalar önceki kopyaya hardlink, yer
# kaplamaz). Her .ckpt sağlamsa ve sunucudaki last.ckpt yedektekinden eski değilse kopya
# BACKUP_DIR/snapshots/<zaman> olur ve BACKUP_DIR/current onu gösterir. Yarım kalan
# indirme sonraki turda kaldığı yerden sürer. Yeni sunucuya geri yükleme: restore.sh.
set -uo pipefail

BACKUP_DIR=${BACKUP_DIR:-$HOME/lr-v2-yedek}
INTERVAL=${INTERVAL:-120}
KEEP=${KEEP:-3}
ONCE=${ONCE:-}
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 1; }
INSTANCE=""; HOST=""; PORT=""
if [ $# -eq 1 ]; then INSTANCE=$1; else HOST=$1; PORT=$2; fi
mkdir -p "$BACKUP_DIR/snapshots"
LOG=$BACKUP_DIR/backup.log

say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
alert() {
    say "UYARI: $*"
    command -v notify-send >/dev/null && notify-send -u critical "lr-v2 yedek" "$*" 2>/dev/null || true
}

# vast API: instance'ın SSH adresi ve durumu. Çıktı: "<host> <port> <durum>"
instance_info() {
    local key
    key=$(tr -d '[:space:]' < ~/.config/vastai/vast_api_key 2>/dev/null) || return 1
    curl -s --max-time 30 -H "Authorization: Bearer $key" "https://console.vast.ai/api/v0/instances/$INSTANCE/" \
        | python3 -c '
import json, sys
d = json.load(sys.stdin)
i = d.get("instances") or {}
if isinstance(i, list):
    i = i[0] if i else {}
print(i.get("ssh_host") or "-", i.get("ssh_port") or "-",
      (i.get("actual_status") or "?") + "/" + (i.get("intended_status") or "?"),
      (i.get("status_msg") or "").replace("\n", " ")[:120])'
}

one_round() {
    local status=""
    if [ -n "$INSTANCE" ]; then
        read -r HOST PORT status rest < <(instance_info) || true
        if [ -z "${HOST:-}" ] || [ "$HOST" = "-" ]; then
            alert "instance $INSTANCE için adres alınamadı (durum: ${status:-?} ${rest:-})"; return 1
        fi
    fi
    SSH=("${SSH_BIN:-ssh}" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30
         -o StrictHostKeyChecking=accept-new)
    local proj
    proj=$("${SSH[@]}" "root@$HOST" 'for d in /workspace /root; do [ -d $d/tr-lipreading/exp ] && { echo $d/tr-lipreading; break; }; done; true' 2>>"$LOG")
    if [ $? -ne 0 ]; then
        alert "sunucuya bağlanılamadı ($HOST:$PORT${status:+, vast durumu: $status ${rest:-}}). Spot makine durdurulmuş olabilir."
        return 1
    fi
    if [ -z "$proj" ]; then say "sunucuda henüz exp/ yok (eğitim başlamadı)"; return 0; fi

    local inc=$BACKUP_DIR/incoming
    local link=()
    [ -d "$BACKUP_DIR/current/" ] && link=(--link-dest="$(readlink -f "$BACKUP_DIR/current")")
    mkdir -p "$inc"
    say "çekiliyor: $HOST:$proj/{exp,logs}"
    local started=$SECONDS
    if ! rsync -a -W --delete --partial-dir=.rsync-partial "${link[@]}" -e "${SSH[*]}" \
            "root@$HOST:$proj/exp" ":$proj/logs" "$inc/" 2>>"$LOG"; then
        alert "rsync yarıda kaldı; sonraki turda kaldığı yerden sürecek"; return 1
    fi

    # Doğrulama: her checkpoint açılabilir bir zip (torch.save biçimi) ve last.ckpt geri gitmemiş
    local exp
    exp=$(cat "$inc/logs/last_exp" 2>/dev/null)
    if [ -z "$exp" ] || [ ! -f "$inc/exp/$exp/checkpoints/last.ckpt" ]; then
        say "sunucuda henüz last.ckpt yok (${exp:-deney yok}); yedek değişmedi"; return 0
    fi
    if ! python3 - "$inc" <<'EOF'
import glob, sys, zipfile
bad = []
for path in glob.glob(f"{sys.argv[1]}/exp/*/checkpoints/*.ckpt"):
    try:
        with zipfile.ZipFile(path) as z:
            if not any(n.endswith("data.pkl") for n in z.namelist()):
                bad.append(path)
    except Exception:
        bad.append(path)
for path in bad:
    print("bozuk:", path)
sys.exit(1 if bad else 0)
EOF
    then
        alert "çekilen checkpoint'lerden biri bozuk; yedek değişmedi"; return 1
    fi
    local cur=$BACKUP_DIR/current/exp/$exp/checkpoints/last.ckpt
    if [ -f "$cur" ] && [ "$cur" -nt "$inc/exp/$exp/checkpoints/last.ckpt" ]; then
        alert "sunucudaki last.ckpt yedektekinden eski ($exp). Yeni bir sunucu sıfırdan mı başladı? Yedek değişmedi"
        return 1
    fi

    local snap
    snap=$BACKUP_DIR/snapshots/$(date +%Y%m%d-%H%M%S)
    mv "$inc" "$snap" && ln -sfn "$snap" "$BACKUP_DIR/current"
    ls -1d "$BACKUP_DIR"/snapshots/* | head -n -"$KEEP" | xargs -r rm -rf
    local when cer
    when=$(date -r "$snap/exp/$exp/checkpoints/last.ckpt" '+%F %T')
    cer=$(python3 - "$snap/exp/$exp" <<'EOF'
import csv, glob, sys
rows = []
for p in sorted(glob.glob(f"{sys.argv[1]}/csv/metrics*.csv")):
    rows += list(csv.DictReader(open(p)))
val = [r for r in rows if r.get("val/cer")]
print(f"epoch {val[-1]['epoch']}, val/cer {float(val[-1]['val/cer']):.3f}" if val else "henüz val/cer yok")
EOF
)
    say "TAMAM ($((SECONDS - started)) sn): $exp, last.ckpt $when, $cer -> $snap (yedeklerin diskte toplamı $(du -sh "$BACKUP_DIR/snapshots" | cut -f1))"
}

while true; do
    one_round
    [ -n "$ONCE" ] && break
    sleep $((INTERVAL * 60))
done
