#!/usr/bin/env bash
# Kurulum: boş bir Ubuntu vast.ai sunucusunda eğitime hazır proje.
#
#   tar -xf lr-v2-vast.tar && cd lr-v2-vast && bash setup.sh
#
# Yaptıkları (her adım bitince işaretlenir; tekrar çalıştırınca bitenler atlanır):
#   1. GPU, disk ve rclone ayarı kontrolü
#   2. Drive'dan dataset, wheelhouse ve pretrained ağırlıkların indirilmesi + sha256 kontrolü
#   3. Kodun, dataset'in ve ağırlıkların proje klasörüne açılması
#   4. Python 3.12 ve kütüphanelerin internetsiz kurulumu (wheelhouse)
# İnternet yalnızca Google Drive için gerekir.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/drive.env"

WORK=${WORK:-/workspace}
if ! mkdir -p "$WORK" 2>/dev/null || [ ! -w "$WORK" ]; then WORK=$HOME; fi
PROJ=$WORK/lr-v2-codebase
DL=$WORK/downloads
DONE=$WORK/.setup-done
mkdir -p "$DL" "$DONE" "$WORK/bin"
export PATH="$WORK/bin:$PATH"

step() { echo; echo "=== $*"; }
done_before() { [ -f "$DONE/$1" ]; }
mark() { touch "$DONE/$1"; }

# ---------------------------------------------------------------- 1. kontrol
step "1/4 Kontrol"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi çalışmıyor: sunucuda GPU görünmüyor." >&2; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
driver_cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)
echo "Sürücünün desteklediği CUDA: ${driver_cuda:-?} (PyTorch 2.13 cu129 için 12.x gerekir; 12.8+ önerilir)"
case "$driver_cuda" in 11.*|10.*) echo "Sürücü çok eski. vast.ai'de 'Max CUDA >= 12.8' filtresiyle sunucu seç." >&2; exit 1;; esac

free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
need_gb=60
done_before extract && need_gb=20
echo "Boş disk ($WORK): ${free_gb} GB"
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "En az ${need_gb} GB boş disk gerekir (dataset 13+14 GB, kütüphaneler 11 GB, checkpoint'ler ~15 GB)." >&2; exit 1
fi

install -m 755 "$HERE/bin/rclone" "$WORK/bin/rclone"
if [ -f "$HERE/rclone.conf" ]; then
    mkdir -p ~/.config/rclone && install -m 600 "$HERE/rclone.conf" ~/.config/rclone/rclone.conf
fi
if ! rclone listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
    cat >&2 <<EOF
rclone'da '${RCLONE_REMOTE}:' adında bir Drive bağlantısı yok.
Laptoptan rclone ayarını sunucuya kopyala (README.md, adım 2):
    scp -P <port> ~/.config/rclone/rclone.conf root@<sunucu>:$HERE/rclone.conf
sonra 'bash setup.sh' komutunu tekrar çalıştır.
EOF
    exit 1
fi

# ---------------------------------------------------------------- 2. indirme
step "2/4 Drive'dan indirme"
if done_before download; then
    echo "daha önce indirildi"
else
    for f in "$DATASET_TAR" "$WHEELHOUSE_TAR" "$PRETRAINED_PTH"; do
        echo "-> $f"
        rclone copy "${RCLONE_REMOTE}:" "$DL" --bind 0.0.0.0 --drive-root-folder-id "$DRIVE_FOLDER_ID" \
            --include "/$f" --multi-thread-streams 8 --stats 30s --stats-one-line -v 2>&1 | grep -v DEBUG || true
        [ -f "$DL/$f" ] || { echo "$f indirilemedi" >&2; exit 1; }
    done
    echo "sha256 kontrolü (birkaç dakika)..."
    (cd "$DL" && sha256sum -c "$HERE/SHA256SUMS.drive")
    mark download
fi

# ---------------------------------------------------------------- 3. açma
step "3/4 Proje klasörü: $PROJ"
if done_before extract; then
    echo "daha önce açıldı"
else
    mkdir -p "$PROJ"
    tar -xf "$HERE/code.tar" -C "$PROJ"
    cp -r "$HERE/assets" "$PROJ/"
    mkdir -p "$PROJ/pretrained"
    cp "$DL/$PRETRAINED_PTH" "$PROJ/pretrained/"
    echo "dataset açılıyor (5-10 dk)..."
    tar -xzf "$DL/$DATASET_TAR" -C "$PROJ"
    sessions=$(find "$PROJ/dataset/tr-lrs-v3" -mindepth 1 -maxdepth 1 -type d | wc -l)
    files=$(find "$PROJ/dataset/tr-lrs-v3" -type f | wc -l)
    echo "oturum: $sessions (beklenen $EXPECT_SESSIONS), dosya: $files (beklenen $EXPECT_FILES)"
    if [ "$sessions" != "$EXPECT_SESSIONS" ] || [ "$files" != "$EXPECT_FILES" ]; then
        echo "Dataset eksik açıldı." >&2; exit 1
    fi
    rm -f "$DL/$DATASET_TAR"   # 13 GB yer açar; sha256 zaten doğrulandı
    mark extract
fi

# ---------------------------------------------------------------- 4. python
step "4/4 Python 3.12 ve kütüphaneler"
if done_before python; then
    echo "daha önce kuruldu"
else
    rm -rf "$WORK/python" && tar -xzf "$HERE/bin/python-3.12.tar.gz" -C "$WORK"
    mkdir -p "$DL/wheelhouse" && tar -xf "$DL/$WHEELHOUSE_TAR" -C "$DL/wheelhouse"
    rm -rf "$PROJ/.venv"
    "$WORK/python/bin/python3" -m venv "$PROJ/.venv"
    "$PROJ/.venv/bin/pip" install -q --no-index --find-links "$DL/wheelhouse" -r "$HERE/requirements-lock.txt"
    rm -rf "$DL/wheelhouse"
    mark python
fi
"$PROJ/.venv/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch GPU'yu göremiyor (sürücü/CUDA uyumsuz)"
x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize(); (x @ x).sum().item()
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
EOF

echo
echo "KURULUM TAMAM. Sıradaki: bash $HERE/check.sh"
echo "WORK=$WORK" > "$HERE/.work"
