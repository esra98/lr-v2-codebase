#!/usr/bin/env bash
# Kurulum: boş bir Ubuntu vast.ai sunucusunda eğitime hazır proje.
#
#   tar -xf lr-v2-vast.tar && cd lr-v2-vast && bash setup.sh
#
# Yaptıkları (her adım bitince işaretlenir; tekrar çalıştırınca bitenler atlanır):
#   1. GPU ve disk kontrolü
#   2. Taşınabilir Python 3.12 ve gdown (paketin içinden, internetsiz)
#   3. Drive'dan dataset, wheelhouse parçaları ve pretrained ağırlıkların
#      gdown ile indirilmesi (DRIVE_FILES.txt) + sha256 kontrolü
#   4. Kodun, dataset'in ve ağırlıkların proje klasörüne açılması
#   5. Kütüphanelerin wheelhouse'tan internetsiz kurulumu
# İnternet yalnızca Google Drive için gerekir; Drive klasörü linkle paylaşılmış
# olmalı (giriş yapmadan indirilir).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/drive.env"

WORK=${WORK:-/workspace}
if ! mkdir -p "$WORK" 2>/dev/null || [ ! -w "$WORK" ]; then WORK=$HOME; fi
PROJ=$WORK/tr-lipreading
DL=$WORK/downloads
DONE=$WORK/.setup-done
mkdir -p "$DL" "$DONE"
echo "WORK=$WORK" > "$HERE/.work"

step() { echo; echo "=== $*"; }
done_before() { [ -f "$DONE/$1" ]; }
mark() { touch "$DONE/$1"; }

# ---------------------------------------------------------------- 1. kontrol
step "1/5 Kontrol"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi çalışmıyor: sunucuda GPU görünmüyor." >&2; exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
driver_cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)
echo "Sürücünün desteklediği CUDA: ${driver_cuda:-?} (PyTorch 2.13 cu129 için 12.x gerekir; 12.8+ önerilir)"
case "$driver_cuda" in 11.*|10.*) echo "Sürücü çok eski. vast.ai'de 'Max CUDA >= 12.8' filtresiyle sunucu seç." >&2; exit 1;; esac

free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
need_gb=${NEED_GB:-60}   # boş disk sınırı (GB)
done_before extract && need_gb=20
echo "Boş disk ($WORK): ${free_gb} GB"
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "En az ${need_gb} GB boş disk gerekir (dataset 13+14 GB, kütüphaneler 11 GB, checkpoint'ler ~15 GB)." >&2; exit 1
fi

# ---------------------------------------------------------------- 2. python + gdown
step "2/5 Python 3.12 ve gdown"
if done_before tools; then
    echo "daha önce kuruldu"
else
    rm -rf "$WORK/python" "$WORK/tools"
    tar -xzf "$HERE/bin/python-3.12.tar.gz" -C "$WORK"
    "$WORK/python/bin/python3" -m venv "$WORK/tools"
    "$WORK/tools/bin/pip" install -q --no-index --find-links "$HERE/bin/gdown-wheels" gdown
    mark tools
fi
GDOWN=$WORK/tools/bin/gdown
"$GDOWN" --version

# ---------------------------------------------------------------- 3. indirme
step "3/5 Drive'dan indirme (gdown)"
if done_before download; then
    echo "daha önce indirildi"
else
    # DRIVE_FILES.txt: "<drive id>  <dosya adı>" satırları
    while read -r id name; do
        case "$id" in ''|'#'*) continue;; esac
        echo "-> $name"
        if ! (cd "$DL" && "$GDOWN" "$id" -O "$name" --continue --retries 5 --no-cookies); then
            echo "$name indirilemedi; 'bash setup.sh' kaldığı yerden sürer (KURULUM.md, 7)." >&2; exit 1
        fi
    done < "$HERE/DRIVE_FILES.txt"
    echo "wheelhouse parçaları birleştiriliyor..."
    cat "$DL/$WHEELHOUSE_TAR".part-* > "$DL/$WHEELHOUSE_TAR"
    echo "sha256 kontrolü (birkaç dakika)..."
    (cd "$DL" && sha256sum -c "$HERE/SHA256SUMS.drive")
    rm -f "$DL/$WHEELHOUSE_TAR".part-*
    mark download
fi

# ---------------------------------------------------------------- 4. açma
step "4/5 Proje klasörü: $PROJ"
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

# ---------------------------------------------------------------- 5. kütüphaneler
step "5/5 Kütüphaneler (wheelhouse, internetsiz)"
if done_before python; then
    echo "daha önce kuruldu"
else
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
