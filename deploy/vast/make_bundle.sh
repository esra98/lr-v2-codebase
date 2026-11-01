#!/usr/bin/env bash
# Laptopta çalışır: sunucuya yüklenecek lr-v2-vast.tar dosyasını yapar.
#
#   bash deploy/vast/make_bundle.sh <yapım klasörü> <çıktı tar>
#
# Yapım klasöründe önceden hazırlanmış olmalı (README'deki "paketin yeniden
# yapılması" bölümü): bin/python-3.12.tar.gz, bin/gdown-wheels/. Büyük dosyalar
# (dataset, wheelhouse, pretrained) pakete girmez; Drive'dan indirilir.
set -euo pipefail

BUILD=$1
OUT=$2
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "Repoda commit edilmemiş değişiklik var; paket HEAD'den yapılır, önce commit et." >&2; exit 1
fi

STAGE=$BUILD/lr-v2-vast
rm -rf "$STAGE" && mkdir -p "$STAGE/bin"
cp deploy/vast/{setup.sh,check.sh,train.sh,finish.sh,drive.env,DRIVE_FILES.txt,SHA256SUMS.drive,requirements-lock.txt} "$STAGE/"
cp deploy/vast/README.md "$STAGE/KURULUM.md"
cp deploy/vast/MODEL_KULLANIMI.md deploy/vast/REMOTE_RUN.md "$STAGE/"
cp "$BUILD/bin/python-3.12.tar.gz" "$STAGE/bin/" && cp -r "$BUILD/bin/gdown-wheels" "$STAGE/bin/"
cp -r assets "$STAGE/assets"
git archive --format=tar HEAD > "$STAGE/code.tar"
git rev-parse HEAD > "$STAGE/COMMIT"
(cd "$STAGE" && find . -type f ! -name SHA256SUMS.bundle -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.bundle)

tar -cf "$OUT" -C "$BUILD" lr-v2-vast
sha256sum "$OUT" | sed "s|  .*/|  |" > "$OUT.sha256"
ls -lh "$OUT"
cat "$OUT.sha256"
