#!/usr/bin/env bash
# Re-download the YouTube sessions at up to 1080p into videos/_incoming/.
# Resumable: yt-dlp continues partial files and skips ids in the archive.
# When a file is complete and ffprobe confirms 1080 rows, it replaces the
# 360p file of the same name in videos/. Run from the repository root.
set -u
cd "$(dirname "$0")/.." || exit 1
IN=videos/_incoming
# Needs a current yt-dlp (.venv/bin/pip install -U yt-dlp) and a JavaScript
# runtime (node) to solve the YouTube URL challenge; without it every
# download beyond the first bytes is refused with HTTP 403.
YTDLP=${YTDLP:-.venv/bin/yt-dlp}
mkdir -p "$IN"
ids=$(awk -F'\t' 'NR>1 && $2=="youtube" && $5<1080 {print $1}' videos/sources.tsv | sed 's/^yt-//')
[ -z "$ids" ] && { echo "nothing to download"; exit 0; }
for id in $ids; do
  out="$IN/yt-$id.mp4"
  # A 1080p session is up to 6 GB and is merged from two parts, so keep headroom.
  while [ "$(df --output=avail -BG videos | tail -1 | tr -dc 0-9)" -lt 15 ]; do
    echo "WAITING: under 15 GB free, sleeping 5 min"; sleep 300
  done
  if [ ! -s "$out" ]; then
    "$YTDLP" --js-runtimes node --continue --no-overwrites --no-playlist --retries 10 --fragment-retries 10 \
      -f 'bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][ext=mp4]+ba/b[height<=1080]' --merge-output-format mp4 \
      --download-archive "$IN/archive.txt" -o "$IN/yt-%(id)s.%(ext)s" \
      "https://www.youtube.com/watch?v=$id" || { echo "FAILED $id"; continue; }
  fi
  h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$out" 2>/dev/null)
  if [ "$h" = "1080" ] || [ "$h" = "720" ]; then
    mv -f "$out" "videos/yt-$id.mp4" && echo "DONE $id ${h}p"
  else
    echo "UNEXPECTED height '$h' for $id, kept in $IN"
  fi
done
echo "ALL DONE"
