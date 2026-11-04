#!/bin/bash
# Start the chord annotator at http://127.0.0.1:8765
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # ffmpeg + deno (YouTube JS runtime for yt-dlp)

for cmd in ffmpeg deno; do
  command -v "$cmd" >/dev/null || { echo "'$cmd' belum terinstall. Mac: brew install python@3.12 ffmpeg deno"; exit 1; }
done

if [ ! -d .venv ]; then
  # yt-dlp needs Python >= 3.10 for current YouTube support.
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || { echo "Butuh Python 3.10+. Mac: brew install python@3.12"; exit 1; }
  "$PY" -m venv .venv && .venv/bin/pip install -q -r requirements.txt || exit 1
fi
exec .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765
