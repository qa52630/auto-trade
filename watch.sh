#!/usr/bin/env bash
# Watch source files; trigger rebuild.sh whenever any tracked file changes.
#
# Requires:
#   macOS:  brew install fswatch
#   Linux:  apt install inotify-tools
#
# Usage:
#   ./watch.sh                  # watches & auto-rebuilds in foreground
#   ./watch.sh nohup &          # run in background
#
# Coalesces rapid bursts of changes (e.g. saving multiple files) into one rebuild
# by waiting WINDOW seconds of quiet before triggering.
set -euo pipefail
cd "$(dirname "$0")"

WINDOW=3   # seconds of quiet before rebuild fires

TRACKED=(
  Dockerfile
  docker-compose.yml
  stocks.json
)
# also track all .py files
PY_FILES=( $(ls *.py 2>/dev/null) )
TRACKED+=( "${PY_FILES[@]}" )

echo "watching: ${TRACKED[*]}"
echo "rebuild on change after ${WINDOW}s of quiet"

trigger_rebuild() {
  echo
  echo "🔄 [$(date '+%H:%M:%S')] change detected → rebuilding..."
  if ./rebuild.sh; then
    echo "✅ [$(date '+%H:%M:%S')] rebuild OK"
  else
    echo "❌ [$(date '+%H:%M:%S')] rebuild FAILED — fix and save again to retry"
  fi
}

# Pick a watcher backend
if command -v fswatch >/dev/null 2>&1; then
  # macOS path
  fswatch -o --event Created --event Updated --event Renamed \
          --latency "$WINDOW" \
          "${TRACKED[@]}" \
    | while read -r _; do trigger_rebuild; done
elif command -v inotifywait >/dev/null 2>&1; then
  # Linux path
  while true; do
    inotifywait -e modify,create,move "${TRACKED[@]}" -qq
    sleep "$WINDOW"   # coalesce burst
    trigger_rebuild
  done
else
  echo "❌ Need fswatch (mac) or inotifywait (linux). Install:"
  echo "    macOS:  brew install fswatch"
  echo "    Linux:  apt install inotify-tools"
  exit 1
fi
