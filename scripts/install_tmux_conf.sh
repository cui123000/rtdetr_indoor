#!/usr/bin/env bash
set -euo pipefail

# Install repository tmux.conf to user's home and optionally reload tmux
# Usage: bash scripts/install_tmux_conf.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/tmux.conf"
DST="$HOME/.tmux.conf"

if [ ! -f "$SRC" ]; then
  echo "ERROR: source tmux.conf not found at $SRC"
  exit 1
fi

cp "$SRC" "$DST"
echo "Copied $SRC -> $DST"

# If inside tmux, reload the config for current session
if [ -n "${TMUX:-}" ]; then
  if command -v tmux >/dev/null 2>&1; then
    tmux source-file "$DST" && echo "Reloaded tmux config in current session"
  else
    echo "tmux command not found; configuration installed but not reloaded"
  fi
else
  echo "Configuration installed. To apply it, either start a tmux session or run:"
  echo "  tmux source-file $DST"
fi

exit 0
