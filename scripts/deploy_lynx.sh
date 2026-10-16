#!/usr/bin/env bash
# Sync this project to the lynx training box (tar over ssh; lynx has no git).
set -euo pipefail
HOST="${XWJEV_TRAIN_HOST:-lynx}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
tar --exclude=.venv --exclude=.git --exclude=assets --exclude=__pycache__ --exclude='*.egg-info' --exclude=puzzles -czf - . | ssh "$HOST" 'mkdir -p ~/crossword-solver && cd ~/crossword-solver && tar -xzf -'
ssh "$HOST" 'cd ~/crossword-solver && source .venv/bin/activate && ~/.local/bin/uv pip install -q -e . >/dev/null 2>&1; echo deployed'
