#!/bin/bash
# setup.sh — Environment setup using uv for Stage 1 Dataset Curation
#
# Usage:
#   bash setup.sh

set -euo pipefail

log() { echo -e "\033[1;32m[setup.sh]\033[0m $1"; }
err() { echo -e "\033[1;31m[setup.sh] ERROR:\033[0m $1" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check and install uv
if ! command -v uv &>/dev/null; then
    log "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
log "uv version: $(uv --version)"

# 2. Create virtual environment with uv
if [ ! -d ".venv" ]; then
    log "Creating virtual environment via uv (.venv/)..."
    uv venv --python 3.12
fi

# 3. Install dependencies via uv
log "Installing dependencies using uv sync..."
uv sync

log "Stage 1 Dataset Curation environment is ready!"
log "Activate with: source .venv/bin/activate (or run directly via: uv run python cli.py)"
