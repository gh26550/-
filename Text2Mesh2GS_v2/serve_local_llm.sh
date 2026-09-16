#!/usr/bin/env bash
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export OLLAMA_MODELS="${T2M_MODELS_DIR:-$HOME/.local/share/text2mesh2gs/models}"
export OLLAMA_NUM_PARALLEL=1
exec "$HOME/.local/share/text2mesh2gs/ollama/bin/ollama" serve
