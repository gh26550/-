#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${T2M_ENV_DIR:-$HOME/.venvs/text2mesh2gs-v2}"
PYTHON_BIN="${T2M_PYTHON:-python3}"
if [[ ! -f "$ENV_DIR/pyvenv.cfg" ]]; then
    if [[ -e "$ENV_DIR" ]]; then
        echo "Refusing to overwrite existing non-venv directory: $ENV_DIR" >&2
        exit 1
    fi
    "$PYTHON_BIN" -m venv --copies "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$ENV_DIR/bin/python" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu128
REQUIREMENTS="$PROJECT_DIR/requirements-runtime.txt"
if [[ -f "$PROJECT_DIR/requirements-runtime-lock.txt" ]]; then
    REQUIREMENTS="$PROJECT_DIR/requirements-runtime-lock.txt"
fi
"$ENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS"
"$ENV_DIR/bin/python" -m pip check
echo "Created environment: $ENV_DIR"
echo "Activate using: source \"$PROJECT_DIR/activate_wsl.sh\""
