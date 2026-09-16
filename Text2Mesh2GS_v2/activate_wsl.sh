#!/usr/bin/env bash
# Source this script from a WSL bash shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source activate_wsl.sh" >&2
    exit 1
fi
T2M_ENV_DIR="${T2M_ENV_DIR:-$HOME/.venvs/text2mesh2gs-v2}"
if [[ ! -f "$T2M_ENV_DIR/bin/activate" ]]; then
    echo "Environment not found: $T2M_ENV_DIR. Run setup_wsl.sh first." >&2
    return 1
fi
source "$T2M_ENV_DIR/bin/activate"
export CUDA_HOME="${T2M_CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ -x "$HOME/software/blender-5.1.2-linux-x64/blender" ]]; then
    export PATH="$HOME/software/blender-5.1.2-linux-x64:$PATH"
fi
echo "Python: $(command -v python)"
