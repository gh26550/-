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
# Ignore architecture/cache settings inherited from another Conda environment.
# Explicit project overrides are available through the T2M_* variables.
_t2m_runtime=$(python -c 'import sys, torch; caps=sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}); print(";".join(f"{a}.{b}" for a,b in caps) or "none"); print(f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch.__version__}_cuda{torch.version.cuda}")') || return 1
_t2m_arch="${_t2m_runtime%%$'\n'*}"
_t2m_tag="${_t2m_runtime#*$'\n'}"
if [[ "$_t2m_arch" == "none" ]]; then
    echo "No CUDA GPU detected. GPU generation requires a visible NVIDIA GPU." >&2
    return 1
fi
export TORCH_CUDA_ARCH_LIST="${T2M_CUDA_ARCH_LIST:-$_t2m_arch}"
export TORCH_EXTENSIONS_DIR="${T2M_EXTENSIONS_DIR:-$T2M_ENV_DIR/torch_extensions/$_t2m_tag/arch_${TORCH_CUDA_ARCH_LIST//;/_}}"
unset _t2m_runtime _t2m_arch _t2m_tag
if [[ -x "$HOME/software/blender-5.1.2-linux-x64/blender" ]]; then
    export PATH="$HOME/software/blender-5.1.2-linux-x64:$PATH"
fi
echo "Python: $(command -v python)"
echo "CUDA architecture: $TORCH_CUDA_ARCH_LIST"
echo "CUDA extension cache: $TORCH_EXTENSIONS_DIR"
