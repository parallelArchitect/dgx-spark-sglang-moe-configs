#!/usr/bin/env bash
# setup_env.sh — SGLang bare metal environment for DGX Spark (GB10, SM121)
# Target: DGX Spark (GB10, SM121), CUDA 13.0, driver 580.142, aarch64
# Run once from repo root. Then use run.sh to execute the benchmark.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$REPO_DIR/.sglang_env"

echo "=== DGX Spark SGLang Environment Setup ==="
echo "  Repo: $REPO_DIR"
echo "  Env:  $ENV_DIR"
echo

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Ensure CUDA 13.0 is installed."
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "  Creating Python 3.10 venv..."
uv venv "$ENV_DIR" --python 3.10
source "$ENV_DIR/bin/activate"

export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

echo "  Installing PyTorch cu130..."
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130

echo "  Installing system dependencies..."
sudo apt-get install -y libnuma-dev libibverbs-dev python3-dev

echo "  Installing SGLang..."
uv pip install build wheel "cmake<4.0" ninja scikit-build-core
uv pip install "sglang[all]" --prerelease=allow --index-url https://docs.sglang.ai/whl/cu130

echo
echo "=== Setup complete. Run: ./run.sh ==="
