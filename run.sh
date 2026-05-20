#!/usr/bin/env bash
# run.sh — run the benchmark from the SGLang bare metal environment

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$REPO_DIR/.sglang_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "ERROR: Environment not found. Run ./setup_env.sh first."
    exit 1
fi

source "$ENV_DIR/bin/activate"

export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

exec python3 "$REPO_DIR/benchmarks/benchmark_ab_instrumented.py" "$@"
