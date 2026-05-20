# Running on GB10 Without Docker
Target: DGX Spark / ASUS GX10 (GB10, SM121), driver 580.142, CUDA 13.0, SGLang bare metal.

---

## Step 1 — Clone

    git clone https://github.com/parallelArchitect/dgx-spark-sglang-moe-configs.git
    cd dgx-spark-sglang-moe-configs

## Step 2 — Requirements

SGLang must be installed in your Python environment. If you run models with SGLang already, you are ready. If not:

    python3 -m venv ~/gb10_env
    source ~/gb10_env/bin/activate
    pip install sglang

## Step 3 — Run

From the same environment you use to run your models:

    python3 benchmarks/benchmark_ab_instrumented.py --report

The script auto-discovers your Triton config directory. If it cannot find it
automatically, it will search your system and prompt you to confirm the path.
Once confirmed the benchmark runs automatically.

Report saves to ~/.spark-moe-benchmark/reports/.
