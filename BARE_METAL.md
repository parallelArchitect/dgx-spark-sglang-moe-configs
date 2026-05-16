# Running on GB10 Without Docker

This guide covers running `benchmark_ab_instrumented.py` directly on a
DGX Spark or ASUS GX10 (GB10, SM121) without Docker.

Tested configuration:
- Hardware: DGX Spark / ASUS GX10 (GB10, 128 GB LPDDR5X)
- Driver: 580.142
- CUDA: 13.0
- SGLang: `lmsysorg/sglang:spark` base or bare metal install

---

## Prerequisites

### 1. SGLang or vLLM installed

The benchmark requires either SGLang or vLLM installed in your Python
environment. The script auto-discovers the Triton MoE config directory
from the installed package.

**SGLang (recommended for GB10):**
```bash
pip install sglang
# or from the spark container base:
# lmsysorg/sglang:spark already has SM121-compatible Triton
```

**vLLM:**
```bash
pip install vllm
```

### 2. SM121-tuned MoE configs

The configs in this repo fix GB10's 101,376-byte shared memory constraint.
Without them, SGLang's default MoE kernels request 147,456 bytes and crash
with `OutOfResources`. EAGLE speculative decoding requires these configs.

Locate your Triton config directory:
```bash
# SGLang
find / -path "*/fused_moe_triton/configs" 2>/dev/null

# vLLM
find / -path "*/fused_moe/configs" 2>/dev/null
```

Copy the configs for your Triton version:
```bash
# Triton 3.5.0 (most common on GB10 bare metal)
cp configs/triton_3_5_0/* <your_triton_config_dir>/

# Triton 3.3.0
cp configs/triton_3_3_0/* <your_triton_config_dir>/
```

Check which Triton version you have:
```bash
python3 -c "import triton; print(triton.__version__)"
```

### 3. Model weights

The benchmark uses `zai-org/GLM-4.7-FP8` (355B MoE, 32B active params).
Weights must be cached locally before running:

```bash
huggingface-cli download zai-org/GLM-4.7-FP8
```

Approximately 32 GB download. Requires a Hugging Face account with access.

### 4. nvidia-uma-fault-probe baseline (optional but recommended)

The instrumented benchmark anchors results against a kernel sweep baseline
from [nvidia-uma-fault-probe](https://github.com/parallelArchitect/nvidia-uma-fault-probe).
This confirms LPDDR5X bandwidth health before the inference run.

If you have already run `nvidia-uma-fault-probe`, the baseline file is at:
```
~/opt_cuda/src/power_correlation/nvidia-uma-fault-probe/uma_bw_results.json
```

The benchmark auto-discovers this path. To specify manually:
```bash
python3 benchmarks/benchmark_ab_instrumented.py \
  --baseline ~/path/to/uma_bw_results.json
```

---

## Running the Benchmark

### Full A/B comparison (recommended)

Runs Test A (SM80 fallback) then Test B (SM121-tuned configs).
30-second thermal cooldown between conditions.

```bash
cd dgx-spark-sglang-moe-configs
python3 benchmarks/benchmark_ab_instrumented.py --report --save-baseline
```

### Test B only (SM121-tuned configs)

If you only want to confirm your tuned config performance with hardware
telemetry:

```bash
python3 benchmarks/benchmark_ab_instrumented.py --mode with --report
```

### Test A only (SM80 fallback baseline)

```bash
python3 benchmarks/benchmark_ab_instrumented.py --mode without --report
```

### With explicit paths

If auto-discovery doesn't find your Triton config dir:

```bash
python3 benchmarks/benchmark_ab_instrumented.py \
  --config-dir /path/to/fused_moe_triton/configs \
  --baseline ~/uma_bw_results.json \
  --report \
  --save-baseline
```

### Compare against saved baseline

After a SGLang or driver update, check for regression:

```bash
python3 benchmarks/benchmark_ab_instrumented.py --mode with --compare
```

---

## Output

The benchmark prints hardware telemetry per condition:

```
Hardware (Test B):
  Clock:  peak=2405 MHz  avg=2403 MHz  stability=0.04% CV
  Power:  avg=138.7 W    peak=144.2 W
  Temp:   avg=57.1°C     peak=61°C
  Thermal trajectory: stable (+0.01 °C/s)
```

And a summary delta:

```
SUMMARY
  Throughput:
    Test A (SM80 fallback):  15.8 tok/s
    Test B (SM121-tuned):    16.8 tok/s
    Delta:                   +1.0 tok/s (+6.3%)

  Hardware delta (A → B):
    Clock peak (MHz)    A=2405  B=2405  +0
    Power avg (W)       A=142.3 B=138.7 -3.6
    Temp peak (°C)      A=62    B=61    -1
```

Full JSON report saved to `~/.spark-moe-benchmark/reports/`.

---

## What the Hardware Telemetry Shows

**Clock locked at 2405 MHz both conditions** — throughput difference is
kernel efficiency, not clock state. SM121-tuned configs do more work per
cycle on GB10's native Tensor Cores.

**Power drops with tuned configs** — SM121 kernels are more efficient per
token. SM80 fallback burns more power for lower throughput.

**Thermal trajectory** — shows whether the platform is thermally constrained
during the run. A rising slope indicates the platform is approaching a
thermal ceiling; stable means headroom exists.

**Kernel sweep baseline anchor** — confirms LPDDR5X bandwidth is healthy
before the inference run starts. Expected idle: GPU read ~165 GB/s,
GPU write ~115 GB/s on GB10.

---

## Sharing Results

If you run this benchmark, please share your results. Useful data:

- Full JSON report from `~/.spark-moe-benchmark/reports/`
- Your Triton version (`python3 -c "import triton; print(triton.__version__)"`)
- SGLang or vLLM version
- Driver version (`nvidia-smi --query-gpu=driver_version --format=csv,noheader`)
- Platform (DGX Spark 128GB / ASUS GX10 / other)

Results can be shared in the
[GB10 Hardware Baseline thread](https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851)
or as a GitHub issue in this repo.

---

## Known GB10 Platform Constraints

- **101,376 bytes shared memory** — SM121 hardware limit. Default SGLang
  MoE configs request 147,456 bytes and crash. Configs in this repo fix this.
- **EAGLE requires tuned configs** — without them, EAGLE crashes at server
  startup with `OutOfResources`. With tuned configs: 20–27 tok/s.
- **CUDA 13.0 recommended** — CUDA 13.1 has a known event timing bug on GB10.
- **CUPTI UVM profiling unavailable** — `CUPTI_ERROR_NOT_READY` on GB10
  with CUDA 13.0, driver 580.142. Known driver limitation.
- **`nvidia-smi clocks.mem` returns N/A** — NVML memory clock not exposed
  by driver on GB10. Use `/proc/meminfo` for memory state.

---

## References

- [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — original configs and benchmarks
- [parallelArchitect/nvidia-uma-fault-probe](https://github.com/parallelArchitect/nvidia-uma-fault-probe) — kernel sweep baseline tool
- [parallelArchitect/sparkview](https://github.com/parallelArchitect/sparkview) — continuous GB10 system monitor
- [parallelArchitect/spark-gpu-throttle-check](https://github.com/parallelArchitect/spark-gpu-throttle-check) — GPU throttle diagnostic
- [CUTLASS issue #2800](https://github.com/NVIDIA/cutlass/issues/2800) — SM121 Python DSL fix (upstream pending)
- [NVIDIA Developer Forum — GB10 Hardware Baseline](https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851)
