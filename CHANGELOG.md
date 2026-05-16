# Changelog

All notable changes to this repository are documented here.

This fork extends [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs)
with hardware instrumentation tooling for DGX Spark (GB10, SM121).

---

## [Unreleased] — parallelArchitect fork

### Added

- `benchmarks/benchmark_ab_instrumented.py` — instrumented A/B benchmark with
  continuous NVML hardware telemetry during MoE kernel config comparison.
  Captures clock, power, temperature, and throttle reasons throughout Test A
  (SM80 fallback kernels) and Test B (SM121-tuned configs). Reports thermal
  trajectory, clock stability (coefficient of variation), power delta, and
  per-round hardware state. Anchors results against `uma_bw_results.json`
  from [nvidia-uma-fault-probe](https://github.com/parallelArchitect/nvidia-uma-fault-probe)
  kernel sweep baseline when present. Supports `--save-baseline`, `--compare`,
  `--report`, `--config-dir`, and `--baseline` flags. `NVMLDirect` telemetry
  layer ported from
  [spark-gpu-throttle-check v2.1.0](https://github.com/parallelArchitect/spark-gpu-throttle-check).

---

## [1.0.0] — 2026-02-07 — baristankut original

### Added

- SM121-tuned MoE kernel configs for Triton 3.3.0 and 3.5.0 targeting
  GB10's 101,376-byte shared memory constraint. Default SGLang MoE configs
  request 147,456 bytes and fail with `OutOfResources` on GB10.
- GLM-4.7 tool call parser backport (`glm47`) for SGLang v0.5.4.
- Pre-configured Dockerfile (`lmsysorg/sglang:spark` base) with configs
  and patches applied at build time.
- `benchmarks/benchmark_ab.py` — A/B throughput comparison: SM80 fallback
  (15.77 tok/s) vs SM121-tuned configs (16.77 tok/s, +6.3%). EAGLE
  speculative decoding requires tuned configs — crashes at startup without
  them (`OutOfResources`). With tuned configs + EAGLE: 20–27 tok/s.
- `benchmarks/benchmark_context_vs_speed.py` — context length vs decode
  speed sweep validating flash3 bandwidth formula (TP=4, 273 GB/s per node).
- `benchmarks/benchmark_eagle_efficiency.py` — EAGLE ON vs OFF comparison
  across context lengths.
- `benchmarks/benchmark_agentic_workflow.py` — multi-turn tool calling
  benchmark.
- `benchmarks/benchmark_thinking_mode.py` — thinking mode impact on
  throughput.
- `MULTI_NODE_SETUP.md` — 4-node cluster guide (200 Gbps RoCE/RDMA,
  MikroTik CRS812 DDQ).
- `TUNING.md` — MoE kernel tuning guide for other models on GB10.
- `RESULTS.md` — full benchmark results, context vs speed analysis,
  agentic workflow data, EAGLE efficiency comparison.

### Environment

- Hardware: 4× DGX Spark (GB10, 128 GB each)
- Network: 200 Gbps RoCE/RDMA dedicated fabric
- Container: `lmsysorg/sglang:spark` (v0.5.4.post2)
- Model: `zai-org/GLM-4.7-FP8` (355B MoE, 32B active)
- CUDA 13.1, CUTLASS 4.4.0

### References

- [CUTLASS issue #2800](https://github.com/NVIDIA/cutlass/issues/2800) —
  `sm_121a` missing from `BlockScaledMmaOp.admissible_archs` in Python DSL.
  Manual patch required; upstream fix pending.
- [NVIDIA Developer Forum thread](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)
