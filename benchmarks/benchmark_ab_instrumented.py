#!/usr/bin/env python3
"""
benchmark_ab_instrumented.py — DGX Spark GB10 (SM121) MoE kernel config
A/B benchmark with continuous NVML hardware telemetry.

Target platform: NVIDIA DGX Spark / ASUS GX10 (GB10, SM121, LPDDR5X UMA)
Driver: 580.142, CUDA 13.0, SGLang :spark container

Test A: Without SM121-tuned MoE configs — GB10 falls back to SM80-class
        kernels. EAGLE speculative decoding will crash (OutOfResources:
        147456 bytes requested, 101376 bytes available on GB10).
Test B: With SM121-tuned MoE configs (baristankut/dgx-spark-sglang-moe-configs)
        Proper tile shapes fit 101KB SMEM budget. EAGLE enabled: 20-27 tok/s.

Measures tok/s per round AND captures thermal, power, clock, and throttle
reasons throughout each condition via NVML direct telemetry. Anchors
results against kernel sweep baseline (uma_bw_results.json from
nvidia-uma-fault-probe) if provided.

Original benchmark_ab.py by baristankut (BTankut/dgx-spark-sglang-moe-configs)
Instrumentation layer by parallelArchitect (parallelArchitect/dgx-spark-sglang-moe-configs)

Usage:
  python3 benchmark_ab_instrumented.py                        # full A/B run
  python3 benchmark_ab_instrumented.py --mode with            # Test B only
  python3 benchmark_ab_instrumented.py --mode without         # Test A only
  python3 benchmark_ab_instrumented.py --save-baseline        # save Test B as baseline
  python3 benchmark_ab_instrumented.py --compare              # compare against baseline
  python3 benchmark_ab_instrumented.py --report               # export full JSON report
  python3 benchmark_ab_instrumented.py --baseline /path/to/uma_bw_results.json
  python3 benchmark_ab_instrumented.py --config-dir /path/to/triton/configs
"""

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
# Target platform: DGX Spark / ASUS GX10 (GB10, SM121, LPDDR5X UMA)
# Driver 580.142, CUDA 13.0, SGLang :spark or vllm-node container

BACKUP_DIR = os.path.expanduser("~/sm121-kernels/gb10_configs_backup/")
MODEL = "zai-org/GLM-4.7-FP8"
BASELINE_DIR = Path.home() / ".spark-moe-benchmark"

# Triton MoE config dir — searched in order, first match wins
# Override with --config-dir
_TRITON_CONFIG_SEARCH = [
    # SGLang :spark container (lmsysorg/sglang:spark)
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs",
    # vLLM container (vllm-node)
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs",
    "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/configs",
    # bare metal miniforge on GB10
    os.path.expanduser("~/miniforge3/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs"),
    os.path.expanduser("~/miniforge3/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs"),
]

# uma_bw_results.json from nvidia-uma-fault-probe kernel sweep
# Override with --baseline
_BASELINE_SEARCH = [
    os.path.expanduser("~/dgx_spark_data/uma_bw_results.json"),
    os.path.expanduser("~/opt_cuda/src/power_correlation/nvidia-uma-fault-probe/uma_bw_results.json"),
    os.path.expanduser("~/uma_bw_results.json"),
]


def find_triton_config_dir(override=None):
    """Auto-discover Triton MoE config dir on GB10.
    If static search fails, runs find command and prompts user to confirm."""
    if override and os.path.isdir(override):
        return override
    for path in _TRITON_CONFIG_SEARCH:
        if os.path.isdir(path):
            return path

    # Static search failed — try find command
    import subprocess
    print(f"  {YELLOW}Triton config dir not found in known paths — searching...{RESET}")
    try:
        result = subprocess.run(
            ["find", "/", "-path", "*/fused_moe_triton/configs", "-type", "d"],
            capture_output=True, text=True, timeout=30
        )
        candidates = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if not candidates:
            return None
        if len(candidates) == 1:
            path = candidates[0]
            print(f"  Found: {path}")
            answer = input("  Use this path? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                return path
            return None
        else:
            print("  Multiple paths found:")
            for i, p in enumerate(candidates):
                print(f"    [{i}] {p}")
            answer = input("  Enter number to use: ").strip()
            try:
                return candidates[int(answer)]
            except (ValueError, IndexError):
                return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def find_kernel_sweep_baseline(override=None):
    """Auto-discover uma_bw_results.json from kernel sweep."""
    if override and os.path.exists(override):
        return override
    for path in _BASELINE_SEARCH:
        if os.path.exists(path):
            return path
    return None

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers efficiently.",
    "What are the key differences between TCP and UDP protocols?",
    "Summarize the history of artificial intelligence in 200 words.",
]

# NVML sample interval during inference
SAMPLE_INTERVAL = 0.5  # seconds

# ─── ANSI ────────────────────────────────────────────────────────────────────

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ─── NVMLDirect (from spark-gpu-throttle-check, parallelArchitect) ───────────

THROTTLE_REASONS = {
    0x0000000000000001: "GPU_IDLE",
    0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000010: "SYNC_BOOST",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
}

PROBLEM_REASONS = {
    0x0000000000000004,
    0x0000000000000008,
    0x0000000000000020,
    0x0000000000000040,
    0x0000000000000080,
}


def decode_throttle_bitmask(bitmask: int) -> list:
    if bitmask == 0:
        return ["NONE"]
    reasons = [name for bit, name in THROTTLE_REASONS.items() if bitmask & bit]
    return reasons if reasons else [f"UNKNOWN(0x{bitmask:016x})"]


def has_problem_throttle(bitmask: int) -> bool:
    return bool(bitmask & sum(PROBLEM_REASONS))


class NVMLDirect:
    """Lightweight NVML wrapper via ctypes. No pynvml dependency.
    Copied from spark-gpu-throttle-check v2.1.0 (parallelArchitect)."""

    def __init__(self):
        self._lib = None
        self._handle = None
        self._available = False
        self._initialized = False

    def _load_lib(self) -> bool:
        if self._initialized:
            return self._lib is not None
        self._initialized = True
        try:
            path = ctypes.util.find_library("nvidia-ml")
            if not path:
                for candidate in [
                    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
                    "/usr/lib64/libnvidia-ml.so.1",
                    "/usr/lib/libnvidia-ml.so.1",
                ]:
                    if os.path.exists(candidate):
                        path = candidate
                        break
            if not path:
                return False
            self._lib = ctypes.CDLL(path)
            return self._lib.nvmlInit_v2() == 0
        except (OSError, AttributeError):
            return False

    def init(self, gpu_index: int = 0) -> bool:
        if not self._load_lib():
            return False
        self._handle = ctypes.c_void_p()
        rc = self._lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(gpu_index), ctypes.byref(self._handle)
        )
        if rc != 0:
            return False
        self._available = True
        return True

    def shutdown(self):
        if self._lib and self._initialized:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._available

    def get_clock_mhz(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetClockInfo(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_power_w(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(val))
        return val.value / 1000.0 if rc == 0 else None

    def get_temperature(self):
        if not self._available:
            return None
        val = ctypes.c_uint()
        rc = self._lib.nvmlDeviceGetTemperature(self._handle, 0, ctypes.byref(val))
        return val.value if rc == 0 else None

    def get_throttle_reasons(self):
        if not self._available:
            return None
        val = ctypes.c_ulonglong()
        rc = self._lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self._handle, ctypes.byref(val)
        )
        return val.value if rc == 0 else None

    def get_gpu_name(self):
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlDeviceGetName(self._handle, buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def get_driver_version(self):
        if not self._available:
            return None
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.nvmlSystemGetDriverVersion(buf, 256)
        return buf.value.decode("utf-8", errors="replace") if rc == 0 else None

    def sample(self) -> dict:
        throttle_raw = self.get_throttle_reasons()
        return {
            "timestamp": time.time(),
            "clk_mhz": self.get_clock_mhz(),
            "power_w": self.get_power_w(),
            "temp_c": self.get_temperature(),
            "throttle_raw": throttle_raw,
            "throttle_reasons": decode_throttle_bitmask(throttle_raw)
            if throttle_raw is not None
            else [],
            "throttle_problem": has_problem_throttle(throttle_raw)
            if throttle_raw is not None
            else False,
        }


# ─── Thermal trajectory (from spark-gpu-throttle-check, parallelArchitect) ──

def compute_thermal_trajectory(samples: list) -> dict:
    temps = [
        (s.get("elapsed", 0), s.get("temp_c"))
        for s in samples
        if s.get("temp_c") is not None
    ]
    if len(temps) < 3:
        return {"slope": 0.0, "direction": "insufficient_data", "stable": True}

    n = len(temps)
    sum_x = sum(t[0] for t in temps)
    sum_y = sum(t[1] for t in temps)
    sum_xy = sum(t[0] * t[1] for t in temps)
    sum_x2 = sum(t[0] ** 2 for t in temps)

    denom = n * sum_x2 - sum_x ** 2
    if denom == 0:
        return {"slope": 0.0, "direction": "flat", "stable": True}

    slope = (n * sum_xy - sum_x * sum_y) / denom

    if slope > 0.1:
        direction = "rising"
    elif slope < -0.1:
        direction = "cooling"
    else:
        direction = "stable"

    return {
        "slope": round(slope, 3),
        "direction": direction,
        "stable": abs(slope) <= 0.1,
        "start_temp": temps[0][1],
        "end_temp": temps[-1][1],
    }


def compute_stability_score(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (math.sqrt(variance) / mean) * 100


# ─── NVML sampling thread ─────────────────────────────────────────────────────

class HardwareSampler:
    """Background thread: continuously samples NVML during inference."""

    def __init__(self, nvml: NVMLDirect, interval: float = SAMPLE_INTERVAL):
        self._nvml = nvml
        self._interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t_start = None

    def start(self):
        self._samples = []
        self._stop.clear()
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._samples

    def _run(self):
        while not self._stop.is_set():
            s = self._nvml.sample()
            s["elapsed"] = time.time() - self._t_start
            self._samples.append(s)
            time.sleep(self._interval)

    def mark_round(self, round_num: int, label: str):
        """Insert a marker into the sample stream at the current time."""
        self._samples.append({
            "marker": True,
            "round": round_num,
            "label": label,
            "elapsed": time.time() - self._t_start,
            "timestamp": time.time(),
        })


# ─── Summarize hardware samples for one test condition ───────────────────────

def summarize_hw(samples: list, label: str) -> dict:
    hw = [s for s in samples if not s.get("marker")]
    if not hw:
        return {}

    clocks = [s["clk_mhz"] for s in hw if s.get("clk_mhz") is not None]
    powers = [s["power_w"] for s in hw if s.get("power_w") is not None]
    temps = [s["temp_c"] for s in hw if s.get("temp_c") is not None]

    throttle_reasons_seen = set()
    problem_throttle = False
    for s in hw:
        for r in s.get("throttle_reasons", []):
            if r not in ("NONE", "GPU_IDLE"):
                throttle_reasons_seen.add(r)
        if s.get("throttle_problem"):
            problem_throttle = True

    thermal = compute_thermal_trajectory(hw)
    clock_stability = compute_stability_score(clocks) if clocks else 0.0

    return {
        "label": label,
        "sample_count": len(hw),
        "clk_peak_mhz": max(clocks) if clocks else None,
        "clk_avg_mhz": round(sum(clocks) / len(clocks), 1) if clocks else None,
        "clk_stability_cv": round(clock_stability, 3),
        "power_avg_w": round(sum(powers) / len(powers), 1) if powers else None,
        "power_peak_w": round(max(powers), 1) if powers else None,
        "temp_avg_c": round(sum(temps) / len(temps), 1) if temps else None,
        "temp_peak_c": max(temps) if temps else None,
        "thermal_trajectory": thermal,
        "throttle_reasons_seen": sorted(throttle_reasons_seen),
        "problem_throttle": problem_throttle,
    }


# ─── Config file management (from baristankut) ───────────────────────────────

def get_gb10_configs(config_dir):
    if not config_dir or not os.path.isdir(config_dir):
        return []
    return [f for f in os.listdir(config_dir) if "NVIDIA_GB10" in f]


def backup_configs(config_dir):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for f in get_gb10_configs(config_dir):
        shutil.copy2(os.path.join(config_dir, f), os.path.join(BACKUP_DIR, f))


def remove_configs(config_dir):
    for f in get_gb10_configs(config_dir):
        os.remove(os.path.join(config_dir, f))


def restore_configs(config_dir):
    if not os.path.isdir(BACKUP_DIR):
        return
    if not config_dir:
        return
    for f in os.listdir(BACKUP_DIR):
        shutil.copy2(os.path.join(BACKUP_DIR, f), os.path.join(config_dir, f))


# ─── Kernel sweep baseline loader ────────────────────────────────────────────

def load_kernel_sweep_baseline(override=None) -> dict:
    """Load uma_bw_results.json from nvidia-uma-fault-probe kernel sweep."""
    path = find_kernel_sweep_baseline(override)
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ─── Baseline save/compare ───────────────────────────────────────────────────

def baseline_path(gpu_index: int = 0) -> Path:
    return BASELINE_DIR / f"moe-baseline-gpu{gpu_index}.json"


def save_baseline(data: dict, gpu_index: int = 0):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    data["saved_at"] = datetime.now(timezone.utc).isoformat()
    path = baseline_path(gpu_index)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  {GREEN}Baseline saved: {path}{RESET}")


def load_baseline(gpu_index: int = 0) -> dict:
    path = baseline_path(gpu_index)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def show_comparison(current: dict, baseline: dict):
    print()
    print("─" * 60)
    print("  BASELINE COMPARISON")
    print("─" * 60)
    print(f"  Baseline from: {baseline.get('saved_at', 'unknown')}")
    print()

    cur_b = current.get("test_b_hw", {})
    base_b = baseline.get("test_b_hw", {})

    fields = [
        ("Peak clock (MHz)", "clk_peak_mhz", ".0f", "higher"),
        ("Avg clock (MHz)", "clk_avg_mhz", ".1f", "higher"),
        ("Avg power (W)", "power_avg_w", ".1f", "lower"),
        ("Peak temp (°C)", "temp_peak_c", ".0f", "lower"),
        ("Clock stability (%CV)", "clk_stability_cv", ".2f", "lower"),
    ]

    for label, key, spec, better in fields:
        cur_val = cur_b.get(key)
        base_val = base_b.get(key)
        if cur_val is not None and base_val is not None:
            delta = cur_val - base_val
            sign = "+" if delta >= 0 else ""
            color = GREEN if (
                (better == "higher" and delta >= 0) or
                (better == "lower" and delta <= 0)
            ) else YELLOW
            print(
                f"  {label:<26s}  now: {cur_val:{spec}}  base: {base_val:{spec}}  "
                f"{color}{sign}{delta:{spec}}{RESET}"
            )

    cur_tps = current.get("test_b", {}).get("avg_tok_per_s")
    base_tps = baseline.get("test_b", {}).get("avg_tok_per_s")
    if cur_tps and base_tps:
        delta = cur_tps - base_tps
        color = GREEN if delta >= 0 else RED
        print(
            f"  {'Avg tok/s (Test B)':<26s}  now: {cur_tps:.1f}  "
            f"base: {base_tps:.1f}  {color}{delta:+.1f}{RESET}"
        )
    print()


# ─── Inference benchmark (from baristankut, instrumented) ────────────────────

def run_benchmark(label: str, prompts: list, sampler: HardwareSampler) -> dict:
    """Run vLLM inference benchmark with hardware telemetry in background."""
    from vllm import LLM, SamplingParams

    print(f"\n=== {label} ===")
    print(f"  Model: {MODEL}")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    print("  Warmup...")
    llm.generate(["Hello"], sampling_params)

    results = []
    for i in range(3):
        sampler.mark_round(i + 1, f"{label} round {i + 1} start")
        t_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - t_start
        sampler.mark_round(i + 1, f"{label} round {i + 1} end")

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed
        results.append({
            "round": i + 1,
            "tokens": total_tokens,
            "time_s": round(elapsed, 3),
            "tok_per_s": round(tps, 2),
        })
        print(
            f"  Round {i + 1}: {total_tokens} tokens in {elapsed:.2f}s = "
            f"{GREEN}{tps:.1f} tok/s{RESET}"
        )

    avg_tps = sum(r["tok_per_s"] for r in results) / len(results)
    print(f"  Average: {BOLD}{avg_tps:.1f} tok/s{RESET}")

    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    return {"label": label, "results": results, "avg_tok_per_s": round(avg_tps, 2)}


# ─── JSON report export ──────────────────────────────────────────────────────

def export_report(results: dict, gpu_index: int = 0):
    report_dir = BASELINE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"moe-ab-benchmark_gpu{gpu_index}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  {GREEN}Report saved: {path}{RESET}")
    return str(path)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GB10 MoE kernel config A/B benchmark with NVML telemetry"
    )
    parser.add_argument(
        "--mode",
        choices=["both", "with", "without"],
        default="both",
        help="Run both conditions, only with configs, or only without (default: both)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index (default: 0)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save Test B (with configs) result as baseline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against saved baseline after run",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Export full JSON report with hardware telemetry timeline",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Triton MoE config directory (auto-discovered if not set)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to uma_bw_results.json from nvidia-uma-fault-probe kernel sweep",
    )
    args = parser.parse_args()

    # ── Resolve config dir ──
    config_dir = find_triton_config_dir(args.config_dir)

    # ── Initialize NVML ──
    nvml = NVMLDirect()
    nvml_ok = nvml.init(gpu_index=args.gpu)

    print("=" * 60)
    print("  DGX Spark GB10 (SM121) MoE Kernel A/B Benchmark")
    print("  Instrumented — parallelArchitect")
    print("=" * 60)

    if nvml_ok:
        gpu_name = nvml.get_gpu_name()
        driver = nvml.get_driver_version()
        print(f"  GPU:    {gpu_name or 'unknown'}")
        print(f"  Driver: {driver or 'unknown'}")
    else:
        print(f"  {YELLOW}NVML unavailable — hardware telemetry disabled{RESET}")

    if config_dir:
        print(f"  Config: {config_dir}")
    else:
        print(f"  {YELLOW}No Triton config dir found — use --config-dir to specify{RESET}")

    # ── Load kernel sweep baseline ──
    ks_baseline = load_kernel_sweep_baseline(args.baseline)
    if ks_baseline:
        idle_read = ks_baseline.get("bw_idle_gpu_read_gbs") or ks_baseline.get("gpu_read_gbs")
        idle_write = ks_baseline.get("bw_idle_gpu_write_gbs") or ks_baseline.get("gpu_write_gbs")
        if idle_read:
            print(f"  Kernel sweep baseline: GPU read {idle_read:.2f} GB/s  "
                  f"write {idle_write:.2f} GB/s")
    else:
        print(f"  {DIM}No kernel sweep baseline found — use --baseline to specify{RESET}")

    # ── Idle hardware snapshot ──
    print()
    if nvml_ok:
        idle = nvml.sample()
        print(f"  Idle state: CLK={idle.get('clk_mhz')} MHz  "
              f"PWR={idle.get('power_w'):.1f}W  "
              f"TMP={idle.get('temp_c')}°C")

    print()
    BOX_W = 56

    # ── Sampler ──
    sampler = HardwareSampler(nvml) if nvml_ok else None

    results = {}
    all_samples = []

    # ── Test A: Without configs ──
    if args.mode in ("both", "without"):
        configs = get_gb10_configs(config_dir)
        if configs:
            backup_configs(config_dir)
            remove_configs(config_dir)
            print(f"  GB10 configs: {YELLOW}REMOVED{RESET} "
                  f"({len(configs)} backed up) — SM80 fallback active")
        else:
            print(f"  {YELLOW}No GB10 configs found to remove — "
                  f"already using fallback kernels{RESET}")

        if sampler:
            sampler.start()

        result_a = run_benchmark(
            "Test A: Without GB10 Configs (SM80 fallback)", PROMPTS, sampler
        )

        if sampler:
            samples_a = sampler.stop()
            all_samples.extend(samples_a)
            hw_a = summarize_hw(samples_a, "Test A")
            result_a["hw"] = hw_a

            print(f"\n  Hardware (Test A):")
            print(f"    Clock:  peak={hw_a.get('clk_peak_mhz')} MHz  "
                  f"avg={hw_a.get('clk_avg_mhz')} MHz  "
                  f"stability={hw_a.get('clk_stability_cv'):.2f}% CV")
            print(f"    Power:  avg={hw_a.get('power_avg_w')} W  "
                  f"peak={hw_a.get('power_peak_w')} W")
            print(f"    Temp:   avg={hw_a.get('temp_avg_c')}°C  "
                  f"peak={hw_a.get('temp_peak_c')}°C")
            traj = hw_a.get("thermal_trajectory", {})
            t_color = GREEN if traj.get("stable") else YELLOW
            print(f"    Thermal trajectory: {t_color}{traj.get('direction')} "
                  f"({traj.get('slope'):+.2f} °C/s){RESET}")
            if hw_a.get("problem_throttle"):
                print(f"    {RED}Problem throttle detected: "
                      f"{', '.join(hw_a.get('throttle_reasons_seen', []))}{RESET}")

        restore_configs(config_dir)
        print(f"\n  GB10 configs: {GREEN}RESTORED{RESET}")
        results["test_a"] = result_a

        if args.mode == "both":
            print("\n  Thermal cooldown (30s)...")
            time.sleep(30)

    # ── Test B: With configs ──
    if args.mode in ("both", "with"):
        configs = get_gb10_configs(config_dir)
        print(f"\n  GB10 configs: {GREEN}{len(configs)} present{RESET} "
              f"— SM121-tuned kernels active")

        if sampler:
            sampler.start()

        result_b = run_benchmark(
            "Test B: With GB10 Configs (SM121-tuned)", PROMPTS, sampler
        )

        if sampler:
            samples_b = sampler.stop()
            all_samples.extend(samples_b)
            hw_b = summarize_hw(samples_b, "Test B")
            result_b["hw"] = hw_b

            print(f"\n  Hardware (Test B):")
            print(f"    Clock:  peak={hw_b.get('clk_peak_mhz')} MHz  "
                  f"avg={hw_b.get('clk_avg_mhz')} MHz  "
                  f"stability={hw_b.get('clk_stability_cv'):.2f}% CV")
            print(f"    Power:  avg={hw_b.get('power_avg_w')} W  "
                  f"peak={hw_b.get('power_peak_w')} W")
            print(f"    Temp:   avg={hw_b.get('temp_avg_c')}°C  "
                  f"peak={hw_b.get('temp_peak_c')}°C")
            traj = hw_b.get("thermal_trajectory", {})
            t_color = GREEN if traj.get("stable") else YELLOW
            print(f"    Thermal trajectory: {t_color}{traj.get('direction')} "
                  f"({traj.get('slope'):+.2f} °C/s){RESET}")
            if hw_b.get("problem_throttle"):
                print(f"    {RED}Problem throttle detected: "
                      f"{', '.join(hw_b.get('throttle_reasons_seen', []))}{RESET}")

        results["test_b"] = result_b
        results["test_b_hw"] = result_b.get("hw", {})

    # ── Summary ──
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if "test_a" in results and "test_b" in results:
        tps_a = results["test_a"]["avg_tok_per_s"]
        tps_b = results["test_b"]["avg_tok_per_s"]
        diff = tps_b - tps_a
        pct = (diff / tps_a) * 100 if tps_a > 0 else 0
        color = GREEN if diff >= 0 else RED

        print(f"  Throughput:")
        print(f"    Test A (SM80 fallback):  {tps_a:.1f} tok/s")
        print(f"    Test B (SM121-tuned):    {tps_b:.1f} tok/s")
        print(f"    Delta:                   {color}{diff:+.1f} tok/s ({pct:+.1f}%){RESET}")

        if sampler:
            hw_a = results["test_a"].get("hw", {})
            hw_b = results["test_b"].get("hw", {})
            print(f"\n  Hardware delta (A → B):")

            for label, key, spec in [
                ("Clock peak (MHz)", "clk_peak_mhz", ".0f"),
                ("Power avg (W)", "power_avg_w", ".1f"),
                ("Temp peak (°C)", "temp_peak_c", ".0f"),
            ]:
                va = hw_a.get(key)
                vb = hw_b.get(key)
                if va is not None and vb is not None:
                    delta = vb - va
                    sign = "+" if delta >= 0 else ""
                    print(f"    {label:<22s}  A={va:{spec}}  B={vb:{spec}}  "
                          f"{sign}{delta:{spec}}")

        results["diff_pct"] = round(pct, 2)
        results["diff_tps"] = round(diff, 2)

    # ── Kernel sweep baseline anchor ──
    if ks_baseline:
        results["kernel_sweep_baseline"] = ks_baseline

    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    results["gpu_index"] = args.gpu
    if nvml_ok:
        results["gpu_name"] = nvml.get_gpu_name()
        results["driver_version"] = nvml.get_driver_version()

    # ── Save results ──
    out_dir = Path.home() / "sm121-kernels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ab_instrumented_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ── Baseline ──
    if args.save_baseline and "test_b" in results:
        save_baseline(results, args.gpu)

    if args.compare:
        bl = load_baseline(args.gpu)
        if bl:
            show_comparison(results, bl)
        else:
            print(f"\n  {YELLOW}No baseline found. Run with --save-baseline first.{RESET}")

    if args.report:
        results["hw_timeline"] = all_samples
        export_report(results, args.gpu)

    nvml.shutdown()


if __name__ == "__main__":
    main()
