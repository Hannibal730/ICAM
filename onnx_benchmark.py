#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onnx_benchmark.py

SCI/EAAI 투고용 ONNX Runtime(ORT) CPU inference 벤치마크 스크립트 (Raspberry Pi 5 포함).

핵심 목표: "통제 강한" 실험 조건 설정 + 지표 정의 명확화 + 재현성.

주요 기능
- (선택) CPU affinity 고정: --cpu-affinity "0-3"  (taskset과 동일한 효과)
- ORT 스레드/실행 모드 고정: --intra / --inter / --execution-mode
- (선택) 외부 스레딩 라이브러리 환경변수 고정: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB
- latency / memory 측정 분리: --mode latency|memory|both
- (선택) fresh-process 반복 실행: --repeat N --fresh-process true

권장(투고용) 실행 예시 (Pi5, 4 cores)
1) Latency (통제 강함):
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
   python3 onnx_benchmark.py --onnx model.onnx --mode latency \
     --cpu-affinity 0-3 --intra 4 --inter 1 --execution-mode sequential \
     --warmup 50 --runs 1000 --repeat 5 --fresh-process true

2) Peak RSS (paper-grade):
   (fresh-process + 외부 측정)
   for i in 1 2 3 4 5; do
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
     /usr/bin/time -v python3 onnx_benchmark.py --onnx model.onnx --mode latency \
       --cpu-affinity 0-3 --intra 4 --inter 1 --execution-mode sequential \
       --warmup 50 --runs 1000 --json-out run_$i.json 1> run_$i.out 2> run_$i.time
   done
   -> 'Maximum resident set size (kbytes)' 를 Peak RSS로 사용

or
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/time -v python3 onnx_benchmark.py \
--onnx /home/pi/Desktop/sewer_binary_cls_v9/onnx/deit_tiny/deit_tiny_model_fp32_20260118_104712.onnx \
--mode latency \
--cpu-affinity 0-3 \
--intra 4 --inter 1 --execution-mode sequential \
--seed 42 --fill random \
--warmup 50 --runs 1000


저장된 time_*.txt에서 Max RSS 평균/분산을 “로그로 출력”
peak_rss_stats.py



3) Memory (보조 지표, Load delta RSS (MB), inference peak delta (MB)):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 onnx_benchmark.py \
  --onnx /home/pi/Desktop/sewer_binary_cls_v9/onnx/deit_tiny/deit_tiny_model_fp32_20260118_104712.onnx \
  --mode memory \
  --cpu-affinity 0-3 \
  --intra 4 --inter 1 --execution-mode sequential \
  --seed 42 --fill random \
  --warmup 50 --runs 1000 \
  --mem-interval-ms 1 \
  --mem-include-warmup true \
  --repeat 5 --fresh-process true \
  --json-out memory_benchmark.json



주의
- 본 스크립트 내부 RSS 샘플링(--mode memory)은 보조 지표입니다.
  논문 본문 Peak RSS는 /usr/bin/time -v의 Maximum RSS를 우선 권장합니다.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -------------------------
# Basic utilities (no heavy imports)
# -------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean (true/false), got: {v!r}")


def parse_cpu_affinity(s: str) -> List[int]:
    """
    Parse cpu list/range:
      "0-3" -> [0,1,2,3]
      "0,2,3" -> [0,2,3]
      "0-2,4" -> [0,1,2,4]
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty --cpu-affinity")

    cpus: List[int] = []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                raise ValueError(f"Invalid range in --cpu-affinity: {p}")
            cpus.extend(list(range(a_i, b_i + 1)))
        else:
            cpus.append(int(p))
    # unique + sorted
    cpus = sorted(set(cpus))
    return cpus


def apply_cpu_affinity(cpus: List[int]) -> None:
    """Apply CPU affinity for current process (taskset equivalent)."""
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, set(cpus))  # Linux only
        return
    except Exception:
        pass
    # Fallback: psutil if available
    try:
        import psutil  # type: ignore
        psutil.Process(os.getpid()).cpu_affinity(cpus)
    except Exception as e:
        raise RuntimeError(f"Failed to set CPU affinity. cpus={cpus}. Error: {e}") from e


def set_env_threads_auto(intra_threads: int, explicit: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    """
    If user enabled env thread control and did not set explicit values,
    set common env vars to intra_threads (or to cpu_count if intra=0).
    """
    cpu_cnt = os.cpu_count() or 1
    base = intra_threads if intra_threads and intra_threads > 0 else cpu_cnt

    out = dict(explicit)
    def _fill(key: str) -> None:
        if out.get(key) is None:
            out[key] = base

    _fill("OMP_NUM_THREADS")
    _fill("OPENBLAS_NUM_THREADS")
    _fill("MKL_NUM_THREADS")
    _fill("NUMEXPR_NUM_THREADS")
    _fill("VECLIB_MAXIMUM_THREADS")
    return out


def apply_env_thread_vars(vals: Dict[str, Optional[int]]) -> None:
    for k, v in vals.items():
        if v is None:
            continue
        os.environ[k] = str(int(v))


def read_cpu_governor() -> Optional[str]:
    # Works on many Linux systems (including Pi) if cpufreq is available
    p = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


# -------------------------
# Heavy imports (after env/affinity)
# -------------------------

def lazy_imports():
    import gc
    import statistics
    import threading
    from dataclasses import dataclass
    from typing import Sequence

    import numpy as np
    import psutil
    import onnxruntime as ort

    return gc, statistics, threading, dataclass, Sequence, np, psutil, ort


# -------------------------
# Core compute helpers
# -------------------------

def dtype_from_onnx(elem_type: str, np) -> Any:
    t = elem_type.lower()
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    if "double" in t or "float64" in t:
        return np.float64
    if "int64" in t:
        return np.int64
    if "int32" in t:
        return np.int32
    if "int8" in t:
        return np.int8
    if "uint8" in t:
        return np.uint8
    if "bool" in t:
        return np.bool_
    return np.float32


def parse_shape(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --input-shape")
    out: List[int] = []
    for p in parts:
        if p in ("-1", "?", "None"):
            out.append(-1)
        else:
            out.append(int(p))
    return out


def percentile(values: Sequence[float], p: float, np) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def mb(x_bytes: int) -> float:
    return x_bytes / (1024.0 * 1024.0)


def summarize_sample(xs: List[float]) -> Dict[str, float]:
    """
    Sample statistics across repeats (recommended for paper reporting):
    - std: sample standard deviation (denominator n-1)
    - var: sample variance (denominator n-1)
    """
    if not xs:
        return {}
    import statistics as _st
    n = len(xs)
    mean = float(_st.mean(xs))
    std = float(_st.stdev(xs)) if n > 1 else 0.0
    var = float(_st.variance(xs)) if n > 1 else 0.0
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "var": var,
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def parse_time_v_maxrss_kb(text: str) -> Optional[int]:
    """
    Parse `/usr/bin/time -v` output and extract:
      Maximum resident set size (kbytes): <int>
    Robust to ANSI escape sequences.
    """
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    text = ansi.sub("", text)
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def summarize_time_files(glob_pattern: str) -> Dict[str, Any]:
    """
    Summarize Max RSS across multiple `time_*.txt` files produced by `/usr/bin/time -v`.
    Returns MB mean/std/var (sample stats) + per-file values.
    """
    import glob
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    rows = []
    vals_mb: List[float] = []
    for fn in files:
        with open(fn, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        kb = parse_time_v_maxrss_kb(txt)
        if kb is None:
            raise ValueError(f"Max RSS not found in file: {fn}")
        mb_val = kb / 1024.0
        rows.append({"file": fn, "max_rss_kb": kb, "max_rss_mb": mb_val})
        vals_mb.append(mb_val)

    summ = summarize_sample(vals_mb)
    return {
        "glob": glob_pattern,
        "files": rows,
        "summary_mb": summ,
    }

class RssSampler:
    """Background RSS sampler (psutil) to estimate peak RSS within a region."""
    def __init__(self, interval_ms: float, psutil, threading):
        self.interval_s = max(interval_ms / 1000.0, 0.001)
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._psutil = psutil

    def start(self, threading):
        self.peak_rss = self._proc.memory_info().rss
        self._stop.clear()

        def _run():
            while not self._stop.is_set():
                rss = self._proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss = self._proc.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        return self.peak_rss


def create_session(onnx_path: str, args, ort):
    so = ort.SessionOptions()

    if args.intra_threads > 0:
        so.intra_op_num_threads = int(args.intra_threads)
    if args.inter_threads > 0:
        so.inter_op_num_threads = int(args.inter_threads)

    if args.execution_mode == "sequential":
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    ol = args.opt_level
    if ol == "disable":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif ol == "basic":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif ol == "extended":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    so.enable_mem_pattern = bool(args.mem_pattern)
    so.enable_cpu_mem_arena = bool(args.cpu_arena)

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def resolve_input_spec(sess, args, np) -> Tuple[str, List[int], Any]:
    inputs = sess.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs")

    if args.input_name is None:
        chosen = inputs[0]
    else:
        chosen = None
        for it in inputs:
            if it.name == args.input_name:
                chosen = it
                break
        if chosen is None:
            names = [it.name for it in inputs]
            raise ValueError(f"Input '{args.input_name}' not found. Available: {names}")

    if args.input_dtype is None:
        np_dt = dtype_from_onnx(chosen.type, np)
    else:
        t = args.input_dtype.lower()
        if t in ("fp32", "float", "float32"):
            np_dt = np.float32
        elif t in ("fp16", "float16"):
            np_dt = np.float16
        elif t == "int64":
            np_dt = np.int64
        elif t == "int32":
            np_dt = np.int32
        elif t == "uint8":
            np_dt = np.uint8
        else:
            raise ValueError(f"Unknown --input-dtype: {args.input_dtype}")

    model_shape: List[int] = []
    for d in chosen.shape:
        if isinstance(d, str) or d is None:
            model_shape.append(-1)
        else:
            model_shape.append(int(d))

    if args.input_shape is None:
        concrete = [1 if d == -1 else int(d) for d in model_shape]
    else:
        user_shape = parse_shape(args.input_shape)
        if len(user_shape) != len(model_shape):
            raise ValueError(f"--input-shape rank mismatch. Model rank={len(model_shape)}, got={len(user_shape)}")
        concrete = []
        for ms, us in zip(model_shape, user_shape):
            if us == -1:
                concrete.append(1 if ms == -1 else int(ms))
            else:
                concrete.append(int(us))

    return chosen.name, concrete, np_dt


def make_input_array(shape: List[int], dtype, args, np):
    rng = np.random.default_rng(args.seed)
    if args.fill == "random":
        if np.issubdtype(dtype, np.floating):
            return rng.standard_normal(size=shape).astype(dtype)
        return rng.integers(low=0, high=127, size=shape, dtype=dtype)
    if args.fill == "zeros":
        return np.zeros(shape, dtype=dtype)
    if args.fill == "ones":
        return np.ones(shape, dtype=dtype)
    raise ValueError(f"Unknown --fill: {args.fill}")


# -------------------------
# Single-run worker (child)
# -------------------------

def run_single(args) -> Dict[str, Any]:
    gc, statistics, threading, dataclass, Sequence, np, psutil, ort = lazy_imports()

    proc = psutil.Process(os.getpid())

    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "onnx": os.path.abspath(args.onnx),
        "mode": args.mode,
        "platform": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(list(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else None,
            "cpu_governor": read_cpu_governor(),
        },
        "ort": {
            "onnxruntime_version": ort.__version__,
            "providers": ["CPUExecutionProvider"],
            "session_options": {
                "intra_threads": args.intra_threads,
                "inter_threads": args.inter_threads,
                "execution_mode": args.execution_mode,
                "opt_level": args.opt_level,
                "enable_mem_pattern": bool(args.mem_pattern),
                "enable_cpu_mem_arena": bool(args.cpu_arena),
            },
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            },
        },
        "warmup": args.warmup,
        "runs": args.runs,
        "input": {},
    }

    # Session (for latency/both). For memory mode we load inside measured block to capture load.
    sess = None
    if args.mode in ("latency", "both"):
        sess = create_session(args.onnx, args, ort)

    if sess is None:
        tmp = create_session(args.onnx, args, ort)
        sess_for_spec = tmp
    else:
        sess_for_spec = sess

    in_name, in_shape, in_dtype = resolve_input_spec(sess_for_spec, args, np)
    x = make_input_array(in_shape, in_dtype, args, np)
    feeds = {in_name: x}
    report["input"] = {"name": in_name, "shape": in_shape, "dtype": str(in_dtype), "fill": args.fill, "seed": args.seed}

    # ---- LATENCY ----
    if args.mode in ("latency", "both"):
        assert sess is not None
        # warmup (excluded)
        for _ in range(max(0, args.warmup)):
            _ = sess.run(None, feeds)

        if args.sync_barrier:
            time.sleep(0.05)

        # reduce Python GC jitter during timing window
        prev_gc = gc.isenabled()
        gc.disable()

        times_ms: List[float] = []
        for _ in range(max(0, args.runs)):
            t0 = time.perf_counter_ns()
            _ = sess.run(None, feeds)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)

        if prev_gc:
            gc.enable()

        mean = statistics.mean(times_ms) if times_ms else float("nan")
        std = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0

        lat = {
            "mean_ms": float(mean),
            "std_ms": float(std),
            "min_ms": float(min(times_ms)) if times_ms else float("nan"),
            "max_ms": float(max(times_ms)) if times_ms else float("nan"),
            "p50_ms": float(percentile(times_ms, 50, np)),
            "p95_ms": float(percentile(times_ms, 95, np)),
            "p99_ms": float(percentile(times_ms, 99, np)),
            "warmup_excluded": True,
            "n_runs": len(times_ms),
        }
        report["latency"] = lat

    # ---- MEMORY (aux RSS sampling) ----
    if args.mode in ("memory", "both"):
        # load phase
        rss_before_load = proc.memory_info().rss
        load_sampler = RssSampler(interval_ms=args.mem_interval_ms, psutil=psutil, threading=threading)
        load_sampler.start(threading)
        sess2 = create_session(args.onnx, args, ort)
        peak_load = load_sampler.stop()
        rss_after_load = proc.memory_info().rss

        # inference phase
        rss_before_infer = proc.memory_info().rss
        infer_sampler = RssSampler(interval_ms=args.mem_interval_ms, psutil=psutil, threading=threading)
        infer_sampler.start(threading)

        if args.mem_include_warmup:
            for _ in range(max(0, args.warmup)):
                _ = sess2.run(None, feeds)
        else:
            for _ in range(max(0, args.warmup)):
                _ = sess2.run(None, feeds)
            infer_sampler.stop()
            rss_before_infer = proc.memory_info().rss
            infer_sampler = RssSampler(interval_ms=args.mem_interval_ms, psutil=psutil, threading=threading)
            infer_sampler.start(threading)

        for _ in range(max(0, args.runs)):
            _ = sess2.run(None, feeds)

        peak_infer = infer_sampler.stop()
        rss_after_infer = proc.memory_info().rss

        mem = {
            "rss_before_load_mb": mb(rss_before_load),
            "rss_after_load_mb": mb(rss_after_load),
            "delta_load_mb": mb(rss_after_load - rss_before_load),
            "peak_rss_during_load_mb": mb(peak_load),
            "rss_before_infer_mb": mb(rss_before_infer),
            "rss_after_infer_mb": mb(rss_after_infer),
            "peak_rss_during_infer_mb": mb(peak_infer),
            "delta_peak_infer_from_before_infer_mb": mb(peak_infer - rss_before_infer),
            "delta_peak_total_from_before_load_mb": mb(peak_infer - rss_before_load),
            "include_warmup_in_peak": bool(args.mem_include_warmup),
            "mem_interval_ms": float(args.mem_interval_ms),
            "note": "Use /usr/bin/time -v for paper-grade peak RSS (Maximum resident set size).",
        }
        report["memory"] = mem

    return report


# -------------------------
# Parent aggregation for repeat/fresh-process
# -------------------------

def aggregate_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-run summaries across repeats (NOT concatenating raw samples)."""
    out: Dict[str, Any] = {
        "repeat": len(reports),
        "runs": reports,
    }

    # ---- Latency across repeats (sample stats; n-1) ----
    lat_mean_ms: List[float] = []
    lat_p50_ms: List[float] = []
    lat_p95_ms: List[float] = []
    lat_p99_ms: List[float] = []

    for r in reports:
        lat = r.get("latency")
        if not isinstance(lat, dict):
            continue
        lat_mean_ms.append(float(lat.get("mean_ms", float("nan"))))
        lat_p50_ms.append(float(lat.get("p50_ms", float("nan"))))
        lat_p95_ms.append(float(lat.get("p95_ms", float("nan"))))
        lat_p99_ms.append(float(lat.get("p99_ms", float("nan"))))

    # Filter NaNs (defensive)
    def _finite(xs: List[float]) -> List[float]:
        import math
        return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]

    lat_mean_ms = _finite(lat_mean_ms)
    lat_p50_ms = _finite(lat_p50_ms)
    lat_p95_ms = _finite(lat_p95_ms)
    lat_p99_ms = _finite(lat_p99_ms)

    if lat_mean_ms:
        out["latency_across_repeats_ms"] = {
            "mean_ms": summarize_sample(lat_mean_ms),
            "p50_ms": summarize_sample(lat_p50_ms),
            "p95_ms": summarize_sample(lat_p95_ms),
            "p99_ms": summarize_sample(lat_p99_ms),
        }

    # ---- Memory across repeats (sample stats; n-1) ----
    mem_peak_infer_mb: List[float] = []
    mem_delta_load_mb: List[float] = []
    mem_delta_peak_infer_mb: List[float] = []
    mem_delta_peak_total_mb: List[float] = []

    for r in reports:
        mem = r.get("memory")
        if not isinstance(mem, dict):
            continue
        mem_peak_infer_mb.append(float(mem.get("peak_rss_during_infer_mb", float("nan"))))
        mem_delta_load_mb.append(float(mem.get("delta_load_mb", float("nan"))))
        mem_delta_peak_infer_mb.append(float(mem.get("delta_peak_infer_from_before_infer_mb", float("nan"))))
        mem_delta_peak_total_mb.append(float(mem.get("delta_peak_total_from_before_load_mb", float("nan"))))

    mem_peak_infer_mb = _finite(mem_peak_infer_mb)
    mem_delta_load_mb = _finite(mem_delta_load_mb)
    mem_delta_peak_infer_mb = _finite(mem_delta_peak_infer_mb)
    mem_delta_peak_total_mb = _finite(mem_delta_peak_total_mb)

    if mem_peak_infer_mb:
        out["memory_across_repeats_mb"] = {
            "peak_rss_during_infer_mb": summarize_sample(mem_peak_infer_mb),
            "delta_load_mb": summarize_sample(mem_delta_load_mb),
            "delta_peak_infer_from_before_infer_mb": summarize_sample(mem_delta_peak_infer_mb),
            "delta_peak_total_from_before_load_mb": summarize_sample(mem_delta_peak_total_mb),
        }

    return out


def run_fresh_process_repeats(args) -> Dict[str, Any]:
    """Run this script as a child process multiple times and aggregate results."""
    cmd_base = [sys.executable, os.path.abspath(__file__), "--_child", "true"]

    # Reconstruct args for child (exclude repeat/fresh-process/json-out and internal flags)
    passthrough = []
    for k, v in vars(args).items():
        if k in ("repeat", "fresh_process", "json_out", "_child"):
            continue
        if v is None:
            continue
        # booleans
        if isinstance(v, bool):
            # only pass if True for store_true flags; for explicit bool args we pass key+value
            if k in ("sync_barrier",):
                if v:
                    passthrough.append(f"--{k.replace('_', '-')}")
            else:
                passthrough += [f"--{k.replace('_', '-')}", "true" if v else "false"]
        else:
            passthrough += [f"--{k.replace('_', '-')}", str(v)]

    reports: List[Dict[str, Any]] = []
    for i in range(int(args.repeat)):
        # Each repeat is a new process => good for peak RSS consistency
        child_cmd = cmd_base + passthrough
        p = subprocess.run(child_cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Child run failed (repeat {i+1}/{args.repeat}). stderr:\n{p.stderr}\nstdout:\n{p.stdout}")
        # Child prints JSON only
        rep = json.loads(p.stdout.strip())
        reports.append(rep)

    return aggregate_reports(reports)


# -------------------------
# CLI / Main
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--mode", choices=["latency", "memory", "both"], default="latency")

    # Strong control knobs
    p.add_argument("--cpu-affinity", default=None, help='CPU affinity like "0-3" or "0,2,3" (taskset equivalent)')
    p.add_argument("--set-env-threads", type=str2bool, default=True, help="Set OMP/BLAS thread env vars for reproducibility")

    # input
    p.add_argument("--input-name", default=None)
    p.add_argument("--input-shape", default=None, help="Comma-separated, e.g. 1,3,224,224 (required for dynamic shapes)")
    p.add_argument("--input-dtype", default=None, help="fp32|fp16|int64|int32|uint8")
    p.add_argument("--fill", choices=["random", "zeros", "ones"], default="random")
    p.add_argument("--seed", type=int, default=42)

    # repetitions
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=1000)

    # ORT threads/options
    p.add_argument("--intra", "--intra-threads", dest="intra_threads", type=int, default=4)
    p.add_argument("--inter", "--inter-threads", dest="inter_threads", type=int, default=1)
    p.add_argument("--execution-mode", choices=["sequential", "parallel"], default="sequential")
    p.add_argument("--opt-level", choices=["disable", "basic", "extended", "all"], default="all")

    # ORT memory behavior (bools)
    p.add_argument("--mem-pattern", type=str2bool, default=True)
    p.add_argument("--cpu-arena", type=str2bool, default=True)

    # env thread overrides
    p.add_argument("--omp-threads", type=int, default=None)
    p.add_argument("--openblas-threads", type=int, default=None)
    p.add_argument("--mkl-threads", type=int, default=None)
    p.add_argument("--numexpr-threads", type=int, default=None)
    p.add_argument("--veclib-threads", type=int, default=None)

    # internal RSS sampling (aux)
    p.add_argument("--mem-interval-ms", type=float, default=1.0)
    p.add_argument("--mem-include-warmup", type=str2bool, default=False)

    # timing stability
    p.add_argument("--sync-barrier", action="store_true", help="Tiny sleep before timing window")

    # repeat control
    p.add_argument("--repeat", type=int, default=1, help="Number of repeats (use with --fresh-process true for strongest reproducibility)")
    p.add_argument("--fresh-process", type=str2bool, default=False, help="If true, repeats are executed as fresh child processes")

    # output
    p.add_argument("--json-out", default=None, help="Write report JSON to path")

    # post-processing helper: summarize `/usr/bin/time -v` outputs
    p.add_argument("--summarize-time-glob", default=None,
                   help='If set, parse matching files (e.g., "time_*.txt") and print Max RSS mean/std/var in MB, then exit.')

    # internal
    p.add_argument("--_child", type=str2bool, default=False, help=argparse.SUPPRESS)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Helper mode: summarize `/usr/bin/time -v` files (Max RSS) and exit
    if not args._child and args.summarize_time_glob:
        rep = summarize_time_files(args.summarize_time_glob)
        s = rep["summary_mb"]
        print(f"[TIME -v] Max RSS across files '{rep['glob']}': {s['mean']:.3f} ± {s['std']:.3f} MB (var {s['var']:.6f}, n={int(s['n'])})")
        print(f"[TIME -v] min/max (MB): {s['min']:.3f} / {s['max']:.3f}")
        return 0

    # ---- Strong control: CPU affinity ----
    if args.cpu_affinity:
        cpus = parse_cpu_affinity(args.cpu_affinity)
        apply_cpu_affinity(cpus)

    # ---- Strong control: env threads ----
    explicit = {
        "OMP_NUM_THREADS": args.omp_threads,
        "OPENBLAS_NUM_THREADS": args.openblas_threads,
        "MKL_NUM_THREADS": args.mkl_threads,
        "NUMEXPR_NUM_THREADS": args.numexpr_threads,
        "VECLIB_MAXIMUM_THREADS": args.veclib_threads,
    }
    if args.set_env_threads:
        vals = set_env_threads_auto(args.intra_threads, explicit)
        apply_env_thread_vars(vals)
    else:
        apply_env_thread_vars(explicit)

    # ---- Repeat handling ----
    if not args._child and args.repeat > 1 and args.fresh_process:
        agg = run_fresh_process_repeats(args)
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2)
        # human-readable summary (sample stats across repeats; var uses n-1)
        if "latency_across_repeats_ms" in agg:
            L = agg["latency_across_repeats_ms"]
            for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
                s = L.get(key, {})
                if s:
                    print(
                        f"[REPEAT x{args.repeat}] latency {key} across repeats: "
                        f"{s['mean']:.4f} ± {s['std']:.4f} ms (var {s['var']:.6f}, min {s['min']:.4f}, max {s['max']:.4f})"
                    )

        if "memory_across_repeats_mb" in agg:
            M = agg["memory_across_repeats_mb"]
            s = M.get("peak_rss_during_infer_mb", {})
            if s:
                print(
                    f"[REPEAT x{args.repeat}] memory peak_rss_during_infer_mb across repeats: "
                    f"{s['mean']:.3f} ± {s['std']:.3f} MB (var {s['var']:.6f}, min {s['min']:.3f}, max {s['max']:.3f})"
                )
            s = M.get("delta_load_mb", {})
            if s:
                print(
                    f"[REPEAT x{args.repeat}] memory delta_load_mb across repeats: "
                    f"{s['mean']:.3f} ± {s['std']:.3f} MB (var {s['var']:.6f}, min {s['min']:.3f}, max {s['max']:.3f})"
                )
            s = M.get("delta_peak_infer_from_before_infer_mb", {})
            if s:
                print(
                    f"[REPEAT x{args.repeat}] memory delta_peak_infer_from_before_infer_mb across repeats: "
                    f"{s['mean']:.3f} ± {s['std']:.3f} MB (var {s['var']:.6f}, min {s['min']:.3f}, max {s['max']:.3f})"
                )
            s = M.get("delta_peak_total_from_before_load_mb", {})
            if s:
                print(
                    f"[REPEAT x{args.repeat}] memory delta_peak_total_from_before_load_mb across repeats: "
                    f"{s['mean']:.3f} ± {s['std']:.3f} MB (var {s['var']:.6f}, min {s['min']:.3f}, max {s['max']:.3f})"
                )
        return 0

    # ---- Single run ----
    rep = run_single(args)

    # If child mode: print JSON only (for parent aggregation)
    if args._child:
        print(json.dumps(rep))
        return 0

    # Otherwise: print human-readable + optionally JSON file
    if args.mode in ("latency", "both") and "latency" in rep:
        lat = rep["latency"]
        print("[LATENCY] ms:")
        for k in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            print(f"  {k}: {lat[k]:.4f}")

    if args.mode in ("memory", "both") and "memory" in rep:
        mem = rep["memory"]
        print("[MEMORY] RSS (MB):")
        for k in (
            "rss_before_load_mb",
            "rss_after_load_mb",
            "delta_load_mb",
            "peak_rss_during_load_mb",
            "rss_before_infer_mb",
            "rss_after_infer_mb",
            "peak_rss_during_infer_mb",
            "delta_peak_infer_from_before_infer_mb",
            "delta_peak_total_from_before_load_mb",
        ):
            print(f"  {k}: {mem[k]:.3f}")
        print(f"  include_warmup_in_peak: {mem['include_warmup_in_peak']}")
        print("  NOTE: For paper-grade peak RSS, prefer external: /usr/bin/time -v ... (Maximum resident set size)")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"[JSON] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
