#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_onnx.py

SCI/EAAI 제출용 ONNX Runtime(ORT) CPU inference 벤치마크 스크립트.

목표
- 라즈베리파이(ARM) 같은 환경에서도 재현 가능한 latency / memory 측정
- 측정 구간(로드/워밍업/실측)을 명확히 분리하고, 보고 지표를 표준화
- latency 측정과 memory 측정을 분리 실행할 수 있게 설계(교란 최소화)

권장 사용법(가장 '국룰'에 가까운 방식)
- Peak RSS(Max resident set size)는 외부에서 측정:
  /usr/bin/time -v python3 benchmark_onnx.py --mode latency --onnx model.onnx ...
  /usr/bin/time -v python3 benchmark_onnx.py --mode memory  --onnx model.onnx ...

- latency는 내부 통계 출력(mean/std/p50/p95/p99)
- memory 모드는 내부적으로도 RSS 샘플링 기반 peak를 제공(보조 지표)

주의
- 이 스크립트는 GPU 측정이 아니라 CPU EP 기준을 기본으로 합니다.
- dynamic shape 모델이면 --input-shape 를 명시하세요.

"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime import failed:", e, file=sys.stderr)
    raise

try:
    import psutil
except Exception as e:
    print("ERROR: psutil import failed (pip install psutil):", e, file=sys.stderr)
    raise


# -------------------------------
# Utilities
# -------------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def set_env_threads(omp: Optional[int], mkl: Optional[int]) -> None:
    """Set environment variables that can influence CPU kernels."""
    if omp is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp)
    if mkl is not None:
        os.environ["MKL_NUM_THREADS"] = str(mkl)


def parse_shape(s: str) -> List[int]:
    """Parse shape string like '1,3,224,224' -> [1,3,224,224]."""
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


def dtype_from_onnx(elem_type: str) -> np.dtype:
    """Best-effort mapping from ORT type string to numpy dtype."""
    # Examples: 'tensor(float)', 'tensor(float16)', 'tensor(int64)'
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
    # Default
    return np.float32


def safe_int(x: Any, default: int = 1) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return default


@dataclass
class RssStats:
    rss_before: int
    rss_after: int
    peak_rss: int

    @property
    def delta_bytes(self) -> int:
        return int(self.rss_after - self.rss_before)


class RssSampler:
    """Background RSS sampler (psutil) to estimate peak RSS within a region."""

    def __init__(self, interval_ms: float = 5.0):
        self.interval_s = max(interval_ms / 1000.0, 0.001)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())
        self.peak_rss = 0

    def start(self) -> None:
        self.peak_rss = self._proc.memory_info().rss
        self._stop.clear()

        def _run() -> None:
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
        # final sample
        rss = self._proc.memory_info().rss
        if rss > self.peak_rss:
            self.peak_rss = rss
        return self.peak_rss


def mb(x_bytes: int) -> float:
    return x_bytes / (1024.0 * 1024.0)


def percentile(values: Sequence[float], p: float) -> float:
    """Compute percentile with linear interpolation."""
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


# -------------------------------
# ORT session & input generation
# -------------------------------

def create_session(
    onnx_path: str,
    intra_threads: int,
    inter_threads: int,
    execution_mode: str,
    opt_level: str,
    enable_mem_pattern: bool,
    enable_cpu_mem_arena: bool,
    providers: Optional[List[str]] = None,
) -> ort.InferenceSession:
    so = ort.SessionOptions()

    # threads
    if intra_threads > 0:
        so.intra_op_num_threads = int(intra_threads)
    if inter_threads > 0:
        so.inter_op_num_threads = int(inter_threads)

    # execution mode
    em = execution_mode.strip().lower()
    if em == "sequential":
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    elif em == "parallel":
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        raise ValueError(f"Unknown --execution-mode: {execution_mode}")

    # graph optimization
    ol = opt_level.strip().lower()
    if ol in ("disable", "none", "0"):
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif ol in ("basic", "1"):
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif ol in ("extended", "2"):
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif ol in ("all", "3"):
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        raise ValueError(f"Unknown --opt-level: {opt_level}")

    # memory behavior
    so.enable_mem_pattern = bool(enable_mem_pattern)
    so.enable_cpu_mem_arena = bool(enable_cpu_mem_arena)

    # providers
    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


def resolve_input_spec(
    sess: ort.InferenceSession,
    input_name: Optional[str],
    input_shape: Optional[List[int]],
    input_dtype: Optional[str],
) -> Tuple[str, List[int], np.dtype]:
    inputs = sess.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs")

    # pick input
    if input_name is None:
        chosen = inputs[0]
    else:
        found = None
        for it in inputs:
            if it.name == input_name:
                found = it
                break
        if found is None:
            names = [it.name for it in inputs]
            raise ValueError(f"Input '{input_name}' not found. Available: {names}")
        chosen = found

    # dtype
    if input_dtype is None:
        np_dt = dtype_from_onnx(chosen.type)
    else:
        t = input_dtype.lower()
        if t in ("fp32", "float", "float32"):
            np_dt = np.float32
        elif t in ("fp16", "float16"):
            np_dt = np.float16
        elif t in ("int64",):
            np_dt = np.int64
        elif t in ("int32",):
            np_dt = np.int32
        elif t in ("uint8",):
            np_dt = np.uint8
        else:
            raise ValueError(f"Unknown --input-dtype: {input_dtype}")

    # shape
    model_shape = []
    for d in chosen.shape:
        if isinstance(d, str) or d is None:
            model_shape.append(-1)
        else:
            model_shape.append(safe_int(d, -1))

    if input_shape is None:
        # Fill dynamic dims with 1 by default (safe but may be wrong)
        concrete = [1 if d == -1 else int(d) for d in model_shape]
    else:
        if len(input_shape) != len(model_shape):
            raise ValueError(
                f"--input-shape rank mismatch. Model rank={len(model_shape)}, got={len(input_shape)}"
            )
        concrete = []
        for ms, us in zip(model_shape, input_shape):
            if us == -1:
                # user left as dynamic -> fill with 1
                concrete.append(1 if ms == -1 else int(ms))
            else:
                concrete.append(int(us))

    return chosen.name, concrete, np_dt


def make_input_array(shape: List[int], dtype: np.dtype, seed: int, fill: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if fill == "random":
        if np.issubdtype(dtype, np.floating):
            return rng.standard_normal(size=shape).astype(dtype)
        # int-like
        return rng.integers(low=0, high=127, size=shape, dtype=dtype)
    if fill == "zeros":
        return np.zeros(shape, dtype=dtype)
    if fill == "ones":
        return np.ones(shape, dtype=dtype)
    raise ValueError(f"Unknown --fill: {fill}")


# -------------------------------
# Bench modes
# -------------------------------

@dataclass
class LatencyReport:
    warmup: int
    runs: int
    times_ms: List[float]

    def summary(self) -> Dict[str, float]:
        xs = self.times_ms
        if not xs:
            return {}
        mean = statistics.mean(xs)
        stdev = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {
            "mean_ms": mean,
            "std_ms": stdev,
            "min_ms": min(xs),
            "max_ms": max(xs),
            "p50_ms": percentile(xs, 50),
            "p95_ms": percentile(xs, 95),
            "p99_ms": percentile(xs, 99),
        }


def run_latency(
    sess: ort.InferenceSession,
    feeds: Dict[str, np.ndarray],
    warmup: int,
    runs: int,
    sync_barrier: bool,
) -> LatencyReport:
    # warmup (not measured)
    for _ in range(max(0, warmup)):
        _ = sess.run(None, feeds)

    # small optional barrier (to reduce OS scheduling noise)
    if sync_barrier:
        time.sleep(0.05)

    times_ms: List[float] = []
    for _ in range(max(0, runs)):
        t0 = time.perf_counter_ns()
        _ = sess.run(None, feeds)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    return LatencyReport(warmup=warmup, runs=runs, times_ms=times_ms)


@dataclass
class MemoryReport:
    rss_before_load: int
    rss_after_load: int
    peak_rss_during_load: int
    rss_before_infer: int
    peak_rss_during_infer: int
    rss_after_infer: int
    include_warmup_in_peak: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rss_before_load_mb": mb(self.rss_before_load),
            "rss_after_load_mb": mb(self.rss_after_load),
            "delta_load_mb": mb(self.rss_after_load - self.rss_before_load),
            "peak_rss_during_load_mb": mb(self.peak_rss_during_load),
            "rss_before_infer_mb": mb(self.rss_before_infer),
            "rss_after_infer_mb": mb(self.rss_after_infer),
            "peak_rss_during_infer_mb": mb(self.peak_rss_during_infer),
            "delta_peak_infer_from_before_infer_mb": mb(self.peak_rss_during_infer - self.rss_before_infer),
            "delta_peak_total_from_before_load_mb": mb(self.peak_rss_during_infer - self.rss_before_load),
            "include_warmup_in_peak": self.include_warmup_in_peak,
        }


def run_memory(
    onnx_path: str,
    sess_kwargs: Dict[str, Any],
    feeds: Dict[str, np.ndarray],
    warmup: int,
    runs: int,
    sample_interval_ms: float,
    include_warmup_in_peak: bool,
) -> MemoryReport:
    proc = psutil.Process(os.getpid())

    # -------- load phase --------
    rss_before_load = proc.memory_info().rss
    load_sampler = RssSampler(interval_ms=sample_interval_ms)
    load_sampler.start()
    sess = create_session(onnx_path=onnx_path, **sess_kwargs)
    peak_rss_during_load = load_sampler.stop()
    rss_after_load = proc.memory_info().rss

    # -------- inference phase --------
    rss_before_infer = proc.memory_info().rss
    infer_sampler = RssSampler(interval_ms=sample_interval_ms)
    infer_sampler.start()

    if include_warmup_in_peak:
        for _ in range(max(0, warmup)):
            _ = sess.run(None, feeds)
    else:
        # warmup outside the monitored window
        for _ in range(max(0, warmup)):
            _ = sess.run(None, feeds)
        # restart sampler for steady-state peak
        infer_sampler.stop()
        infer_sampler = RssSampler(interval_ms=sample_interval_ms)
        rss_before_infer = proc.memory_info().rss
        infer_sampler.start()

    for _ in range(max(0, runs)):
        _ = sess.run(None, feeds)

    peak_rss_during_infer = infer_sampler.stop()
    rss_after_infer = proc.memory_info().rss

    return MemoryReport(
        rss_before_load=rss_before_load,
        rss_after_load=rss_after_load,
        peak_rss_during_load=peak_rss_during_load,
        rss_before_infer=rss_before_infer,
        peak_rss_during_infer=peak_rss_during_infer,
        rss_after_infer=rss_after_infer,
        include_warmup_in_peak=include_warmup_in_peak,
    )


# -------------------------------
# Main
# -------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--onnx", required=True, help="Path to ONNX model")

    # mode
    p.add_argument(
        "--mode",
        choices=["latency", "memory", "both"],
        default="latency",
        help="What to measure. Use separate runs for best practice: latency then memory.",
    )

    # input
    p.add_argument("--input-name", default=None, help="Model input name (default: first input)")
    p.add_argument(
        "--input-shape",
        default=None,
        help="Comma-separated shape, e.g. 1,3,224,224. Required for dynamic shapes.",
    )
    p.add_argument(
        "--input-dtype",
        default=None,
        help="Override input dtype: fp32|fp16|int64|int32|uint8",
    )
    p.add_argument("--fill", choices=["random", "zeros", "ones"], default="random")
    p.add_argument("--seed", type=int, default=0)

    # repetitions
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=200)

    # ORT session options
    p.add_argument("--intra-threads", type=int, default=0, help="0 = ORT default")
    p.add_argument("--inter-threads", type=int, default=0, help="0 = ORT default")
    p.add_argument("--execution-mode", choices=["sequential", "parallel"], default="sequential")
    p.add_argument("--opt-level", choices=["disable", "basic", "extended", "all"], default="all")
    p.add_argument("--mem-pattern", type=int, default=1, help="1=enable (default), 0=disable")
    p.add_argument("--cpu-arena", type=int, default=1, help="1=enable (default), 0=disable")

    # env threads (affects underlying kernels)
    p.add_argument("--omp-threads", type=int, default=None)
    p.add_argument("--mkl-threads", type=int, default=None)

    # memory measurement knobs
    p.add_argument("--mem-sample-ms", type=float, default=5.0, help="RSS sampler interval (ms)")
    p.add_argument(
        "--mem-include-warmup",
        action="store_true",
        help="If set, peak RSS during inference includes warmup phase (cold/first-run peak)",
    )

    # misc
    p.add_argument("--sync-barrier", action="store_true", help="Small sleep before timing runs")
    p.add_argument(
        "--json-out",
        default=None,
        help="If set, write full report JSON to this path.",
    )

    args = p.parse_args(argv)

    set_env_threads(args.omp_threads, args.mkl_threads)

    enable_mem_pattern = bool(args.mem_pattern)
    enable_cpu_mem_arena = bool(args.cpu_arena)

    sess_kwargs = dict(
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        execution_mode=args.execution_mode,
        opt_level=args.opt_level,
        enable_mem_pattern=enable_mem_pattern,
        enable_cpu_mem_arena=enable_cpu_mem_arena,
        providers=["CPUExecutionProvider"],
    )

    # create a session once for latency/both (memory mode creates its own to isolate load-phase)
    sess = None
    if args.mode in ("latency", "both"):
        sess = create_session(args.onnx, **sess_kwargs)

    # determine input spec
    if sess is None:
        # temp session to query input spec (memory mode will recreate for measured phases)
        tmp = create_session(args.onnx, **sess_kwargs)
        sess_for_spec = tmp
    else:
        sess_for_spec = sess

    input_shape = parse_shape(args.input_shape) if args.input_shape else None
    in_name, in_shape, in_dtype = resolve_input_spec(
        sess_for_spec,
        input_name=args.input_name,
        input_shape=input_shape,
        input_dtype=args.input_dtype,
    )

    x = make_input_array(in_shape, in_dtype, seed=args.seed, fill=args.fill)
    feeds = {in_name: x}

    # build report
    report: Dict[str, Any] = {
        "timestamp": now_iso(),
        "onnx": os.path.abspath(args.onnx),
        "mode": args.mode,
        "input": {
            "name": in_name,
            "shape": in_shape,
            "dtype": str(in_dtype),
            "fill": args.fill,
            "seed": args.seed,
        },
        "ort": {
            "onnxruntime_version": ort.__version__,
            "providers": ["CPUExecutionProvider"],
            "session_options": {
                "intra_threads": args.intra_threads,
                "inter_threads": args.inter_threads,
                "execution_mode": args.execution_mode,
                "opt_level": args.opt_level,
                "enable_mem_pattern": enable_mem_pattern,
                "enable_cpu_mem_arena": enable_cpu_mem_arena,
            },
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
        "warmup": args.warmup,
        "runs": args.runs,
    }

    # LATENCY
    if args.mode in ("latency", "both"):
        assert sess is not None
        lat = run_latency(sess, feeds, warmup=args.warmup, runs=args.runs, sync_barrier=args.sync_barrier)
        lat_sum = lat.summary()
        report["latency"] = {
            **lat_sum,
            "warmup_excluded": True,
            "n_runs": len(lat.times_ms),
        }
        # Print human-readable
        print("[LATENCY] ms:")
        for k in ("mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            if k in lat_sum:
                print(f"  {k}: {lat_sum[k]:.4f}")

    # MEMORY
    if args.mode in ("memory", "both"):
        mem = run_memory(
            onnx_path=args.onnx,
            sess_kwargs=sess_kwargs,
            feeds=feeds,
            warmup=args.warmup,
            runs=args.runs,
            sample_interval_ms=args.mem_sample_ms,
            include_warmup_in_peak=args.mem_include_warmup,
        )
        mem_dict = mem.as_dict()
        report["memory"] = mem_dict
        # Print human-readable
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
            print(f"  {k}: {mem_dict[k]:.3f}")
        print(f"  include_warmup_in_peak: {mem_dict['include_warmup_in_peak']}")
        print("  NOTE: For paper-grade peak RSS, prefer external: /usr/bin/time -v ... (Maximum resident set size)")

    # JSON output
    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[JSON] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
