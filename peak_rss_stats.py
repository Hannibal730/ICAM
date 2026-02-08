#!/usr/bin/env python3
"""
Summarize Peak RSS from `/usr/bin/time -v` output files.

Typical usage with the benchmark loop:
  python3 peak_rss_stats.py

If your files use a different naming pattern:
  python3 peak_rss_stats.py --glob "time_*.txt"
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from typing import Any, Dict, List, Optional


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MAX_RSS_RE = re.compile(
    r"Maximum resident set size \(kbytes\):\s*(\d+)",
    flags=re.IGNORECASE,
)


def parse_maxrss_kb(text: str) -> Optional[int]:
    cleaned = ANSI_RE.sub("", text)
    m = MAX_RSS_RE.search(cleaned)
    if not m:
        return None
    return int(m.group(1))


def summarize(vals_mb: List[float]) -> Dict[str, float]:
    n = len(vals_mb)
    mean = float(statistics.mean(vals_mb))
    std_sample = float(statistics.stdev(vals_mb)) if n > 1 else 0.0
    var_sample = float(statistics.variance(vals_mb)) if n > 1 else 0.0
    std_pop = float(statistics.pstdev(vals_mb)) if n > 1 else 0.0
    var_pop = float(statistics.pvariance(vals_mb)) if n > 1 else 0.0
    return {
        "n": float(n),
        "mean_mb": mean,
        "std_sample_mb": std_sample,
        "var_sample_mb2": var_sample,
        "std_pop_mb": std_pop,
        "var_pop_mb2": var_pop,
        "min_mb": float(min(vals_mb)),
        "max_mb": float(max(vals_mb)),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute Peak RSS statistics from /usr/bin/time -v outputs."
    )
    p.add_argument(
        "--glob",
        default="run_*.time",
        help='File glob to parse, e.g. "run_*.time" or "time_*.txt".',
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save detailed summary JSON.",
    )
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows: List[Dict[str, Any]] = []
    vals_mb: List[float] = []

    for fn in files:
        with open(fn, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        kb = parse_maxrss_kb(txt)
        if kb is None:
            raise SystemExit(f"Max RSS not found in {fn}")

        mb = kb / 1024.0
        rows.append({"file": fn, "max_rss_kb": kb, "max_rss_mb": mb})
        vals_mb.append(mb)

    s = summarize(vals_mb)

    print("Per-run Peak RSS (Maximum resident set size):")
    for r in rows:
        print(
            f"  {r['file']}: {r['max_rss_kb']} kB = {r['max_rss_mb']:.3f} MB"
        )

    print("\nSummary (table-ready):")
    print(
        f"  Peak RSS (MB): {s['mean_mb']:.3f} +/- {s['std_sample_mb']:.3f} "
        f"(sample std, n={int(s['n'])})"
    )
    print(
        f"  Peak RSS (MB): {s['mean_mb']:.3f} +/- {s['std_pop_mb']:.3f} "
        f"(population std, n={int(s['n'])})"
    )
    print(f"  sample var (MB^2): {s['var_sample_mb2']:.6f}")
    print(f"  min/max (MB): {s['min_mb']:.3f} / {s['max_mb']:.3f}")
    print(
        "  LaTeX: "
        f"{s['mean_mb']:.2f} \\\\pm {s['std_sample_mb']:.2f}"
    )

    if args.json_out:
        out = {
            "glob": args.glob,
            "files": rows,
            "summary": s,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[JSON] wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

