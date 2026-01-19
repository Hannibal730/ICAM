python3 - << 'PY'
import glob, re, math, statistics

files = sorted(glob.glob("time_*.txt"))
if not files:
    raise SystemExit("No files matched: time_*.txt")

ansi = re.compile(r"\x1b\[[0-9;]*m")
pat  = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", re.I)

vals_kb = []
rows = []

for fn in files:
    txt = open(fn, "r", encoding="utf-8", errors="ignore").read()
    txt = ansi.sub("", txt)  # ANSI 색 코드 제거
    m = pat.search(txt)
    if not m:
        raise SystemExit(f"Max RSS not found in {fn}")
    kb = int(m.group(1))
    vals_kb.append(kb)
    rows.append((fn, kb, kb/1024.0))

# 논문 표에는 보통 MB로 보고
vals_mb = [kb/1024.0 for kb in vals_kb]
n = len(vals_mb)

mean = statistics.mean(vals_mb)
# 표준편차: 논문에서는 보통 'sample std'(N-1) 사용
std_sample = statistics.stdev(vals_mb) if n > 1 else 0.0
# 참고용: population std (N)도 함께
std_pop = statistics.pstdev(vals_mb) if n > 1 else 0.0

print("Per-run Peak RSS (Max RSS from /usr/bin/time -v):")
for fn, kb, mb in rows:
    print(f"  {fn}: {kb} kB = {mb:.3f} MB")

print("\nSummary (table-ready):")
print(f"  Peak RSS (MB): {mean:.3f} ± {std_sample:.3f}  (sample std, n={n})")
print(f"  Peak RSS (MB): {mean:.3f} ± {std_pop:.3f}  (population std, n={n})")
print(f"  min/max (MB): {min(vals_mb):.3f} / {max(vals_mb):.3f}")
PY
