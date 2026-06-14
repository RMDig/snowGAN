#!/usr/bin/env python3
"""Summarize a memtrace JSONL (written by snowgan.memtrace during a real run).

    python scripts/analyze_memtrace.py keras/snowgan/core/memtrace-<pid>.jsonl

Reports the overall RSS leak rate, regresses each native counter to say WHERE
the bytes go (live arena vs mmap vs fragmentation), averages the per-substep
attribution, and prints a verdict against the four signatures documented in
src/snowgan/memtrace.py. Pure stdlib -- runs anywhere, no TF import.
"""
import argparse
import json
import sys


def _linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy * sxy / (sxx * syy)) if syy > 0 else None
    return slope, r2


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a torn last line from an OOM-kill mid-write
    return rows


def _slope_per_batch(rows, getter):
    xs, ys = [], []
    for r in rows:
        v = getter(r)
        if v is not None:
            xs.append(float(r["batch"]))
            ys.append(float(v))
    return _linfit(xs, ys)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="memtrace JSONL file")
    args = ap.parse_args(argv)

    rows = _load(args.path)
    if len(rows) < 2:
        print(f"Need >=2 records to fit a trend; found {len(rows)}.")
        return 1

    b0, b1 = rows[0]["batch"], rows[-1]["batch"]
    span_h = (rows[-1]["ts"] - rows[0]["ts"]) / 3600.0
    print(f"== memtrace: {len(rows)} snapshots, batches {b0}..{b1}, {span_h:.2f}h ==\n")

    rss0, rss1 = rows[0]["rss_mib"], rows[-1]["rss_mib"]
    rss_slope, rss_r2 = _slope_per_batch(rows, lambda r: r.get("rss_mib"))
    print(f"RSS: {rss0:.0f} -> {rss1:.0f} MiB  (delta {rss1 - rss0:+.0f})")
    if rss_slope is not None:
        per_h = rss_slope * (b1 - b0) / span_h if span_h > 0 else float("nan")
        print(f"  leak rate: {rss_slope:+.4f} MiB/batch  (~{per_h:+.0f} MiB/h)  R^2={rss_r2:.3f}\n")

    # Which native counter tracks the RSS climb?
    counters = {
        "uordblks (live arena)": lambda r: (r.get("mallinfo2") or {}).get("uordblks_mib"),
        "fordblks (free/frag) ": lambda r: (r.get("mallinfo2") or {}).get("fordblks_mib"),
        "hblkhd   (mmap'd)    ": lambda r: (r.get("mallinfo2") or {}).get("hblkhd_mib"),
        "arena                ": lambda r: (r.get("mallinfo2") or {}).get("arena_mib"),
        "maps_count (VMAs)    ": lambda r: r.get("maps_count"),
        "smaps anon           ": lambda r: (r.get("smaps") or {}).get("anon_mib"),
        "tracemalloc (python) ": lambda r: r.get("tracemalloc_cur_mib"),
    }
    print("Native counters (slope per batch, R^2):")
    slopes = {}
    for name, get in counters.items():
        s, r2 = _slope_per_batch(rows, get)
        slopes[name.strip()] = s
        if s is not None:
            unit = "VMAs" if "maps" in name else "MiB"
            print(f"  {name}: {s:+.4f} {unit}/batch  R^2={r2 if r2 is None else round(r2, 3)}")
    print()

    # Per-substep attribution (disc loop / gen loop / post-step).
    subs = [r.get("substeps") for r in rows if r.get("substeps")]
    if subs:
        print("Per-substep mean RSS delta (MiB/step):")
        for key in ("disc_loop_mib", "gen_loop_mib", "poststep_mib", "total_mib"):
            m = _mean([s.get(key) for s in subs])
            if m is not None:
                print(f"  {key:14s}: {m:+.3f}")
        print()

    # malloc_trim reclaimability, if the run enabled SNOWGAN_MEMTRACE_TRIM.
    trims = [r["trim"]["reclaimed_mib"] for r in rows if r.get("trim")]
    if trims:
        print(f"malloc_trim reclaimed: mean {_mean(trims):+.2f} MiB/tick "
              f"(max {max(trims):+.2f})\n")

    # Verdict heuristic.
    rss_s = rss_slope or 0.0
    uord = slopes.get("uordblks (live arena)") or 0.0
    ford = slopes.get("fordblks (free/frag)") or 0.0
    hblk = slopes.get("hblkhd   (mmap'd)") or 0.0
    maps = slopes.get("maps_count (VMAs)") or 0.0
    tmal = slopes.get("tracemalloc (python)")
    print("Verdict:")
    if abs(rss_s) < 0.05:
        print("  RSS is essentially flat -- no meaningful leak in this window.")
    elif tmal is not None and tmal > 0.5 * rss_s and rss_s > 0:
        print("  PYTHON leak -- tracemalloc tracks RSS. Check top_growers in the JSONL.")
    elif hblk > 0.3 * rss_s or maps > 0.5:
        print("  MMAP growth -- unbounded mmap regions (hblkhd / maps_count rising).")
        print("  Signature of the cuDNN/cuda JIT module-cache-per-step (Blackwell) theory.")
    elif uord > 0.5 * rss_s:
        print("  NATIVE arena leak -- live in-use bytes (uordblks) climb with RSS.")
        print("  malloc_trim won't help (bytes are live). See substep attribution above.")
    elif ford > 0.5 * rss_s:
        print("  FRAGMENTATION -- free-in-arena (fordblks) grows; malloc_trim should reclaim.")
    else:
        print("  RSS climbs but no single counter dominates -- inspect the JSONL directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
