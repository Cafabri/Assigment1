"""Every number quoted in the text of the report, computed from the raw data.

Usage (from any directory):  python analysis/report_numbers.py [results_dir]

Writes  report_numbers.txt  growth exponents (global and by segment), doubling factors,
        variability, process-to-process spread, warm-up and tracemalloc ratios,
        allocation share of the kernel time and peak-memory slopes.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (TABLE_ORDER, load_peaks, load_raw, path,  # noqa: E402
                          summarise)

MIB = 2 ** 20
out = []


def say(text=""):
    out.append(text)
    print(text)


def median_time(summary, lang, n):
    row = summary[(summary["language"] == lang) & (summary["n"] == n)]
    return None if row.empty else row["median_time"].iloc[0]


def fitted_exponent(d):
    return np.polyfit(np.log(d["n"]), np.log(d["median_time"]), 1)[0]


def main():
    raw = load_raw()
    summary = summarise(raw, load_peaks())

    say("== Growth exponent: least-squares slope of log(median time) vs log(n)")
    for lo, hi in [(100, 1000), (1000, 10 ** 9)]:
        parts = []
        for lang in TABLE_ORDER:
            d = summary[(summary["language"] == lang) & summary["n"].between(lo, hi)]
            if len(d) >= 2:
                parts.append(f"{lang} {fitted_exponent(d):.2f} (n={list(d['n'])})")
        say(f"  {lo} <= n <= {hi if hi < 10 ** 9 else 'max'}: " + "; ".join(parts))

    say("\n== Exponent by segment, log(t2/t1)/log(n2/n1), with ns per multiply-add at n2")
    for lang in TABLE_ORDER:
        d = summary[summary["language"] == lang].sort_values("n")
        segs = []
        for (n1, t1), (n2, t2) in zip(d[["n", "median_time"]].values[:-1],
                                      d[["n", "median_time"]].values[1:]):
            segs.append(f"{int(n1)}-{int(n2)}: {np.log(t2 / t1) / np.log(n2 / n1):.2f} "
                        f"[{t2 * 1e9 / n2 ** 3:.3g} ns]")
        say(f"  {lang}: " + ", ".join(segs))

    # LaTeX rows of the segment table (n >= 500)
    rows = []
    edges = sorted(summary[summary["n"] >= 500]["n"].unique())
    for n1, n2 in zip(edges[:-1], edges[1:]):
        cells = []
        for lang in TABLE_ORDER:
            t1, t2 = median_time(summary, lang, n1), median_time(summary, lang, n2)
            if t1 and t2:
                ns = t2 * 1e9 / n2 ** 3
                cells += [f"{np.log(t2 / t1) / np.log(n2 / n1):.2f}",
                          f"{ns:.2f}" if ns < 10 else f"{ns:.1f}"]
            else:
                cells += ["---", "---"]
        rows.append(f"{n1}--{n2} & " + " & ".join(cells) + " \\\\")
    os.makedirs(path("tables"), exist_ok=True)
    with open(path("tables", "segments.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    say("\n== Time factor when n doubles (cubic model: 8)")
    for n1, n2 in [(50, 100), (100, 200), (500, 1000), (750, 1500), (1000, 2000),
                   (1500, 3000)]:
        parts = []
        for lang in TABLE_ORDER:
            t1, t2 = median_time(summary, lang, n1), median_time(summary, lang, n2)
            if t1 and t2:
                parts.append(f"{lang} {t2 / t1:.1f}")
        say(f"  {n1} -> {n2}: " + ", ".join(parts))

    say("\n== Ratio to C (median / median of C)")
    for lo, hi in [(10, 100), (100, 1000), (1500, 1500), (2000, 3000), (10, 10 ** 9)]:
        parts = []
        for lang in ["java", "python"]:
            r = summary[(summary["language"] == lang) & summary["n"].between(lo, hi)]["ratio_to_c"]
            if len(r):
                parts.append(f"{lang} {r.min():.2f}-{r.max():.2f}")
        say(f"  {lo} <= n <= {hi}: " + ", ".join(parts))

    say("\n== Variability")
    long = summary[summary["median_time"] > 1e-3]
    worst = long.loc[long["iqr_over_median_pct"].idxmax()]
    say(f"  configurations: {len(summary)}; max IQR/median among medians > 1 ms: "
        f"{worst['iqr_over_median_pct']:.2f} % ({worst['language']}, n={worst['n']})")
    worst_all = summary.loc[summary["iqr_over_median_pct"].idxmax()]
    say(f"  max IQR/median overall: {worst_all['iqr_over_median_pct']:.2f} % "
        f"({worst_all['language']}, n={worst_all['n']})")
    say("  largest IQR/median: " + ", ".join(
        f"{r.language} n={r.n} {r.iqr_over_median_pct:.1f} %"
        for r in summary.nlargest(4, "iqr_over_median_pct").itertuples()))
    say("  largest spread of process medians: " + ", ".join(
        f"{r.language} n={r.n} {r.process_spread_pct:.1f} %"
        for r in summary.nlargest(4, "process_spread_pct").itertuples()))
    multi = summary[summary["processes"] > 1]
    w = multi.loc[multi["process_spread_pct"].idxmax()]
    say(f"  configurations with >1 process: {len(multi)}; max (max-min) of the process "
        f"medians: {w['process_spread_pct']:.2f} % of the median ({w['language']}, n={w['n']})")
    say(f"  runs per configuration: " + ", ".join(
        f"{lang} {sorted(set(summary[summary['language'] == lang]['runs']))}"
        for lang in TABLE_ORDER))
    say(f"  batch sizes (n <= 100): " + ", ".join(
        f"{r.language} n={r.n}: {int(r.batch_size)}"
        for r in summary[summary["n"] <= 100].itertuples()))

    say("\n== Java n = 1500 and neighbours: median of each process (s)")
    m = raw[(raw["run_type"] == "measured") & (raw["language"] == "java")]
    for n in [1000, 1500, 2000]:
        per = m[m["n"] == n].groupby("process")["time_seconds"].median()
        say(f"  n={n}: " + ", ".join(f"p{p} {t:.4g}" for p, t in per.items()))

    say("\n== Warm-up")
    warm = raw[raw["run_type"] == "warmup"]
    for lang in TABLE_ORDER:
        w = warm[warm["language"] == lang]
        if w.empty:
            continue
        parts = []
        for n in sorted(w["n"].unique()):
            first = w[(w["n"] == n) & (w["repetition"] == 1)]["time_seconds"].median()
            count = w[w["n"] == n].groupby("process")["repetition"].max()
            parts.append(f"n={n}: first/median {first / median_time(summary, lang, n):.2f}"
                         + (f", warm-ups {sorted(set(count))}" if lang == "java" else ""))
        say(f"  {lang}: " + "; ".join(parts))
    jw = warm[warm["language"] == "java"]
    unstable = []
    for (n, p), g in jw.groupby(["n", "process"]):
        t = g.sort_values("repetition")["time_seconds"].values
        if len(t) < 2 or abs(t[-1] - t[-2]) / t[-2] >= 0.02:
            unstable.append(f"n={n} p{p}")
    say(f"  Java processes that reached the warm-up maximum without stabilising: "
        f"{unstable or 'none'}")

    say("\n== tracemalloc run time / median timed run (Python)")
    mem = raw[(raw["language"] == "python") & (raw["run_type"] == "memory")]
    ratios = {int(r.n): r.time_seconds / median_time(summary, "python", r.n)
              for r in mem.itertuples()}
    say("  " + ", ".join(f"n={n}: {v:.1f}" for n, v in ratios.items()))
    big = {n: v for n, v in ratios.items() if n >= 100}
    if big:
        say(f"  range for n >= 100: {min(big.values()):.1f}-{max(big.values()):.1f}")

    say("\n== Allocation of the result matrix alone, median / median kernel time")
    alloc_file = path("allocation_times.csv")
    if os.path.exists(alloc_file):
        alloc = pd.read_csv(alloc_file).groupby(["language", "n"])["time_seconds"].median()
        for lang in ["java", "python"]:
            parts = []
            for n, t in alloc[lang].items():
                k = median_time(summary, lang, n)
                if k:
                    parts.append(f"n={n}: {t:.3g} s = {100 * t / k:.3g} %")
            say(f"  {lang}: " + "; ".join(parts))

    say("\n== Peak working set of the timing processes (median over processes)")
    for lang in TABLE_ORDER:
        d = summary[summary["language"] == lang].dropna(subset=["peak_bytes"]).sort_values("n")
        base = d["peak_bytes"].iloc[0]
        large = d[d["n"] >= 1000]
        slope = np.polyfit(large["n"] ** 2, large["peak_bytes"], 1)[0] if len(large) >= 2 else np.nan
        say(f"  {lang}: n={int(d['n'].iloc[0])} {base / MIB:.1f} MiB, "
            f"max {d['peak_bytes'].max() / MIB:.1f} MiB at n={int(d.loc[d['peak_bytes'].idxmax(), 'n'])}; "
            f"slope vs n^2 for n >= 1000: {slope:.1f} B/element; (peak - peak at n=10)/n^2: "
            + ", ".join(f"n={int(r.n)} {(r.peak_bytes - base) / r.n ** 2:.1f}"
                        for r in d[d["n"] >= 500].itertuples()))
    top = summary.loc[summary["peak_bytes"].idxmax()]
    say(f"  largest peak: {top['peak_bytes'] / MIB:.1f} MiB ({top['language']}, n={top['n']})")

    with open(path("report_numbers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSaved {path('report_numbers.txt')}")


if __name__ == "__main__":
    main()
