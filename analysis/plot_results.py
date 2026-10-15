"""Summary table, LaTeX tables and figures of the report.

Usage (from any directory):  python analysis/plot_results.py [results_dir]

Reads   raw_measurements.csv, process_memory.csv, run_status.csv
Writes  summary.csv, plot_time.png, plot_cost_per_op.png, plot_memory.png,
        plot_peak_memory.png and tables/*.tex (included by the report)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, LogLocator, NullLocator

RESULTS_DIR = (sys.argv[1] if len(sys.argv) > 1 else
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

COLORS = {"python": "#e07b39", "java": "#4c8eda", "c": "#5baa5b"}
LABELS = {"python": "Python (CPython)", "java": "Java (HotSpot JIT)", "c": "C (gcc -O2)"}
ORDER = ["python", "java", "c"]
TABLE_ORDER = ["c", "java", "python"]
ANCHOR_N = 100          # the cubic reference lines go through the median at this size
MIB = 2 ** 20


def path(*parts):
    return os.path.join(RESULTS_DIR, *parts)


def add_grid(ax, ns):
    """One labelled tick per measured n, one horizontal line per decade."""
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) < 4:   # few decades: add 2 and 5 marks so values are easy to read
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=30))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        nice = [m * 10.0 ** e for e in range(-6, 7) for m in (1, 2, 5)]
        ax.set_ylim(bottom=max(v for v in nice if v <= lo))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)


# --------------------------------------------------------------------------- data

def load_raw():
    return pd.read_csv(path("raw_measurements.csv"), float_precision="round_trip")


def load_peaks():
    """Peak working set of the timing processes (the tracemalloc processes are excluded)."""
    peaks = pd.read_csv(path("process_memory.csv"))
    return peaks[peaks["mode"] == "run"]


def summarise(df, peaks):
    """One row per (language, n). Time statistics use the measured runs of all processes;
    warm-up, check and memory runs are excluded."""
    measured = df[df["run_type"] == "measured"]
    summary = (
        measured.groupby(["language", "n"])
        .agg(runs=("time_seconds", "size"),
             processes=("process", "nunique"),
             batch_size=("batch_size", "max"),
             median_time=("time_seconds", "median"),
             q1_time=("time_seconds", lambda x: x.quantile(0.25)),
             q3_time=("time_seconds", lambda x: x.quantile(0.75)))
        .reset_index()
    )
    # spread of the per-process medians, relative to the overall median
    per_process = measured.groupby(["language", "n", "process"])["time_seconds"].median()
    spread = per_process.groupby(["language", "n"]).agg(lambda x: x.max() - x.min())
    summary = summary.merge(spread.rename("process_spread").reset_index(), on=["language", "n"])

    # language-specific memory metric: C computed, Java heap delta, Python tracemalloc peak
    lang_mem = df[((df["language"] == "c") & (df["run_type"] == "measured"))
                  | ((df["language"] == "java") & df["run_type"].isin(["measured", "check"]))
                  | ((df["language"] == "python") & (df["run_type"] == "memory"))]
    mem = lang_mem.dropna(subset=["memory_bytes"]).groupby(["language", "n"])["memory_bytes"]
    summary = summary.merge(mem.median().rename("median_mem").reset_index(),
                            on=["language", "n"], how="left")

    peak = peaks.groupby(["language", "n"])["peak_bytes"].median().rename("peak_bytes")
    summary = summary.merge(peak.reset_index(), on=["language", "n"], how="left")

    summary["iqr_time"] = summary["q3_time"] - summary["q1_time"]
    summary["iqr_over_median_pct"] = 100 * summary["iqr_time"] / summary["median_time"]
    summary["process_spread_pct"] = 100 * summary["process_spread"] / summary["median_time"]
    summary["ns_per_multiply_add"] = summary["median_time"] * 1e9 / summary["n"] ** 3
    c_time = summary[summary["language"] == "c"].set_index("n")["median_time"]
    summary["ratio_to_c"] = summary["median_time"] / summary["n"].map(c_time)
    summary["bytes_per_element"] = summary["median_mem"] / summary["n"] ** 2
    return summary


def status_table():
    return pd.read_csv(path("run_status.csv"))


# --------------------------------------------------------------------------- figures

def save(fig, name):
    fig.tight_layout()
    fig.savefig(path(name), dpi=150)
    plt.close(fig)
    print(f"Saved {path(name)}")


def plot_time(summary):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n")
        anchor = d[d["n"] == ANCHOR_N]["median_time"].iloc[0]
        ax.plot(d["n"], anchor * (d["n"] / ANCHOR_N) ** 3, color=COLORS[lang],
                linewidth=0.8, linestyle="--", alpha=0.6)
        ax.errorbar(d["n"], d["median_time"],
                    yerr=[d["median_time"] - d["q1_time"], d["q3_time"] - d["median_time"]],
                    label=LABELS[lang], color=COLORS[lang], marker="o", capsize=4,
                    linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time per multiplication (s, log scale)", fontsize=12)
    ax.set_title(f"Runtime vs matrix size (dashed: n³ through n = {ANCHOR_N})", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, np.sort(summary["n"].unique()))
    save(fig, "plot_time.png")


def plot_cost_per_op(summary):
    """Time / n³ in nanoseconds per multiply-add. Flat = perfect cubic scaling."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n")
        ax.plot(d["n"], d["ns_per_multiply_add"], label=LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time / n³ (ns per multiply-add, log scale)", fontsize=12)
    ax.set_title("Cost per operation (flat line = perfect cubic scaling)", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, np.sort(summary["n"].unique()))
    save(fig, "plot_cost_per_op.png")


MEMORY_LABELS = {"c": "C: computed 3×8n²", "java": "Java: heap delta (result matrix)",
                 "python": "Python: tracemalloc peak (result matrix)"}


def plot_memory(summary):
    """Language-specific metrics: different quantities, labelled as such."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    all_n = np.sort(summary["n"].unique())
    ax.plot(all_n, 8 * all_n ** 2 / 1024, color="gray", linewidth=1, linestyle="--",
            label="8n² bytes (one float64 matrix)")
    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n").dropna(subset=["median_mem"])
        ax.plot(d["n"], d["median_mem"] / 1024, label=MEMORY_LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Memory (KiB, log scale)", fontsize=12)
    ax.set_title("Language-specific memory metrics (not comparable)", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, all_n)
    save(fig, "plot_memory.png")


def plot_peak_memory(summary):
    """Common metric: peak working set of the whole benchmark process."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    all_n = np.sort(summary["n"].unique())
    ax.plot(all_n, 24 * all_n ** 2 / MIB, color="gray", linewidth=1, linestyle="--",
            label="24n² bytes (three float64 matrices)")
    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n").dropna(subset=["peak_bytes"])
        ax.plot(d["n"], d["peak_bytes"] / MIB, label=LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Peak working set of the process (MiB, log scale)", fontsize=12)
    ax.set_title("Peak process memory (same metric for the three languages)", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, all_n)
    ax.set_ylim(bottom=1)
    save(fig, "plot_peak_memory.png")


# --------------------------------------------------------------------------- LaTeX tables

def fmt_time(t):
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 0.01:
        return f"{t:#.3g}"
    mantissa, exponent = f"{t:.2e}".split("e")
    return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"


def fmt_sig(x, digits=3):
    return "---" if pd.isna(x) else f"{float(f'{x:.{digits}g}'):g}"


def write_table(name, lines):
    os.makedirs(path("tables"), exist_ok=True)
    with open(path("tables", name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path('tables', name)}")


def cell(summary, lang, n, column):
    row = summary[(summary["language"] == lang) & (summary["n"] == n)]
    return None if row.empty else row.iloc[0][column]


def table_runtime(summary, status):
    lines = []
    for n in sorted(status["n"].unique()):
        cells, runs = [], []
        for lang in TABLE_ORDER:
            t = cell(summary, lang, n, "median_time")
            if t is None:
                st = status[(status["language"] == lang) & (status["n"] == n)
                            & (status["process"] == 1) & (status["mode"] == "run")]
                cells.append(st["status"].iloc[0] if not st.empty else "---")
                runs.append("--")
            else:
                iqr = cell(summary, lang, n, "iqr_over_median_pct")
                cells.append(f"{fmt_time(t)} ({iqr:.1f}\\%)")
                runs.append(str(int(cell(summary, lang, n, "runs"))))
        lines.append(f"{n} & " + " & ".join(cells) + f" & {'/'.join(runs)} \\\\")
    write_table("runtime.tex", lines)


def table_perop(summary):
    lines = []
    for n in sorted(summary["n"].unique()):
        ns = [fmt_sig(cell(summary, lang, n, "ns_per_multiply_add")) for lang in TABLE_ORDER]
        ratios = [fmt_sig(cell(summary, lang, n, "ratio_to_c"), 2 if lang == "java" else 3)
                  for lang in ["java", "python"]]
        lines.append(f"{n} & " + " & ".join(ns + ratios) + " \\\\")
    write_table("perop.tex", lines)


def table_memory(summary):
    """Peak working set (MiB) and language-specific metrics (KiB) side by side."""
    def fmt(v, unit):
        return "---" if v is None or pd.isna(v) else f"{v / unit:.1f}"

    lines = []
    for n in sorted(summary["n"].unique()):
        peak = [fmt(cell(summary, lang, n, "peak_bytes"), MIB) for lang in TABLE_ORDER]
        own = [fmt(cell(summary, lang, n, "median_mem"), 1024) for lang in TABLE_ORDER]
        lines.append(f"{n} & " + " & ".join(peak + own) + " \\\\")
    lines.append("\\midrule")
    big = summary[summary["n"] >= 200]
    ranges = []
    for lang in TABLE_ORDER:
        b = big[big["language"] == lang]["bytes_per_element"].dropna()
        lo, hi = b.min(), b.max()
        ranges.append(f"{lo:.1f}" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}--{hi:.1f}")
    lines.append("\\multicolumn{4}{l}{\\textit{Bytes / $n^2$ for $n \\ge 200$}} & "
                 + " & ".join(ranges) + " \\\\")
    write_table("memory.tex", lines)


if __name__ == "__main__":
    raw = load_raw()
    summary = summarise(raw, load_peaks())
    summary.to_csv(path("summary.csv"), index=False, float_format="%.6g")
    print(f"Saved {path('summary.csv')}")
    plot_time(summary)
    plot_cost_per_op(summary)
    plot_memory(summary)
    plot_peak_memory(summary)
    status = status_table()
    table_runtime(summary, status)
    table_perop(summary)
    table_memory(summary)

