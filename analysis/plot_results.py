import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, LogLocator, NullLocator
import os

RESULTS_DIR = os.path.join("..", "results")
FIGURES_DIR = os.path.join("..", "results")

COLORS = {"python": "#e07b39", "java": "#4c8eda", "c": "#5baa5b"}
LABELS = {"python": "Python (CPython)", "java": "Java (HotSpot JIT)", "c": "C (gcc -O2)"}
ORDER  = ["python", "java", "c"]


def add_grid(ax, ns):
    """Clean grid: one vertical line per measured n (labelled), one horizontal line per decade."""
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


def load_data():
    return pd.read_csv(os.path.join(RESULTS_DIR, "raw_measurements.csv"), float_precision="round_trip")


def summarise(df):
    """Time statistics from the measured runs only (warm-ups excluded).

    Memory: C and Java report it in every measured run; Python in a separate memory run.
    """
    measured = df[df["run_type"] == "measured"]
    summary = (
        measured.groupby(["language", "n"])
        .agg(
            runs=("time_seconds", "size"),
            median_time=("time_seconds", "median"),
            q1_time=("time_seconds", lambda x: x.quantile(0.25)),
            q3_time=("time_seconds", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    mem_runs = df[((df["language"] != "python") & (df["run_type"] == "measured"))
                  | ((df["language"] == "python") & (df["run_type"] == "memory"))]
    mem = mem_runs.groupby(["language", "n"])["memory_bytes"].median().rename("median_mem")
    return summary.merge(mem.reset_index(), on=["language", "n"], how="left")


def plot_time(summary):
    fig, ax = plt.subplots(figsize=(7, 5))

    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n")
        d = d[d["median_time"] > 0]          # skip unmeasurable points (t=0)
        if d.empty:
            continue

        # cubic reference line anchored at the smallest point
        n0, t0 = d["n"].iloc[0], d["median_time"].iloc[0]
        n_ref = np.array(d["n"])
        t_ref = t0 * (n_ref / n0) ** 3
        ax.plot(n_ref, t_ref, color=COLORS[lang], linewidth=0.8,
                linestyle="--", alpha=0.6)

        yerr_low  = d["median_time"] - d["q1_time"]
        yerr_high = d["q3_time"]     - d["median_time"]
        ax.errorbar(
            d["n"], d["median_time"],
            yerr=[yerr_low, yerr_high],
            label=LABELS[lang], color=COLORS[lang],
            marker="o", capsize=4, linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Execution time (s, log scale)", fontsize=12)
    ax.set_title("Runtime vs matrix size", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, np.sort(summary["n"].unique()))

    path = os.path.join(FIGURES_DIR, "plot_time.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_memory(summary):
    fig, ax = plt.subplots(figsize=(7, 5))

    # theoretical raw storage: 3 * 8 * n^2 bytes → KB
    all_n = np.sort(summary["n"].unique())
    raw_kb = 3 * 8 * all_n ** 2 / 1024
    ax.plot(all_n, raw_kb, color="gray", linewidth=1, linestyle="--",
            label="3 × 8n² bytes (raw storage)")

    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n").dropna(subset=["median_mem"])
        mem_kb = d["median_mem"] / 1024
        ax.plot(d["n"], mem_kb, label=LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Memory (KB, log scale)", fontsize=12)
    ax.set_title("Memory usage vs matrix size", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, np.sort(summary["n"].unique()))

    path = os.path.join(FIGURES_DIR, "plot_memory.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_cost_per_op(summary):
    """Time / n³ in nanoseconds per multiply-add. Flat = perfect cubic scaling."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for lang in ORDER:
        d = summary[summary["language"] == lang].sort_values("n")
        d = d[d["median_time"] > 0]
        if d.empty:
            continue
        ns_per_op = d["median_time"] * 1e9 / (d["n"] ** 3)
        ax.plot(d["n"], ns_per_op, label=LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Median time / n³ (ns per multiply-add, log scale)", fontsize=12)
    ax.set_title("Cost per operation — flat line = perfect cubic scaling", fontsize=12)
    ax.legend(fontsize=9)
    add_grid(ax, np.sort(summary["n"].unique()))

    path = os.path.join(FIGURES_DIR, "plot_cost_per_op.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def save_summary(summary):
    """Writes the table used in the report: one row per (language, n)."""
    out = summary.copy()
    out["iqr_time"] = out["q3_time"] - out["q1_time"]
    out["iqr_over_median_pct"] = 100 * out["iqr_time"] / out["median_time"]
    out["ns_per_multiply_add"] = out["median_time"] * 1e9 / out["n"] ** 3
    c_time = out[out["language"] == "c"].set_index("n")["median_time"]
    out["ratio_to_c"] = out["median_time"] / out["n"].map(c_time)
    out["bytes_per_element"] = out["median_mem"] / out["n"] ** 2
    path = os.path.join(RESULTS_DIR, "summary.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"Saved {path}")


def print_summary_table(summary):
    print("\n--- Runtime summary (median seconds) ---")
    pivot = summary.pivot_table(index="n", columns="language", values="median_time")
    pivot = pivot.rename(columns=LABELS)
    print(pivot.to_string(float_format="{:.6f}".format))

    print("\n--- Memory summary (median KB) ---")
    pivot_mem = summary.pivot_table(index="n", columns="language", values="median_mem") / 1024
    pivot_mem = pivot_mem.rename(columns=LABELS)
    print(pivot_mem.to_string(float_format="{:.1f}".format))

    print("\n--- Runtime relative to C (ratio of medians) ---")
    raw = summary.pivot_table(index="n", columns="language", values="median_time")
    raw = raw[raw["c"] > 0]
    print(raw[["java", "python"]].div(raw["c"], axis=0).to_string(float_format="{:.1f}".format))

    print("\n--- Fitted growth exponent, slope of log(time) vs log(n) for n >= 100 ---")
    for lang in ORDER:
        d = summary[(summary["language"] == lang) & (summary["n"] >= 100)]
        slope = np.polyfit(np.log10(d["n"]), np.log10(d["median_time"]), 1)[0]
        print(f"{LABELS[lang]}: {slope:.2f}")

    print("\n--- Memory per matrix element (median bytes / n^2) ---")
    raw_mem = summary.pivot_table(index="n", columns="language", values="median_mem")
    print(raw_mem.div(raw_mem.index.to_series() ** 2, axis=0).to_string(float_format="{:.1f}".format))


if __name__ == "__main__":
    df = load_data()
    summary = summarise(df)
    print_summary_table(summary)
    save_summary(summary)
    plot_time(summary)
    plot_memory(summary)
    plot_cost_per_op(summary)
