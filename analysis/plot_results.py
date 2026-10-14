import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join("..", "results")
FIGURES_DIR = os.path.join("..", "results")

COLORS = {"python": "#e07b39", "java": "#4c8eda", "c": "#5baa5b"}
LABELS = {"python": "Python (CPython)", "java": "Java (HotSpot JIT)", "c": "C (gcc -O2)"}
ORDER  = ["python", "java", "c"]


def load_data():
    files = ["results_python.csv", "results_java.csv", "results_c.csv"]
    frames = []
    for f in files:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def summarise(df):
    return (
        df.groupby(["language", "n"])
        .agg(
            median_time=("time_seconds", "median"),
            q1_time=("time_seconds", lambda x: x.quantile(0.25)),
            q3_time=("time_seconds", lambda x: x.quantile(0.75)),
            median_mem=("memory_bytes", "median"),
            q1_mem=("memory_bytes", lambda x: x.quantile(0.25)),
            q3_mem=("memory_bytes", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )


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
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

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
        d = summary[summary["language"] == lang].sort_values("n")
        mem_kb = d["median_mem"] / 1024
        ax.plot(d["n"], mem_kb, label=LABELS[lang], color=COLORS[lang],
                marker="o", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (log scale)", fontsize=12)
    ax.set_ylabel("Memory (KB, log scale)", fontsize=12)
    ax.set_title("Memory usage vs matrix size", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

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
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    path = os.path.join(FIGURES_DIR, "plot_cost_per_op.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def print_summary_table(summary):
    print("\n--- Runtime summary (median seconds) ---")
    pivot = summary.pivot_table(index="n", columns="language", values="median_time")
    pivot = pivot.rename(columns=LABELS)
    print(pivot.to_string(float_format="{:.6f}".format))

    print("\n--- Memory summary (median KB) ---")
    pivot_mem = summary.pivot_table(index="n", columns="language", values="median_mem") / 1024
    pivot_mem = pivot_mem.rename(columns=LABELS)
    print(pivot_mem.to_string(float_format="{:.1f}".format))


if __name__ == "__main__":
    df = load_data()
    summary = summarise(df)
    print_summary_table(summary)
    plot_time(summary)
    plot_memory(summary)
    plot_cost_per_op(summary)
