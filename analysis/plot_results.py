import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join("..", "results")
FIGURES_DIR = os.path.join("..", "results")

COLORS = {"python": "#e07b39", "java": "#4c8eda", "c": "#5baa5b"}
LABELS = {"python": "Python", "java": "Java", "c": "C"}


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

    for lang in ["python", "java", "c"]:
        d = summary[summary["language"] == lang].sort_values("n")
        yerr_low  = d["median_time"] - d["q1_time"]
        yerr_high = d["q3_time"]     - d["median_time"]
        ax.errorbar(
            d["n"], d["median_time"],
            yerr=[yerr_low, yerr_high],
            label=LABELS[lang], color=COLORS[lang],
            marker="o", capsize=4, linewidth=1.5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Execution time (seconds, log scale)", fontsize=12)
    ax.set_title("Triple-loop matrix multiplication — runtime vs matrix size", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    path = os.path.join(FIGURES_DIR, "plot_time.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_memory(summary):
    fig, ax = plt.subplots(figsize=(7, 5))

    for lang in ["python", "java", "c"]:
        d = summary[summary["language"] == lang].sort_values("n")
        mem_kb = d["median_mem"] / 1024
        ax.plot(
            d["n"], mem_kb,
            label=LABELS[lang], color=COLORS[lang],
            marker="o", linewidth=1.5,
        )

    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Memory (KB)", fontsize=12)
    ax.set_title("Triple-loop matrix multiplication — memory vs matrix size", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(FIGURES_DIR, "plot_memory.png")
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
