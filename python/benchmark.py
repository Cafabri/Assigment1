import time
import tracemalloc
import random
import csv
import os
from matmul import multiply, check_correctness

SIZES = [10, 50, 100, 150, 200]
REPETITIONS = 5
WARMUP_RUNS = 1
SEED = 42
RESULTS_FILE = os.path.join("..", "results", "results_python.csv")


def generate_matrix(n, seed_offset=0):
    random.seed(SEED + seed_offset)
    return [[random.uniform(0, 10) for _ in range(n)] for _ in range(n)]


def measure(A, B):
    tracemalloc.start()
    t_start = time.perf_counter()
    C = multiply(A, B)
    t_end = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _ = C  # ensure result is consumed
    return t_end - t_start, peak_bytes


def run_benchmark():
    check_correctness()

    rows = []
    for n in SIZES:
        print(f"Running n={n}...")
        A = generate_matrix(n, seed_offset=0)
        B = generate_matrix(n, seed_offset=1)

        for _ in range(WARMUP_RUNS):
            multiply(A, B)

        for rep in range(1, REPETITIONS + 1):
            elapsed, peak_mem = measure(A, B)
            rows.append({
                "language": "python",
                "n": n,
                "repetition": rep,
                "time_seconds": round(elapsed, 6),
                "memory_bytes": peak_mem,
            })
            print(f"  rep {rep}: {elapsed:.4f}s  {peak_mem/1024:.1f} KB")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_benchmark()
