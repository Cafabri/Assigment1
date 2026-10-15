"""Validates the three implementations against NumPy (A @ B) on random inputs and checks the
checksums of every benchmark run.

Usage (from the repository root, after run_experiment.py has built the programs):
    python analysis/check_correctness.py

Outputs:
  results/correctness.csv   element-wise comparison with NumPy for the verification sizes
  results/checksums.csv     checksum of the result of every configuration vs NumPy
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from run_experiment import COMMANDS  # noqa: E402

SEED_A, SEED_B = 42, 43
VERIFY_SIZES = [50, 257]

# All inputs are non-negative, so every product is non-negative and the sums have no
# cancellation. The relative error of a sum of n such terms is then at most about
# n * u, with u = 2^-53 ~ 1.1e-16 (n = 257: ~3e-14; n = 3000: ~3.3e-13).
# A relative tolerance of 1e-11 leaves a margin of more than 30x for every tested size,
# while an indexing error changes the result by a relative amount of order 1.
RTOL = 1e-11


def splitmix64_matrix(n, seed):
    """Vectorised SplitMix64, same sequence as the C, Java and Python programs."""
    k = np.arange(1, n * n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + k * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    values = (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 * 10.0
    return values.reshape(n, n)


def reference(n):
    return splitmix64_matrix(n, SEED_A) @ splitmix64_matrix(n, SEED_B)


def check_against_numpy():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in VERIFY_SIZES:
            ref = reference(n)
            results = {}
            for lang, cmd in COMMANDS.items():
                path = os.path.join(tmp, f"{lang}_{n}.txt")
                subprocess.run(cmd + ["verify", str(n), path], check=True, cwd=ROOT)
                results[lang] = np.loadtxt(path).reshape(n, n)
                abs_err = np.abs(results[lang] - ref)
                rel_err = abs_err / np.abs(ref)
                rows.append({
                    "test": "random vs NumPy", "n": n, "language": lang,
                    "max_abs_error": abs_err.max(), "max_rel_error": rel_err.max(),
                    "tolerance": f"rtol={RTOL}", "passed": bool((rel_err <= RTOL).all()),
                })
            same = (np.array_equal(results["c"], results["java"])
                    and np.array_equal(results["c"], results["python"]))
            rows.append({"test": "C = Java = Python (bitwise)", "n": n, "language": "all",
                         "max_abs_error": 0.0 if same else np.nan, "max_rel_error": np.nan,
                         "tolerance": "exact", "passed": same})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "correctness.csv"), index=False)
    print(df.to_string(index=False))
    return df["passed"].all()


def check_checksums():
    raw = pd.read_csv(os.path.join(RESULTS, "raw_measurements.csv"), float_precision="round_trip")
    rows = []
    for n in sorted(raw["n"].unique()):
        ref = reference(n).sum()
        d = raw[raw["n"] == n]
        for lang in ["c", "java", "python"]:
            values = d[d["language"] == lang]["checksum"].unique()
            if len(values) == 0:
                continue
            rel = np.abs(values - ref).max() / ref
            rows.append({"n": n, "language": lang, "runs": int((d["language"] == lang).sum()),
                         "distinct_checksums": len(values), "checksum": repr(float(values[0])),
                         "numpy_checksum": repr(float(ref)), "rel_diff_vs_numpy": rel,
                         "passed": bool(rel <= RTOL)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "checksums.csv"), index=False)
    print(df.to_string(index=False))
    return df["passed"].all()


if __name__ == "__main__":
    ok_numpy = check_against_numpy()
    ok_sums = check_checksums()
    print("\nALL CHECKS PASSED" if ok_numpy and ok_sums else "\nSOME CHECKS FAILED")
