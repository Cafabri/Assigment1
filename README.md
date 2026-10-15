# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

**BIG DATA · Academic year 2026-2027**  
Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria  
**Author:** Carlos Falcón Brito

## Research question

How does matrix size affect the runtime and memory usage of a naive O(n³) triple-loop
matrix multiplication implemented in Python, Java and C, when each language runs in its
standard execution environment?

## Platform

**The study runs on Windows only.** The C timer (`QueryPerformanceCounter`), the processor
affinity (`SetProcessAffinityMask`), the peak-memory measurement (`GetProcessMemoryInfo`)
and the hardware queries (PowerShell, `powercfg`) use Windows APIs. Running the study on
another Windows machine reproduces the procedure, not the numbers: times depend on the
processor, its caches and the toolchain versions.

## Repository structure

```
├── run_experiment.py          # Builds C and Java, runs every configuration under the budget
├── c/
│   ├── matmul.h / matmul.c    # Triple-loop kernel
│   └── benchmark.c            # Input generation, timing, batching, checksum, verify mode
├── java/
│   ├── MatMul.java            # Triple-loop kernel + small correctness tests
│   └── Benchmark.java         # Input generation, timing, warm-up rule, alloc and verify modes
├── python/
│   ├── matmul.py              # Triple-loop kernel + small correctness tests
│   └── benchmark.py           # Input generation, timing, tracemalloc, alloc and verify modes
├── analysis/
│   ├── check_correctness.py   # Comparison with NumPy and checksum checks
│   ├── plot_results.py        # summary.csv, figures and the LaTeX tables of the report
│   └── report_numbers.py      # Every number quoted in the text of the report
├── results/
│   ├── raw_measurements.csv   # Every timed run (warm-up, measured, check and memory runs)
│   ├── run_status.csv         # OK / TIMEOUT / SKIPPED for every process
│   ├── process_memory.csv     # Peak working set of every benchmark process
│   ├── allocation_times.csv   # Time of allocating the result matrix alone (Java, Python)
│   ├── environment.txt        # Hardware, OS, toolchain, JVM settings, affinity, power plan
│   ├── correctness.csv        # Element-wise comparison with NumPy
│   ├── checksums.csv          # Checksum of every configuration vs NumPy
│   ├── summary.csv            # Median, IQR, process spread, ns per multiply-add, memory
│   ├── report_numbers.txt     # Exponents, doubling factors, ratios quoted in the report
│   ├── tables/*.tex           # Tables included by the report
│   └── plot_*.png             # Figures used in the report
└── report/
    ├── Assignment1_Report.tex
    └── Assignment01_FalconBrito_Report.pdf
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | Intel Core i7-13700KF: 8 P-cores (logical processors 0–15) + 8 E-cores (16–23) |
| Cache / RAM | 2 MB L2 per P-core, 30 MB shared L3 / 31.8 GiB |
| Affinity | runner and all benchmark processes restricted to the P-cores (0–15) |
| Power plan | Balanced (Windows default) |
| C compiler | gcc 16.1.0 (MSYS2/MinGW64), flags: `-O2 -ffp-contract=off` |
| Java | OpenJDK 21.0.12 (Microsoft Build), default options: G1 GC, max heap 8 GiB |
| Python | CPython 3.11.9 (benchmark uses the standard library only) |
| Analysis | numpy 2.2, pandas 2.2, matplotlib 3.10 |

`results/environment.txt` is written by `run_experiment.py` and records all of these,
including the JVM thread settings (`ParallelGCThreads`, `ConcGCThreads`,
`CICompilerCount`) read with `-XX:+PrintFlagsFinal` under the same affinity.

## Reproducing the study

Requirements: MSYS2 with `mingw-w64-x86_64-gcc`, a JDK 21, Python 3.11 and, for the
analysis, `pip install numpy pandas matplotlib`. The paths to `gcc.exe` and to the JDK
`bin` directory are set at the top of `run_experiment.py` (`GCC`, `JDK_BIN`); change them
if they are installed elsewhere.

```bash
python run_experiment.py                # build + all benchmarks (about 3 hours)
python analysis/check_correctness.py    # validation against NumPy + checksums
python analysis/plot_results.py         # summary.csv, figures and tables
python analysis/report_numbers.py       # numbers quoted in the text
```

To compile the report (run twice so that references resolve):

```bash
cd report
pdflatex -jobname=Assignment01_FalconBrito_Report Assignment1_Report.tex
```

Each program can also be run on its own, e.g. for one size:

```bash
c/benchmark.exe run 500 2 5                       # n, warm-up runs, measured runs
java -cp java Benchmark run 500 10 5              # n, maximum warm-up runs, measured runs
python python/benchmark.py run 500 1 5
python python/benchmark.py memory 500             # tracemalloc run (separate process)
java -cp java Benchmark alloc 500 5               # allocation of the result matrix only
c/benchmark.exe verify 50 out.txt                 # writes the full result matrix
```

## Experimental settings

- **Inputs:** the three programs use the same SplitMix64 generator (seed 42 for A, 43 for
  B, uniform values in [0, 10), filled row by row), so they multiply bit-identical
  matrices. NumPy reproduces the same matrices in `check_correctness.py`.
- **Sizes:** n ∈ {10, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000}. A language
  stops at its first size over budget; larger sizes are recorded as SKIPPED.
- **Budget:** 15 minutes per individual run (the runner kills the process when it is
  exceeded and records TIMEOUT) and 30 GB of memory.
- **Short sizes (n ≤ 100):** timed in batches of repeated calls lasting at least 10 ms; the
  time per multiplication is reported, and the checksum comes from a separate single call.
- **Warm-up:** C 2 runs; Java until two consecutive runs differ by less than 2 % (minimum
  2, maximum 10); Python 1 run (none for n ≥ 1500).
- **Repetitions:** 5 measured runs per process. Configurations whose first process has a
  median below 60 s are run in 3 independent processes, interleaved across languages
  (15 runs); the others in one process (5 runs; Python n ≥ 1500: 3 runs).
- **Memory:** peak working set of every timing process (common metric), plus a
  language-specific metric: C the computed size of A, B and C; Java the heap growth during
  the call; Python the `tracemalloc` peak in a separate process (n ≤ 1000).

## Raw data format (`results/raw_measurements.csv`)

| Column | Description |
|--------|-------------|
| `language` | `c`, `java` or `python` |
| `n` | Matrix side length |
| `dtype` | Numeric type (`float64`) |
| `threads` | Threads used by the kernel (1) |
| `run_type` | `warmup`, `measured`, `check` (separate checksum call of a batched size) or `memory` (Python tracemalloc process) |
| `repetition` | Run index within its type |
| `batch_size` | Calls per timed batch (1 for n > 100) |
| `time_seconds` | Wall-clock time per multiplication (batch time / batch size) |
| `memory_bytes` | Language-specific memory metric (empty when not measured) |
| `checksum` | Sum of all elements of the result (empty for batched runs) |
| `process` | Index of the independent process (1–3) |
