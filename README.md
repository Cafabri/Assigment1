# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

**BIG DATA · Academic year 2026-2027**  
Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria  
**Author:** Carlos Falcón Brito

## Research question

How does matrix size affect the runtime and memory usage of a naive O(n³) triple-loop
matrix multiplication implemented in Python, Java and C, when each language runs in its
standard execution environment?

## Repository structure

```
├── run_experiment.py        # Builds C and Java, runs every configuration under the budget
├── c/
│   ├── matmul.h / matmul.c  # Triple-loop kernel
│   └── benchmark.c          # Input generation, timing, checksum, verify mode
├── java/
│   ├── MatMul.java          # Triple-loop kernel + small correctness tests
│   └── Benchmark.java       # Input generation, timing, checksum, verify mode
├── python/
│   ├── matmul.py            # Triple-loop kernel + small correctness tests
│   └── benchmark.py         # Input generation, timing, checksum, verify mode
├── analysis/
│   ├── check_correctness.py # Comparison with NumPy and checksum checks
│   └── plot_results.py      # Summary table and figures
├── results/
│   ├── raw_measurements.csv # Every run (warm-up, measured and memory runs)
│   ├── run_status.csv       # OK / TIMEOUT / SKIPPED for every configuration
│   ├── environment.txt      # Tool versions and CPU of the run
│   ├── correctness.csv      # Element-wise comparison with NumPy
│   ├── checksums.csv        # Checksum of every configuration vs NumPy
│   ├── summary.csv          # Median, IQR, ns per multiply-add, ratio to C, bytes/element
│   └── plot_*.png           # Figures used in the report
└── report/
    ├── Assignment1_Report.tex
    └── Assignment01_FalconBrito_Report.pdf
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | Intel Core i7-13700KF (8 P-cores + 8 E-cores, 24 logical processors) |
| RAM | 31.8 GB |
| C compiler | gcc 16.1.0 (MSYS2/MinGW64), flags: `-O2` |
| Java | OpenJDK 21.0.12 (Microsoft Build), default JIT and GC |
| Python | CPython 3.11.9 (benchmark uses the standard library only) |
| Analysis | numpy 2.2, pandas 2.2, matplotlib 3.10 |

The paths to `gcc` and the JDK are set at the top of `run_experiment.py`; change them if
they are installed elsewhere.

## Reproducing the study

```bash
pip install numpy pandas matplotlib     # only needed for the analysis scripts

python run_experiment.py                # build + all benchmarks (about 3 hours)
python analysis/check_correctness.py    # validation against NumPy + checksums
cd analysis && python plot_results.py   # summary.csv and figures
```

To compile the report: `cd report && pdflatex Assignment1_Report.tex` (twice).

Each program can also be run on its own, e.g. for one size:

```bash
c/benchmark.exe run 500 2 5                       # n, warm-up runs, measured runs
java -cp java Benchmark run 500 3 5
python python/benchmark.py run 500 1 5
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
- **Repetitions:** 5 measured runs after 2 (C), 3 (Java) or 1 (Python) warm-up runs.
- **Memory:** C reports the allocated size of A, B and C; Java the heap growth during the
  call; Python the `tracemalloc` peak in one extra run (n ≤ 1000), separate from the timed
  runs.

## Raw data format (`results/raw_measurements.csv`)

| Column | Description |
|--------|-------------|
| `language` | `c`, `java` or `python` |
| `n` | Matrix side length |
| `dtype` | Numeric type (`float64`) |
| `threads` | Threads used by the kernel (1) |
| `run_type` | `warmup`, `measured` or `memory` (Python memory run) |
| `repetition` | Run index within its type |
| `time_seconds` | Wall-clock time of the multiplication kernel |
| `memory_bytes` | Language-specific memory metric (empty for Python timed runs) |
| `checksum` | Sum of all elements of the result matrix |
