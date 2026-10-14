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
├── python/
│   ├── matmul.py          # Triple-loop implementation + correctness checks
│   └── benchmark.py       # Benchmark runner — outputs results/results_python.csv
├── java/
│   ├── MatMul.java        # Triple-loop implementation + correctness checks
│   └── Benchmark.java     # Benchmark runner — outputs results/results_java.csv
├── c/
│   ├── matmul.h           # Function declaration
│   ├── matmul.c           # Triple-loop implementation
│   └── benchmark.c        # Benchmark runner — outputs results/results_c.csv
├── results/
│   ├── results_python.csv # Raw measurements (Python)
│   ├── results_java.csv   # Raw measurements (Java)
│   ├── results_c.csv      # Raw measurements (C)
│   ├── plot_time.png      # Runtime figure
│   └── plot_memory.png    # Memory figure
├── analysis/
│   └── plot_results.py    # Reads CSVs, prints summary tables, saves figures
├── report/
│   └── Assignment01_FalconBrito_Report.pdf
└── README.md
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | Intel Core i7-13700KF (16 cores, 24 logical processors) |
| RAM | 31.8 GB |
| Python | CPython 3.11.9 |
| Java | OpenJDK 21.0.12 (Microsoft Build of OpenJDK with Hotspot) |
| C compiler | gcc 16.1.0 (MSYS2/MinGW64), flags: `-O0` |
| pandas | 2.x (used only for analysis script) |
| matplotlib | 3.x (used only for analysis script) |

## Build and run

### Python

```bash
cd python
python benchmark.py
```

No external dependencies required for the benchmark. `pandas` and `matplotlib` are only
needed for the analysis script.

### Java

```bash
cd java
javac MatMul.java Benchmark.java
java Benchmark
```

Requires JDK 21 or later.

### C

```bash
cd c
gcc -O0 -o benchmark matmul.c benchmark.c
./benchmark          # Linux/macOS
benchmark.exe        # Windows
```

Flag `-O0` disables all compiler optimisations. Each implementation runs on a single
thread with no internal library threads.

## Reproducing the full study

Run the three benchmarks in order (Python is the slowest — allow ~15 minutes):

```bash
# 1. Python
cd python && python benchmark.py && cd ..

# 2. Java
cd java && javac MatMul.java Benchmark.java && java Benchmark && cd ..

# 3. C
cd c && gcc -O0 -o benchmark matmul.c benchmark.c && ./benchmark && cd ..

# 4. Generate figures and summary tables
cd analysis && python plot_results.py
```

Raw measurements are saved as CSV files in `results/`. Figures are saved as PNG files
in `results/`.

## Measurements

Each CSV file has the following columns:

| Column | Description |
|--------|-------------|
| `language` | `python`, `java` or `c` |
| `n` | Matrix side length |
| `repetition` | Run index (1–5) |
| `time_seconds` | Wall-clock time of the multiplication kernel |
| `memory_bytes` | Language-specific memory metric (see report Section 3) |

Matrix sizes tested: n ∈ {10, 50, 100, 200, 300, 500, 750, 1000}.  
Repetitions per configuration: 5 (preceded by warm-up runs).  
Random seed: 42 for matrix A, 43 for matrix B.
