# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

**BIG DATA · Academic year 2026-2027**  
Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria

## Research question

How does matrix size affect the runtime and memory usage of a basic triple-loop matrix multiplication implemented in Python, Java and C?

## Repository structure

```
├── python/       # Python implementation and benchmark
├── java/         # Java implementation and benchmark
├── c/            # C implementation and benchmark
├── results/      # Raw measurements in CSV format
├── analysis/     # Scripts to generate tables and plots
└── README.md
```

## Implementations

Each implementation uses the straightforward O(n³) triple-loop algorithm with no optimised library calls.

| Language | File | Notes |
|----------|------|-------|
| Python   | `python/matmul.py` | CPython, pure loops |
| Java     | `java/MatMul.java` | JVM warm-up applied |
| C        | `c/matmul.c` | Compiled with gcc, flags documented below |

## Build and run

### Python
```bash
cd python
python matmul.py
```
Requires Python 3.x. No external dependencies.

### Java
```bash
cd java
javac MatMul.java
java MatMul
```

### C
```bash
cd c
gcc -O0 -o matmul matmul.c
./matmul
```
Flag `-O0` disables compiler optimisations to keep the comparison fair.

## Measurements

Raw results are saved to `results/results.csv` with columns:
`language, n, repetition, time_seconds, memory_bytes`

Run `analysis/plot_results.py` to regenerate tables and figures.

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | (to be filled) |
| RAM | (to be filled) |
| Python version | (to be filled) |
| Java version | (to be filled) |
| gcc version | (to be filled) |
