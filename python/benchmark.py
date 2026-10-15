import sys
import time
import tracemalloc
from matmul import multiply, check_correctness

SEED_A = 42
SEED_B = 43
# tracemalloc slows the kernel down about 5x, so the memory run is limited to n <= 1000
# to keep it within the per-run time budget
MEMORY_MAX_N = 1000
MASK64 = 0xFFFFFFFFFFFFFFFF


def splitmix64_values(seed, count):
    """SplitMix64, identical to the generator used in the C and Java programs."""
    state = seed
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        yield z ^ (z >> 31)


def generate_matrix(n, seed):
    """Row-major fill with uniform values in [0, 10): top 53 bits -> [0, 1) -> x 10"""
    values = [(x >> 11) * 2.0 ** -53 * 10.0 for x in splitmix64_values(seed, n * n)]
    return [values[i * n:(i + 1) * n] for i in range(n)]


def checksum(C):
    s = 0.0
    for row in C:
        for v in row:
            s += v
    return s


def run(n, warmups, reps):
    """One CSV line per run on stdout.

    Timed runs are executed without tracemalloc, because tracing every allocation would
    slow down the kernel. Memory is measured in one extra run whose time is not used.
    """
    A = generate_matrix(n, SEED_A)
    B = generate_matrix(n, SEED_B)

    for r in range(1, warmups + reps + 1):
        t0 = time.perf_counter()
        C = multiply(A, B)
        t1 = time.perf_counter()
        warm = r <= warmups
        print(f"python,{n},float64,1,{'warmup' if warm else 'measured'},"
              f"{r if warm else r - warmups},{t1 - t0:.9f},,{checksum(C)!r}", flush=True)

    if n > MEMORY_MAX_N:
        return
    tracemalloc.start()
    t0 = time.perf_counter()
    C = multiply(A, B)
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"python,{n},float64,1,memory,1,{t1 - t0:.9f},{peak_bytes},{checksum(C)!r}",
          flush=True)


def verify(n, path):
    """Writes the full result matrix for the correctness check."""
    C = multiply(generate_matrix(n, SEED_A), generate_matrix(n, SEED_B))
    with open(path, "w") as f:
        for row in C:
            for v in row:
                f.write(f"{v!r}\n")


if __name__ == "__main__":
    check_correctness()
    if len(sys.argv) == 5 and sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        verify(int(sys.argv[2]), sys.argv[3])
    else:
        sys.exit("usage: python benchmark.py run <n> <warmups> <reps>\n"
                 "       python benchmark.py verify <n> <output_file>")
