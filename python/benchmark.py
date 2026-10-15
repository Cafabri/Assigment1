import sys
import time
import tracemalloc
from matmul import multiply, check_correctness

SEED_A = 42
SEED_B = 43
# Sizes up to BATCH_MAX_N are timed in batches of repeated calls lasting at least
# BATCH_MIN_SECONDS, and the time per multiplication is reported.
BATCH_MAX_N = 100
BATCH_MIN_SECONDS = 0.010
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


def time_batch(A, B, batch):
    """Time of `batch` consecutive calls; the last result is returned so it is consumed."""
    t0 = time.perf_counter()
    for _ in range(batch):
        C = multiply(A, B)
    t1 = time.perf_counter()
    return t1 - t0, C


def row(n, run_type, rep, batch, seconds, memory="", total=""):
    print(f"python,{n},float64,1,{run_type},{rep},{batch},{seconds:.9e},{memory},{total}",
          flush=True)


def run(n, warmups, reps):
    """One CSV line per run on stdout. tracemalloc is not active in these runs."""
    A = generate_matrix(n, SEED_A)
    B = generate_matrix(n, SEED_B)

    batch = 1
    if n <= BATCH_MAX_N:
        # calibration (not recorded): double the batch until it lasts BATCH_MIN_SECONDS
        while time_batch(A, B, batch)[0] < BATCH_MIN_SECONDS:
            batch *= 2
        # the checksum is validated in a separate single call
        seconds, C = time_batch(A, B, 1)
        row(n, "check", 1, 1, seconds, total=repr(checksum(C)))

    for r in range(1, warmups + reps + 1):
        C = None                                  # release the previous result first
        seconds, C = time_batch(A, B, batch)
        warm = r <= warmups
        row(n, "warmup" if warm else "measured", r if warm else r - warmups, batch,
            seconds / batch, total=repr(checksum(C)) if batch == 1 else "")


def memory(n):
    """Separate process: one call traced by tracemalloc, whose time is not used."""
    A = generate_matrix(n, SEED_A)
    B = generate_matrix(n, SEED_B)
    tracemalloc.start()
    t0 = time.perf_counter()
    C = multiply(A, B)
    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    row(n, "memory", 1, 1, t1 - t0, peak_bytes, repr(checksum(C)))


def alloc(n, reps):
    """Time of creating the n x n result matrix alone, as done inside multiply()."""
    for r in range(-4, reps + 1):                 # 5 unrecorded warm-up allocations
        t0 = time.perf_counter()
        C = [[0.0] * n for _ in range(n)]
        t1 = time.perf_counter()
        del C
        if r >= 1:
            print(f"python,{n},{r},{t1 - t0:.9e}", flush=True)


def verify(n, path):
    """Writes the full result matrix for the correctness check."""
    C = multiply(generate_matrix(n, SEED_A), generate_matrix(n, SEED_B))
    with open(path, "w") as f:
        for r in C:
            for v in r:
                f.write(f"{v!r}\n")


if __name__ == "__main__":
    check_correctness()
    if len(sys.argv) == 5 and sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 3 and sys.argv[1] == "memory":
        memory(int(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "alloc":
        alloc(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        verify(int(sys.argv[2]), sys.argv[3])
    else:
        sys.exit("usage: python benchmark.py run <n> <warmups> <reps>\n"
                 "       python benchmark.py memory <n>\n"
                 "       python benchmark.py alloc <n> <reps>\n"
                 "       python benchmark.py verify <n> <output_file>")
