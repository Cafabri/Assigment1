#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <windows.h>
#include "matmul.h"

#define SEED_A 42
#define SEED_B 43
/* Sizes up to BATCH_MAX_N are timed in batches of repeated calls lasting at least
   BATCH_MIN_SECONDS, and the time per multiplication is reported. */
#define BATCH_MAX_N 100
#define BATCH_MIN_SECONDS 0.010

/* Monotonic clock in seconds (Windows QueryPerformanceCounter) */
static double now(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/*
 * SplitMix64 generator. The same constants and the same conversion to double are used
 * in the Java and Python programs, so the three languages multiply bit-identical matrices.
 */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Row-major fill with uniform values in [0, 10): top 53 bits -> [0, 1) -> x 10 */
static void fill_matrix(double *M, int n, uint64_t seed) {
    uint64_t state = seed;
    for (int64_t i = 0; i < (int64_t)n * n; i++)
        M[i] = (double)(splitmix64(&state) >> 11) * 0x1.0p-53 * 10.0;
}

static double *alloc_matrix(int n) {
    double *M = calloc((size_t)n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(2); }
    return M;
}

static double checksum(const double *C, int n) {
    double s = 0.0;
    for (int64_t i = 0; i < (int64_t)n * n; i++)
        s += C[i];
    return s;
}

static void check_correctness(void) {
    double A[] = {1,2, 3,4};
    double I[] = {1,0, 0,1};
    double Z[] = {0,0, 0,0};
    double B[] = {5,6, 7,8};
    double C[4];

    memset(C, 0, sizeof(C));
    multiply(A, I, C, 2);
    if (C[0]!=1||C[1]!=2||C[2]!=3||C[3]!=4)
        { fprintf(stderr,"Identity test failed\n"); exit(1); }

    memset(C, 0, sizeof(C));
    multiply(A, Z, C, 2);
    if (C[0]||C[1]||C[2]||C[3])
        { fprintf(stderr,"Zero matrix test failed\n"); exit(1); }

    memset(C, 0, sizeof(C));
    multiply(A, B, C, 2);
    if (C[0]!=19||C[1]!=22||C[2]!=43||C[3]!=50)
        { fprintf(stderr,"Known result test failed\n"); exit(1); }
}

/* Times `batch` consecutive calls. C is zeroed before the batch (outside the timer) and
   accumulates inside it, so every call does the same n^3 multiply-adds. */
static double time_batch(const double *A, const double *B, double *C, int n, long batch) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    double t0 = now();
    for (long b = 0; b < batch; b++)
        multiply(A, B, C, n);
    return now() - t0;
}

/* benchmark run <n> <warmups> <reps>: one CSV line per run on stdout */
static void run(int n, int warmups, int reps) {
    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);
    fill_matrix(A, n, SEED_A);
    fill_matrix(B, n, SEED_B);

    int64_t mem_bytes = (int64_t)3 * n * n * (int64_t)sizeof(double);

    long batch = 1;
    if (n <= BATCH_MAX_N) {
        /* calibration (not recorded): double the batch until it lasts BATCH_MIN_SECONDS */
        while (time_batch(A, B, C, n, batch) < BATCH_MIN_SECONDS)
            batch *= 2;
        /* the checksum is validated in a separate single call */
        memset(C, 0, (size_t)n * n * sizeof(double));
        double t0 = now();
        multiply(A, B, C, n);
        double elapsed = now() - t0;
        printf("c,%d,float64,1,check,1,1,%.9e,%" PRId64 ",%.17g\n", n, elapsed, mem_bytes,
               checksum(C, n));
    }

    for (int r = 1; r <= warmups + reps; r++) {
        double per_call = time_batch(A, B, C, n, batch) / batch;
        int warm = r <= warmups;
        if (batch == 1)
            printf("c,%d,float64,1,%s,%d,1,%.9e,%" PRId64 ",%.17g\n", n,
                   warm ? "warmup" : "measured", warm ? r : r - warmups, per_call, mem_bytes,
                   checksum(C, n));
        else
            printf("c,%d,float64,1,%s,%d,%ld,%.9e,%" PRId64 ",\n", n,
                   warm ? "warmup" : "measured", warm ? r : r - warmups, batch, per_call,
                   mem_bytes);
        fflush(stdout);
    }
    free(A); free(B); free(C);
}

/* benchmark verify <n> <file>: writes the full result matrix for the correctness check */
static void verify(int n, const char *path) {
    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);
    fill_matrix(A, n, SEED_A);
    fill_matrix(B, n, SEED_B);
    multiply(A, B, C, n);

    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(2); }
    for (int64_t i = 0; i < (int64_t)n * n; i++)
        fprintf(f, "%.17g\n", C[i]);
    fclose(f);
    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    check_correctness();

    if (argc == 5 && strcmp(argv[1], "run") == 0) {
        run(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    } else if (argc == 4 && strcmp(argv[1], "verify") == 0) {
        verify(atoi(argv[2]), argv[3]);
    } else {
        fprintf(stderr, "usage: benchmark run <n> <warmups> <reps>\n"
                        "       benchmark verify <n> <output_file>\n");
        return 2;
    }
    return 0;
}
