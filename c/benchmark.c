#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include "matmul.h"

#define SEED_A 42
#define SEED_B 43

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
    for (long i = 0; i < (long)n * n; i++)
        M[i] = (double)(splitmix64(&state) >> 11) * 0x1.0p-53 * 10.0;
}

static double *alloc_matrix(int n) {
    double *M = calloc((size_t)n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(2); }
    return M;
}

static double checksum(const double *C, int n) {
    double s = 0.0;
    for (long i = 0; i < (long)n * n; i++)
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

/* benchmark run <n> <warmups> <reps>: one CSV line per run on stdout */
static void run(int n, int warmups, int reps) {
    double *A = alloc_matrix(n);
    double *B = alloc_matrix(n);
    double *C = alloc_matrix(n);
    fill_matrix(A, n, SEED_A);
    fill_matrix(B, n, SEED_B);

    long mem_bytes = 3L * n * n * (long)sizeof(double);
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    for (int r = 1; r <= warmups + reps; r++) {
        memset(C, 0, (size_t)n * n * sizeof(double));

        LARGE_INTEGER t0, t1;
        QueryPerformanceCounter(&t0);
        multiply(A, B, C, n);
        QueryPerformanceCounter(&t1);

        double elapsed = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;
        int warm = r <= warmups;
        printf("c,%d,float64,1,%s,%d,%.9f,%ld,%.17g\n", n, warm ? "warmup" : "measured",
               warm ? r : r - warmups, elapsed, mem_bytes, checksum(C, n));
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
    for (long i = 0; i < (long)n * n; i++)
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
