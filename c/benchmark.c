#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "matmul.h"

#define SEED        42
#define REPETITIONS 5
#define WARMUP_RUNS 2

static const int SIZES[]   = {10, 50, 100, 150, 200};
static const int NUM_SIZES = 5;

static double *alloc_matrix(int n) {
    double *M = calloc(n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(1); }
    return M;
}

static void fill_matrix(double *M, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX * 10.0;
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

    printf("All correctness checks passed.\n");
}

int main(void) {
    check_correctness();

    FILE *csv = fopen("../results/results_c.csv", "w");
    if (!csv) { fprintf(stderr, "Cannot open CSV file\n"); exit(1); }
    fprintf(csv, "language,n,repetition,time_seconds,memory_bytes\n");

    for (int s = 0; s < NUM_SIZES; s++) {
        int n = SIZES[s];
        printf("Running n=%d...\n", n);

        double *A = alloc_matrix(n);
        double *B = alloc_matrix(n);
        double *C = alloc_matrix(n);
        fill_matrix(A, n, SEED);
        fill_matrix(B, n, SEED + 1);

        long mem_bytes = 3L * n * n * (long)sizeof(double);

        for (int w = 0; w < WARMUP_RUNS; w++) {
            memset(C, 0, n * n * sizeof(double));
            multiply(A, B, C, n);
        }

        for (int rep = 1; rep <= REPETITIONS; rep++) {
            memset(C, 0, n * n * sizeof(double));

            LARGE_INTEGER freq, t0, t1;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&t0);
            multiply(A, B, C, n);
            QueryPerformanceCounter(&t1);

            double elapsed = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;

            printf("  rep %d: %.6fs  %.1f KB\n", rep, elapsed, mem_bytes / 1024.0);
            fprintf(csv, "c,%d,%d,%.6f,%ld\n", n, rep, elapsed, mem_bytes);
        }

        free(A); free(B); free(C);
    }

    fclose(csv);
    printf("Results saved to ../results/results_c.csv\n");
    return 0;
}
