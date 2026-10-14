import java.io.*;
import java.util.Random;

public class Benchmark {

    static final int[] SIZES = {10, 50, 100, 150, 200};
    static final int REPETITIONS = 5;
    static final int WARMUP_RUNS = 3;
    static final long SEED = 42;

    static double[][] generateMatrix(int n, long seed) {
        Random rng = new Random(seed);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] = rng.nextDouble() * 10;
        return M;
    }

    public static void main(String[] args) throws IOException {
        MatMul.checkCorrectness();

        Runtime rt = Runtime.getRuntime();
        PrintWriter csv = new PrintWriter(new FileWriter("../results/results_java.csv"));
        csv.println("language,n,repetition,time_seconds,memory_bytes");

        for (int n : SIZES) {
            System.out.printf("Running n=%d...%n", n);
            double[][] A = generateMatrix(n, SEED);
            double[][] B = generateMatrix(n, SEED + 1);

            for (int w = 0; w < WARMUP_RUNS; w++)
                MatMul.multiply(A, B);

            for (int rep = 1; rep <= REPETITIONS; rep++) {
                rt.gc();
                long memBefore = rt.totalMemory() - rt.freeMemory();

                long t0 = System.nanoTime();
                double[][] C = MatMul.multiply(A, B);
                long t1 = System.nanoTime();

                long memAfter = rt.totalMemory() - rt.freeMemory();
                double elapsed = (t1 - t0) / 1e9;
                long memUsed = Math.max(0, memAfter - memBefore);

                double dummy = C[0][0]; // prevent dead code elimination

                System.out.printf("  rep %d: %.4fs  %.1f KB%n", rep, elapsed, memUsed / 1024.0);
                csv.printf("java,%d,%d,%.6f,%d%n", n, rep, elapsed, memUsed);
            }
        }

        csv.close();
        System.out.println("Results saved to ../results/results_java.csv");
    }
}
