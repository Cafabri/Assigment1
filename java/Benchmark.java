import java.io.*;
import java.util.Locale;

public class Benchmark {

    static final long SEED_A = 42;
    static final long SEED_B = 43;

    /** SplitMix64 state, identical to the generator used in the C and Python programs. */
    static long state;

    static long splitmix64() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Row-major fill with uniform values in [0, 10): top 53 bits -> [0, 1) -> x 10 */
    static double[][] generateMatrix(int n, long seed) {
        state = seed;
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] = (double) (splitmix64() >>> 11) * 0x1.0p-53 * 10.0;
        return M;
    }

    static double checksum(double[][] C) {
        double s = 0.0;
        for (double[] row : C)
            for (double v : row)
                s += v;
        return s;
    }

    /** Benchmark run n warmups reps: one CSV line per run on stdout. */
    static void run(int n, int warmups, int reps) {
        double[][] A = generateMatrix(n, SEED_A);
        double[][] B = generateMatrix(n, SEED_B);
        Runtime rt = Runtime.getRuntime();

        for (int r = 1; r <= warmups + reps; r++) {
            rt.gc();
            long memBefore = rt.totalMemory() - rt.freeMemory();

            long t0 = System.nanoTime();
            double[][] C = MatMul.multiply(A, B);
            long t1 = System.nanoTime();

            long memAfter = rt.totalMemory() - rt.freeMemory();
            long memUsed = Math.max(0, memAfter - memBefore);
            boolean warm = r <= warmups;

            System.out.printf(Locale.US, "java,%d,float64,1,%s,%d,%.9f,%d,%s%n", n,
                    warm ? "warmup" : "measured", warm ? r : r - warmups,
                    (t1 - t0) / 1e9, memUsed, Double.toString(checksum(C)));
            System.out.flush();
        }
    }

    /** Benchmark verify n file: writes the full result matrix for the correctness check. */
    static void verify(int n, String path) throws IOException {
        double[][] C = MatMul.multiply(generateMatrix(n, SEED_A), generateMatrix(n, SEED_B));
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {
            for (double[] row : C)
                for (double v : row)
                    out.println(Double.toString(v));
        }
    }

    public static void main(String[] args) throws IOException {
        MatMul.checkCorrectness();

        if (args.length == 4 && args[0].equals("run")) {
            run(Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
        } else if (args.length == 3 && args[0].equals("verify")) {
            verify(Integer.parseInt(args[1]), args[2]);
        } else {
            System.err.println("usage: java Benchmark run <n> <warmups> <reps>");
            System.err.println("       java Benchmark verify <n> <output_file>");
            System.exit(2);
        }
    }
}
