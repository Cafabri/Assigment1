import java.io.*;
import java.util.Locale;

public class Benchmark {

    static final long SEED_A = 42;
    static final long SEED_B = 43;
    /** Sizes up to BATCH_MAX_N are timed in batches of repeated calls lasting at least
     *  BATCH_MIN_SECONDS, and the time per multiplication is reported. */
    static final int BATCH_MAX_N = 100;
    static final double BATCH_MIN_SECONDS = 0.010;
    /** Warm-up continues until two consecutive runs differ by less than WARMUP_TOLERANCE,
     *  with at least MIN_WARMUPS and at most the number given on the command line. */
    static final int MIN_WARMUPS = 2;
    static final double WARMUP_TOLERANCE = 0.02;

    /** SplitMix64 state, identical to the generator used in the C and Python programs. */
    static long state;
    /** Receives one element of every result so that no call can be removed by the JIT. */
    static volatile double sink;

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

    /** Time of `batch` consecutive calls, in seconds. */
    static double timeBatch(double[][] A, double[][] B, int batch) {
        int n = A.length;
        double s = 0.0;
        long t0 = System.nanoTime();
        for (int b = 0; b < batch; b++)
            s += MatMul.multiply(A, B)[n - 1][n - 1];
        long t1 = System.nanoTime();
        sink = s;
        return (t1 - t0) / 1e9;
    }

    static void print(int n, String type, int rep, int batch, double seconds, String mem,
                      String sum) {
        System.out.printf(Locale.US, "java,%d,float64,1,%s,%d,%d,%.9e,%s,%s%n", n, type, rep,
                batch, seconds, mem, sum);
        System.out.flush();
    }

    /** One call with heap measurement: used for every run when batch = 1, and as the
     *  separate checksum/memory call when the timed runs are batched. */
    static void singleRun(double[][] A, double[][] B, String type, int rep) {
        Runtime rt = Runtime.getRuntime();
        rt.gc();
        long memBefore = rt.totalMemory() - rt.freeMemory();

        long t0 = System.nanoTime();
        double[][] C = MatMul.multiply(A, B);
        long t1 = System.nanoTime();

        long memAfter = rt.totalMemory() - rt.freeMemory();
        print(A.length, type, rep, 1, (t1 - t0) / 1e9,
                Long.toString(Math.max(0, memAfter - memBefore)), Double.toString(checksum(C)));
    }

    /** Benchmark run n maxWarmups reps: one CSV line per run on stdout. */
    static void run(int n, int maxWarmups, int reps) {
        double[][] A = generateMatrix(n, SEED_A);
        double[][] B = generateMatrix(n, SEED_B);

        int batch = 1;
        if (n <= BATCH_MAX_N) {
            // calibration (not recorded): double the batch until it lasts BATCH_MIN_SECONDS
            while (timeBatch(A, B, batch) < BATCH_MIN_SECONDS)
                batch *= 2;
            singleRun(A, B, "check", 1);
        }

        // warm-up until two consecutive runs agree within WARMUP_TOLERANCE
        double previous = -1;
        for (int w = 1; w <= maxWarmups; w++) {
            double t;
            if (batch == 1) {
                Runtime.getRuntime().gc();
                long t0 = System.nanoTime();
                double[][] C = MatMul.multiply(A, B);
                t = (System.nanoTime() - t0) / 1e9;
                print(n, "warmup", w, 1, t, "", Double.toString(checksum(C)));
            } else {
                t = timeBatch(A, B, batch) / batch;
                print(n, "warmup", w, batch, t, "", "");
            }
            if (w >= MIN_WARMUPS && Math.abs(t - previous) / previous < WARMUP_TOLERANCE)
                break;
            previous = t;
        }

        for (int r = 1; r <= reps; r++) {
            if (batch == 1)
                singleRun(A, B, "measured", r);
            else
                print(n, "measured", r, batch, timeBatch(A, B, batch) / batch, "", "");
        }
    }

    /** Benchmark alloc n reps: time of allocating the n x n result matrix alone. */
    static void alloc(int n, int reps) {
        for (int r = -4; r <= reps; r++) {      // 5 unrecorded warm-up allocations
            long t0 = System.nanoTime();
            double[][] C = new double[n][n];
            long t1 = System.nanoTime();
            C[n - 1][n - 1] = 1.0;
            sink = C[n - 1][n - 1];
            if (r >= 1)
                System.out.printf(Locale.US, "java,%d,%d,%.9e%n", n, r, (t1 - t0) / 1e9);
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
        } else if (args.length == 3 && args[0].equals("alloc")) {
            alloc(Integer.parseInt(args[1]), Integer.parseInt(args[2]));
        } else if (args.length == 3 && args[0].equals("verify")) {
            verify(Integer.parseInt(args[1]), args[2]);
        } else {
            System.err.println("usage: java Benchmark run <n> <max_warmups> <reps>");
            System.err.println("       java Benchmark alloc <n> <reps>");
            System.err.println("       java Benchmark verify <n> <output_file>");
            System.exit(2);
        }
    }
}
