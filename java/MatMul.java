public class MatMul {

    public static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    public static void checkCorrectness() {
        double[][] A = {{1, 2}, {3, 4}};

        // Identity: A * I = A
        double[][] I = {{1, 0}, {0, 1}};
        double[][] result = multiply(A, I);
        if (result[0][0] != 1 || result[0][1] != 2 || result[1][0] != 3 || result[1][1] != 4)
            throw new RuntimeException("Identity test failed");

        // Zero matrix: A * 0 = 0
        double[][] Z = {{0, 0}, {0, 0}};
        result = multiply(A, Z);
        if (result[0][0] != 0 || result[0][1] != 0 || result[1][0] != 0 || result[1][1] != 0)
            throw new RuntimeException("Zero matrix test failed");

        // Known result: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        double[][] B = {{5, 6}, {7, 8}};
        result = multiply(A, B);
        if (result[0][0] != 19 || result[0][1] != 22 || result[1][0] != 43 || result[1][1] != 50)
            throw new RuntimeException("Known result test failed");

        System.out.println("All correctness checks passed.");
    }
}
