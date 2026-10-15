import sys


def multiply(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def check_correctness():
    # Identity matrix: A * I = A
    A = [[1.0, 2.0], [3.0, 4.0]]
    I = [[1.0, 0.0], [0.0, 1.0]]
    result = multiply(A, I)
    assert result == A, f"Identity test failed: {result}"

    # Zero matrix: A * 0 = 0
    Z = [[0.0, 0.0], [0.0, 0.0]]
    result = multiply(A, Z)
    assert result == Z, f"Zero matrix test failed: {result}"

    # Known result: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]
    result = multiply(A, B)
    assert result == expected, f"Known result test failed: {result}"

    print("All correctness checks passed.", file=sys.stderr)


if __name__ == "__main__":
    check_correctness()
