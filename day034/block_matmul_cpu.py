import numpy as np


def block_matmul_cpu(A, B, C, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    for i in range(0, M, BLOCK_SIZE_M):
        for j in range(0, N, BLOCK_SIZE_N):
            acc = np.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=np.float32)
            for k in range(0, K, BLOCK_SIZE_K):
                a = A[i: i+BLOCK_SIZE_M, k: k+BLOCK_SIZE_K]
                b = B[k: k+BLOCK_SIZE_K, j: j+BLOCK_SIZE_N]

                acc += np.dot(a, b)

            C[i: i+BLOCK_SIZE_M, j: j+BLOCK_SIZE_N] = acc


if __name__ == "__main__":
    M, N, K = 8, 8, 8
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_K = 4

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    # Run block matrix multiplication
    block_matmul_cpu(A, B, C, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # Compare with NumPy's result
    C_expected = np.dot(A, B)

    np.testing.assert_allclose(C, C_expected, atol=1e-5)
