import numpy as np

# A shape: (M, K)
# B shape: (K, N)
# C = A @ B shape: (M, N)


def naive_matmul_cpu(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
