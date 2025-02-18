#include <cuda_runtime.h>
#include <mma.h>

#include <iostream>

// Matrix dimensions
#define M 16
#define N 16
#define K 16

// Use NVIDIA's Warp Matrix Multiply-Accumulate namespace
using namespace nvcuda;

__global__ void tensorCoreMatmulKernel(half *a, half *b, float *c) {
    // declare the 16x16 fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load input matrices
    wmma::load_matrix_sync(a_frag, a, K);
    wmma::load_matrix_sync(b_frag, b, K);

    // Matrix multiply using Tensor Cores
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // Store result
    wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

void initialize_matrices(half *A, half *B) {
    for (int i = 0; i < M * K; i++)
        A[i] = __float2half((i % 5) + 1);  // Fill with some values

    for (int i = 0; i < K * N; i++)
        B[i] = __float2half((i % 3) + 1);  // Fill with some values
}

int main() {
    // Host memory allocation
    half h_A[M * K], h_B[K * N];
    float h_C[M * N] = {0};

    // Initialize matrices
    initialize_matrices(h_A, h_B);

    // Device memory allocation
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(32);
    dim3 gridDim(1);
    tensorCoreMatmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result matrix
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
