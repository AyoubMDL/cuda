#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include <vector>

#include "common.h"

__global__ void conv1DGpu(const float *x, const float *w, float *y, int N,
                          int K) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N - K + 1) {
        float result = 0.0f;
        for (int i = 0; i < K; ++i) {
            result += x[index + i] * w[i];
        }
        y[index] = result;
    }
}

void conv1dCpu(const float *x, const float *w, float *y, int N, int K) {
    int M = N - K + 1;
    for (int i = 0; i < M; ++i) {
        y[i] = 0.0f;
        for (int j = 0; j < K; ++j) {
            y[i] += x[i + j] * w[j];
        }
    }
}

int main() {
    const int N = 1 << 20;    // Input size
    const int K = 7;          // Kernel size
    const int M = N - K + 1;  // Output size

    // Allocate memory for input, kernel, and output
    std::vector<float> h_x(N);
    std::vector<float> h_w(K);
    std::vector<float> cpu_result(M, 0.0f);
    std::vector<float> gpu_result(M, 0.0f);

    // Measure CPU time for matrix multiplication
    double cpu_time = measureExecutionTime(
        [&]() { conv1dCpu(h_x.data(), h_w.data(), cpu_result.data(), N, K); });
    std::cout << "CPU Time: " << cpu_time << " ms\n";

    // Allocate device memory
    float *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_w, K * sizeof(float));
    cudaMalloc(&d_y, (N - K + 1) * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel and measure time
    double gpu_time = measureExecutionTime([&]() {
        int blockSize = 256;
        int numBlocks = ceil(M / blockSize);
        conv1DGpu<<<numBlocks, blockSize>>>(d_x, d_w, d_y, N, K);
        cudaDeviceSynchronize();
    });
    std::cout << "GPU Time: " << gpu_time << " ms\n";

    // Copy result back to host
    cudaMemcpy(gpu_result.data(), d_y, M * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool success = compareResults(cpu_result.data(), gpu_result.data(), M);
    std::cout << (success ? "CPU and GPU results match!" : "Results mismatch!")
              << std::endl;

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    return 0;
}