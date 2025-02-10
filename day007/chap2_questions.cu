#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include "common.h"


__global__ void question1(const float *A, const float *B, float *C, int N) {
    // block_id * block_size + thread_id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cuda launches 
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


// CUDA kernel for vector addition
__global__ void question2(const float *A, const float *B, float *C, int N) {
    // Each thread processes two adjacent elements
    int i = (blockIdx.x*blockDim.x + threadIdx.x) * 2;

    // Cuda launches 
    if (i < N) {
        C[i] = A[i] + B[i];
    }

    if (i + 1 < N) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

__global__ void question3(const float *A, const float *B, float *C, int N) {
    // Each thread block will process 2*blockDim
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Cuda launches 
    if (i < N) {
        C[i] = A[i] + B[i];
    }

    if (i + blockDim.x < N) {
        C[i + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
    }
}

void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}


int main() {
    // 1M elements
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *A_host = (float*)malloc(size);
    float *B_host = (float*)malloc(size);
    float *C_mat_cpu = (float*)malloc(size);
    float *C_mat_gpu = (float*)malloc(size);

    // Initialize input vectors
    initializeVectors(A_host, B_host, N);

    // Measure CPU time for vector addition
    double cpuTime = measureExecutionTime([&]() {
        vectorAddCPU(A_host, B_host, C_mat_cpu, N);
    });
    std::cout << "CPU execution time: " << cpuTime << " ms" << std::endl;

    // Allocate memory on the device
    float *A_device, *B_device, *C_device;
    cudaMalloc((void**)&A_device, size);
    cudaMalloc((void**)&B_device, size);
    cudaMalloc((void**)&C_device, size);

    // Copy data from host to device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Measure GPU time for vector addition
    double gpuTime = measureExecutionTime([&]() {
        question3<<<blocksPerGrid, threadsPerBlock>>>(A_device, B_device, C_device, N);
        cudaDeviceSynchronize();
    });
    std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(C_mat_gpu, C_device, size, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    bool success = compareResults(C_mat_cpu, C_mat_gpu, N);
    std::cout << (success ? "CPU and GPU results match!" : "Results mismatch!") << std::endl;

    // Free device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Free host memory
    free(A_host);
    free(B_host);
    free(C_mat_cpu);
    free(C_mat_gpu);

    return 0;

}
