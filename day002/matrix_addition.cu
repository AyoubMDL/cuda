#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#include "common.h"


__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int columns) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIndex < rows && columnIndex < columns) {
        int index = rowIndex * columns + columnIndex;
        C[index] = A[index] + B[index];
    }
}

void matrixAddCpu(const float *A, const float *B, float *C, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            int index = i * columns + j;
            C[index] = A[index] + B[index];
        }
    }
}


int main() {
    int rows = 1 << 14;
    int block_size_rows = 32;
    int columns = 1 << 14;
    int block_size_columns = 32;

    size_t size = rows * columns * sizeof(float);
    
    // Allocate memory on the host
    float *A_host = (float *)malloc(size);
    float *B_host = (float *)malloc(size);
    float *C_mat_cpu = (float *)malloc(size);
    float *C_mat_gpu = (float *)malloc(size);

    // Initialize input vectors
    initializeVectors(A_host, B_host, rows * columns);

    // Measure CPU time for vector addition
    double cpuTime = measureExecutionTime([&]() {
        matrixAddCpu(A_host, B_host, C_mat_cpu, rows, columns);
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

    // Define grid
    int num_blocks_rows = ceil(rows / block_size_rows);
    int num_blocks_columns = ceil(columns / block_size_columns);

    dim3 grid(num_blocks_columns, num_blocks_rows, 1);
    dim3 block(block_size_columns, block_size_rows, 1);

    double gpuTime = measureExecutionTime([&]() {
        matrixAdd<<<grid, block>>>(A_device, B_device, C_device, rows, columns);
        cudaDeviceSynchronize();
    });
    std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(C_mat_gpu, C_device, size, cudaMemcpyDeviceToHost);

    bool success = compareResults(C_mat_cpu, C_mat_gpu, rows * columns);
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