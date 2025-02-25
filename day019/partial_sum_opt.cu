#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256

// Kernel: each thread loads two elements (if in bounds) and reduces them in shared memory.
__global__ void partialSumKernel(const float *X, float *result, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // Each block processes two elements per thread
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n)
        sum = X[idx];
    if (idx + blockDim.x < n)
        sum += X[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result
    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

// Host function: reduce the array using successive kernel launches.
float reduceSumCuda(float *h_input, int size) {
    float *d_in = nullptr, *d_out = nullptr;
    float h_sum = 0.0f;

    // Calculate initial grid size.
    int grid = (size + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    
    // Allocate device memory.
    cudaMalloc((void**)&d_in, size * sizeof(float));
    cudaMalloc((void**)&d_out, grid * sizeof(float));

    // Copy input data from host to device.
    cudaMemcpy(d_in, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int currentSize = size;
    // Pointers for ping-ponging between device buffers.
    float *d_current_in = d_in;
    float *d_current_out = d_out;

    // Continue reducing until a single sum remains.
    while (true) {
        grid = (currentSize + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        partialSumKernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_current_in, d_current_out, currentSize);
        cudaDeviceSynchronize();

        if (grid == 1)
            break;

        // Prepare for the next iteration.
        currentSize = grid;
        std::swap(d_current_in, d_current_out);
    }

    // Copy the final result back to host.
    cudaMemcpy(&h_sum, d_current_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up.
    cudaFree(d_in);
    cudaFree(d_out);

    return h_sum;
}

int main() {
    int size = 10000000; // 10 million elements
    float *h_input = new float[size];

    // Initialize the host array (for example, with all ones).
    for (int i = 0; i < size; ++i) {
        h_input[i] = 1.0f;
    }

    // Compute the sum on the GPU.
    float sum = reduceSumCuda(h_input, size);
    std::cout << "Sum: " << sum << std::endl;

    delete[] h_input;
    return 0;
}
