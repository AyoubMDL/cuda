#include <cuda_runtime.h>
#include <iostream>


__global__ void blockReduceSum(const float *X, float *d_sum, int N) {
    extern __shared__ float sharedMemory[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sharedMemory[tid] = 0.0f;
    if (index < N) {
        sharedMemory[tid] = X[index];
    }
    __syncthreads();
    
    // Reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMemory[tid] += sharedMemory[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_sum[blockIdx.x] = sharedMemory[0];
    }
}


float cudaReduceSum(float *X, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_output;
    cudaMalloc(&d_output, numBlocks * sizeof(float));


    while (numBlocks > 1) {
        int sharedMemSize = blockSize * sizeof(float);
        blockReduceSum<<<numBlocks, blockSize, sharedMemSize>>>(X, d_output, N);
        cudaDeviceSynchronize();

        // Update N to the number of partial sums (numBlocks)
        N = numBlocks;
        numBlocks = (N + blockSize - 1) / blockSize;

        // Swap input and output for the next pass
        std::swap(X, d_output);
    }

    // Perform the final reduction (if there's only one block left)
    if (numBlocks == 1) {
        int sharedMemSize = blockSize * sizeof(float);
        blockReduceSum<<<1, blockSize, sharedMemSize>>>(X, d_output, N);
        cudaDeviceSynchronize();
        std::swap(X, d_output);
    }

    // Copy the final result back to the host
    float result;
    cudaMemcpy(&result, X, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    return result;
}


float cpuReduceSum(const float *X, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += X[i];
    }
    return sum;
}


int main() {
    int n = 10000;
    size_t size = n * sizeof(float);

    // Allocate and initialize host memory
    float* h_input = new float[n];
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Initialize with 1.0 for simplicity
    }

    // Allocate device memory
    float* d_input;
    cudaMalloc((void **)&d_input, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // compare CPU/GPU
    float cudaSum = cudaReduceSum(d_input, n);
    float cpuSum = cpuReduceSum(h_input, n);

    std::cout << "GPU Sum: " << cudaSum << std::endl;
    std::cout << "CPU Sum: " << cpuSum << std::endl;

    // Free memory
    delete[] h_input;
    cudaFree(d_input);

    return 0;
}