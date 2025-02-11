#include <cuda_runtime.h>
#include <float.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>


__global__ void blockMaxKernel(const float *X, float *d_max, int N) {
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    extern __shared__ float sharedMemory[];
    sharedMemory[tid] = -FLT_MAX;

    if (index < N) {
        sharedMemory[tid] = X[index];
    }
    __syncthreads();

    // Perform parallel reduction to find the max in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMemory[tid] =
                max(sharedMemory[tid], sharedMemory[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_max[blockIdx.x] = sharedMemory[0];
    }
}

__global__ void globalMaxKernel(float *d_max, int numBlocks) {
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // Load d_max into shared memory
    if (tid < numBlocks) {
        shared[tid] = d_max[tid];
    } else {
        shared[tid] = -FLT_MAX;
    }
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Store the final result in d_max[0]
    if (tid == 0) {
        d_max[0] = shared[0];
    }
}

float cpuMax(const float *X, int N) {
    float maxVal = -FLT_MAX;
    for (int i = 0; i < N; ++i) {
        if (X[i] > maxVal) {
            maxVal = X[i];
        }
    }
    return maxVal;
}


int main() {
    int N = 10000;
    int blockSize = 256;
    int numBlocks = ceil(N / blockSize);

    float *X, *d_X, *d_max;
    
    X = (float *)malloc(N * sizeof(float));
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        X[i] = static_cast<float>(rand()) / RAND_MAX * 1000.f;
    }

    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_max, numBlocks * sizeof(float));

    // Copy input data
    cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice);

    // First kernel: compute local max for each block
    blockMaxKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_X, d_max, N);

    // Second kernel: reduce the block max values to get the global max
    globalMaxKernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_max,
                                                                 numBlocks);

    // Copy the final result back to host
    float maxVal;
    cudaMemcpy(&maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    float cpuMaxVal = cpuMax(X, N);

    // Compare results
    printf("CPU Max: %f\n", cpuMaxVal);
    printf("CUDA Max: %f\n", maxVal);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_max);
    free(X);
}