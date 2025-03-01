#include <cuda_runtime.h>

#include "kernel_launcher.h"

__device__ void warpReduce(volatile float *sdata, int tid) {
    // no additional synchronization is required because the threads are
    // synchronized within their warp.
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(const float *input, float *output, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] =
        (index < size) ? input[index] + input[index + blockDim.x] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        // Sequential addressing is conflict free but Idle threads
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduce4(const float *d_input, float *d_output, int size,
                    int numBlocks, int threadsPerBlock) {
    reduce4<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_output, size);
}
