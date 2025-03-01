#include <cuda_runtime.h>

#include "kernel_launcher.h"

__global__ void reduce3(const float *input, float *output, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] =
        (index < size) ? input[index] + input[index + blockDim.x] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Sequential addressing is conflict free but Idle threads
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduce3(const float *d_input, float *d_output, int size,
                    int numBlocks, int threadsPerBlock) {
    reduce3<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_output, size);
}
