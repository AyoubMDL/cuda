#include <cuda_runtime.h>

#include "kernel_launcher.h"

__global__ void reduce0(const float *input, float *output, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (index < size) ? input[index] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // highly divergent warps are very inefficient, and % operator is very
        // slow
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduce0(const float *d_input, float *d_output, int size,
                    int numBlocks, int threadsPerBlock) {
    reduce0<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_output, size);
}
