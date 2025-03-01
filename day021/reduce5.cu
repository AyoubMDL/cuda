#include <cuda_runtime.h>

#include "kernel_launcher.h"

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid) {
    // no additional synchronization is required because the threads are
    // synchronized within their warp.
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5(const float *input, float *output, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (index < size) {
        sdata[tid] += input[index] + input[index + blockSize];
        index += gridSize;
    }

    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void launch_reduce5(const float *d_input, float *d_output, int size,
                    int numBlocks, int threadsPerBlock) {
    switch (threadsPerBlock) {
        case 512:
            reduce5<512>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 256:
            reduce5<256>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 128:
            reduce5<128>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 64:
            reduce5<64>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 32:
            reduce5<32>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 16:
            reduce5<16>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 8:
            reduce5<8>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 4:
            reduce5<4>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 2:
            reduce5<2>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
        case 1:
            reduce5<1>
                <<<numBlocks, threadsPerBlock,
                   threadsPerBlock * sizeof(float)>>>(d_input, d_output, size);
            break;
    }
}
