#include <cuda_runtime.h>

#define KERNEL_SIZE 9
#define TILE_SIZE 256


// Tested in day023/conv1d.cu


__constant__ float W[KERNEL_SIZE];

__global__ void conv1d_kernel_v2(const float *input, float *output, const int kernel_size, const int input_size) {
    unsigned int tid = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + tid;

    __shared__ float sharedMem[TILE_SIZE + KERNEL_SIZE - 1];
    int halo_offset = kernel_size / 2;

    // Load left halo from previous block
    // We map the thread index to element index into the previous 
    // tile with the expression (blockIdx.x-1)*blockDim.x+threadIdx.x. We then pick 
    // only the last "halo_offset" threads to load the needed left halo elements
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + tid;
    if (tid >= blockDim.x - halo_offset) {
        // check if the halo cells are actually ghost cells
        sharedMem[tid - (blockDim.x - halo_offset)] = (halo_index_left < 0) ? 0 : input[halo_index_left];
    }

    // Internal elements (normal mapping blockIdx.x * blockDim.x + tid)
    // Since the first "halo_offset" elements of the sharedMem array already contain the left halo cells, the 
    // center elements need to be loaded into the next section of sharedMem. This is done by adding halo_offset
    // to threadIdx.x
    sharedMem[halo_offset + tid] = input[blockIdx.x * blockDim.x + tid];

    // Load Right halo from next block
    int halo_index_right = (blockIdx.x + 1) * blockDim.x + tid;
    if (tid < halo_offset) {
        sharedMem[halo_offset + blockDim.x + tid] = (halo_index_right >= input_size) ? 0 : input[halo_index_right];
    }

    __syncthreads();

    // Perform convolution using shared memory
    float out_value = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        out_value += sharedMem[tid + i] * W[i];
    }
    output[index] = out_value;
}