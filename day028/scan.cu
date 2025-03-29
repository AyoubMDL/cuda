#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256


__global__ void inclusive_scan_kernel(const float *input, float *output, int input_size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < input_size) {
        shared[tid] = input[index];
    } else {
        shared[tid] = 0.0f;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (tid >= stride) {
            shared[tid] += shared[tid - stride]; 
        }
    }
    output[index] = shared[tid];
}


__global__ void exclusive_scan_kernel(const float *input, float *output, int input_size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < input_size && index > 0) {
        shared[tid] = input[index - 1];
    } else {
        shared[tid] = 0.0f;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        if (tid >= stride) {
            shared[tid] += shared[tid - stride]; 
        }
    }
    output[index] = shared[tid];
}


int main() {
    const int input_size = 1024;
    float input[input_size], output_inclusive[input_size], output_exclusive[input_size];
    for (int i = 0; i < input_size; i++) {
        input[i] = static_cast<float>(i + 1);
    }
    
    float *d_input, *d_output_inclusive, *d_output_exclusive;
    size_t size = input_size * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output_inclusive, size);
    cudaMalloc((void**)&d_output_exclusive, size);
    
    // Copy input data to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    // Launch kernel (multiple blocks, 256 threads per block)
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (input_size + threads_per_block - 1) / threads_per_block;

    // inclusive scan
    inclusive_scan_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output_inclusive, input_size);
    cudaMemcpy(output_inclusive, d_output_inclusive, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // exclusive scan
    exclusive_scan_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output_exclusive, input_size);
    // Copy result back to host
    cudaMemcpy(output_exclusive, d_output_exclusive, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output_inclusive);
    cudaFree(d_output_exclusive);
    
    // Print result
    std::cout << "Inclusive scan result: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output_inclusive[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Exclusive scan result: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output_exclusive[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}