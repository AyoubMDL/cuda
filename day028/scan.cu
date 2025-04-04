#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 1024
const float EPSILON = 1e-6;

typedef void (*ScanKernel)(const float*, float*, float*, int);


__global__ void add_block_sums_kernel(float *output, float *block_sums, int input_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && index < input_size) {
        output[index] += block_sums[blockIdx.x - 1];
    }
}

__global__ void inclusive_scan_kernel(const float *input, float *output, float *block_sums, int input_size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    shared[tid] = 0.0f;
    if (index < input_size) {
        shared[tid] = input[index];
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp;
        // Read value and synchronize (to avoid race)
        if (tid >= stride) {
            temp = shared[tid - stride];
        }
        __syncthreads();
        
        // Write value and synchronize
        if (tid >= stride) {
            shared[tid] += temp;
        }
        __syncthreads();
    }
    output[index] = shared[tid];

    // Store the last element of each block for block sum scan
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = shared[tid];
    }
}


__global__ void exclusive_scan_kernel(const float *input, float *output, float *block_sums, int input_size) {
    __shared__ float shared[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    shared[tid] = 0.0f;
    if (index < input_size && index > 0) {
        shared[tid] = input[index - 1];
    }
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp;
       // Read value and synchronize (to avoid race)
       if (tid >= stride) {
            temp = shared[tid - stride];
        }
        __syncthreads();
        
        // Write value and synchronize
        if (tid >= stride) {
            shared[tid] += temp;
        }
        __syncthreads();
    }
    output[index] = shared[tid];
    
    // Store the last element of each block for block sum scan
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = shared[tid];
    }
}


void scan(float *d_input, float *d_output, int input_size, ScanKernel scan_kernel) {
    // Config
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (input_size + threads_per_block - 1) / threads_per_block;
    
    float *d_block_sums;
    cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float));
    
    // Step 1: Block-wise scan using the provided kernel
    scan_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, d_block_sums, input_size);
    cudaDeviceSynchronize();

    if(num_blocks > 1) {
        // Scan block sums
        scan(d_block_sums, d_block_sums, num_blocks, inclusive_scan_kernel);

        // Add block sums to each block's output
        add_block_sums_kernel<<<num_blocks, threads_per_block>>>(d_output, d_block_sums, input_size);
        cudaDeviceSynchronize();
    }
    
    // Free mem
    cudaFree(d_block_sums);
}


void scan_cpu(const float *input, float *output, const int input_size) {
    output[0] = input[0];
    for (int i = 1; i < input_size; ++i) {
        output[i] = input[i] + output[i - 1];
    }
}


int main() {
    const int input_size = 4000;
    float input[input_size], output_inclusive[input_size], output_exclusive[input_size];
    for (int i = 0; i < input_size; i++) {
        input[i] = static_cast<float>(i + 1);
    }
    
    float *d_input, *d_output, *d_output_exclusive, *output_cpu;
    size_t size = input_size * sizeof(float);
    output_cpu = (float *)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMalloc((void**)&d_output_exclusive, size);
    
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    // Inclusive scan
    scan(d_input, d_output, input_size, inclusive_scan_kernel);
    cudaMemcpy(output_inclusive, d_output, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Exclusive scan
    scan(d_input, d_output_exclusive, input_size, exclusive_scan_kernel);
    cudaMemcpy(output_exclusive, d_output_exclusive, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    scan_cpu(input, output_cpu, input_size);
    cudaFree(d_input);
    cudaFree(d_output_exclusive);

    
    std::cout << "CPU scan result: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output_cpu[i] << " ";
    }
    std::cout << std::endl;
    
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

    // Check outputs
    bool success = true;

    for (int i = 0; i < input_size; i++) {
        // Check inclusive scan
        if (std::fabs(output_inclusive[i] - output_cpu[i]) >= EPSILON) {
            std::cerr << "Inclusive scan mismatch at index " << i
                    << ": expected " << output_cpu[i]
                    << ", got " << output_inclusive[i] << std::endl;
            success = false;
            break;
        }

        // Check exclusive scan
        float expected_exclusive = (i == 0) ? 0.0f : output_cpu[i - 1];
        if (std::fabs(output_exclusive[i] - expected_exclusive) >= EPSILON) {
            std::cerr << "Exclusive scan mismatch at index " << i
                    << ": expected " << expected_exclusive
                    << ", got " << output_exclusive[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "✅ All results are correct!" << std::endl;
    }

    return 0;
}