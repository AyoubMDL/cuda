#include <cuda_runtime.h>
#include <iostream>


#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << " - " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define BLOCK_SIZE 256
const float EPSILON = 1e-6;

typedef void (*ScanKernel)(const float*, float*, float*, int);


void scan_cpu(const float *input, float *output, const int input_size) {
    output[0] = input[0];
    for (int i = 1; i < input_size; ++i) {
        output[i] = input[i] + output[i - 1];
    }
}


__global__ void add_block_sums_kernel(float *output, float *block_sums, int input_size) {
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (blockIdx.x > 0) {
        float add_val = block_sums[blockIdx.x - 1];
        if (index < input_size) {
            output[index] += add_val;
        }
        if (index + BLOCK_SIZE < input_size) {
            output[index + BLOCK_SIZE] += add_val;
        }
    }
}



// Brent Kung 
__global__ void inclusive_scan_kernel(const float *input, float *output, float *block_sums, int input_size) {
    __shared__ float shared[BLOCK_SIZE * 2];
    
    // we have twice input elements as we have threads
    int segment = blockIdx.x * blockDim.x * 2;
    int tid = threadIdx.x;
    int index = segment + tid;

    shared[tid] = (index < input_size) ? input[index] : 0.0f;
    shared[tid + BLOCK_SIZE] = (index + BLOCK_SIZE < input_size) ? input[index + BLOCK_SIZE] : 0.0f;

    __syncthreads();

    // Reduction step
    for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int i = (tid + 1) * 2 * stride - 1;

        if (i < BLOCK_SIZE * 2) {
            shared[i] += shared[i - stride];
        }
        __syncthreads();
    }


    // Post-Reduction step
    for(int stride = BLOCK_SIZE / 2; stride >=1; stride /= 2) {
        int i = (tid + 1) * 2 * stride - 1;
        
        if (i + stride < BLOCK_SIZE * 2) {
            shared[i + stride] += shared[i];
        }
        __syncthreads();
    }

    if (index < input_size) {
        output[index] = shared[tid];
    }

    if (index + BLOCK_SIZE < input_size) {
        output[index + BLOCK_SIZE] = shared[tid + BLOCK_SIZE];
    }
    
    // Store the last element of each block for block sum scan
    if (tid == 0) {
        block_sums[blockIdx.x] = shared[2 * BLOCK_SIZE - 1];
    }

}


void scan(float *d_input, float *d_output, int input_size, ScanKernel scan_kernel) {
    // Config
    int threads_per_block = BLOCK_SIZE;
    int num_elements_per_block = 2 * threads_per_block;
    int num_blocks = (input_size + num_elements_per_block - 1) / num_elements_per_block;
    printf("Num blocks %d\n", num_blocks);
    
    float *d_block_sums;
    CHECK_CUDA(cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float)));
    
    // Step 1: Block-wise scan using the provided kernel
    scan_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, d_block_sums, input_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    if(num_blocks > 1) {
        // Scan block sums
        scan(d_block_sums, d_block_sums, num_blocks, inclusive_scan_kernel);

        // Add block sums to each block's output
        add_block_sums_kernel<<<num_blocks, threads_per_block>>>(d_output, d_block_sums, input_size);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Free mem
    CHECK_CUDA(cudaFree(d_block_sums));
}


int main() {
    const int input_size = 4000;
    float input[input_size], output_inclusive[input_size];
    for (int i = 0; i < input_size; i++) {
        input[i] = static_cast<float>(i + 1);
    }
    
    float *d_input, *d_output, *output_cpu;
    size_t size = input_size * sizeof(float);
    output_cpu = (float *)malloc(size);
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));
    
    CHECK_CUDA(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    
    // Inclusive scan
    scan(d_input, d_output, input_size, inclusive_scan_kernel);
    CHECK_CUDA(cudaMemcpy(output_inclusive, d_output, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    scan_cpu(input, output_cpu, input_size);
    CHECK_CUDA(cudaFree(d_input));
    
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
    }

    if (success) {
        std::cout << "✅ All results are correct!" << std::endl;
    }

    return 0;
}