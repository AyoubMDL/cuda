#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#define INPUT_SIZE 1000000
#define KERNEL_SIZE 9

__constant__ float W[KERNEL_SIZE];


__global__ void conv1d_kernel_v0(const float *input, const float *weights, float *output, const int kernel_size, const int input_size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < input_size) {
        int starting_idx = index - (kernel_size / 2);
        float out_value = 0.0f;

        for (int i = 0; i < kernel_size; ++i) {
            if (starting_idx + i >= 0 && starting_idx + i < input_size) {
                out_value += input[starting_idx + i] * weights[i];
            }
        }

        output[index] = out_value;
    }
}


// using constant memory
__global__ void conv1d_kernel_v1(const float *input, float *output, const int kernel_size, const int input_size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < input_size) {
        int starting_idx = index - (kernel_size / 2);
        float out_value = 0.0f;

        for (int i = 0; i < kernel_size; ++i) {
            if (starting_idx + i >= 0 && starting_idx + i < input_size) {
                out_value += input[starting_idx + i] * W[i];
            }
        }

        output[index] = out_value;
    }
}


void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void run_conv1d(float *d_input, float *d_weights, float *d_output, int kernel_size, int input_size, const std::string& version) {
    int threads_per_block = 256;
    int num_blocks = (input_size + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (version == "v1") {
        checkCuda(cudaMemcpyToSymbol(W, d_weights, kernel_size * sizeof(float)), "Copy weights to constant memory");
    }

    cudaEventRecord(start);
    
    if (version == "v0") {
        conv1d_kernel_v0<<<num_blocks, threads_per_block>>>(d_input, d_weights, d_output, kernel_size, input_size);
    } else if (version == "v1") {
        conv1d_kernel_v1<<<num_blocks, threads_per_block>>>(d_input, d_output, kernel_size, input_size);
    } else {
        fprintf(stderr, "Error: Unknown kernel version '%s'\n", version.c_str());
        return;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time (%s): %.5f ms\n", version.c_str(), milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    size_t input_size = INPUT_SIZE * sizeof(float);
    size_t kernel_size = KERNEL_SIZE * sizeof(float);

    // Allocate and Initialize Host Memory
    float *h_input = (float*)malloc(input_size);
    float *h_weights = (float*)malloc(kernel_size);
    float *h_output = (float*)malloc(input_size);

    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = 1.0f;
    }
    for (int i = 0; i < KERNEL_SIZE; i++) {
        h_weights[i] = 1.0f / KERNEL_SIZE;
    }

    // Allocate Device Memory
    float *d_input, *d_weights, *d_output;
    checkCuda(cudaMalloc(&d_input, input_size), "Alloc d_input");
    checkCuda(cudaMalloc(&d_weights, kernel_size), "Alloc d_weights");
    checkCuda(cudaMalloc(&d_output, input_size), "Alloc d_output");

    checkCuda(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "Copy input");
    checkCuda(cudaMemcpy(d_weights, h_weights, kernel_size, cudaMemcpyHostToDevice), "Copy weights");

    // Run and Compare Both Kernels
    run_conv1d(d_input, d_weights, d_output, KERNEL_SIZE, INPUT_SIZE, "v0"); // Naive
    run_conv1d(d_input, d_weights, d_output, KERNEL_SIZE, INPUT_SIZE, "v1");  // using constant memory

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    free(h_input);
    free(h_weights);
    free(h_output);

    return 0;
}