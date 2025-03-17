#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

#define INPUT_SIZE 10000000
#define KERNEL_SIZE 9
#define TILE_SIZE 256


__constant__ float W[KERNEL_SIZE];


void conv1d_cpu(const float *input, float *output, const float *kernel, int input_size, int kernel_size) {
    int half_k = kernel_size / 2;

    // Perform convolution
    for (int i = 0; i < input_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            int input_index = i + j - half_k;
            if (input_index >= 0 && input_index < input_size) {
                sum += input[input_index] * kernel[j];
            }
        }
        output[i] = sum;
    }
}


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


// using shared memory and halo cells
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


// Using general caching for halo cells
__global__ void conv1d_kernel_v3(const float *input, float *output, const int kernel_size, const int input_size) {
    unsigned int tid = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + tid;

    __shared__ float sharedMem[TILE_SIZE];
    sharedMem[tid] = input[index];

    __syncthreads();

    int this_tile_starting_point = blockIdx.x * blockDim.x;
    int next_tile_starting_point = (blockIdx.x + 1) * blockDim.x;
    int input_start_point = index - (kernel_size / 2);

    float out_value = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        int input_index = input_start_point + i;

        // Handle ghost cells
        if (input_index >=0 && input_index < input_size) {
            // Handle halo cells that are not ghost
            if((input_index >= this_tile_starting_point) && (input_index < next_tile_starting_point)) {
                out_value += sharedMem[tid + i - (kernel_size / 2)] * W[i];
            } else {
                out_value += input[input_index] * W[i];
            }
        }
    }
    output[index] = out_value;
}



void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void run_conv1d(float *d_input, float *d_weights, float *d_output, float *expected, int kernel_size, int input_size, const std::string& version) {
    int threads_per_block = TILE_SIZE;
    int num_blocks = (input_size + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (version == "v1" || version == "v2") {
        checkCuda(cudaMemcpyToSymbol(W, d_weights, kernel_size * sizeof(float)), "Copy weights to constant memory");
    }

    cudaEventRecord(start);
    
    if (version == "v0") {
        conv1d_kernel_v0<<<num_blocks, threads_per_block>>>(d_input, d_weights, d_output, kernel_size, input_size);
    } else if (version == "v1") {
        conv1d_kernel_v1<<<num_blocks, threads_per_block>>>(d_input, d_output, kernel_size, input_size);
    } else if (version == "v2") {
        conv1d_kernel_v2<<<num_blocks, threads_per_block>>>(d_input, d_output, kernel_size, input_size);
    } else if (version == "v3") {
        conv1d_kernel_v3<<<num_blocks, threads_per_block>>>(d_input, d_output, kernel_size, input_size);
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv1d_kernel_%s: %s\n", version.c_str(), cudaGetErrorString(err));
        return;
    }

    // Copy result from device to host
    float *h_output = (float*)malloc(input_size * sizeof(float));
    checkCuda(cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost), "Copy output from device to host");

    // Compare results
    int mismatch_count = 0;
    for (int i = 0; i < input_size; i++) {
        if (fabs(h_output[i] - expected[i]) > 1e-5) {
            printf("Mismatch at index %d: Expected %.6f, Got %.6f\n", i, expected[i], h_output[i]);
            mismatch_count++;
            if (mismatch_count > 10) {  // Stop after 10 mismatches
                printf("Too many mismatches, stopping comparison...\n");
                break;
            }
        }
    }

    if (mismatch_count == 0) {
        printf("All values match expected results! ✅\n");
    } else {
        printf("Total mismatches: %d ❌\n", mismatch_count);
    }

    // Cleanup
    free(h_output);

}


int main() {
    size_t input_size = INPUT_SIZE * sizeof(float);
    size_t kernel_size = KERNEL_SIZE * sizeof(float);

    // Allocate and Initialize Host Memory
    float *h_input = (float*)malloc(input_size);
    float *h_weights = (float*)malloc(kernel_size);
    float *h_output = (float*)malloc(input_size);
    float *cpu_output = (float*)malloc(input_size);

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

    // CPU
    conv1d_cpu(h_input, cpu_output, h_weights, INPUT_SIZE, KERNEL_SIZE);

    // Run and Compare Both Kernels
    run_conv1d(d_input, d_weights, d_output, cpu_output, KERNEL_SIZE, INPUT_SIZE, "v0"); // Naive
    run_conv1d(d_input, d_weights, d_output, cpu_output, KERNEL_SIZE, INPUT_SIZE, "v1");  // using constant memory
    run_conv1d(d_input, d_weights, d_output, cpu_output, KERNEL_SIZE, INPUT_SIZE, "v2");  // using halo cells
    run_conv1d(d_input, d_weights, d_output, cpu_output, KERNEL_SIZE, INPUT_SIZE, "v3");  // using general caching

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    free(h_input);
    free(h_weights);
    free(h_output);

    return 0;
}