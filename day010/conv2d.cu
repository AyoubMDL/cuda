#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void conv2dKernel(const float* input, const float* kernel, float* output,
                              int input_height, int input_width,
                              int kernel_height, int kernel_width,
                              int output_height, int output_width) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < output_height && out_col < output_width) {
        float value = 0.0f;
        for (int k_row = 0; k_row < kernel_height; ++k_row) {
            for (int k_col = 0; k_col < kernel_width; ++k_col) {
                int in_row = out_row + k_row;
                int in_col = out_col + k_col;
                value += input[in_row * input_width + in_col] * kernel[k_row * kernel_width + k_col];
            }
        }
        output[out_row * output_width + out_col] = value;
    }
}

void conv2d(const float* input, const float* kernel, float* output,
            int input_height, int input_width,
            int kernel_height, int kernel_width) {
    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    float *d_input, *d_kernel, *d_output;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_input, input_height * input_width * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_kernel, kernel_height * kernel_width * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, output_height * output_width * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, input, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, kernel, kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    conv2dKernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                         input_height, input_width,
                                         kernel_height, kernel_width,
                                         output_height, output_width);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, d_output, output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    // Example input image and kernel
    const int input_height = 5;
    const int input_width = 5;
    const int kernel_height = 3;
    const int kernel_width = 3;

    float input[input_height * input_width] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    float kernel[kernel_height * kernel_width] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    float output[output_height * output_width];

    // Perform convolution
    conv2d(input, kernel, output, input_height, input_width, kernel_height, kernel_width);

    // Print output
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            std::cout << output[i * output_width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}