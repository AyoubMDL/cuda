#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define O_TILE_SIZE 2
#define KERNEL_SIZE 3

__constant__ float W[KERNEL_SIZE][KERNEL_SIZE];


__global__ void tiled_conv2d_kernel(const float *input, float *output, int height, int width, int pitch) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_o = blockIdx.y * O_TILE_SIZE + ty;
    int column_o = blockIdx.x * O_TILE_SIZE + tx;

    int row_i = row_o - (KERNEL_SIZE / 2);
    int column_i = column_o - (KERNEL_SIZE / 2);
    
    // Declare shared memory
    __shared__ float sharedMem[O_TILE_SIZE + KERNEL_SIZE - 1][O_TILE_SIZE + KERNEL_SIZE - 1];

    // Load to shared memory
    if ((row_i >= 0 && row_i < height) && (column_i >= 0 && column_i < width)) {
        sharedMem[ty][tx] = input[row_i * pitch + column_i];
    } else {
        sharedMem[ty][tx] = 0.0f;
    }

    float out_value = 0.0f;

    if (ty < O_TILE_SIZE && tx < O_TILE_SIZE) {
        for(int i = 0; i < KERNEL_SIZE; ++i) {
            for(int j = 0; j < KERNEL_SIZE; ++j) {
                out_value += W[i][j] * sharedMem[i + ty][j + tx];
            }
        }
        if(row_o < height && column_o < width){
            output[row_o * width + column_o] = out_value;
        }
    }
}


void conv2d_cpu(const float *input, float *output, const float *kernel, int width, int height, int kernel_size) {
    int half_k = kernel_size / 2;
  
    // Perform convolution
    for (int i = 0; i < height; ++i) {  // Iterate over height first for better memory access
        for (int j = 0; j < width; ++j) {
            int output_index = i * width + j;  // linearize index
            float out_value = 0.0f;
  
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    int input_index_i = i + ki - half_k;
                    int input_index_j = j + kj - half_k;
  
                    // Ensure input indices are within bounds
                    if (input_index_i >= 0 && input_index_i < height && 
                        input_index_j >= 0 && input_index_j < width) {
                        out_value += input[input_index_i * width + input_index_j] * kernel[ki * kernel_size + kj];
                    }
                }
            }
            output[output_index] = out_value;
        }
    }
}


bool compare_results(float *cpu, float *gpu, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu[i] << ", GPU=" << gpu[i] << "\n";
            return false;
        }
    }
    return true;
}


void print_matrix(const float *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << std::setw(4) << matrix[i * width + j] << " ";
        }
        std::cout << "\n";
    }
  }


int main() {
    const int width = 5, height = 5;
    float h_input[width * height] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    float h_output_cpu[width * height] = {0};
    float h_output_gpu[width * height] = {0};

    // Run CPU version
    conv2d_cpu(h_input, h_output_cpu, h_kernel, width, height, KERNEL_SIZE);

    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(W, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Define block and grid sizes
    int tile_size = O_TILE_SIZE + KERNEL_SIZE - 1;
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim((height + O_TILE_SIZE - 1) / O_TILE_SIZE, 
                (width + O_TILE_SIZE - 1) / O_TILE_SIZE);


    // Launch kernel
    tiled_conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_output, height, width, width);
    cudaDeviceSynchronize();

    // Copy GPU result back to host
    cudaMemcpy(h_output_gpu, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    if (compare_results(h_output_cpu, h_output_gpu, width * height)) {
        std::cout << "Results match!\n";
    } else {
        std::cout << "Results do not match!\n";
    }

    std::cout << "\nOutput Image (Before Convolution):\n";
    print_matrix(h_input, width, height);

    std::cout << "\nOutput Image CPU (After Convolution):\n";
    print_matrix(h_output_cpu, width, height);

    std::cout << "\nOutput Image GPU (After Convolution):\n";
    print_matrix(h_output_gpu, width, height);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}