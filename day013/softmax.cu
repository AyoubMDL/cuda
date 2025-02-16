#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

float benchmarkKernel(void (*kernel)(float *, float *, int), float *d_input,
                      float *d_output, int size) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel(d_input, d_output, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

__global__ void reduceMaxKernel(const float *input, float *output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (index < size) ? input[index] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceSumKernel(const float *input, float *output,
                                float max_val, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (index < size) ? expf(input[index] - max_val) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void optimizedSoftmaxKernel(const float *input, float *output,
                                       float max_val, float sum, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = expf(input[index] - max_val) / sum;
    }
}

void optimizedSoftmax(float *input, float *output, int size) {
    float *d_input, *d_output, *d_max_vals, *d_sums;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMalloc(&d_max_vals, blocksPerGrid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, blocksPerGrid * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    reduceMaxKernel<<<blocksPerGrid, threadsPerBlock,
                      threadsPerBlock * sizeof(float)>>>(d_input, d_max_vals,
                                                         size);

    float h_max_vals[blocksPerGrid];
    CUDA_CHECK(cudaMemcpy(h_max_vals, d_max_vals, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_val = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) {
        max_val = fmaxf(max_val, h_max_vals[i]);
    }

    reduceSumKernel<<<blocksPerGrid, threadsPerBlock,
                      threadsPerBlock * sizeof(float)>>>(d_input, d_sums,
                                                         max_val, size);

    float h_sums[blocksPerGrid];
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_sums[i];
    }

    optimizedSoftmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, max_val, sum, size);

    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_max_vals));
    CUDA_CHECK(cudaFree(d_sums));
}

__global__ void naiveSoftmaxKernel(const float *input, float *output,
                                   int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float max_val = input[0];
        for (int i = 1; i < size; ++i) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += expf(input[i] - max_val);
        }

        output[index] = expf(input[index] - max_val) / sum;
    }
}

int main() {
    const int size = 100000;
    float *h_input = new float[size];
    float *h_output_naive = new float[size];
    float *h_output_optimized = new float[size];

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_input, *d_output_naive, *d_output_optimized;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_naive, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_optimized, size * sizeof(float)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch Naïve Softmax
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    naiveSoftmaxKernel<<<gridSize, blockSize>>>(d_input, d_output_naive, size);

    // Measure Naïve Softmax time
    float naive_time = benchmarkKernel(
        [](float *d_input, float *d_output, int size) {
            naiveSoftmaxKernel<<<(size + 255) / 256, 256>>>(d_input, d_output,
                                                            size);
            cudaDeviceSynchronize();
        },
        d_input, d_output_naive, size);

    // Measure Optimized Softmax time
    float optimized_time =
        benchmarkKernel(optimizedSoftmax, d_input, d_output_optimized, size);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_optimized, d_output_optimized,
                          size * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate results
    bool match = true;
    for (int i = 0; i < size; i++) {
        float diff = fabs(h_output_naive[i] - h_output_optimized[i]);
        if (diff > 1e-8) match = false;
    }

    std::cout << "Naïve Softmax: " << naive_time << " ms" << std::endl;
    std::cout << "Optimized Softmax: " << optimized_time << " ms" << std::endl;
    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do NOT match!" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_naive));
    CUDA_CHECK(cudaFree(d_output_optimized));
    delete[] h_input;
    delete[] h_output_naive;
    delete[] h_output_optimized;

    return 0;
}