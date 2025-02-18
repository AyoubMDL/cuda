#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

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

    sdata[tid] = (index < size) ? fabsf(input[index]) : 0.0f;
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

__global__ void quantizeKernel(const float *input, int8_t *output, float scale,
                               int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        int8_t quantized = roundf(input[index] / scale);
        quantized = max(-128, min(127, quantized));
        output[index] = quantized;
    }
}

__global__ void dequantizeKernel(const int8_t *input, float *output,
                                 float scale, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        // Convert back to float
        output[index] = scale * input[index];
    }
}

void quantizeDequantize(float *h_input, float *h_output, int size) {
    float *d_input, *d_output, *d_max_vals;
    int8_t *d_quantized;

    CUDA_CHECK(cudaMalloc((void **)&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_quantized, size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(float(size) / threadsPerBlock);

    // Compute scale
    CUDA_CHECK(cudaMalloc(&d_max_vals, blocksPerGrid * sizeof(float)));
    reduceMaxKernel<<<blocksPerGrid, threadsPerBlock,
                      threadsPerBlock * sizeof(float)>>>(d_input, d_max_vals,
                                                         size);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_max_vals[blocksPerGrid];
    CUDA_CHECK(cudaMemcpy(h_max_vals, d_max_vals, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_val = *std::max_element(h_max_vals, h_max_vals + blocksPerGrid);

    float scale = (max_val > 0) ? (max_val / 127.0f) : 1e-6f;

    // Quantization
    quantizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_quantized,
                                                       scale, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantization
    dequantizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_quantized, d_output,
                                                         scale, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    const int size = 1000000;
    float *h_input = new float[size];
    float *h_output = new float[size];

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 100;
    }

    float time = benchmarkKernel(quantizeDequantize, h_input, h_output, size);

    std::cout << "Original -> Dequantized:" << std::endl;
    for (int i = 100; i < 200; i++) {
        std::cout << h_input[i] << " -> " << h_output[i] << std::endl;
    }
    std::cout << "Number of elements : " << size << std::endl;
    std::cout << "Quantization time: " << time << " ms" << std::endl;

    delete[] h_input;
    delete[] h_output;
    return 0;
}