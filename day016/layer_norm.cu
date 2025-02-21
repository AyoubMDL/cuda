#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <chrono>


#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

float benchmarkCPUKernel(void (*kernel)(const float *, float *, int, float, float, float), 
    const float *input, float *output, int size, float gamma, float beta, float epsilon = 1e-8) {
auto start = std::chrono::high_resolution_clock::now();

kernel(input, output, size, gamma, beta, epsilon);

auto stop = std::chrono::high_resolution_clock::now();
std::chrono::duration<float, std::milli> duration = stop - start;

return duration.count(); // Return time in milliseconds
}


void cpuLayerNorm(const float *input, float *output, int size, float gamma, float beta, float epsilon = 1e-8) {
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    float mean = sum / size;

    // Compute variance
    float variance_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / size;
    float std = std::sqrt(variance + epsilon);

    // Normalize and apply gamma, beta
    for (int i = 0; i < size; i++) {
        output[i] = ((input[i] - mean) / std) * gamma + beta;
    }
}

__global__ void sumKernel(const float *input, float *sum, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (index < size) ? input[index] : 0.0f;
    __syncthreads();

    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] =  sdata[0];
    }
}

__global__ void varianceKernel(const float *input, float *variance, float mean, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    float diff = (index < size) ? (input[index] - mean) : 0.0f;
    sdata[tid] = diff * diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        variance[blockIdx.x] =  sdata[0];
    }

}


__global__ void layerNormKernel(const float *input, float *output, float mean, float std, float gamma, float beta, int size, const float epsilon) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float numerator = input[index] - mean;
        float denominator = std + epsilon;
        output[index] = ((numerator * gamma) / denominator) + beta;
    }
}


void cudaLayerNorm(const float *input, float *output, int size, float gamma, float beta, const float epsilon = 1e-8) {
    float *d_input, *d_output, *d_sum, *d_variance;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMalloc(&d_sum, blocksPerGrid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_variance, blocksPerGrid * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Compute sum and then the threadsPerBlock
    sumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_sum, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<float> h_sums(blocksPerGrid);
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sum, blocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        sum += h_sums[i];
    }
    float mean = sum / size;

    // Compute variance and then std
    varianceKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_variance, mean, size);
    CUDA_CHECK(cudaDeviceSynchronize());
   
    std::vector<float> h_variance(blocksPerGrid);
    CUDA_CHECK(cudaMemcpy(h_variance.data(), d_variance, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    float square_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        square_sum += h_variance[i];
    }
    // Divide by size to get the final variance
    float variance = square_sum / size;
    // Square root to get std
    float std = sqrtf(variance);


    layerNormKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, mean, std, gamma, beta, size, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_variance));
}


int main() {
    int size = 100000;
    std::vector<float> h_input(size);
    std::vector<float> h_output(size);
    std::vector<float> h_output_cpu(size);

    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i % 10);
    }

    float gamma = 1.0f;
    float beta = 0.0f;

    float gpu_time = benchmarkCPUKernel(cudaLayerNorm, h_input.data(), h_output.data(), size, gamma, beta);
    float cpu_time = benchmarkCPUKernel(cpuLayerNorm, h_input.data(), h_output_cpu.data(), size, gamma, beta);
    
    std::cout << "GPU LayerNorm Time: " << gpu_time << " ms" << std::endl;
    std::cout << "CPU LayerNorm Time: " << cpu_time << " ms" << std::endl;

    bool match = true;
    for (int i = 0; i < size; i++) {
        float diff = fabs(h_output[i] - h_output_cpu[i]);
        if (diff > 1e-5) match = false;
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do NOT match!" << std::endl;
    }

    return 0;

}