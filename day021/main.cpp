#include <cuda_runtime.h>
#include <iostream>
#include "kernel_launcher.h"


bool check_result(const float *d_output, int numBlocks, float expected_sum) {
    float *h_output = new float[numBlocks];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum all block results
    float total_sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        total_sum += h_output[i];
    }

    delete[] h_output;
    return (total_sum == expected_sum);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_number>" << std::endl;
        return 1;
    }

    int kernel_num = std::stoi(argv[1]);

    const int N = 4 * 1024 * 1024;  // 4M floats
    const int bytes = N * sizeof(float);

    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA(cudaMalloc((void**)&d_output, numBlocks * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Call the appropriate kernel launcher
    switch (kernel_num) {
        case 0: launch_reduce0(d_input, d_output, N, numBlocks, threadsPerBlock); break;
        case 1: launch_reduce1(d_input, d_output, N, numBlocks, threadsPerBlock); break;
        case 2: launch_reduce2(d_input, d_output, N, numBlocks, threadsPerBlock); break;
        case 3: launch_reduce3(d_input, d_output, N, numBlocks / 2, threadsPerBlock); break; // halve the number of blocks
        case 4: launch_reduce4(d_input, d_output, N, numBlocks / 2, threadsPerBlock); break; // halve the number of blocks
        case 5: launch_reduce5(d_input, d_output, N, numBlocks / 2, threadsPerBlock); break;
        default:
            std::cerr << "Unknown kernel number: " << kernel_num << std::endl;
            return 1;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    double seconds = milliseconds / 1000.0;
    double totalBytes = bytes + numBlocks * sizeof(float);  // Read input and write output
    double bandwidth = (totalBytes / seconds) / 1.0e9;  // Bandwidth in GB/s

    std::cout << "Kernel " << kernel_num << " Time: " << milliseconds << " ms\n";
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s\n";

    // Check result
    float expected_sum = static_cast<float>(N);  // Since each input element is 1.0f, expected sum is N
    bool result_correct = check_result(d_output, numBlocks, expected_sum);

    if (result_correct) {
        std::cout << "Result is correct!" << std::endl;
    } else {
        std::cerr << "Result is incorrect!" << std::endl;
    }

    delete[] h_input;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
