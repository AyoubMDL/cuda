#pragma once

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void launch_reduce0(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);
void launch_reduce1(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);
void launch_reduce2(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);
void launch_reduce3(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);
void launch_reduce4(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);
void launch_reduce5(const float *d_input, float *d_output, int size, int numBlocks, int threadsPerBlock);

