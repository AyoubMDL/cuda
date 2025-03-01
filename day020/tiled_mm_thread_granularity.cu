#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 32


__global__ void tiledMatMulOptimized(const float *A, const float *B, float *C, int width) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    // Load two columns of B matrix
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH * 2];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + ty;

    int col1 = blockIdx.x * TILE_WIDTH * 2 + tx;
    int col2 = col1 + TILE_WIDTH;

    float Cvalue1 = 0;
    float Cvalue2 = 0;

    for (int t = 0; t < ceil((float)width / TILE_WIDTH); ++t) {
        // Load A tile
        int tileRow = row;
        int tileCol = t * TILE_WIDTH + tx;
        sharedA[ty][tx] = A[tileRow * width + tileCol];
        
        // Load two columns of N tile
        tileRow = t * TILE_WIDTH + ty;
        tileCol = col1;
        sharedB[ty][tx] = B[tileRow * width + tileCol];
        if (col2 < width) {
            sharedB[ty][tx + TILE_WIDTH] = B[tileRow * width + col2];
        }
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue1 += sharedA[ty][k] * sharedB[k][tx];
            if (col2 < width) {
                Cvalue2 += sharedA[ty][k] * sharedB[k][tx + TILE_WIDTH];
            }
        }
        __syncthreads();
    }
    C[row * width + col1] = Cvalue1;
    if (col2 < width) {
        C[row * width + col2] = Cvalue2;
    }
}


int main() {
    const int width = 2048;
    float *A, *B, *C;

    // Allocate pinned host memory (for better transfer performance)
    cudaMallocHost((void**)&A, width * width * sizeof(float));
    cudaMallocHost((void**)&B, width * width * sizeof(float));
    cudaMallocHost((void**)&C, width * width * sizeof(float));

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            A[i * width + j] = 1.0f;
            B[i * width + j] = (i == j) ? 1.0f : 0.0f;  // Identity matrix
            C[i * width + j] = 0.0f;
        }
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, width * width * sizeof(float));
    cudaMalloc(&dB, width * width * sizeof(float));
    cudaMalloc(&dC, width * width * sizeof(float));

    cudaMemcpy(dA, A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        ceil((float) width / (TILE_WIDTH * 2)), // cols
        ceil((float) width / TILE_WIDTH) // rows
    );

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiledMatMulOptimized<<<grid, block>>>(dA, dB, dC, width);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, dC, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tiled Matmul Time with Thread granularity: " << milliseconds << " ms" << std::endl;

    // Validate result (C = A @ B = A ; B = I)
    bool valid = true;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (C[i * width + j] != A[i * width + j]) {
                valid = false;
                break;
            }
        }
    }
    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}