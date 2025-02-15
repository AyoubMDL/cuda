#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "common.h"
#define TILE_SIZE 32 // Tile shape and block shape must be the same

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


bool compareResults(const std::vector<float>& A, const std::vector<float>& B, float epsilon = 1e-5) {
    for (size_t i = 0; i < A.size(); ++i) {
        if (std::fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

__global__ void naiveMatmul(const float *A, const float *B, float *C, int rowsA,
    int colsA, int colsB) {
int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;

if (rowIndex < rowsA && columnIndex < colsB) {
float sum = 0.0f;
for (int k = 0; k < colsA; ++k) {
int aIndex = rowIndex * colsA + k;
int bIndex = k * colsB + columnIndex;
sum += A[aIndex] * B[bIndex];
}
C[rowIndex * colsB + columnIndex] = sum;
}
} 


__global__ void tiledMatmul(const float *A, const float *B, float *C, int rowsA, int colsA, int colsB) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * blockDim.y + ty;
    int column = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over tiles (colsA as it is the common dim between A and B)
    for (int t = 0; t < ceil((float)colsA / TILE_SIZE); ++t) {
        // Load tiles data to shared memory
        // 1. Row index is same as global row (e.g. C)
        // 2. Column index depends on the Tile number (TileNumber * TileSize + tx)
        int tileRow = row;
        int tileCol = t * TILE_SIZE + tx;
        sharedA[ty][tx] = (tileRow < rowsA && tileCol < colsA) ? A[tileRow * colsA + tileCol] : 0.0f;
        
        // 1. column index is the same as global column
        // 2. row index depends on the tile number (TileNumber * TileSize + ty)
        tileRow = t * TILE_SIZE + ty;
        tileCol = column;
        sharedB[ty][tx] = (tileRow < colsA && tileCol < colsB) ? B[tileRow * colsB + tileCol] : 0.0f; 
        
        // Sync
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }

    if (row < rowsA && column < colsB) {
        C[row * colsB + column] = sum;
    }
}


int main() {
    int rowsA = 2048, colsA = 1024, colsB = 2048;
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    std::vector<float> h_A(rowsA * colsA);
    std::vector<float> h_B(colsA * colsB);
    std::vector<float> h_C(rowsA * colsB, 0);
    std::vector<float> h_C_naive(rowsA * colsB, 0);

    // Initialize matrices A and B
    for (int i = 0; i < rowsA * colsA; ++i) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < colsA * colsB; ++i) h_B[i] = static_cast<float>((i + 1) * 2);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_naive;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));
    CUDA_CHECK(cudaMalloc(&d_C_naive, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    // Grid and Block
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid(ceil(colsB / TILE_SIZE), ceil(rowsA / TILE_SIZE), 1);

    // Timing setup
    cudaEvent_t start, stop;
    float elapsedTime;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run naive matrix multiplication
    CUDA_CHECK(cudaEventRecord(start));
    naiveMatmul<<<grid, block>>>(d_A, d_B, d_C_naive, rowsA, colsA, colsB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Naive Matmul Time: " << elapsedTime << " ms" << std::endl;

    // Run tiled matrix multiplication
    CUDA_CHECK(cudaEventRecord(start));
    tiledMatmul<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Tiled Matmul Time: " << elapsedTime << " ms" << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C_naive, sizeC, cudaMemcpyDeviceToHost));

    // Compare results
    if (compareResults(h_C, h_C_naive)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do NOT match!" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}