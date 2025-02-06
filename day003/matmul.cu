#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

typedef struct {
    int rows;
    int columns;
    float *data;
} Matrix;

Matrix createMatrix(int rows, int columns) {
    Matrix mat;
    mat.rows = rows;
    mat.columns = columns;
    mat.data = (float *)malloc(rows * columns * sizeof(float));
    return mat;
}

__global__ void matmul(const float *A, const float *B, float *C, int rowsA,
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

void matmulCpu(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A->columns != B->rows) {
        printf("Matrix dimensions are incompatible for multiplication!\n");
        return;
    }

    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->columns; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A->columns; ++k) {
                sum +=
                    A->data[i * A->columns + k] * B->data[k * B->columns + j];
            }
            C->data[i * B->columns + j] = sum;
        }
    }
}

int main() {
    int aRows = 1000;
    int aColumns = 1000;
    int bRows = 1000;
    int bColumns = 1000;

    Matrix A = createMatrix(aRows, aColumns);
    Matrix B = createMatrix(bRows, bColumns);
    Matrix cCPU = createMatrix(aRows, bColumns);
    Matrix cGPU = createMatrix(aRows, bColumns);

    // Initialize input vectors
    initializeVectors(A.data, B.data, A.rows * A.columns);

    // Measure CPU time for matrix multiplication
    double cpuTime = measureExecutionTime([&]() { matmulCpu(&A, &B, &cCPU); });
    std::cout << "CPU execution time: " << cpuTime << " ms" << std::endl;

    // Allocate device memory
    float *A_device, *B_device, *C_device;
    int sizeA = A.rows * A.columns * sizeof(float);
    int sizeB = B.rows * B.columns * sizeof(float);
    int sizeC = A.rows * B.columns * sizeof(float);
    cudaMalloc((void **)&A_device, sizeA);
    cudaMalloc((void **)&B_device, sizeB);
    cudaMalloc((void **)&C_device, sizeC);

    // Copy data from host to device
    cudaMemcpy(A_device, A.data, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B.data, sizeB, cudaMemcpyHostToDevice);

    // define grid and block dims
    int numBlocks = 64;
    int blockSize = 16;
    dim3 grid(numBlocks, numBlocks);
    dim3 block(blockSize, blockSize, 1);

    double gpuTime = measureExecutionTime([&]() {
        matmul<<<grid, block>>>(A_device, B_device, C_device, A.rows, A.columns,
                                B.columns);
        cudaDeviceSynchronize();
    });
    std::cout << "GPU execution time: " << gpuTime << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(cGPU.data, C_device, sizeC, cudaMemcpyDeviceToHost);

    bool success =
        compareResults(cCPU.data, cGPU.data, cCPU.rows * cCPU.columns);
    std::cout << (success ? "CPU and GPU results match!" : "Results mismatch!")
              << std::endl;

    // Free device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Free host memory
    free(A.data);
    free(B.data);
    free(cCPU.data);
    free(cGPU.data);

    return 0;
}