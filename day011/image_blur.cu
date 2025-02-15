#include <iostream>
#include <cuda_runtime.h>

#define BLUR_SIZE 1

__global__ void imageBlurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    unsigned int columnIndex  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowIndex < h && columnIndex < w) {
        int pixelVal = 0;
        int numPixels = 0;
        for (int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i) {
            for (int j = -BLUR_SIZE; j < BLUR_SIZE + 1; ++j) {
                int currentRow = rowIndex + i;
                int currentColumn = columnIndex + j;
                if (currentRow > -1 && currentRow < h && currentColumn > -1 && currentColumn < w) {
                    pixelVal += in[currentRow * w + currentColumn];
                    numPixels++;
                }
            }
        }
        out[rowIndex * w + columnIndex] = (unsigned char)(pixelVal / numPixels);
    }
}

void testImageBlurKernel() {
    const int width = 5, height = 5;
    unsigned char h_in[width * height];
    unsigned char h_out[width * height];

    for (int i = 0; i < width * height; ++i) {
        h_in[i] = i * 10 % 256;
    }

    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, width * height * sizeof(unsigned char));
    cudaMemcpy(d_in, h_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    imageBlurKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);
    cudaMemcpy(h_out, d_out, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::cout << "Blurred Image Output:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)h_out[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    testImageBlurKernel();
    return 0;
}
