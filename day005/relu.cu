#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>

__global__ void reluCuda(const float *X, float *Y, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        Y[index] = fmaxf(0.0f, X[index]);
    }
}

torch::Tensor reluForward(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "Input must be a CUDA tensor");
    int N = x.numel();
    auto y = torch::zeros_like(x);

    int threads = 256;
    int blocks = ceil(N / threads);
    reluCuda<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &reluForward, "ReLU forward (CUDA)");
}