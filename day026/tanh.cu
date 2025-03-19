#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>

#define BLOCK_SIZE 256


__global__ void tanh_forward_kernel(const float *input, float *output, const int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = tanhf(input[index]);
    }
}


__global__ void tanh_backward_kernel(const float* input, const float* grad_output, float* grad_input, const int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float tanh_val = tanhf(input[index]);
        grad_input[index] = grad_output[index] * (1.0f - tanh_val * tanh_val);
    }
}


torch::Tensor tanh_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int size = input.numel();
    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil((float)(size) / BLOCK_SIZE));

    tanh_forward_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}


torch::Tensor tanh_backward(torch::Tensor input, torch::Tensor grad_output) {
    auto grad_input = torch::empty_like(grad_output);

    const int size = input.numel();
    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil((float)(size) / BLOCK_SIZE));

    tanh_backward_kernel<<<grid, block>>>(input.data_ptr<float>(), grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), size);
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tanh_forward, "Tanh forward (CUDA)");
    m.def("backward", &tanh_backward, "Tanh backward (CUDA)");
}