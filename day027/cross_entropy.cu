#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void cross_entropy_forward_kernel(const float* logits, const long *targets, float* outputs, const int batch_size, const int num_classes) {
    // logits shape (batch_size, num_classes)
    // targets shape (batch_size,)
    // outputs shape (batch_size,)
    // CE(logits, target) = -x_class + log(sum_exp)
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        const float* sample_logits = logits + batch_idx * num_classes; // pointer arithmetics (row major)
        int target = targets[batch_idx];
        
        // Step 1: Compute the denominator (sum of exp)
        float max_logit = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, sample_logits[i]);  // Avoid overflow
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(sample_logits[i] - max_logit);
        }

        float log_prob = sample_logits[target] - max_logit - logf(sum_exp);
        outputs[batch_idx] = -log_prob;
    }
}


__global__ void cross_entropy_backward_kernel(const float* logits, const long *targets, float* grad_logits, const int batch_size, const int num_classes) {
    // derivative(CE, xi) = softmax(xi) + 1(i = target_class)
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        const float* sample_logits = logits + batch_idx * num_classes;
        float* sample_grad = grad_logits + batch_idx * num_classes;
        int target = targets[batch_idx];

        float max_logit = -FLT_MAX;
        for (int i = 0; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, sample_logits[i]);  // Avoid overflow
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(sample_logits[i] - max_logit);
        }

        for (int i = 0; i < num_classes; i++) {
            float softmax_i = expf(sample_logits[i] - max_logit) / sum_exp;
            sample_grad[i] = (softmax_i - (i == target ? 1.0f : 0.0f));
        }
    }
}


torch::Tensor cross_entropy_forward(torch::Tensor input, torch::Tensor target, std::string reduction = "mean") {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor (batch_size, num_classes)");
    TORCH_CHECK(target.dim() == 1, "Target must be a 1D tensor (batch_size)");
    TORCH_CHECK(input.size(0) == target.size(0), "Batch size of input and target must match");

    int batch_size = input.size(0);
    int num_classes = input.size(1);

    auto output = torch::empty({batch_size}, input.options());

    const int threads = min(1024, batch_size);
    const int blocks = ceil(batch_size / 256.);

    cross_entropy_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        target.data_ptr<long>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes
    );

    if (reduction == "mean") {
        return output.mean();
    } else { // "none"
        return output;
    }
    return output;
}


torch::Tensor cross_entropy_backward(torch::Tensor input, torch::Tensor target, torch::Tensor grad_outputs = torch::Tensor(), std::string reduction = "mean") {
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor (batch_size, num_classes)");
    TORCH_CHECK(target.dim() == 1, "Target must be a 1D tensor (batch_size)");
    TORCH_CHECK(input.size(0) == target.size(0), "Batch size of input and target must match");
    // TORCH_CHECK(input.size(0) == grad_outputs.size(0), "Batch size of grad_outputs must match input");

    if (grad_outputs.numel() == 0) {
        grad_outputs = torch::ones_like(input);
    }

    // Ensure grad_outputs has the same shape as input (expand if necessary)
    if (grad_outputs.dim() == 1) {
        grad_outputs = grad_outputs.view({input.size(0), 1}).expand({input.size(0), input.size(1)});
    }

    int batch_size = input.size(0);
    int num_classes = input.size(1);

    auto grad_logits = torch::empty_like(input);

    const int threads = min(1024, batch_size);
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_backward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        target.data_ptr<long>(),
        grad_logits.data_ptr<float>(),
        batch_size,
        num_classes
    );

    grad_logits.mul_(grad_outputs);

    if (reduction == "mean") {
        grad_logits.div_(input.size(0)); // Divide by batch_size
    }

    return grad_logits;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cross_entropy_forward, "Cross Entropy forward");
    m.def("backward", &cross_entropy_backward, "Cross Entropy forward");
}
