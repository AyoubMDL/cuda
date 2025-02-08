import torch
import custom_cuda
from common.utils import compute_time_execution, compare_outputs

x = torch.randn(10_000_000, device="cuda")


def pytorch_relu():
    return torch.relu(x)


def cuda_relu():
    return custom_cuda.relu_forward(x)


# Compute and print execution times
print("Running PyTorch ReLU...")
pytorch_output = compute_time_execution(pytorch_relu)

print("\nRunning Custom CUDA ReLU...")
cuda_output = compute_time_execution(cuda_relu)

# Compare results
print("\nComparing outputs...")
compare_outputs(cuda_output, pytorch_output, tolerance=1e-5)
