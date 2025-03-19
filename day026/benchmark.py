import time
import torch
from tanh_custom import tanh

# Create input tensor
x = torch.randn(1000000, device="cuda", requires_grad=True)
grad_output = torch.ones_like(x)

# Measure PyTorch's execution time
torch.cuda.synchronize()
start = time.time()
y_torch = torch.tanh(x)
torch.cuda.synchronize()
torch_forward_time = time.time() - start

torch.cuda.synchronize()
start = time.time()
y_torch.backward(grad_output, retain_graph=True)
torch.cuda.synchronize()
torch_backward_time = time.time() - start

# Measure Custom CUDA execution time
torch.cuda.synchronize()
start = time.time()
y_cuda = tanh(x)
torch.cuda.synchronize()
cuda_forward_time = time.time() - start

torch.cuda.synchronize()
start = time.time()
y_cuda.backward(grad_output)
torch.cuda.synchronize()
cuda_backward_time = time.time() - start

print(f"PyTorch Forward Time: {torch_forward_time:.6f} sec")
print(f"CUDA Forward Time:   {cuda_forward_time:.6f} sec")
print(f"PyTorch Backward Time: {torch_backward_time:.6f} sec")
print(f"CUDA Backward Time:   {cuda_backward_time:.6f} sec")

# Speedup factors
print(f"Forward Speedup: {torch_forward_time / cuda_forward_time:.2f}x")
print(f"Backward Speedup: {torch_backward_time / cuda_backward_time:.2f}x")
