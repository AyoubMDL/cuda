import time
import torch
from cross_entropy_custom import cross_entropy

# Create input tensor
batch_size = 10000
num_classes = 100
input_tensor = torch.randn(batch_size, num_classes, device='cuda', requires_grad=True)
target = torch.randint(0, num_classes, (batch_size,), device='cuda')

# Measure PyTorch's execution time
torch.cuda.synchronize()
start = time.time()
y_torch = torch.nn.functional.cross_entropy(input_tensor, target)
torch.cuda.synchronize()
torch_forward_time = time.time() - start

torch.cuda.synchronize()
start = time.time()
y_torch.backward()
torch.cuda.synchronize()
torch_backward_time = time.time() - start

# Measure Custom CUDA execution time
torch.cuda.synchronize()
start = time.time()
y_cuda = cross_entropy(input_tensor, target)
torch.cuda.synchronize()
cuda_forward_time = time.time() - start

torch.cuda.synchronize()
start = time.time()
y_cuda.backward()
torch.cuda.synchronize()
cuda_backward_time = time.time() - start


print(f"PyTorch Forward Time: {torch_forward_time:.6f} sec")
print(f"CUDA Forward Time:   {cuda_forward_time:.6f} sec")
print(f"PyTorch Backward Time: {torch_backward_time:.6f} sec")
print(f"CUDA Backward Time:   {cuda_backward_time:.6f} sec")

# Speedup factors
print(f"Forward Speedup: {torch_forward_time / cuda_forward_time:.2f}x")
print(f"Backward Speedup: {torch_backward_time / cuda_backward_time:.2f}x")
