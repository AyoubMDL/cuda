import torch
from tanh_custom import tanh

# Set seed for reproducibility
torch.manual_seed(42)

# Create input tensor on GPU
x = torch.randn(1000000, device="cuda", requires_grad=True)

# Forward pass
y_torch = torch.tanh(x)
y_cuda = tanh(x)

# Check forward correctness
torch.testing.assert_close(y_torch, y_cuda, atol=1e-8, rtol=1e-8)

# Backward pass
grad_output = torch.ones_like(x)
y_torch.backward(grad_output, retain_graph=True)
grad_torch = x.grad.clone()
x.grad.zero_()

y_cuda.backward(grad_output)
grad_cuda = x.grad.clone()

# Check backward correctness
torch.testing.assert_close(grad_torch, grad_cuda, atol=1e-8, rtol=1e-8)

print("All tests passed !")
