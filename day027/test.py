import torch
import pytest
from cross_entropy_custom import cross_entropy


@pytest.mark.parametrize("batch_size", [10, 1000])
@pytest.mark.parametrize("num_classes", [10, 100, 1000])
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_cross_entropy(batch_size, num_classes, reduction):
    input_tensor = torch.randn(batch_size, num_classes, device='cuda', requires_grad=True)
    target = torch.randint(0, num_classes, (batch_size,), device='cuda')

    # PyTorch's built-in CrossEntropy Loss
    torch_loss = torch.nn.functional.cross_entropy(input_tensor, target, reduction=reduction)
    grad_output = torch.ones_like(target) if reduction == "none" else None

    torch_loss.backward(grad_output)
    grad_torch = input_tensor.grad.clone()
    input_tensor.grad.zero_()  # Reset gradients

    # Custom CUDA CrossEntropy Loss
    cuda_loss = cross_entropy(input_tensor, target, reduction=reduction)

    cuda_loss.backward(grad_output)
    grad_cuda = input_tensor.grad.clone()

    # Check forward correctness
    torch.testing.assert_close(torch_loss, cuda_loss, atol=1e-6, rtol=1e-6)

    # Check backward correctness
    torch.testing.assert_close(grad_torch, grad_cuda, atol=1e-6, rtol=1e-6)
