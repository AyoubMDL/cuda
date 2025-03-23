import torch
import cross_entropy_cuda


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, reduction="mean"):
        ctx.save_for_backward(input, target)
        ctx.reduction = reduction
        return cross_entropy_cuda.forward(input, target, reduction)

    @staticmethod
    def backward(ctx, grad_output=None):
        input, target = ctx.saved_tensors
        reduction = ctx.reduction
        grad_logits = cross_entropy_cuda.backward(input, target, grad_output, reduction)
        return grad_logits, None, None  # Return None for targets


def cross_entropy(input, target, reduction="mean"):
    return CrossEntropyFunction.apply(input, target, reduction)
