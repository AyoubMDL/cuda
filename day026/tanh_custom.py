import torch
import tanh_cuda

class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return tanh_cuda.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return tanh_cuda.backward(input, grad_output)


def tanh(input):
    return TanhFunction.apply(input)
