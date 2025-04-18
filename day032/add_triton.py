import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_current_target().backend
assert DEVICE == "cuda"


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr,
               n_elements, BLOCK_SIZE: tl.constexpr):
    # "Same" as BlockIdx in CUDA
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # handle out of bounds accesses
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device.type == DEVICE and y.device.type == DEVICE and output.device.type == DEVICE
    n_elements = output.numel()

    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    size = 14243
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    output_torch = x + y
    output_triton = add(x, y)

    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
