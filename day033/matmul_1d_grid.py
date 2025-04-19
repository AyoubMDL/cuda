import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    # program ID, one program = one (m, n) pair
    pid = tl.program_id(axis=0)

    # Compute coordinates
    # pid = m * N + n so we do the inverse of this to find m and n
    m = pid // N  # Row index in output
    n = pid % N  # Column index in output

    # initialize accumulator
    acc = 0.0

    for k in range(K):
        a = tl.load(a_ptr + m * stride_am + k * stride_ak)
        b = tl.load(b_ptr + k * stride_bk + n * stride_bn)
        acc += a * b

    tl.store(c_ptr + m * stride_cm + n * stride_cn, acc)


def naive_matmul_1d_grid(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"

    M, K = A.shape
    _, N = B.shape

    C = torch.empty((M, N), device="cuda", dtype=A.dtype)

    # grid: one program per output element
    def grid(meta): return (M * N,)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(4, 3, device='cuda')
    B = torch.randn(3, 5, device='cuda')

    C_triton = naive_matmul_1d_grid(A, B)
    C_torch = A @ B

    print("Triton:\n", C_triton)
    print("Torch:\n", C_torch)
    print("Max Error:", (C_triton - C_torch).abs().max().item())
