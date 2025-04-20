import torch
import triton
import triton.language as tl


@triton.jit
def block_matmul_kernel(a_ptr, b_ptr, c_ptr,
                        M, N, K,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_SIZE_M: tl.constexpr,
                        BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n  # vertical position
    pid_n = pid % num_pid_n   # horizontal position

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        a_mask = None  # to be computed
        b_mask = None  # to be computed

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = None  # to be computed
    tl.store(c_ptrs, acc, mask=c_mask)


def block_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def grid(meta): return (triton.cdiv(
        M, meta["BLOCK_SIZE_M"]) * tl.cdiv(N, meta["BLOCK_SIZE_N"]), )

    print(C.stride(0), C.stride(1))
    block_matmul_kernel[grid](A, B, C,
                              M, N, K,
                              A.stride(0), A.stride(1),
                              B.stride(0), B.stride(1),
                              C.stride(0), C.stride(1),
                              BLOCK_SIZE_M=16,
                              BLOCK_SIZE_N=16,
                              BLOCK_SIZE_K=16)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(8, 4, device='cuda')
    B = torch.randn(4, 8, device='cuda')

    C_triton = block_matmul(A, B)
    C_torch = A @ B

    print("Triton:\n", C_triton)
    print("Torch:\n", C_torch)
    print("Max Error:", (C_triton - C_torch).abs().max().item())
