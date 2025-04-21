import torch
import triton
import triton.language as tl


autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                  'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
                  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
                  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2)
]


@triton.autotune(configs=autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def block_matmul_2d_grid_kernel(a_ptr, b_ptr, c_ptr,
                                M, N, K,
                                stride_am, stride_ak,
                                stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                BLOCK_SIZE_M: tl.constexpr,
                                BLOCK_SIZE_N: tl.constexpr,
                                BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Get chunks along m/n/k dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Offsets of A and B
    offsets_a = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    offsets_b = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(offsets_a)
        b = tl.load(offsets_b)

        acc = tl.dot(a, b, acc=acc)

        offsets_a += BLOCK_SIZE_K * stride_ak
        offsets_b += BLOCK_SIZE_K * stride_bk

    c_offests = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_offests, acc.to(tl.float16), mask=c_mask)


def block_matmul_2d_grid(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    block_matmul_2d_grid_kernel[grid](A, B, C,
                                      M, N, K,
                                      A.stride(0), A.stride(1),
                                      B.stride(0), B.stride(1),
                                      C.stride(0), C.stride(1))
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(8, 4, device='cuda', dtype=torch.float16)
    B = torch.randn(4, 8, device='cuda', dtype=torch.float16)

    C_triton = block_matmul_2d_grid(A, B)
    C_torch = A @ B

    print("Triton:\n", C_triton)
    print("Torch:\n", C_torch)
    print("Max Error:", (C_triton - C_torch).abs().max().item())
