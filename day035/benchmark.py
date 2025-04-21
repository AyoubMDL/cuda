import triton
import torch
from block_matmul_2d_grid import block_matmul_2d_grid


DEVICE = "cuda"
configs = [
    triton.testing.Benchmark(
        # we can increase multiple dimensions simultaneously while benchmarking
        x_names=['size'],
        x_vals=[2**i for i in range(5, 13)],  # Matrix sizes from 32x32 to 4096x4096
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPs",
        plot_name="matmul-performance",
        args={},
    )
]


@triton.testing.perf_report(configs)
def benchmark(size, provider):
    M = N = K = size
    A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
    B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B),
                                                     quantiles=quantiles)

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: block_matmul_2d_grid(A, B), quantiles=quantiles)

    def perf(ms): return 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    benchmark.run(save_path="bench/", print_data=False)
