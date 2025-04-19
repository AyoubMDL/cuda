import triton
import torch
from matmul_2d_grid import naive_matmul_2d_grid
from matmul_1d_grid import naive_matmul_1d_grid


DEVICE = "cuda"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(5, 13)],  # Matrix sizes from 32x32 to 4096x4096
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',
        line_vals=['triton_1d_grid', 'triton_2d_grid', 'torch'],
        line_names=['Triton 1D', 'Triton 2D', 'Torch'],
        styles=[('blue', '-'), ('orange', '-'), ('green', '-')],
        ylabel='TFLOPs',
        plot_name='Matmul performance',
        args={},
    )
)
def benchmark(size, provider):
    M = N = K = size
    A = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    B = torch.randn((K, N), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B),
                                                     quantiles=quantiles)

    if provider == 'triton_1d_grid':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul_1d_grid(A, B),
                                                     quantiles=quantiles)

    if provider == 'triton_2d_grid':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_matmul_2d_grid(A, B),
                                                     quantiles=quantiles)

    def perf(ms): return 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
