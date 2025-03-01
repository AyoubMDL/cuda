## Learning Summary

* Read half of chapter 6

### Parallel Reduction

* Kernel launch has negligible HW overhead, low SW overhead

#### Optimization Goals
1. Reach GPU peak performance
2. GFLOP/s: for compute-bound kernels
3. Bandwidth: for memory-bound kernels

Reductions have very low arithmetic intensity 1 flop per element loaded (bandwidth-optimal). Therefore we should strive for peak bandwidth

Nvidia RTX 2080 Ti (352-bit memory interface, 700 MHZ DDR)
memory bandwidth = (352 * 700 * 2) / 8 = 61.6 GB/s

### Kernels
1. interleaved addressing with divergent branching (highly divergent warps are very inefficient, and % operator is very slow)
2. interleaved addressing with bank conflicts (Shared Memory Bank Conflicts)
3. sequential addressing (Idle Threads, Half of the threads are idle on first loop iteration)
4. first add during global load (sdata[tid] = (index < size) ? input[index] + input[index + blockDim.x] : 0.0f;)
5. Unroll the Last Warp
6. Unrolling with Templates + multiple elements per thread

