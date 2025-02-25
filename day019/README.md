##  Learning summary

Chapter 5 exercices
 
1. Kernel (5.15) optimization (``Launche blockDim.x / 2 threads``)
2. See partial_sum_opt.cu
3. See partial_sum_opt.cu
4. See partial_sum_opt.cu

6. For the simple matrix multiplication (P = M*N) based on row-major layout, 
which input matrix will have coalesced accesses? (`M`)

7. For the tiled matrix–matrix multiplication (M*N) based on row-major layout, 
which input matrix will have coalesced accesses? (``M, N``)

8. For the simple reduction kernel, if the block size is 1024 and warp size is 32, 
how many warps in a block will have divergence during the 5th iteration? (`32`)
```bash
Each warp consists of 32 threads. For a warp to have divergence, some threads in the warp must satisfy 
t%32==0 (and execute the if statement), while others do not.
However, in the 5th iteration, the condition  t%32==0 is only true for one thread in each warp (the thread with 
t being a multiple of 32). All other threads in the warp will not satisfy the condition.
Therefore, every warp in the block will have divergence during the 5th iteration because only one thread in each warp will execute the if statement, while the other 31 threads will not.
```

9. For the improved reduction kernel, if the block size is 1024 and warp size is 
32, how many warps will have divergence during the 5th iteration? (`0`)