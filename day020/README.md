## Learning Summary

Exercices chapter 5 (10, 11, 12)

1. Write a matrix multiplication kernel function that corresponds to the design 
illustrated in Figure 5.17. (`See tiled_mm_thread_granularity`)
2. For tiled matrix multiplication out of the possible range of values for 
BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid 
un-coalesced accesses to global memory? (You need to consider only square 
blocks.) (`MATRIX_SIZE needs to be divisible by BLOCK_SIZE`)
3. In an attempt to improve performance, a bright young engineer changed the 
reduction kernel into the following. (A) Do you believe that the performance 
will improve? Why or why not? (B) Should the engineer receive a reward or a 
lecture? Why?
```bash
__shared__ float partialSum[];
unsigned int tid=threadIdx.x;
for (unsigned int stride=n>>1; stride >= 32; stride >>= 1) {
    __syncthreads();
if (tid < stride)
    shared[tid] += shared[tid + stride];
}
__syncthreads();
if (tid < 32) { // unroll last 5 predicated steps
    shared[tid] += shared[tid + 16];
    shared[tid] += shared[tid + 8];
    shared[tid] += shared[tid + 4];
    shared[tid] += shared[tid + 2];
    shared[tid] += shared[tid + 1];
}
```
The issue with this reduction kernel is that in the unrolled section (the final warp-level reduction), the values being accessed in shared[] may not be fully up-to-date. This happens because there is no __syncthreads() to ensure all preceding reductions (from the larger strides) have completed before these final operations begin.
For example, when thread tid reads from shared[tid + 8], it assumes that shared[tid + 8] already contains the final reduced value from all earlier stages. However, if other threads (which contribute to shared[tid + 8]) have not yet completed their work, the value will be stale or partially updated.