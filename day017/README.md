## Learning summary

* Chapter 4 of pmpp

### Exercices's answers
1. Consider matrix addition. Can one use shared memory to reduce the global 
memory bandwidth consumption? Hint: Analyze the elements accessed by 
each thread and see if there is any commonality between threads.
`Using shared memory has no effect in this case because eachelement is accessed only once by a thread (no overlaps)`

2.

3. What type of incorrect execution behavior can happen if one or both 
__syncthreads() are omitted in the kernel ? 
`Access a non loaded data in the shared memory, some hardwares will returns
random values, others will stop the program`.

4. Assuming that capacity is not an issue for registers or shared memory, give 
one important reason why it would be valuable to use shared memory instead 
of registers to hold values fetched from global memory?
`multiple threads within the same block need to access the same data fetched from global memory.`

5. For our tiled matrix–matrix multiplication kernel, if we use a 32x32 tile, what 
is the reduction of memory bandwidth usage for input matrices M and N?
`1/32 of the original usage`

6. Assume that a CUDA kernel is launched with 1,000 thread blocks, with each 
having 512 threads. If a variable is declared as a local variable in the kernel, 
how many versions of the variable will be created through the lifetime of the 
execution of the kernel?
`every thread gets its own separate copy of the variable: 1,000×512=512,000 threads`

7. In the previous question, if a variable is declared as a shared memory 
variable, how many versions of the variable will be created throughout the 
lifetime of the execution of the kernel ?
`1000 (shared memory for each block)`

8. Consider performing a matrix multiplication of two input matrices with 
dimensions N × N. How many times is each element in the input matrices 
requested from global memory in the following situations?
A. There is no tiling. `(N times)`
B. Tiles of size T × T are used. `(N / T times)`

9. A kernel performs 36 floating-point operations and 7 32-bit word global 
memory accesses per thread. For each of the following device properties, 
indicate whether this kernel is compute- or memory-bound.
A. Peak FLOPS= 200 GFLOPS, Peak Memory Bandwidth= 100 GB/s `memory bound`
B. Peak FLOPS= 300 GFLOPS, Peak Memory Bandwidth= 250 GB/s `compute bound`

~~~
Operational intensity = Bytes accessed per thread / FLOP per thread
compute-to-memory ratio = Peak FLOPS / Peak Memory Bandwidth~
~~~
