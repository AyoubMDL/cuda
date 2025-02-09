## Learning summary

* Implemented conv1d using shared memory

1. Shared memory allocation:
    * extern __shared__ float shared_mem[]; is used to define the shared memory space.
    * We allocate memory for both the filter weights w and the portion of the input x that the block will work on.
2. Loading data into shared memory:
    * We load the weights w into shared_w.
    * We load the relevant portion of x into shared_x for the current block.
3. Synchronization (__syncthreads()):
    * This ensures that all threads in the block have finished loading data into shared memory before any thread starts the convolution operation.
4. Memory layout:
    * shared_w holds the filter weights, which are the same for all threads in the block.
    * shared_x holds the input data that is relevant to the current block, and each thread accesses a different position in it.
5. Shared memory usage:
    * The kernel now takes advantage of shared memory for both the input data and the weights, reducing the number of global memory accesses.