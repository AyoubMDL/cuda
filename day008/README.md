## Learning summary

* Implemented Reduction Max in CUDA:
Learned how to find the maximum of a large vector using parallel reduction in CUDA.

* Shared Memory and Synchronization:
Used shared memory to store intermediate results for each block and applied synchronization to ensure safe access.

* Reduction Pattern:
Understood the tree-based reduction pattern (stride /= 2) to efficiently reduce values in parallel.

* Two-Step Kernel Approach:
    1. First kernel: Block-wise reduction to compute partial maximums.
    2. Second kernel: Global reduction to find the final maximum.
