## Learning summary

Implemented tiled conv1d with halo cells (see day023/conv1d.cu for comparison with other conv1d implementation)

* The idea is to have a shared memory with size of block (or tile) size + conv_kernel - 1, with this we are
sure that all elements necessary for calculation are included in shared memory
* Second, we need to load elements to shared memory
* For left halo cells, we map the thread index to element index into the previous 
tile with the expression ``(blockIdx.x - 1)  * blockDim.x + threadIdx.x``. We then pick 
only the last n threads to load the needed left halo elements (n = conv_kernel / 2)
* For right halo cells, it is same as left but for the next tile, so we use this formula for mapping
``(blockIdx.x + 1)  * blockDim.x + threadIdx.x``
* For internal cells, we use normal mapping ``blockIdx.x * blockDim.x + threadIdx.x``

### Loading elements to shared memory:
Threads in first positions load right halo cells elements as they have the same indexing within the block. The same goes for the left halo cells.
