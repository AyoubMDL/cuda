## Learning summary

* Image blur
* Completed Chapter 3 of pmpp

### Questions
3. If the SM of a CUDA device can take up to 1536 threads and up to 4 thread 
blocks. Which of the following block configuration would result in the largest 
number of threads in the SM? (``512 threads per block``)

4. For a vector addition, assume that the vector length is 2000, each thread 
calculates one output element, and the thread block size is 512 threads. How 
many threads will be in the grid? (``2048``)

5. With reference to the previous question, how many warps do you expect to 
have divergence due to the boundary check on vector length? (``2``)