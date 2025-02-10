## Learning summary

* Completed Chapter 2 of pmpp

Answers for questions:

1. Basic Addition (``see day001``)
2. Assume that we want to use each thread to calculate two (adjacent) elements 
of a vector addition. What would be the expression for mapping the thread/
block indices to i, the data index of the first element to be processed by a 
thread? (``See day007/chap2_questions.cu`` ``question2`` function)
3. We want to use each thread to calculate two elements of a vector addition. 
Each thread block processes 2*blockDim.x consecutive elements that form 
two sections. All threads in each block will first process a section first, each 
processing one element. They will then all move to the next section, each 
processing one element. Assume that variable i should be the index for the 
first element to be processed by a thread. What would be the expression for 
mapping the thread/block indices to data index of the first element? (``See day007/chap2_questions.cu`` ``question3`` function)
4. For a vector addition, assume that the vector length is 8000, each thread 
calculates one output element, and the thread block size is 1024 threads. The 
programmer configures the kernel launch to have a minimal number of thread 
blocks to cover all output elements. How many threads will be in the grid? (``8192``)
5. If we want to allocate an array of v integer elements in CUDA device global 
memory, what would be an appropriate expression for the second argument of 
the cudaMalloc call? (``v * sizeof(int)``)
6. If we want to allocate an array of n floating-point elements and have a 
floating-point pointer variable d_A to point to the allocated memory, what 
would be an appropriate expression for the first argument of the cudaMalloc() 
call? (``(void **) &d_A``)
7. If we want to copy 3000 bytes of data from host array h_A (h_A is a pointer 
to element 0 of the source array) to device array d_A (d_A is a pointer to 
element 0 of the destination array), what would be an appropriate API call for 
this data copy in CUDA? (``cudaMemcpy(d_A, h_A, 3000, cudaMemcpyHostToDevice);``)
8. How would one declare a variable err that can appropriately receive returned 
value of a CUDA API call? (``cudaError_t err;``)
9. A new summer intern was frustrated with CUDA. He has been complaining 
that CUDA is very tedious: he had to declare many functions that he plans 
to execute on both the host and the device twice, once as a host function and 
once as a device function. What is your response? (``__host__ __device__``)