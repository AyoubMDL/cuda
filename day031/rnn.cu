#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

struct RNNWeights {
    float* W_xh;  // [H x N]
    float* W_hh;  // [H x H]
    float* b_h;   // [H]
    float* W_hy;  // [O x H]
    float* b_y;   // [O]

    int N, H, O;

    RNNWeights(int input_size, int hidden_size, int output_size)
        : N(input_size),
          H(hidden_size),
          O(output_size),
          W_xh(nullptr),
          W_hh(nullptr),
          b_h(nullptr),
          W_hy(nullptr),
          b_y(nullptr) {}

    void allocate_device_memory() {
        CHECK_CUDA(cudaMalloc(&W_xh, H * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&W_hh, sizeof(float) * H * H));
        CHECK_CUDA(cudaMalloc(&b_h, sizeof(float) * H));
        CHECK_CUDA(cudaMalloc(&W_hy, sizeof(float) * O * H));
        CHECK_CUDA(cudaMalloc(&b_y, sizeof(float) * O));
    }

    void copy_to_device(const float* h_W_xh, const float* h_W_hh,
                        const float* h_b_h, const float* h_W_hy,
                        const float* h_b_y) {
        allocate_device_memory();
        CHECK_CUDA(cudaMemcpy(W_xh, h_W_xh, H * N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(W_hh, h_W_hh, sizeof(float) * H * H,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(
            cudaMemcpy(b_h, h_b_h, sizeof(float) * H, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(W_hy, h_W_hy, sizeof(float) * O * H,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(
            cudaMemcpy(b_y, h_b_y, sizeof(float) * O, cudaMemcpyHostToDevice));
    }

    void free_memory() {
        cudaFree(W_xh);
        cudaFree(W_hh);
        cudaFree(b_h);
        cudaFree(W_hy);
        cudaFree(b_y);
    }
};

// -------------------- CUDA Kernel --------------------

__global__ void update_step_kernel(float* x_t, float* h_prev, float* W_xh,
                                   float* W_hh, float* b_h, float* h_t, int N,
                                   int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < H) {
        float sum = b_h[i];

        for (int j = 0; j < N; ++j) {
            sum += W_xh[i * N + j] * x_t[j];
        }

        for (int k = 0; k < H; ++k) {
            sum += W_hh[i * H + k] * h_prev[k];
        }

        h_t[i] = tanhf(sum);
    }
}

__global__ void output_step_kernel(float* h_t, float* W_hy, float* b_y,
                                   float* y_t, int H, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o < O) {
        float sum = b_y[o];

        for (int j = 0; j < H; ++j) {
            sum += W_hy[o * H + j] * h_t[j];
        }

        y_t[o] = sum;
    }
}

class RNNLayer {
   public:
    RNNLayer(RNNWeights& w, int seq_len)
        : weights(w), N(w.N), H(w.H), O(w.O), T(seq_len) {
        allocate_states();
    }

    // d_inputs shape (T, N) flattened
    // d_outputs shape (T, O) flattened
    void forward(float* d_inputs, float* d_outputs) {
        int threads = 256;
        int h_blocks = (H + threads - 1) / threads;
        int o_blocks = (O + threads - 1) / threads;

        for (int t = 0; t < T; ++t) {
            float* x_t = d_inputs + t * N;
            float* y_t = d_outputs + t * O;

            // Call the kernel to update h_new
            update_step_kernel<<<h_blocks, threads>>>(
                x_t, d_h_prev, weights.W_xh, weights.W_hh, weights.b_h, d_h_new,
                N, H);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Call the kernel for output
            output_step_kernel<<<o_blocks, threads>>>(d_h_new, weights.W_hy,
                                                      weights.b_y, y_t, H, O);
            CHECK_CUDA(cudaDeviceSynchronize());

            std::swap(d_h_prev, d_h_new);
        }
    }

    ~RNNLayer() {
        CHECK_CUDA(cudaFree(d_h_prev));
        CHECK_CUDA(cudaFree(d_h_new));
        weights.free_memory();
    }

    RNNWeights& weights;

   private:
    int N, H, O, T;
    float* d_h_prev;
    float* d_h_new;

    void allocate_states() {
        CHECK_CUDA(cudaMalloc(&d_h_prev, H * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_h_new, H * sizeof(float)));

        // Initialize the hidden state h_prev to zeros before starting the RNN
        // forward pass.
        CHECK_CUDA(cudaMemset(d_h_prev, 0, sizeof(float) * H));
    }
};

__global__ void print_device_memory(float* data, int T, int O) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < T * O) {
        printf("d_output[%d] = %.4f\n", idx, data[idx]);
    }
}

int main() {
    const int N = 4;
    const int H = 5;
    const int O = 2;
    const int T = 3;

    // Host Mem
    float h_input[T * N];
    float h_W_xh[H * N], h_W_hh[H * H], h_b_h[H];
    float h_W_hy[O * H], h_b_y[O];

    // init weights
    for (int i = 0; i < T * N; ++i) h_input[i] = 0.3f;
    for (int i = 0; i < H * N; ++i) h_W_xh[i] = 0.1f;
    for (int i = 0; i < H * H; ++i) h_W_hh[i] = 0.1f;
    for (int i = 0; i < H; ++i) h_b_h[i] = 0.1f;
    for (int i = 0; i < O * H; ++i) h_W_hy[i] = 0.1f;
    for (int i = 0; i < O; ++i) h_b_y[i] = 0.1f;

    // Device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float) * T * N));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float) * T * O));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float) * T * N,
                          cudaMemcpyHostToDevice));
    // print_device_memory<<<1, 10>>>(d_input, T, N);

    // Create and copy weights to device
    RNNWeights weights(N, H, O);
    weights.copy_to_device(h_W_xh, h_W_hh, h_b_h, h_W_hy, h_b_y);

    // RNN layer
    RNNLayer rnn(weights, T);
    rnn.forward(d_input, d_output);

    // Get result
    float h_output[T * O];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeof(float) * T * O,
                          cudaMemcpyDeviceToHost));

    // Print output
    printf("RNN Output:\n");
    for (int t = 0; t < T; ++t) {
        printf("y[%d] = ", t);
        for (int o = 0; o < O; ++o) printf("%.4f ", h_output[t * O + o]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}