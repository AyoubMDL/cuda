## Learning summary

* Implemented simple RNN operation

Given:

$$
x_t \in \mathbb{R}^n
$$

$$
h_t \in \mathbb{R}^h
$$

$$
y_t \in \mathbb{R}^o
$$

**Update rule**:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$


* Learned How to Debug with ``cuda-gdb``


### Code structure

```mermaid
classDiagram
    class RNNWeights {
        +int N
        +int H
        +int O
        +float* W_xh
        +float* W_hh
        +float* b_h
        +float* W_hy
        +float* b_y
        +copy_to_device(...)
    }

    class RNNLayer {
        +int N
        +int H
        +int O
        +int T
        +float* d_h_prev
        +float* d_h_new
        +RNNWeights& weights
        +forward(float* d_inputs, float* d_outputs)
    }

    RNNLayer --> RNNWeights : uses
```
