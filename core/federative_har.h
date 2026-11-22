#ifndef FEDERATIVE_HAR_H
#define FEDERATIVE_HAR_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * Returns the total number of float weights for a 2-layer MLP (in_dim -> hid_dim -> out_dim):
 * (in_dim * hid_dim) + hid_dim + (hid_dim * out_dim) + out_dim.
 */
EXPORT int fh_weights_len(int in_dim, int hid_dim, int out_dim);

/**
 * Deterministic He-scaled initialization using fixed LCG PRNG (seed = seed * 1664525u + 1013904223u).
 * Biases are initialized to 0.0f.
 */
EXPORT void fh_init_weights(float* weights, int in_dim, int hid_dim, int out_dim, unsigned int seed);

/**
 * Preprocesses interleaved 3-axis accelerometer samples [x0, y0, z0, x1, y1, z1, ...]
 * (length: 3 * window_len), scales by scale_to_g, and writes orientation-invariant
 * vector magnitude sqrtf(x*x + y*y + z*z) into out_features (length: window_len).
 */
EXPORT void fh_preprocess_window(const float* raw_xyz, float* out_features, int window_len, float scale_to_g);

/**
 * Forward inference pass: Linear(in_dim -> hid_dim) -> ReLU -> Linear(hid_dim -> out_dim) -> stable Softmax.
 * Uses stack scratch buffers when hid_dim <= 256 to avoid heap allocations.
 */
EXPORT void fh_predict(const float* weights, const float* features, float* probs_out, int in_dim, int hid_dim, int out_dim);

/**
 * Mini-batch SGD training step with Cross-Entropy loss (-logf(probs[target] + 1e-7f)).
 * Computes exact analytical gradients across the batch, updates weights in-place with
 * step = lr / batch_size, and returns the average batch loss.
 */
EXPORT float fh_train_batch(float* weights, const float* X, const int* y, int batch_size, int in_dim, int hid_dim, int out_dim, float lr);

#ifdef __cplusplus
}
#endif

#endif /* FEDERATIVE_HAR_H */
