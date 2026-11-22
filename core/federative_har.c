#include "federative_har.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int fh_weights_len(int in_dim, int hid_dim, int out_dim) {
    return (in_dim * hid_dim) + hid_dim + (hid_dim * out_dim) + out_dim;
}

void fh_init_weights(float* weights, int in_dim, int hid_dim, int out_dim, unsigned int seed) {
    float* W1 = weights;
    float* b1 = W1 + (in_dim * hid_dim);
    float* W2 = b1 + hid_dim;
    float* b2 = W2 + (hid_dim * out_dim);

    /* He uniform initialization for Layer 1: limit = sqrt(6 / fan_in) */
    float limit1 = sqrtf(6.0f / (float)in_dim);
    int num_w1 = in_dim * hid_dim;
    for (int i = 0; i < num_w1; i++) {
        seed = seed * 1664525u + 1013904223u;
        float u = (float)seed / 4294967296.0f;
        W1[i] = (u * 2.0f - 1.0f) * limit1;
    }

    for (int j = 0; j < hid_dim; j++) {
        b1[j] = 0.0f;
    }

    /* He uniform initialization for Layer 2: limit = sqrt(6 / fan_in) */
    float limit2 = sqrtf(6.0f / (float)hid_dim);
    int num_w2 = hid_dim * out_dim;
    for (int i = 0; i < num_w2; i++) {
        seed = seed * 1664525u + 1013904223u;
        float u = (float)seed / 4294967296.0f;
        W2[i] = (u * 2.0f - 1.0f) * limit2;
    }

    for (int k = 0; k < out_dim; k++) {
        b2[k] = 0.0f;
    }
}

void fh_preprocess_window(const float* raw_xyz, float* out_features, int window_len, float scale_to_g) {
    for (int i = 0; i < window_len; i++) {
        float x = raw_xyz[3 * i + 0] * scale_to_g;
        float y = raw_xyz[3 * i + 1] * scale_to_g;
        float z = raw_xyz[3 * i + 2] * scale_to_g;
        out_features[i] = sqrtf(x * x + y * y + z * z);
    }
}

void fh_predict(const float* weights, const float* features, float* probs_out, int in_dim, int hid_dim, int out_dim) {
    const float* W1 = weights;
    const float* b1 = W1 + (in_dim * hid_dim);
    const float* W2 = b1 + hid_dim;
    const float* b2 = W2 + (hid_dim * out_dim);

    /* Use stack scratch buffer when hid_dim <= 256 to avoid heap allocation */
    float stack_h[256];
    float* h = (hid_dim <= 256) ? stack_h : (float*)malloc((size_t)hid_dim * sizeof(float));

    /* Layer 1: Linear + ReLU */
    for (int j = 0; j < hid_dim; j++) {
        float sum = b1[j];
        const float* w_row = W1 + (j * in_dim);
        for (int i = 0; i < in_dim; i++) {
            sum += w_row[i] * features[i];
        }
        h[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* Layer 2: Linear + numerically stable Softmax */
    float max_z = -1e30f;
    for (int k = 0; k < out_dim; k++) {
        float sum = b2[k];
        const float* w_row = W2 + (k * hid_dim);
        for (int j = 0; j < hid_dim; j++) {
            sum += w_row[j] * h[j];
        }
        probs_out[k] = sum;
        if (sum > max_z) {
            max_z = sum;
        }
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < out_dim; k++) {
        float e = expf(probs_out[k] - max_z);
        probs_out[k] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < out_dim; k++) {
        probs_out[k] *= inv_sum;
    }

    if (hid_dim > 256) {
        free(h);
    }
}

float fh_train_batch(float* weights, const float* X, const int* y, int batch_size, int in_dim, int hid_dim, int out_dim, float lr) {
    if (batch_size <= 0) {
        return 0.0f;
    }

    float* W1 = weights;
    float* b1 = W1 + (in_dim * hid_dim);
    float* W2 = b1 + hid_dim;
    float* b2 = W2 + (hid_dim * out_dim);

    int total_weights = (in_dim * hid_dim) + hid_dim + (hid_dim * out_dim) + out_dim;

    /* Stack buffers for default / reasonable dimensions; heap fallback for larger */
    float stack_dweights[16384];
    float* dweights = (total_weights <= 16384)
                          ? stack_dweights
                          : (float*)malloc((size_t)total_weights * sizeof(float));
    memset(dweights, 0, (size_t)total_weights * sizeof(float));

    float* dW1 = dweights;
    float* db1 = dW1 + (in_dim * hid_dim);
    float* dW2 = db1 + hid_dim;
    float* db2 = dW2 + (hid_dim * out_dim);

    float stack_z1[256];
    float stack_h[256];
    float* z1 = (hid_dim <= 256) ? stack_z1 : (float*)malloc((size_t)hid_dim * sizeof(float));
    float* h = (hid_dim <= 256) ? stack_h : (float*)malloc((size_t)hid_dim * sizeof(float));

    float stack_probs[64];
    float stack_delta2[64];
    float* probs = (out_dim <= 64) ? stack_probs : (float*)malloc((size_t)out_dim * sizeof(float));
    float* delta2 = (out_dim <= 64) ? stack_delta2 : (float*)malloc((size_t)out_dim * sizeof(float));

    float total_loss = 0.0f;

    for (int s = 0; s < batch_size; s++) {
        const float* x_sample = X + (s * in_dim);
        int target = y[s];

        /* Forward Layer 1: z1 = W1 * x + b1; h = ReLU(z1) */
        for (int j = 0; j < hid_dim; j++) {
            float sum = b1[j];
            const float* w_row = W1 + (j * in_dim);
            for (int i = 0; i < in_dim; i++) {
                sum += w_row[i] * x_sample[i];
            }
            z1[j] = sum;
            h[j] = (sum > 0.0f) ? sum : 0.0f;
        }

        /* Forward Layer 2: z2 = W2 * h + b2 */
        float max_z = -1e30f;
        for (int k = 0; k < out_dim; k++) {
            float sum = b2[k];
            const float* w_row = W2 + (k * hid_dim);
            for (int j = 0; j < hid_dim; j++) {
                sum += w_row[j] * h[j];
            }
            probs[k] = sum;
            if (sum > max_z) {
                max_z = sum;
            }
        }

        /* Numerically stable Softmax */
        float sum_exp = 0.0f;
        for (int k = 0; k < out_dim; k++) {
            float e = expf(probs[k] - max_z);
            probs[k] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < out_dim; k++) {
            probs[k] *= inv_sum;
        }

        /* Cross-Entropy Loss: -log(probs[target] + 1e-7f) */
        float p_target = (target >= 0 && target < out_dim) ? probs[target] : 0.0f;
        total_loss += -logf(p_target + 1e-7f);

        /* Backward Layer 2: delta2 = probs - 1(k == target) */
        for (int k = 0; k < out_dim; k++) {
            float d2 = probs[k] - ((k == target) ? 1.0f : 0.0f);
            delta2[k] = d2;
            db2[k] += d2;

            float* dw_row = dW2 + (k * hid_dim);
            for (int j = 0; j < hid_dim; j++) {
                dw_row[j] += d2 * h[j];
            }
        }

        /* Backward Layer 1: Backprop to hidden + ReLU derivative */
        for (int j = 0; j < hid_dim; j++) {
            if (z1[j] > 0.0f) {
                float dh = 0.0f;
                for (int k = 0; k < out_dim; k++) {
                    dh += delta2[k] * W2[k * hid_dim + j];
                }
                db1[j] += dh;

                float* dw_row = dW1 + (j * in_dim);
                for (int i = 0; i < in_dim; i++) {
                    dw_row[i] += dh * x_sample[i];
                }
            }
        }
    }

    /* In-place SGD update: weights -= (lr / batch_size) * dweights */
    float step = lr / (float)batch_size;
    for (int p = 0; p < total_weights; p++) {
        weights[p] -= step * dweights[p];
    }

    /* Clean up heap allocations if used */
    if (total_weights > 16384) {
        free(dweights);
    }
    if (hid_dim > 256) {
        free(z1);
        free(h);
    }
    if (out_dim > 64) {
        free(probs);
        free(delta2);
    }

    return total_loss / (float)batch_size;
}
