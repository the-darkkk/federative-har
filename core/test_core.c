#include "federative_har.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_weights_len(void) {
    printf("[TEST 1/5] Verifying fh_weights_len...\n");
    int len_har = fh_weights_len(128, 64, 6);
    assert(len_har == 8646);

    int len_small = fh_weights_len(4, 3, 2);
    assert(len_small == 23); /* (4*3) + 3 + (3*2) + 2 = 12 + 3 + 6 + 2 = 23 */
    printf("  PASS: fh_weights_len(128, 64, 6) == 8646\n");
}

static void test_preprocess_window(void) {
    printf("[TEST 2/5] Verifying fh_preprocess_window...\n");
    float raw_xyz[9] = {
        0.0f, 0.0f, 9.80665f,  /* sample 0 */
        0.0f, 9.80665f, 0.0f,  /* sample 1 */
        3.0f, 4.0f, 0.0f       /* sample 2 */
    };
    float out_features[3];

    /* Test 1: scale_to_g for m/s^2 (1.0f / 9.80665f) */
    fh_preprocess_window(raw_xyz, out_features, 2, 1.0f / 9.80665f);
    assert(fabsf(out_features[0] - 1.0f) < 1e-5f);
    assert(fabsf(out_features[1] - 1.0f) < 1e-5f);

    /* Test 2: scale_to_g for raw g (1.0f) on sample 2: sqrt(3^2 + 4^2 + 0^2) = 5.0f */
    fh_preprocess_window(raw_xyz + 6, out_features + 2, 1, 1.0f);
    assert(fabsf(out_features[2] - 5.0f) < 1e-5f);

    printf("  PASS: fh_preprocess_window accurate for both g and m/s^2 scaling\n");
}

static void test_determinism(void) {
    printf("[TEST 3/5] Verifying determinism across independent runs...\n");
    const int in_dim = 128;
    const int hid_dim = 64;
    const int out_dim = 6;
    const int w_len = fh_weights_len(in_dim, hid_dim, out_dim);

    float* w1 = (float*)malloc((size_t)w_len * sizeof(float));
    float* w2 = (float*)malloc((size_t)w_len * sizeof(float));
    assert(w1 != NULL && w2 != NULL);

    /* Deterministic init */
    fh_init_weights(w1, in_dim, hid_dim, out_dim, 42u);
    fh_init_weights(w2, in_dim, hid_dim, out_dim, 42u);
    assert(memcmp(w1, w2, (size_t)w_len * sizeof(float)) == 0);

    /* Deterministic batch training */
    const int batch_size = 8;
    float X[8 * 128];
    int y[8];
    for (int i = 0; i < 8 * 128; i++) {
        X[i] = sinf((float)i * 0.1f);
    }
    for (int i = 0; i < 8; i++) {
        y[i] = i % out_dim;
    }

    float loss1 = fh_train_batch(w1, X, y, batch_size, in_dim, hid_dim, out_dim, 0.05f);
    float loss2 = fh_train_batch(w2, X, y, batch_size, in_dim, hid_dim, out_dim, 0.05f);
    assert(fabsf(loss1 - loss2) < 1e-7f);
    assert(memcmp(w1, w2, (size_t)w_len * sizeof(float)) == 0);

    free(w1);
    free(w2);
    printf("  PASS: Deterministic initialization and training step bit-identical\n");
}

static float eval_batch_loss(const float* weights, const float* X, const int* y, int batch_size, int in_dim, int hid_dim, int out_dim) {
    float total_loss = 0.0f;
    float probs[64];
    for (int s = 0; s < batch_size; s++) {
        fh_predict(weights, X + s * in_dim, probs, in_dim, hid_dim, out_dim);
        int target = y[s];
        float p = (target >= 0 && target < out_dim) ? probs[target] : 0.0f;
        total_loss += -logf(p + 1e-7f);
    }
    return total_loss / (float)batch_size;
}

static void test_finite_difference_gradient(void) {
    printf("[TEST 4/5] Verifying finite-difference numerical gradients (4 -> 3 -> 2)...\n");
    const int in_dim = 4;
    const int hid_dim = 3;
    const int out_dim = 2;
    const int w_len = fh_weights_len(in_dim, hid_dim, out_dim);
    assert(w_len == 23);

    float w_init[23];
    fh_init_weights(w_init, in_dim, hid_dim, out_dim, 1337u);

    const int batch_size = 3;
    float X[3 * 4] = {
        0.5f, -0.2f, 0.8f, -0.1f,
        0.1f,  0.9f, -0.4f, 0.3f,
       -0.7f,  0.2f,  0.5f, 0.6f
    };
    int y[3] = {0, 1, 0};

    /* Analytical gradient via fh_train_batch */
    float w_step[23];
    memcpy(w_step, w_init, sizeof(w_step));
    const float lr = 0.01f;
    fh_train_batch(w_step, X, y, batch_size, in_dim, hid_dim, out_dim, lr);

    float analytical_grad[23];
    for (int p = 0; p < w_len; p++) {
        analytical_grad[p] = (w_init[p] - w_step[p]) / lr;
    }

    /* Central finite differences: (L(w + eps) - L(w - eps)) / (2 * eps) */
    const float eps = 1e-4f;
    float numerical_grad[23];
    float w_perturb[23];

    float max_error = 0.0f;
    for (int p = 0; p < w_len; p++) {
        memcpy(w_perturb, w_init, sizeof(w_perturb));
        w_perturb[p] += eps;
        float loss_plus = eval_batch_loss(w_perturb, X, y, batch_size, in_dim, hid_dim, out_dim);

        memcpy(w_perturb, w_init, sizeof(w_perturb));
        w_perturb[p] -= eps;
        float loss_minus = eval_batch_loss(w_perturb, X, y, batch_size, in_dim, hid_dim, out_dim);

        numerical_grad[p] = (loss_plus - loss_minus) / (2.0f * eps);
        float err = fabsf(analytical_grad[p] - numerical_grad[p]);
        if (err > max_error) {
            max_error = err;
        }
        assert(err < 1e-3f);
    }

    printf("  PASS: Backprop verified against central finite differences (max diff: %.2e < 1e-3)\n", max_error);
}

static void test_overfit_sanity(void) {
    printf("[TEST 5/5] Verifying overfit sanity check (128 -> 32 -> 6, 12 samples, 200 steps)...\n");
    const int in_dim = 128;
    const int hid_dim = 32;
    const int out_dim = 6;
    const int w_len = fh_weights_len(in_dim, hid_dim, out_dim);

    float* weights = (float*)malloc((size_t)w_len * sizeof(float));
    assert(weights != NULL);
    fh_init_weights(weights, in_dim, hid_dim, out_dim, 42u);

    const int batch_size = 12;
    float X[12 * 128];
    int y[12];
    unsigned int x_seed = 12345u;
    for (int s = 0; s < batch_size; s++) {
        y[s] = s % out_dim;
        for (int i = 0; i < in_dim; i++) {
            x_seed = x_seed * 1664525u + 1013904223u;
            float u = (float)x_seed / 4294967296.0f;
            X[s * in_dim + i] = u * 2.0f - 1.0f;
        }
    }

    float final_loss = 0.0f;
    for (int step = 0; step < 200; step++) {
        final_loss = fh_train_batch(weights, X, y, batch_size, in_dim, hid_dim, out_dim, 0.5f);
    }

    int correct = 0;
    float probs[6];
    for (int s = 0; s < batch_size; s++) {
        fh_predict(weights, X + s * in_dim, probs, in_dim, hid_dim, out_dim);
        int pred = 0;
        float best_p = probs[0];
        for (int k = 1; k < out_dim; k++) {
            if (probs[k] > best_p) {
                best_p = probs[k];
                pred = k;
            }
        }
        if (pred == y[s]) {
            correct++;
        }
    }

    printf("  Final Loss: %.6f (assert < 0.05f)\n", final_loss);
    printf("  Accuracy: %d / %d (assert 100%%)\n", correct, batch_size);

    assert(final_loss < 0.05f);
    assert(correct == batch_size);

    free(weights);
    printf("  PASS: Successfully overfitted synthetic batch with 100%% accuracy\n");
}

int main(void) {
    printf("==================================================\n");
    printf("STAGE 1: RUNNING NATIVE C99 UNIT & SANITY TESTS\n");
    printf("==================================================\n");

    test_weights_len();
    test_preprocess_window();
    test_determinism();
    test_finite_difference_gradient();
    test_overfit_sanity();

    printf("==================================================\n");
    printf("ALL NATIVE C TESTS PASSED SUCCESSFULLY!\n");
    printf("==================================================\n");
    return 0;
}
