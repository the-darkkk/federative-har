# Federative HAR Compute Core (`core/`)

Zero-dependency, deterministic C99 compute core for on-device Human Activity Recognition (HAR) and federated client training.

---

## 1. C ABI Functions

All functions use C linkage (`extern "C"`) and standard platform export decorators (`__declspec(dllexport)` on Windows, `__attribute__((visibility("default")))` on Unix/macOS):

| Function | Signature | Description |
| :--- | :--- | :--- |
| `fh_weights_len` | `int fh_weights_len(int in_dim, int hid_dim, int out_dim)` | Returns total float count: `(in * hid) + hid + (hid * out) + out`. For `128 -> 64 -> 6`, returns `8646`. |
| `fh_init_weights` | `void fh_init_weights(float* weights, int in_dim, int hid_dim, int out_dim, unsigned int seed)` | Deterministic He uniform initialization using LCG PRNG (`seed = seed * 1664525u + 1013904223u`). Biases initialized to `0.0f`. |
| `fh_preprocess_window` | `void fh_preprocess_window(const float* raw_xyz, float* out_features, int window_len, float scale_to_g)` | Scales interleaved 3-axis samples $[x_0, y_0, z_0, \dots]$ by `scale_to_g` and computes $a_{mag} = \sqrt{x^2 + y^2 + z^2}$ into `out_features`. |
| `fh_predict` | `void fh_predict(const float* weights, const float* features, float* probs_out, int in_dim, int hid_dim, int out_dim)` | Forward pass: `Linear -> ReLU -> Linear -> Stable Softmax`. Uses stack buffer when `hid_dim <= 256` (zero heap allocations). |
| `fh_train_batch` | `float fh_train_batch(float* weights, const float* X, const int* y, int batch_size, int in_dim, int hid_dim, int out_dim, float lr)` | Mini-batch SGD step with Cross-Entropy loss. Updates `weights` in-place with `step = lr / batch_size` and returns average batch loss. |

---

## 2. Tensor Memory Layout

A model's parameters are packed contiguously into a single flat 1D array of `float32` values:

```
[----------- W1 -----------][--- b1 ---][----------- W2 -----------][--- b2 ---]
 0                           offset1     offset2                     offset3     total
```

1. **Layer 1 Weights (`W1`)**:
   - Shape: `[hid_dim, in_dim]` (row-major).
   - Offset: `0`
   - Length: `in_dim * hid_dim` floats
2. **Layer 1 Biases (`b1`)**:
   - Shape: `[hid_dim]`
   - Offset: `in_dim * hid_dim`
   - Length: `hid_dim` floats
3. **Layer 2 Weights (`W2`)**:
   - Shape: `[out_dim, hid_dim]` (row-major).
   - Offset: `(in_dim * hid_dim) + hid_dim`
   - Length: `hid_dim * out_dim` floats
4. **Layer 2 Biases (`b2`)**:
   - Shape: `[out_dim]`
   - Offset: `(in_dim * hid_dim) + hid_dim + (hid_dim * out_dim)`
   - Length: `out_dim` floats

---

## 3. Compilation Commands (No CMake)

### Windows (`federative_har.dll`)
Using MinGW / GCC:
```bash
gcc -O3 -std=c99 -Wall -Wextra -shared -o core/build/federative_har.dll core/federative_har.c
```

### Linux (`libfederative_har.so`)
```bash
gcc -O3 -std=c99 -Wall -Wextra -fPIC -shared -o core/build/libfederative_har.so core/federative_har.c -lm
```

### macOS (`libfederative_har.dylib`)
```bash
gcc -O3 -std=c99 -Wall -Wextra -dynamiclib -o core/build/libfederative_har.dylib core/federative_har.c
```

### Building and Running the Native Test Runner
Compile and execute Tier 1 native verification tests (`test_core.c`):

**On Windows:**
```bash
gcc -O3 -std=c99 -Wall -Wextra -Werror -Icore -o core/build/test_core.exe core/test_core.c core/federative_har.c -lm
./core/build/test_core.exe
```

**On Linux / macOS:**
```bash
gcc -O3 -std=c99 -Wall -Wextra -fsanitize=address,undefined -Icore -o core/build/test_core core/test_core.c core/federative_har.c -lm
./core/build/test_core
```

---

## 4. Helper Script (`core/build.py`)
To build automatically for the current host platform:
```bash
python core/build.py
```
To build and execute native tests:
```bash
python core/build.py --test
```
