"""Tier 2 Python FFI Contract Tests for federative_har C99 core library.

Validates ABI signatures, zero-copy pointer semantics, numerical stability,
determinism, and training on real MotionSense centralized samples.
"""

import ctypes
import os
import platform
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "core"
BUILD_DIR = CORE_DIR / "build"


def get_shared_lib_path() -> Path:
    system = platform.system().lower()
    if system == "windows":
        lib_name = "federative_har.dll"
    elif system == "darwin":
        lib_name = "libfederative_har.dylib"
    else:
        lib_name = "libfederative_har.so"

    lib_path = BUILD_DIR / lib_name
    if not lib_path.exists():
        # Build if missing
        build_script = CORE_DIR / "build.py"
        subprocess.run(["python", str(build_script)], check=True)

    assert lib_path.exists(), f"Shared library not found at {lib_path}"
    return lib_path


@pytest.fixture(scope="session")
def har_lib():
    """Load and bind ctypes function signatures for federative_har."""
    lib_path = get_shared_lib_path()
    lib = ctypes.CDLL(str(lib_path))

    # 1. int fh_weights_len(int in_dim, int hid_dim, int out_dim)
    lib.fh_weights_len.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.fh_weights_len.restype = ctypes.c_int

    # 2. void fh_init_weights(float* weights, int in_dim, int hid_dim, int out_dim, unsigned int seed)
    lib.fh_init_weights.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.fh_init_weights.restype = None

    # 3. void fh_preprocess_window(const float* raw_xyz, float* out_features, int window_len, float scale_to_g)
    lib.fh_preprocess_window.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.fh_preprocess_window.restype = None

    # 4. void fh_predict(const float* weights, const float* features, float* probs_out, int in_dim, int hid_dim, int out_dim)
    lib.fh_predict.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fh_predict.restype = None

    # 5. float fh_train_batch(float* weights, const float* X, const int* y, int batch_size, int in_dim, int hid_dim, int out_dim, float lr)
    lib.fh_train_batch.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.fh_train_batch.restype = ctypes.c_float

    return lib


def test_weights_len(har_lib):
    """Verify weight array lengths for default and arbitrary architectures."""
    # 128 -> 64 -> 6: (128*64) + 64 + (64*6) + 6 = 8192 + 64 + 384 + 6 = 8646
    assert har_lib.fh_weights_len(128, 64, 6) == 8646
    # 4 -> 3 -> 2: 12 + 3 + 6 + 2 = 23
    assert har_lib.fh_weights_len(4, 3, 2) == 23


def test_preprocess_window(har_lib):
    """Test raw sensor preprocessing matches numpy magnitude computation."""
    window_len = 128
    rng = np.random.default_rng(42)
    # Generate interleaved 3-axis accelerometer values in m/s^2
    raw_xyz_2d = rng.normal(loc=0.0, scale=9.8, size=(window_len, 3)).astype(np.float32)
    raw_xyz_flat = raw_xyz_2d.flatten()

    scale_to_g = 1.0 / 9.80665
    c_out = np.empty(window_len, dtype=np.float32)
    har_lib.fh_preprocess_window(raw_xyz_flat, c_out, window_len, scale_to_g)

    expected = np.linalg.norm(raw_xyz_2d * scale_to_g, axis=-1).astype(np.float32)
    np.testing.assert_allclose(c_out, expected, rtol=1e-5, atol=1e-5)


def test_predict_softmax_sum_and_shapes(har_lib):
    """Test forward inference generates normalized probabilities that sum to 1.0."""
    in_dim, hid_dim, out_dim = 128, 64, 6
    w_len = har_lib.fh_weights_len(in_dim, hid_dim, out_dim)

    weights = np.empty(w_len, dtype=np.float32)
    har_lib.fh_init_weights(weights, in_dim, hid_dim, out_dim, 42)

    features = np.ones(in_dim, dtype=np.float32)
    probs = np.empty(out_dim, dtype=np.float32)
    har_lib.fh_predict(weights, features, probs, in_dim, hid_dim, out_dim)

    assert probs.shape == (out_dim,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()


def test_predict_extreme_inputs(har_lib):
    """Verify numerical stability of Softmax under extreme positive and negative inputs."""
    in_dim, hid_dim, out_dim = 128, 64, 6
    w_len = har_lib.fh_weights_len(in_dim, hid_dim, out_dim)
    weights = np.empty(w_len, dtype=np.float32)
    har_lib.fh_init_weights(weights, in_dim, hid_dim, out_dim, 42)

    probs = np.empty(out_dim, dtype=np.float32)

    # 1. Very large positive features
    large_features = np.full(in_dim, 100.0, dtype=np.float32)
    har_lib.fh_predict(weights, large_features, probs, in_dim, hid_dim, out_dim)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    # 2. Large negative features
    neg_features = np.full(in_dim, -100.0, dtype=np.float32)
    har_lib.fh_predict(weights, neg_features, probs, in_dim, hid_dim, out_dim)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    # 3. All zeros
    zero_features = np.zeros(in_dim, dtype=np.float32)
    har_lib.fh_predict(weights, zero_features, probs, in_dim, hid_dim, out_dim)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_determinism_ctypes(har_lib):
    """Verify that multiple initialization calls with the same seed yield bit-identical arrays."""
    in_dim, hid_dim, out_dim = 128, 64, 6
    w_len = har_lib.fh_weights_len(in_dim, hid_dim, out_dim)

    w1 = np.empty(w_len, dtype=np.float32)
    w2 = np.empty(w_len, dtype=np.float32)

    har_lib.fh_init_weights(w1, in_dim, hid_dim, out_dim, 999)
    har_lib.fh_init_weights(w2, in_dim, hid_dim, out_dim, 999)

    assert np.array_equal(w1, w2)


def test_real_training_step(har_lib):
    """Test training step on real MotionSense samples from centralized_train.npz."""
    data_path = REPO_ROOT / "data" / "processed" / "centralized_train.npz"
    assert data_path.exists(), "data/processed/centralized_train.npz missing (run Stage 0 first)"

    data = np.load(data_path)
    X_train = data["X"][:64].astype(np.float32)
    y_train = data["y"][:64].astype(np.int32)
    batch_size = len(y_train)

    in_dim, hid_dim, out_dim = 128, 64, 6
    w_len = har_lib.fh_weights_len(in_dim, hid_dim, out_dim)
    weights = np.empty(w_len, dtype=np.float32)
    har_lib.fh_init_weights(weights, in_dim, hid_dim, out_dim, 42)

    lr = 0.1
    epochs = 20
    losses = []

    for _ in range(epochs):
        loss = har_lib.fh_train_batch(weights, X_train, y_train, batch_size, in_dim, hid_dim, out_dim, lr)
        losses.append(loss)

    assert not np.isnan(losses).any()
    # Loss should decrease over 20 epochs
    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    print(f"\n[Real Data Training] Initial Loss: {losses[0]:.4f} -> Final Loss (Epoch 20): {losses[-1]:.4f}")
