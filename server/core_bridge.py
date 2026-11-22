import ctypes
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from server import config

_core_lib = None


def get_core_lib():
    global _core_lib
    if _core_lib is not None:
        return _core_lib

    core_dir = config.REPO_ROOT / "core"
    build_dir = core_dir / "build"
    system = platform.system().lower()

    if system == "windows":
        lib_name = "federative_har.dll"
    elif system == "darwin":
        lib_name = "libfederative_har.dylib"
    else:
        lib_name = "libfederative_har.so"

    lib_path = build_dir / lib_name
    if not lib_path.exists():
        build_script = core_dir / "build.py"
        subprocess.run(["python", str(build_script)], check=True)

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

    _core_lib = lib
    return _core_lib


def generate_initial_weights(
    seed: int = config.INIT_SEED,
    in_dim: int = config.IN_DIM,
    hid_dim: int = config.HID_DIM,
    out_dim: int = config.OUT_DIM,
) -> np.ndarray:
    lib = get_core_lib()
    w_len = lib.fh_weights_len(in_dim, hid_dim, out_dim)
    weights = np.empty(w_len, dtype=np.float32)
    lib.fh_init_weights(weights, in_dim, hid_dim, out_dim, seed)
    return weights


def evaluate_weights(
    weights: np.ndarray,
    test_npz_path: Optional[Path | str] = None,
    in_dim: int = config.IN_DIM,
    hid_dim: int = config.HID_DIM,
    out_dim: int = config.OUT_DIM,
) -> Optional[float]:
    path = Path(test_npz_path) if test_npz_path is not None else config.GLOBAL_TEST_PATH
    if not path.is_file():
        return None

    data = np.load(path)
    X_test = np.ascontiguousarray(data["X"], dtype=np.float32)
    y_test = np.ascontiguousarray(data["y"], dtype=np.int32)

    total_samples = len(y_test)
    if total_samples == 0:
        return None

    lib = get_core_lib()
    probs = np.empty(out_dim, dtype=np.float32)
    correct = 0

    for i in range(total_samples):
        x_sample = X_test[i]
        lib.fh_predict(weights, x_sample, probs, in_dim, hid_dim, out_dim)
        if int(np.argmax(probs)) == int(y_test[i]):
            correct += 1

    return float(correct) / float(total_samples)
