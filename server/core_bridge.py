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

    lib.fh_weights_len.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.fh_weights_len.restype = ctypes.c_int

    lib.fh_init_weights.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.fh_init_weights.restype = None

    lib.fh_predict.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fh_predict.restype = None

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
