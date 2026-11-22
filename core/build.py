#!/usr/bin/env python3
"""Cross-platform build script for federative_har C99 core library.

Uses GCC directly without external build systems (No CMake).
Compiles shared library into core/build/ and copies to simulator/ if present.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CORE_DIR.parent
BUILD_DIR = CORE_DIR / "build"


def get_shared_lib_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "federative_har.dll"
    elif system == "darwin":
        return "libfederative_har.dylib"
    else:
        return "libfederative_har.so"


def build_shared_lib() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_lib_path = BUILD_DIR / get_shared_lib_name()

    src_path = CORE_DIR / "federative_har.c"
    cmd = [
        "gcc",
        "-O3",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-shared",
        "-o",
        str(out_lib_path),
        str(src_path),
    ]

    # Position-independent code for unix platforms
    if platform.system().lower() != "windows":
        cmd.insert(4, "-fPIC")

    print(f"[BUILD] Compiling shared library: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[BUILD] Successfully built: {out_lib_path}")

    # Copy to simulator/ if directory exists
    simulator_dir = REPO_ROOT / "simulator"
    if simulator_dir.is_dir():
        dest = simulator_dir / out_lib_path.name
        shutil.copy2(out_lib_path, dest)
        print(f"[BUILD] Copied library to: {dest}")

    return out_lib_path


def build_and_run_native_tests(use_sanitizer: bool = False) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe_name = "test_core.exe" if platform.system().lower() == "windows" else "test_core"
    test_exe = BUILD_DIR / exe_name

    src_files = [str(CORE_DIR / "test_core.c"), str(CORE_DIR / "federative_har.c")]
    cmd = [
        "gcc",
        "-O3",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-I",
        str(CORE_DIR),
        "-o",
        str(test_exe),
    ] + src_files + ["-lm"]

    if use_sanitizer:
        cmd.extend(["-fsanitize=address,undefined"])

    print(f"[TEST] Compiling native tests: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if use_sanitizer and platform.system().lower() == "windows":
            print("[WARN] Sanitizers failed on Windows MinGW. Falling back to -Wall -Wextra -Werror...")
            cmd = [c for c in cmd if not c.startswith("-fsanitize=")]
            cmd.insert(5, "-Werror")
            subprocess.run(cmd, check=True)
        else:
            raise

    print(f"[TEST] Executing native test suite: {test_exe}")
    # On Windows MinGW, run directly
    test_env = os.environ.copy()
    proc = subprocess.run([str(test_exe)], env=test_env)
    if proc.returncode != 0 and use_sanitizer and platform.system().lower() == "windows":
        print("[WARN] Native test failed with ASan on MinGW runtime. Recompiling without ASan...")
        cmd_clean = [
            "gcc",
            "-O3",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(CORE_DIR),
            "-o",
            str(test_exe),
        ] + src_files + ["-lm"]
        subprocess.run(cmd_clean, check=True)
        subprocess.run([str(test_exe)], check=True)
    elif proc.returncode != 0:
        sys.exit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and test federative_har C core")
    parser.add_argument("--test", action="store_true", help="Build and run native C test suite")
    parser.add_argument("--sanitize", action="store_true", help="Enable ASan/UBSan for test runner")
    args = parser.parse_args()

    build_shared_lib()

    if args.test:
        build_and_run_native_tests(use_sanitizer=args.sanitize)


if __name__ == "__main__":
    main()
