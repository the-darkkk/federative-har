import argparse
import ctypes
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
import numpy as np

# Top-of-file Configuration Constants
SERVER_URL = "http://127.0.0.1:8000"
DATASET_PATH = "data/processed/non_iid/client_0.npz"
CLIENT_ID = "dummy_client_0"
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.05
ROUNDS = 1

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_c_core():
    core_dir = REPO_ROOT / "core"
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
        subprocess.run([sys.executable, str(build_script)], check=True)

    lib = ctypes.CDLL(str(lib_path))

    lib.fh_weights_len.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.fh_weights_len.restype = ctypes.c_int

    lib.fh_predict.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS, WRITEABLE"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fh_predict.restype = None

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


def evaluate_dataset(lib, weights: np.ndarray, X: np.ndarray, y: np.ndarray, in_dim: int, hid_dim: int, out_dim: int) -> float:
    n_samples = len(y)
    if n_samples == 0:
        return 0.0

    probs = np.empty(out_dim, dtype=np.float32)
    correct = 0
    for i in range(n_samples):
        lib.fh_predict(weights, X[i], probs, in_dim, hid_dim, out_dim)
        if int(np.argmax(probs)) == int(y[i]):
            correct += 1

    return float(correct) / float(n_samples)


def run_client_round(
    server_url: str = SERVER_URL,
    dataset_path: str = DATASET_PATH,
    client_id: str = CLIENT_ID,
    epochs: int = LOCAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    http_client: Optional[Any] = None,
) -> Dict[str, Any]:
    lib = load_c_core()

    data_file = Path(dataset_path)
    if not data_file.is_absolute():
        data_file = REPO_ROOT / data_file

    if not data_file.exists():
        raise FileNotFoundError(f"Local shard not found: {data_file}")

    data = np.load(data_file)
    X = np.ascontiguousarray(data["X"], dtype=np.float32)
    y = np.ascontiguousarray(data["y"], dtype=np.int32)
    n_samples = len(y)

    client = http_client if http_client is not None else httpx.Client(base_url=server_url, timeout=30.0)
    close_client = http_client is None

    try:
        # 1. Fetch Model Metadata
        meta_url = "/api/v1/model/latest" if http_client is not None else f"{server_url}/api/v1/model/latest"
        meta_res = client.get(meta_url)
        meta_res.raise_for_status()
        meta = meta_res.json()

        version = meta["version"]
        in_dim = meta["in_dim"]
        hid_dim = meta["hid_dim"]
        out_dim = meta["out_dim"]
        weights_count = meta["weights_count"]
        expected_sha = meta["sha256"]

        # 2. Download Global Weights
        weights_url = "/api/v1/model/latest/weights" if http_client is not None else f"{server_url}/api/v1/model/latest/weights"
        weights_res = client.get(weights_url)
        weights_res.raise_for_status()
        raw_bytes = weights_res.content

        expected_bytes = weights_count * 4
        if len(raw_bytes) != expected_bytes:
            raise ValueError(f"Downloaded weights length {len(raw_bytes)} != expected {expected_bytes}")

        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")

        weights = np.frombuffer(raw_bytes, dtype=np.float32).copy()

        # 3. Pre-Train Evaluation
        pre_acc = evaluate_dataset(lib, weights, X, y, in_dim, hid_dim, out_dim)

        # 4. Local On-Device Training
        # Deterministic seed per (client_id, version)
        seed = abs(hash(f"{client_id}_{version}")) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        final_epoch_loss = 0.0

        for epoch in range(epochs):
            perm = rng.permutation(n_samples)
            epoch_loss_sum = 0.0
            num_batches = 0

            for start in range(0, n_samples, batch_size):
                b_idx = perm[start : start + batch_size]
                bs = len(b_idx)
                x_batch = np.ascontiguousarray(X[b_idx], dtype=np.float32)
                y_batch = np.ascontiguousarray(y[b_idx], dtype=np.int32)

                loss = lib.fh_train_batch(weights, x_batch, y_batch, bs, in_dim, hid_dim, out_dim, lr)
                epoch_loss_sum += loss
                num_batches += 1

            final_epoch_loss = epoch_loss_sum / max(1, num_batches)

        t1 = time.perf_counter()
        train_time_ms = (t1 - t0) * 1000.0

        # 5. Post-Train Evaluation
        post_acc = evaluate_dataset(lib, weights, X, y, in_dim, hid_dim, out_dim)

        # 6. Submit Update & Telemetry
        submit_url = "/api/v1/model/submit" if http_client is not None else f"{server_url}/api/v1/model/submit"
        data_fields = {
            "client_id": client_id,
            "base_version": version,
            "sample_count": n_samples,
            "local_loss": round(float(final_epoch_loss), 6),
            "local_accuracy": round(float(post_acc), 6),
            "platform": "dummy_pc",
            "input_mode": "dataset",
            "train_time_ms": round(float(train_time_ms), 2),
        }
        files = {
            "weights": ("weights.bin", weights.tobytes(), "application/octet-stream"),
        }

        submit_res = client.post(submit_url, data=data_fields, files=files)

        if submit_res.status_code == 200:
            res_json = submit_res.json()
            status_val = res_json.get("status", "accepted")
            print(
                f"[{client_id}] base_v={version} -> status={status_val} samples={n_samples} "
                f"pre_acc={pre_acc:.3f} post_acc={post_acc:.3f} loss={final_epoch_loss:.4f} train_ms={train_time_ms:.1f}"
            )
            return {
                "status": status_val,
                "version": version,
                "pre_acc": pre_acc,
                "post_acc": post_acc,
                "local_loss": final_epoch_loss,
                "train_time_ms": train_time_ms,
                "response": res_json,
            }
        elif submit_res.status_code == 409:
            print(f"[{client_id}] base_v={version} -> status=stale_version (server advanced while training)")
            return {
                "status": "stale_version",
                "version": version,
                "pre_acc": pre_acc,
                "post_acc": post_acc,
                "local_loss": final_epoch_loss,
                "train_time_ms": train_time_ms,
                "response": submit_res.json(),
            }
        else:
            submit_res.raise_for_status()
            return {"status": "unknown"}

    finally:
        if close_client:
            client.close()


def run_client_loop(
    server_url: str = SERVER_URL,
    dataset_path: str = DATASET_PATH,
    client_id: str = CLIENT_ID,
    epochs: int = LOCAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    rounds: int = ROUNDS,
) -> None:
    client = httpx.Client(base_url=server_url, timeout=30.0)
    try:
        for r in range(rounds):
            result = run_client_round(
                server_url=server_url,
                dataset_path=dataset_path,
                client_id=client_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                http_client=client,
            )

            # If more rounds remaining, wait until server advances round target version
            if r < rounds - 1:
                cur_v = result.get("version", 0)
                poll_deadline = time.time() + 60.0
                while time.time() < poll_deadline:
                    try:
                        status_res = client.get("/api/v1/round/status")
                        if status_res.status_code == 200:
                            s = status_res.json()
                            if s["target_version"] > cur_v + 1:
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Federative HAR Reference Dummy Client")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="Server base URL")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to local dataset shard (.npz)")
    parser.add_argument("--client-id", type=str, default=CLIENT_ID, help="Unique client ID")
    parser.add_argument("--epochs", type=int, default=LOCAL_EPOCHS, help="Number of local training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="Number of federated rounds to run")

    args = parser.parse_args()

    run_client_loop(
        server_url=args.server,
        dataset_path=args.dataset,
        client_id=args.client_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
