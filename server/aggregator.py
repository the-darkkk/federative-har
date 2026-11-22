from typing import List, Tuple

import numpy as np

from server import config


def validate_weights_buffer(
    raw_bytes: bytes,
    expected_len: int = config.WEIGHTS_COUNT,
) -> np.ndarray:
    expected_bytes = expected_len * 4
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Invalid weights byte length: received {len(raw_bytes)}, expected {expected_bytes} bytes ({expected_len} float32 values)"
        )

    arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()

    if not np.isfinite(arr).all():
        raise ValueError("Weights buffer contains non-finite values (NaN or Inf)")

    return arr


def compute_fedavg(updates: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    if not updates:
        raise ValueError("Cannot compute FedAvg on an empty list of updates")

    weights_list = [u[0] for u in updates]
    sample_counts = np.array([u[1] for u in updates], dtype=np.float64)

    total_samples = np.sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total sample count across submissions must be positive")

    weights_matrix = np.stack(weights_list, axis=0)
    averaged = np.average(weights_matrix, axis=0, weights=sample_counts).astype(np.float32)
    return averaged
