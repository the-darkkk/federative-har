"""MotionSense Accelerometer Dataset Preparation Pipeline for Federated HAR.

Processes raw MotionSense accelerometer CSVs into standardized ML-ready datasets:
- Sample rate: 50 Hz
- Window size: 128 samples (2.56s), 50% overlap (stride 64)
- Activity classes (0..5): dws (0), ups (1), wlk (2), jog (3), sit (4), std (5)
- Feature: Orientation-invariant vector magnitude a_mag = sqrt(x^2 + y^2 + z^2)
- Subject holdout: Subjects 1..20 for train, 21..24 for global test
- Partitions: Centralized, 10-client IID, 10-client Non-IID (2 subjects/client)
- Mobile Demo Bundle: 180 balanced raw 3-axis windows (30/class) rounded to 4 decimals
"""

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Constants and specifications
CLASS_MAP = {
    "dws": 0,
    "ups": 1,
    "wlk": 2,
    "jog": 3,
    "sit": 4,
    "std": 5,
}

CLASS_NAMES = [
    "Downstairs",  # 0: dws
    "Upstairs",    # 1: ups
    "Walking",     # 2: wlk
    "Jogging",     # 3: jog
    "Sitting",     # 4: sit
    "Standing",    # 5: std
]

NUM_CLASSES = 6
SAMPLE_RATE_HZ = 50
WINDOW_SIZE = 128
STRIDE = 64
RANDOM_SEED = 42

TRAIN_SUBJECTS = list(range(1, 21))  # 1..20 (inclusive)
TEST_SUBJECTS = list(range(21, 25))  # 21..24 (inclusive)
NUM_CLIENTS = 10


def ensure_dataset_extracted(raw_dir: Path) -> Path:
    """Ensure MotionSense accelerometer data is present and extracted."""
    extracted_accel_dir = raw_dir / "extracted" / "B_Accelerometer_data"
    if extracted_accel_dir.exists() and any(extracted_accel_dir.iterdir()):
        return extracted_accel_dir

    repo_dir = raw_dir / "motion-sense"
    zip_path = repo_dir / "data" / "B_Accelerometer_data.zip"

    # Clone repository if zip is missing
    if not zip_path.exists():
        print(f"[INFO] Dataset archive not found at {zip_path}. Cloning repository...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git", "clone", "--depth", "1",
            "https://github.com/mmalekzadeh/motion-sense.git",
            str(repo_dir)
        ]
        subprocess.run(clone_cmd, check=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Expected accelerometer archive not found at {zip_path}")

    print(f"[INFO] Extracting {zip_path} -> {raw_dir / 'extracted'}...")
    extract_target = raw_dir / "extracted"
    extract_target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_target)

    if not extracted_accel_dir.exists():
        raise FileNotFoundError(f"Failed to locate extracted directory: {extracted_accel_dir}")

    return extracted_accel_dir


def load_and_window_data(
    accel_dir: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    """Parse all trial CSVs and partition windowed samples per subject."""
    subject_records: Dict[int, Dict[str, List[Any]]] = {
        sub: {"raw": [], "mag": [], "labels": []}
        for sub in range(1, 25)
    }

    subdirs = sorted(os.listdir(accel_dir))
    total_files = 0
    total_dropped_windows = 0

    for folder_name in subdirs:
        folder_path = accel_dir / folder_name
        if not folder_path.is_dir():
            continue

        prefix = folder_name.split("_")[0].lower()
        if prefix not in CLASS_MAP:
            continue
        label = CLASS_MAP[prefix]

        for file_name in sorted(os.listdir(folder_path)):
            match = re.match(r"^sub_(\d+)\.csv$", file_name, re.IGNORECASE)
            if not match:
                continue

            sub_id = int(match.group(1))
            if sub_id not in subject_records:
                continue

            file_path = folder_path / file_name
            total_files += 1

            df = pd.read_csv(file_path)
            if not {"x", "y", "z"}.issubset(df.columns):
                raise ValueError(f"Missing required coordinate columns in {file_path}")

            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            num_rows = len(xyz)

            # Slide window across trial recording
            for start_idx in range(0, num_rows - window_size + 1, stride):
                end_idx = start_idx + window_size
                w_xyz = xyz[start_idx:end_idx]

                # Drop invalid windows containing NaN or Inf
                if np.isnan(w_xyz).any() or np.isinf(w_xyz).any():
                    total_dropped_windows += 1
                    continue

                # Orientation-invariant vector magnitude: a_mag = sqrt(x^2 + y^2 + z^2)
                w_mag = np.sqrt(np.sum(np.square(w_xyz), axis=-1, dtype=np.float32)).astype(np.float32)

                subject_records[sub_id]["raw"].append(w_xyz)
                subject_records[sub_id]["mag"].append(w_mag)
                subject_records[sub_id]["labels"].append(label)

    print(f"[INFO] Processed {total_files} CSV files. Dropped {total_dropped_windows} invalid windows.")
    return subject_records


def build_splits(
    subject_records: Dict[int, Dict[str, List[Any]]],
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Construct centralized train, global test, IID, non-IID shards, and mobile bundle."""
    rng = np.random.default_rng(seed)

    # 1. Centralized Train (Subjects 1..20)
    train_mag_list: List[np.ndarray] = []
    train_raw_list: List[np.ndarray] = []
    train_labels_list: List[int] = []

    for sub in TRAIN_SUBJECTS:
        train_mag_list.extend(subject_records[sub]["mag"])
        train_raw_list.extend(subject_records[sub]["raw"])
        train_labels_list.extend(subject_records[sub]["labels"])

    X_train = np.array(train_mag_list, dtype=np.float32)
    y_train = np.array(train_labels_list, dtype=np.int32)
    X_train_raw = np.array(train_raw_list, dtype=np.float32)

    # Deterministic permutation for centralized train
    train_perm = rng.permutation(len(y_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]
    X_train_raw = X_train_raw[train_perm]

    # 2. Global Test (Subjects 21..24)
    test_mag_list: List[np.ndarray] = []
    test_raw_list: List[np.ndarray] = []
    test_labels_list: List[int] = []

    for sub in TEST_SUBJECTS:
        test_mag_list.extend(subject_records[sub]["mag"])
        test_raw_list.extend(subject_records[sub]["raw"])
        test_labels_list.extend(subject_records[sub]["labels"])

    X_test = np.array(test_mag_list, dtype=np.float32)
    y_test = np.array(test_labels_list, dtype=np.int32)
    X_test_raw = np.array(test_raw_list, dtype=np.float32)

    # Deterministic permutation for global test
    test_perm = rng.permutation(len(y_test))
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]
    X_test_raw = X_test_raw[test_perm]

    # 3. IID Partitions (Randomly shuffled, evenly partitioned from centralized train)
    iid_shards: Dict[str, Dict[str, np.ndarray]] = {}
    iid_splits = np.array_split(np.arange(len(y_train)), NUM_CLIENTS)

    for client_id, shard_indices in enumerate(iid_splits):
        client_name = f"client_{client_id}"
        iid_shards[client_name] = {
            "X": X_train[shard_indices].astype(np.float32),
            "y": y_train[shard_indices].astype(np.int32),
        }

    # 4. Realistic Non-IID Partitions (Subject-grouped: each client gets 2 distinct subjects)
    non_iid_shards: Dict[str, Dict[str, Any]] = {}
    for client_id in range(NUM_CLIENTS):
        sub_a = 2 * client_id + 1
        sub_b = 2 * client_id + 2
        assigned_subs = [sub_a, sub_b]

        c_mag = subject_records[sub_a]["mag"] + subject_records[sub_b]["mag"]
        c_labels = subject_records[sub_a]["labels"] + subject_records[sub_b]["labels"]

        c_X = np.array(c_mag, dtype=np.float32)
        c_y = np.array(c_labels, dtype=np.int32)

        # Shuffle each client's partition deterministically
        c_rng = np.random.default_rng(seed + client_id + 100)
        c_perm = c_rng.permutation(len(c_y))
        c_X = c_X[c_perm]
        c_y = c_y[c_perm]

        client_name = f"client_{client_id}"
        non_iid_shards[client_name] = {
            "subjects": assigned_subs,
            "X": c_X,
            "y": c_y,
        }

    # 5. Mobile Demo Bundle: 30 balanced examples per class from global_test
    bundle_rng = np.random.default_rng(seed)
    selected_indices: List[int] = []

    for c in range(NUM_CLASSES):
        class_candidates = np.where(y_test == c)[0]
        if len(class_candidates) < 30:
            raise ValueError(
                f"Class {c} ({CLASS_NAMES[c]}) only has {len(class_candidates)} test samples, needed 30"
            )
        chosen = bundle_rng.choice(class_candidates, size=30, replace=False)
        selected_indices.extend(chosen)

    bundle_raw = X_test_raw[selected_indices]
    bundle_labels = y_test[selected_indices]

    # Convert to JSON serializable structures with 4 decimal places
    demo_bundle = {
        "dataset": "MotionSense",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "num_samples": len(bundle_labels),
        "samples_per_class": 30,
        "labels": [int(lbl) for lbl in bundle_labels],
        "windows_xyz": [
            [[round(float(coord), 4) for coord in sample] for sample in window]
            for window in bundle_raw
        ],
    }

    return {
        "centralized_train": {"X": X_train, "y": y_train},
        "global_test": {"X": X_test, "y": y_test},
        "iid_shards": iid_shards,
        "non_iid_shards": non_iid_shards,
        "mobile_demo_bundle": demo_bundle,
    }


def compute_class_distribution(y: np.ndarray) -> Dict[str, int]:
    """Compute per-class instance counts."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return {str(cls_id): int(counts[cls_id]) for cls_id in range(NUM_CLASSES)}


def save_dataset_artifacts(splits: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Write all processed artifacts and metadata to data/processed/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iid_dir = output_dir / "iid"
    non_iid_dir = output_dir / "non_iid"
    iid_dir.mkdir(parents=True, exist_ok=True)
    non_iid_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save global_test.npz
    gt = splits["global_test"]
    gt_path = output_dir / "global_test.npz"
    np.savez_compressed(gt_path, X=gt["X"], y=gt["y"])

    # 2. Save centralized_train.npz
    ct = splits["centralized_train"]
    ct_path = output_dir / "centralized_train.npz"
    np.savez_compressed(ct_path, X=ct["X"], y=ct["y"])

    # 3. Save IID shards
    iid_summary = {}
    for client_name, shard in splits["iid_shards"].items():
        client_path = iid_dir / f"{client_name}.npz"
        np.savez_compressed(client_path, X=shard["X"], y=shard["y"])
        iid_summary[client_name] = {
            "num_samples": len(shard["y"]),
            "class_distribution": compute_class_distribution(shard["y"]),
        }

    # 4. Save Non-IID shards
    non_iid_summary = {}
    for client_name, shard in splits["non_iid_shards"].items():
        client_path = non_iid_dir / f"{client_name}.npz"
        np.savez_compressed(client_path, X=shard["X"], y=shard["y"])
        non_iid_summary[client_name] = {
            "subjects": shard["subjects"],
            "num_samples": len(shard["y"]),
            "class_distribution": compute_class_distribution(shard["y"]),
        }

    # 5. Save mobile_demo_bundle.json
    mobile_bundle_path = output_dir / "mobile_demo_bundle.json"
    with open(mobile_bundle_path, "w", encoding="utf-8") as f:
        json.dump(splits["mobile_demo_bundle"], f, indent=2)

    # 6. Save metadata.json
    metadata = {
        "dataset": "MotionSense",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_mapping": CLASS_MAP,
        "feature_type": "orientation_invariant_vector_magnitude",
        "feature_formula": "sqrt(x^2 + y^2 + z^2)",
        "feature_shape": [WINDOW_SIZE],
        "random_seed": RANDOM_SEED,
        "num_clients": NUM_CLIENTS,
        "subjects": {
            "train_subjects": TRAIN_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
        },
        "sample_counts": {
            "centralized_train": len(ct["y"]),
            "global_test": len(gt["y"]),
            "total_windows": len(ct["y"]) + len(gt["y"]),
            "mobile_demo_bundle": splits["mobile_demo_bundle"]["num_samples"],
        },
        "splits": {
            "centralized_train": {
                "num_samples": len(ct["y"]),
                "class_distribution": compute_class_distribution(ct["y"]),
            },
            "global_test": {
                "num_samples": len(gt["y"]),
                "class_distribution": compute_class_distribution(gt["y"]),
            },
            "iid": iid_summary,
            "non_iid": non_iid_summary,
            "mobile_demo_bundle": {
                "num_samples": splits["mobile_demo_bundle"]["num_samples"],
                "samples_per_class": splits["mobile_demo_bundle"]["samples_per_class"],
                "class_distribution": compute_class_distribution(
                    np.array(splits["mobile_demo_bundle"]["labels"], dtype=np.int32)
                ),
            },
        },
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def print_verification_summary(output_dir: Path) -> None:
    """Read back saved files and print a detailed verification table."""
    header = (
        f"{'Artifact':<32} | {'X Shape':<14} | {'X Dtype':<9} | "
        f"{'y Shape':<12} | {'y Dtype':<7} | {'Class Distribution [0..5]'}"
    )
    separator = "-" * 115

    print("\n" + "=" * 115)
    print("STAGE 0: DATASET PREPARATION VERIFICATION SUMMARY")
    print("=" * 115)
    print(header)
    print(separator)

    # Centralized and Test
    for filename in ["centralized_train.npz", "global_test.npz"]:
        file_path = output_dir / filename
        data = np.load(file_path)
        X, y = data["X"], data["y"]
        counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
        print(
            f"{filename:<32} | {str(X.shape):<14} | {str(X.dtype):<9} | "
            f"{str(y.shape):<12} | {str(y.dtype):<7} | {counts}"
        )

    print(separator)

    # IID shards
    for i in range(NUM_CLIENTS):
        name = f"iid/client_{i}.npz"
        file_path = output_dir / "iid" / f"client_{i}.npz"
        data = np.load(file_path)
        X, y = data["X"], data["y"]
        counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
        print(
            f"{name:<32} | {str(X.shape):<14} | {str(X.dtype):<9} | "
            f"{str(y.shape):<12} | {str(y.dtype):<7} | {counts}"
        )

    print(separator)

    # Non-IID shards
    for i in range(NUM_CLIENTS):
        name = f"non_iid/client_{i}.npz (S{2*i+1},{2*i+2})"
        file_path = output_dir / "non_iid" / f"client_{i}.npz"
        data = np.load(file_path)
        X, y = data["X"], data["y"]
        counts = [int(np.sum(y == c)) for c in range(NUM_CLASSES)]
        print(
            f"{name:<32} | {str(X.shape):<14} | {str(X.dtype):<9} | "
            f"{str(y.shape):<12} | {str(y.dtype):<7} | {counts}"
        )

    print(separator)

    # Mobile demo bundle
    bundle_path = output_dir / "mobile_demo_bundle.json"
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    b_windows = np.array(bundle["windows_xyz"], dtype=np.float32)
    b_labels = np.array(bundle["labels"], dtype=np.int32)
    b_counts = [int(np.sum(b_labels == c)) for c in range(NUM_CLASSES)]
    print(
        f"{'mobile_demo_bundle.json':<32} | {str(b_windows.shape):<14} | "
        f"{str(b_windows.dtype):<9} | {str(b_labels.shape):<12} | {str(b_labels.dtype):<7} | {b_counts}"
    )

    print("=" * 115)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MotionSense dataset for Federated HAR")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Path to save ML-ready processed datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("[INFO] Starting MotionSense preparation pipeline...")
    accel_dir = ensure_dataset_extracted(args.raw_dir)
    print(f"[INFO] Using accelerometer data from: {accel_dir}")

    subject_records = load_and_window_data(accel_dir)
    splits = build_splits(subject_records, seed=args.seed)

    print(f"[INFO] Saving processed dataset artifacts to: {args.output_dir}")
    save_dataset_artifacts(splits, args.output_dir)

    print_verification_summary(args.output_dir)
    print("[SUCCESS] Dataset pipeline executed successfully.")


if __name__ == "__main__":
    main()
