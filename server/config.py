import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{REPO_ROOT / 'server' / 'federative_har.db'}")

# Model Architecture Dimensions
IN_DIM = int(os.getenv("IN_DIM", "128"))
HID_DIM = int(os.getenv("HID_DIM", "64"))
OUT_DIM = int(os.getenv("OUT_DIM", "6"))
WEIGHTS_COUNT = int(os.getenv("WEIGHTS_COUNT", "8646"))
INIT_SEED = int(os.getenv("INIT_SEED", "42"))

# Federated Learning Round Settings
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "3"))
ROUND_TIMEOUT_SEC = float(os.getenv("ROUND_TIMEOUT_SEC", "30.0"))

# File and Artifact Storage Paths
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(REPO_ROOT / "server" / "storage")))
GLOBAL_TEST_PATH = Path(os.getenv("GLOBAL_TEST_PATH", str(REPO_ROOT / "data" / "processed" / "global_test.npz")))
