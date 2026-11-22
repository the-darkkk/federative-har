import asyncio
import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import config, core_bridge, database
from server.database import SessionLocal, init_db, reset_db_engine
from server.main import app
from server.round_manager import round_manager


@pytest.fixture
def test_setup(tmp_path, monkeypatch):
    """Isolate database, storage directory, and round manager for each test."""
    test_db = tmp_path / "test_federative_har.db"
    test_storage = tmp_path / "storage"
    test_storage.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{test_db}"
    monkeypatch.setattr(config, "DATABASE_URL", db_url)
    monkeypatch.setattr(config, "STORAGE_DIR", test_storage)
    monkeypatch.setattr(config, "MIN_CLIENTS", 3)
    monkeypatch.setattr(config, "ROUND_TIMEOUT_SEC", 30.0)

    reset_db_engine()
    init_db()

    with TestClient(app) as client:
        yield client

    asyncio.run(round_manager.shutdown())
    reset_db_engine()


def test_startup_v0_model(test_setup):
    """Verify v0 model initialization, metadata schema, and weight download."""
    client = test_setup

    health_res = client.get("/api/v1/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "ok"
    assert health_data["current_model_version"] == 0
    assert health_data["database_backend"] == "sqlite"

    meta_res = client.get("/api/v1/model/latest")
    assert meta_res.status_code == 200
    meta = meta_res.json()
    assert meta["version"] == 0
    assert meta["in_dim"] == 128
    assert meta["hid_dim"] == 64
    assert meta["out_dim"] == 6
    assert meta["weights_count"] == 8646
    assert meta["participants_count"] == 0
    assert meta["val_accuracy"] is not None

    weights_res = client.get("/api/v1/model/latest/weights")
    assert weights_res.status_code == 200
    assert len(weights_res.content) == 34584  # 8646 floats * 4 bytes
    assert weights_res.headers["X-Model-Version"] == "0"

    expected_sha = hashlib.sha256(weights_res.content).hexdigest()
    assert weights_res.headers["X-Weights-Sha256"] == expected_sha
    assert meta["sha256"] == expected_sha


def test_fedavg_aggregation_advances_version(test_setup):
    """Verify that submitting 3 valid client updates triggers aggregation from v0 to v1."""
    client = test_setup

    initial_meta = client.get("/api/v1/model/latest").json()
    assert initial_meta["version"] == 0

    base_weights = core_bridge.generate_initial_weights(seed=42)

    # Submit 3 distinct client updates (MIN_CLIENTS = 3)
    for i in range(3):
        client_w = (base_weights + float(i) * 0.01).astype(np.float32)
        submit_res = client.post(
            "/api/v1/model/submit",
            data={
                "client_id": f"client_{i}",
                "base_version": 0,
                "sample_count": 100 * (i + 1),
                "local_loss": 1.2 - float(i) * 0.1,
                "local_accuracy": 0.85,
                "platform": "dummy_pc",
                "input_mode": "dataset",
                "train_time_ms": 150.0,
            },
            files={"weights": ("weights.bin", client_w.tobytes(), "application/octet-stream")},
        )
        assert submit_res.status_code == 200
        data = submit_res.json()
        assert data["status"] == "accepted"

    # Active round should have aggregated into v1
    new_meta = client.get("/api/v1/model/latest").json()
    assert new_meta["version"] == 1
    assert new_meta["participants_count"] == 3
    assert new_meta["total_samples"] == 100 + 200 + 300
    assert new_meta["val_accuracy"] is not None


def test_stale_submission_rejected(test_setup):
    """Verify that submitting against an outdated base_version returns 409 Conflict."""
    client = test_setup
    base_weights = core_bridge.generate_initial_weights(seed=42)

    # Advance model to v1 by submitting 3 updates
    for i in range(3):
        client_w = (base_weights + float(i) * 0.01).astype(np.float32)
        client.post(
            "/api/v1/model/submit",
            data={
                "client_id": f"worker_{i}",
                "base_version": 0,
                "sample_count": 50,
                "local_loss": 1.0,
                "platform": "android",
                "input_mode": "sensor",
            },
            files={"weights": ("weights.bin", client_w.tobytes(), "application/octet-stream")},
        )

    assert client.get("/api/v1/model/latest").json()["version"] == 1

    # Now submit an update with base_version = 0 (stale)
    stale_res = client.post(
        "/api/v1/model/submit",
        data={
            "client_id": "stale_worker",
            "base_version": 0,
            "sample_count": 50,
            "local_loss": 1.1,
            "platform": "android",
            "input_mode": "sensor",
        },
        files={"weights": ("weights.bin", base_weights.tobytes(), "application/octet-stream")},
    )

    assert stale_res.status_code == 409
    body = stale_res.json()
    assert "stale" in body["detail"].lower()
    assert body["latest_model"]["version"] == 1

    # Check telemetry logged rejection
    telem_res = client.get("/api/v1/telemetry?limit=5")
    events = telem_res.json()
    stale_events = [e for e in events if e["client_id"] == "stale_worker"]
    assert len(stale_events) == 1
    assert stale_events[0]["status"] == "rejected_stale"


def test_duplicate_submission_idempotency(test_setup):
    """Verify that a second submission by the same client in one round returns 200 (duplicate)."""
    client = test_setup
    w_bytes = core_bridge.generate_initial_weights(seed=42).tobytes()

    payload = {
        "client_id": "client_dup",
        "base_version": 0,
        "sample_count": 100,
        "local_loss": 0.9,
        "platform": "ios",
        "input_mode": "dataset",
    }
    files = {"weights": ("weights.bin", w_bytes, "application/octet-stream")}

    res1 = client.post("/api/v1/model/submit", data=payload, files=files)
    assert res1.status_code == 200
    assert res1.json()["status"] == "accepted"

    status1 = client.get("/api/v1/round/status").json()
    assert status1["accepted_clients"] == 1

    # Duplicate submission
    res2 = client.post("/api/v1/model/submit", data=payload, files=files)
    assert res2.status_code == 200
    assert res2.json()["status"] == "duplicate"

    status2 = client.get("/api/v1/round/status").json()
    # Accepted clients must NOT have incremented
    assert status2["accepted_clients"] == 1


def test_invalid_weights_payload(test_setup):
    """Verify payload rejection (422) on incorrect byte length or NaN/Inf values."""
    client = test_setup

    # 1. Truncated weights buffer (wrong length)
    bad_bytes = b"\x00" * 100
    res_trunc = client.post(
        "/api/v1/model/submit",
        data={
            "client_id": "c_bad_len",
            "base_version": 0,
            "sample_count": 50,
            "local_loss": 1.0,
            "platform": "dummy_pc",
            "input_mode": "dataset",
        },
        files={"weights": ("weights.bin", bad_bytes, "application/octet-stream")},
    )
    assert res_trunc.status_code == 422
    assert "Invalid weights byte length" in res_trunc.json()["detail"]

    # 2. NaN in weights buffer
    nan_weights = core_bridge.generate_initial_weights(seed=42)
    nan_weights[10] = np.nan
    res_nan = client.post(
        "/api/v1/model/submit",
        data={
            "client_id": "c_nan",
            "base_version": 0,
            "sample_count": 50,
            "local_loss": 1.0,
            "platform": "dummy_pc",
            "input_mode": "dataset",
        },
        files={"weights": ("weights.bin", nan_weights.tobytes(), "application/octet-stream")},
    )
    assert res_nan.status_code == 422
    assert "non-finite" in res_nan.json()["detail"].lower()

    # Check telemetry logged rejected_invalid
    telem = client.get("/api/v1/telemetry?limit=5").json()
    rejected_ids = [t["client_id"] for t in telem if t["status"] == "rejected_invalid"]
    assert "c_bad_len" in rejected_ids
    assert "c_nan" in rejected_ids


def test_round_timeout_aggregation(tmp_path, monkeypatch):
    """Verify that a round automatically aggregates upon timeout when clients < MIN_CLIENTS."""
    test_db = tmp_path / "test_timeout.db"
    test_storage = tmp_path / "storage_timeout"
    test_storage.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{test_db}"
    monkeypatch.setattr(config, "DATABASE_URL", db_url)
    monkeypatch.setattr(config, "STORAGE_DIR", test_storage)
    monkeypatch.setattr(config, "MIN_CLIENTS", 5)
    monkeypatch.setattr(config, "ROUND_TIMEOUT_SEC", 0.3)

    reset_db_engine()
    init_db()

    with TestClient(app) as client:
        w_bytes = core_bridge.generate_initial_weights().tobytes()
        res = client.post(
            "/api/v1/model/submit",
            data={
                "client_id": "single_client",
                "base_version": 0,
                "sample_count": 64,
                "local_loss": 0.75,
                "platform": "steam_deck",
                "input_mode": "sensor",
            },
            files={"weights": ("weights.bin", w_bytes, "application/octet-stream")},
        )
        assert res.status_code == 200

        # Wait for timeout (0.3s)
        time.sleep(0.6)

        # Model should have aggregated to v1 with 1 client
        latest = client.get("/api/v1/model/latest").json()
        assert latest["version"] == 1
        assert latest["participants_count"] == 1
        assert latest["total_samples"] == 64

    asyncio.run(round_manager.shutdown())
    reset_db_engine()


def test_history_and_telemetry_endpoints(test_setup):
    """Verify chronological model history and telemetry endpoints."""
    client = test_setup
    base_weights = core_bridge.generate_initial_weights(seed=42)

    # Submit 3 clients to produce v1
    for i in range(3):
        client.post(
            "/api/v1/model/submit",
            data={
                "client_id": f"history_client_{i}",
                "base_version": 0,
                "sample_count": 50,
                "local_loss": 1.0,
                "platform": "android",
                "input_mode": "sensor",
            },
            files={"weights": ("weights.bin", base_weights.tobytes(), "application/octet-stream")},
        )

    # Check history contains v0 and v1
    hist_res = client.get("/api/v1/models/history")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 2
    assert history[0]["version"] == 0
    assert history[1]["version"] == 1

    # Check telemetry contains submission events
    telem_res = client.get("/api/v1/telemetry?limit=10")
    assert telem_res.status_code == 200
    telemetry = telem_res.json()
    assert len(telemetry) >= 3
    client_ids = {t["client_id"] for t in telemetry}
    assert "history_client_0" in client_ids
