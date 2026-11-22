import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from client.dummy.dummy_client import run_client_round
from server import config, database
from server.database import Base, get_db, init_db, reset_db_engine
from server.main import app
from server.round_manager import round_manager

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def e2e_setup(tmp_path, monkeypatch):
    """Isolate database, storage directory, and round manager for end-to-end testing."""
    test_db = tmp_path / "e2e_federative_har.db"
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


def test_dummy_clients_e2e_round(e2e_setup):
    """Verify that 3 dummy clients train locally, submit updates, trigger FedAvg,

    advance the model version from v0 to v1, improve validation accuracy, and record telemetry.
    """
    client = e2e_setup

    # 1. Fetch initial model v0 metadata and accuracy
    res_v0 = client.get("/api/v1/model/latest")
    assert res_v0.status_code == 200
    meta_v0 = res_v0.json()
    assert meta_v0["version"] == 0
    acc_v0 = meta_v0["val_accuracy"]
    assert acc_v0 is not None

    # 2. Run local training and submission for 3 distinct clients
    clients = [
        ("dummy_client_0", "data/processed/non_iid/client_0.npz"),
        ("dummy_client_1", "data/processed/non_iid/client_1.npz"),
        ("dummy_client_2", "data/processed/non_iid/client_2.npz"),
    ]

    for client_id, shard_path in clients:
        full_shard_path = str(REPO_ROOT / shard_path)
        result = run_client_round(
            server_url="http://testserver",
            dataset_path=full_shard_path,
            client_id=client_id,
            epochs=5,
            batch_size=32,
            lr=0.05,
            http_client=client,
        )
        assert result["status"] == "accepted"
        assert result["version"] == 0
        assert result["post_acc"] > 0.0

    # 3. Verify server advanced to v1 with increased validation accuracy
    res_v1 = client.get("/api/v1/model/latest")
    assert res_v1.status_code == 200
    meta_v1 = res_v1.json()
    assert meta_v1["version"] == 1
    assert meta_v1["participants_count"] == 3

    acc_v1 = meta_v1["val_accuracy"]
    assert acc_v1 is not None
    # Confirm real learning occurred (v1 accuracy strictly higher than v0)
    assert acc_v1 > acc_v0, f"Expected acc_v1 ({acc_v1:.4f}) > acc_v0 ({acc_v0:.4f})"

    # 4. Verify telemetry records
    telem_res = client.get("/api/v1/telemetry?limit=10")
    assert telem_res.status_code == 200
    telemetry = telem_res.json()
    assert len(telemetry) >= 3

    accepted_telem = [t for t in telemetry if t["status"] == "accepted"]
    assert len(accepted_telem) == 3

    client_ids = {t["client_id"] for t in accepted_telem}
    assert client_ids == {"dummy_client_0", "dummy_client_1", "dummy_client_2"}

    for t in accepted_telem:
        assert t["platform"] == "dummy_pc"
        assert t["input_mode"] == "dataset"
        assert t["sample_count"] > 0
        assert t["local_loss"] > 0.0
