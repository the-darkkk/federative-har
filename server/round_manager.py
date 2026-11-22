import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from server import config, core_bridge
from server.aggregator import compute_fedavg, validate_weights_buffer
from server.database import ModelVersion, RoundSubmission, SessionLocal, TelemetryLog
from server.schemas import ModelMetadata


class StaleModelException(Exception):
    def __init__(self, message: str, latest_model: ModelMetadata):
        super().__init__(message)
        self.latest_model = latest_model


class InvalidWeightsException(Exception):
    pass


class RoundManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.current_version: int = 0
        self.current_model: Optional[ModelVersion] = None
        self.current_weights: Optional[np.ndarray] = None
        self.target_version: int = 1
        self.started_at: float = time.time()
        self.pending_updates: Dict[str, Dict[str, Any]] = {}
        self.timeout_task: Optional[asyncio.Task] = None

    async def initialize(self, db: Session) -> None:
        async with self.lock:
            config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

            # Check if any model version exists
            stmt = select(ModelVersion).order_by(ModelVersion.version.desc()).limit(1)
            latest = db.execute(stmt).scalar_one_or_none()

            if latest is None:
                # Bootstrap initial model v0
                v0_weights = core_bridge.generate_initial_weights(
                    seed=config.INIT_SEED,
                    in_dim=config.IN_DIM,
                    hid_dim=config.HID_DIM,
                    out_dim=config.OUT_DIM,
                )
                v0_bytes = v0_weights.tobytes()
                v0_sha = hashlib.sha256(v0_bytes).hexdigest()
                v0_path = config.STORAGE_DIR / "model_v0.bin"
                with open(v0_path, "wb") as f:
                    f.write(v0_bytes)

                val_acc = core_bridge.evaluate_weights(
                    v0_weights,
                    test_npz_path=config.GLOBAL_TEST_PATH,
                    in_dim=config.IN_DIM,
                    hid_dim=config.HID_DIM,
                    out_dim=config.OUT_DIM,
                )

                model_v0 = ModelVersion(
                    version=0,
                    file_path=str(v0_path),
                    sha256=v0_sha,
                    participants_count=0,
                    total_samples=0,
                    avg_local_loss=0.0,
                    val_accuracy=val_acc,
                )
                db.add(model_v0)
                db.commit()
                db.refresh(model_v0)

                self.current_version = 0
                self.current_model = model_v0
                self.current_weights = v0_weights
            else:
                self.current_version = latest.version
                self.current_model = latest
                with open(latest.file_path, "rb") as f:
                    self.current_weights = np.frombuffer(f.read(), dtype=np.float32).copy()

            self.target_version = self.current_version + 1
            self.started_at = time.time()
            self.pending_updates.clear()
            self._schedule_timeout_locked()

    def _schedule_timeout_locked(self) -> None:
        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()

        if config.ROUND_TIMEOUT_SEC > 0:
            self.timeout_task = asyncio.create_task(
                self._timeout_worker(self.target_version)
            )

    async def _timeout_worker(self, target_version: int) -> None:
        try:
            await asyncio.sleep(config.ROUND_TIMEOUT_SEC)
            async with self.lock:
                if self.target_version == target_version:
                    if len(self.pending_updates) >= 1:
                        # Timeout triggered aggregation with at least 1 update
                        with SessionLocal() as db:
                            await self._aggregate_locked(db)
                    else:
                        # No updates submitted, renew round start timer
                        self.started_at = time.time()
                        self._schedule_timeout_locked()
        except asyncio.CancelledError:
            pass

    async def submit_update(
        self,
        client_id: str,
        base_version: int,
        sample_count: int,
        local_loss: float,
        local_accuracy: Optional[float],
        platform: str,
        input_mode: str,
        train_time_ms: Optional[float],
        weights_bytes: bytes,
        db: Session,
    ) -> Dict[str, Any]:
        async with self.lock:
            # 1. Stale model check
            if base_version != self.current_version:
                telemetry = TelemetryLog(
                    round_version=base_version,
                    client_id=client_id,
                    platform=platform,
                    input_mode=input_mode,
                    sample_count=sample_count,
                    local_loss=local_loss,
                    train_time_ms=train_time_ms,
                    status="rejected_stale",
                    detail=f"Base version {base_version} != current {self.current_version}",
                )
                db.add(telemetry)
                db.commit()

                latest_meta = self.get_latest_metadata()
                raise StaleModelException(
                    message=f"Model version {base_version} is stale. Current version is {self.current_version}.",
                    latest_model=latest_meta,
                )

            # 2. Corrupt / invalid weights buffer check
            try:
                weights_arr = validate_weights_buffer(
                    weights_bytes, expected_len=config.WEIGHTS_COUNT
                )
            except ValueError as e:
                telemetry = TelemetryLog(
                    round_version=base_version,
                    client_id=client_id,
                    platform=platform,
                    input_mode=input_mode,
                    sample_count=sample_count,
                    local_loss=local_loss,
                    train_time_ms=train_time_ms,
                    status="rejected_invalid",
                    detail=str(e),
                )
                db.add(telemetry)
                db.commit()
                raise InvalidWeightsException(str(e))

            # 3. Duplicate submission check
            if client_id in self.pending_updates:
                telemetry = TelemetryLog(
                    round_version=base_version,
                    client_id=client_id,
                    platform=platform,
                    input_mode=input_mode,
                    sample_count=sample_count,
                    local_loss=local_loss,
                    train_time_ms=train_time_ms,
                    status="duplicate",
                    detail=f"Client {client_id} already submitted update for version {self.current_version}",
                )
                db.add(telemetry)
                db.commit()
                return {
                    "status": "duplicate",
                    "message": "Submission already recorded for this active round",
                    "accepted_clients": len(self.pending_updates),
                    "min_clients": config.MIN_CLIENTS,
                }

            # 4. Valid update acceptance
            weights_sha = hashlib.sha256(weights_bytes).hexdigest()

            submission = RoundSubmission(
                round_version=self.current_version,
                client_id=client_id,
                sample_count=sample_count,
                local_loss=local_loss,
                local_accuracy=local_accuracy,
                weights_sha256=weights_sha,
            )
            db.add(submission)

            telemetry = TelemetryLog(
                round_version=self.current_version,
                client_id=client_id,
                platform=platform,
                input_mode=input_mode,
                sample_count=sample_count,
                local_loss=local_loss,
                train_time_ms=train_time_ms,
                status="accepted",
                detail=None,
            )
            db.add(telemetry)
            db.commit()

            self.pending_updates[client_id] = {
                "weights": weights_arr,
                "sample_count": sample_count,
                "local_loss": local_loss,
                "local_accuracy": local_accuracy,
            }

            accepted_count = len(self.pending_updates)

            # Check if threshold reached for immediate aggregation
            if accepted_count >= config.MIN_CLIENTS:
                await self._aggregate_locked(db)

            return {
                "status": "accepted",
                "message": "Model update accepted",
                "accepted_clients": accepted_count,
                "min_clients": config.MIN_CLIENTS,
            }

    async def _aggregate_locked(self, db: Session) -> None:
        if not self.pending_updates:
            return

        updates = [(u["weights"], u["sample_count"]) for u in self.pending_updates.values()]
        new_weights = compute_fedavg(updates)

        val_acc = core_bridge.evaluate_weights(
            new_weights,
            test_npz_path=config.GLOBAL_TEST_PATH,
            in_dim=config.IN_DIM,
            hid_dim=config.HID_DIM,
            out_dim=config.OUT_DIM,
        )

        participants_count = len(self.pending_updates)
        total_samples = sum(u["sample_count"] for u in self.pending_updates.values())
        avg_local_loss = (
            sum(u["local_loss"] * u["sample_count"] for u in self.pending_updates.values())
            / total_samples
        )

        new_version = self.current_version + 1
        new_bytes = new_weights.tobytes()
        new_sha = hashlib.sha256(new_bytes).hexdigest()
        new_path = config.STORAGE_DIR / f"model_v{new_version}.bin"

        with open(new_path, "wb") as f:
            f.write(new_bytes)

        new_model = ModelVersion(
            version=new_version,
            file_path=str(new_path),
            sha256=new_sha,
            participants_count=participants_count,
            total_samples=total_samples,
            avg_local_loss=float(avg_local_loss),
            val_accuracy=val_acc,
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)

        self.current_version = new_version
        self.current_model = new_model
        self.current_weights = new_weights
        self.target_version = self.current_version + 1
        self.started_at = time.time()
        self.pending_updates.clear()
        self._schedule_timeout_locked()

    def get_latest_metadata(self) -> ModelMetadata:
        assert self.current_model is not None
        return ModelMetadata(
            version=self.current_model.version,
            in_dim=config.IN_DIM,
            hid_dim=config.HID_DIM,
            out_dim=config.OUT_DIM,
            weights_count=config.WEIGHTS_COUNT,
            sha256=self.current_model.sha256,
            val_accuracy=self.current_model.val_accuracy,
            avg_local_loss=self.current_model.avg_local_loss,
            participants_count=self.current_model.participants_count,
            total_samples=self.current_model.total_samples,
            created_at=self.current_model.created_at,
        )

    def get_round_status(self) -> Dict[str, Any]:
        elapsed = time.time() - self.started_at
        remaining = max(0.0, config.ROUND_TIMEOUT_SEC - elapsed)
        return {
            "target_version": self.target_version,
            "accepted_clients": len(self.pending_updates),
            "min_clients": config.MIN_CLIENTS,
            "seconds_remaining": round(remaining, 2),
        }

    async def shutdown(self) -> None:
        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()


round_manager = RoundManager()
