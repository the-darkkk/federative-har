from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from server import config
from server.database import ModelVersion, SessionLocal, TelemetryLog, get_db, init_db
from server.round_manager import InvalidWeightsException, StaleModelException, round_manager
from server.schemas import (
    HealthResponse,
    ModelMetadata,
    RoundStatusResponse,
    StaleConflictDetail,
    SubmitResponse,
    TelemetryItem,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as db:
        await round_manager.initialize(db)
    yield
    await round_manager.shutdown()


tags_metadata = [
    {
        "name": "Health",
        "description": "System health status and operational parameters.",
    },
    {
        "name": "Model",
        "description": "Global model metadata retrieval and raw binary weights distribution.",
    },
    {
        "name": "Round",
        "description": "Federated round orchestration, status monitoring, and client weight submissions.",
    },
    {
        "name": "Telemetry",
        "description": "Historical training progression, accuracy metrics, and device telemetry.",
    },
]

app = FastAPI(
    title="Federative HAR Server",
    version="1.0.0",
    description="Central Federated Learning aggregation server for on-device Human Activity Recognition.",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Returns service availability, active global model version, and connected database dialect.",
)
async def get_health():
    backend = "sqlite" if config.DATABASE_URL.startswith("sqlite") else "postgresql"
    return HealthResponse(
        status="ok",
        current_model_version=round_manager.current_version,
        database_backend=backend,
    )


@app.get(
    "/api/v1/model/latest",
    response_model=ModelMetadata,
    tags=["Model"],
    summary="Get Latest Model Metadata",
    description="Returns metadata and validation metrics for the current global model.",
)
async def get_latest_model():
    return round_manager.get_latest_metadata()


@app.get(
    "/api/v1/model/latest/weights",
    tags=["Model"],
    summary="Download Latest Model Weights",
    description="Streams raw float32 parameter array as application/octet-stream with validation headers.",
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Binary float32 buffer of model parameters.",
        }
    },
)
async def get_latest_weights():
    assert round_manager.current_weights is not None
    assert round_manager.current_model is not None

    weights_bytes = round_manager.current_weights.tobytes()
    headers = {
        "X-Model-Version": str(round_manager.current_version),
        "X-Weights-Sha256": round_manager.current_model.sha256,
        "Content-Disposition": f'attachment; filename="model_v{round_manager.current_version}.bin"',
    }
    return Response(
        content=weights_bytes,
        media_type="application/octet-stream",
        headers=headers,
    )


@app.post(
    "/api/v1/model/submit",
    response_model=SubmitResponse,
    tags=["Round"],
    summary="Submit Client Model Update",
    description="Receives local client training weights and telemetry via multipart/form-data.",
    responses={
        200: {"model": SubmitResponse, "description": "Update accepted or recognized duplicate."},
        409: {"model": StaleConflictDetail, "description": "Stale update rejected against outdated base model."},
        422: {"description": "Unprocessable payload (corrupt buffer, NaN/Inf, or shape mismatch)."},
    },
)
async def submit_model_update(
    client_id: str = Form(..., description="Unique client identifier"),
    base_version: int = Form(..., description="Base model version client started training from"),
    sample_count: int = Form(..., description="Number of local training samples utilized"),
    local_loss: float = Form(..., description="Average local training loss"),
    local_accuracy: Optional[float] = Form(None, description="Optional local accuracy metric"),
    platform: str = Form(..., description="Client runtime platform (android, ios, dummy_pc, etc.)"),
    input_mode: str = Form(..., description="Source of training data ('dataset' or 'sensor')"),
    train_time_ms: Optional[float] = Form(None, description="Local training wall-clock time in milliseconds"),
    weights: UploadFile = File(..., description="Binary float32 weights buffer (8646 floats = 34584 bytes)"),
    db: Session = Depends(get_db),
):
    raw_bytes = await weights.read()

    try:
        result = await round_manager.submit_update(
            client_id=client_id,
            base_version=base_version,
            sample_count=sample_count,
            local_loss=local_loss,
            local_accuracy=local_accuracy,
            platform=platform,
            input_mode=input_mode,
            train_time_ms=train_time_ms,
            weights_bytes=raw_bytes,
            db=db,
        )
        return SubmitResponse(**result)

    except StaleModelException as e:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": str(e),
                "latest_model": e.latest_model.model_dump(mode="json"),
            },
        )
    except InvalidWeightsException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@app.get(
    "/api/v1/round/status",
    response_model=RoundStatusResponse,
    tags=["Round"],
    summary="Get Active Round Status",
    description="Returns current active round target version, participant counts, and remaining timeout.",
)
async def get_round_status():
    return RoundStatusResponse(**round_manager.get_round_status())


@app.get(
    "/api/v1/models/history",
    response_model=List[ModelMetadata],
    tags=["Telemetry"],
    summary="List Model History",
    description="Chronological log of all aggregated model versions and their validation metrics.",
)
async def get_models_history(db: Session = Depends(get_db)):
    stmt = select(ModelVersion).order_by(ModelVersion.version.asc())
    records = db.execute(stmt).scalars().all()
    return [
        ModelMetadata(
            version=r.version,
            in_dim=config.IN_DIM,
            hid_dim=config.HID_DIM,
            out_dim=config.OUT_DIM,
            weights_count=config.WEIGHTS_COUNT,
            sha256=r.sha256,
            val_accuracy=r.val_accuracy,
            avg_local_loss=r.avg_local_loss,
            participants_count=r.participants_count,
            total_samples=r.total_samples,
            created_at=r.created_at,
        )
        for r in records
    ]


@app.get(
    "/api/v1/telemetry",
    response_model=List[TelemetryItem],
    tags=["Telemetry"],
    summary="List Telemetry Events",
    description="Recent submission telemetry events covering accepted, duplicate, and rejected updates.",
)
async def get_telemetry(
    limit: int = Query(50, ge=1, le=500, description="Max telemetry records to return"),
    db: Session = Depends(get_db),
):
    stmt = select(TelemetryLog).order_by(TelemetryLog.id.desc()).limit(limit)
    records = db.execute(stmt).scalars().all()
    return [TelemetryItem.model_validate(r) for r in records]
