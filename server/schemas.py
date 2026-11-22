from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ok')")
    current_model_version: int = Field(..., description="Currently active global model version")
    database_backend: str = Field(..., description="Backend database engine (sqlite, postgresql, etc.)")


class ModelMetadata(BaseModel):
    version: int = Field(..., description="Model version index (0, 1, 2...)")
    in_dim: int = Field(..., description="Input feature dimension (128)")
    hid_dim: int = Field(..., description="Hidden layer dimension (64)")
    out_dim: int = Field(..., description="Output classes dimension (6)")
    weights_count: int = Field(..., description="Total parameter count (8646)")
    sha256: str = Field(..., description="Hex SHA-256 digest of weights binary")
    val_accuracy: Optional[float] = Field(None, description="Validation accuracy on global_test (0.0 to 1.0)")
    avg_local_loss: float = Field(..., description="Weighted average client training loss")
    participants_count: int = Field(..., description="Number of contributing clients in this round")
    total_samples: int = Field(..., description="Total training samples aggregated across clients")
    created_at: datetime = Field(..., description="UTC creation timestamp")

    class Config:
        from_attributes = True


class SubmitResponse(BaseModel):
    status: str = Field(..., description="'accepted' or 'duplicate'")
    message: str = Field(..., description="Descriptive status message")
    accepted_clients: Optional[int] = Field(None, description="Current count of accepted client updates")
    min_clients: Optional[int] = Field(None, description="Minimum clients required to trigger aggregation")


class StaleConflictDetail(BaseModel):
    detail: str = Field(..., description="Conflict description")
    latest_model: ModelMetadata = Field(..., description="Current server model version and metadata")


class RoundStatusResponse(BaseModel):
    target_version: int = Field(..., description="Target model version being trained in active round")
    accepted_clients: int = Field(..., description="Count of client updates accepted in active round")
    min_clients: int = Field(..., description="Threshold to trigger immediate FedAvg aggregation")
    seconds_remaining: float = Field(..., description="Remaining seconds before round timeout triggers aggregation")


class TelemetryItem(BaseModel):
    id: int
    round_version: int
    client_id: str
    platform: str
    input_mode: str
    sample_count: int
    local_loss: float
    train_time_ms: Optional[float]
    status: str
    detail: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
