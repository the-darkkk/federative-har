from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from server import config

_engine = None
_session_factory = None


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    participants_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_local_loss: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class RoundSubmission(Base):
    __tablename__ = "round_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_version: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    client_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    local_loss: Mapped[float] = mapped_column(Float, nullable=False)
    local_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    weights_sha256: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("round_version", "client_id", name="uq_round_client"),
    )


class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_version: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    client_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String, nullable=False)
    input_mode: Mapped[str] = mapped_column(String, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    local_loss: Mapped[float] = mapped_column(Float, nullable=False)
    train_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {}
        if config.DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(config.DATABASE_URL, connect_args=connect_args)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _session_factory


def SessionLocal() -> Session:
    return get_session_factory()()


def reset_db_engine():
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def init_db(target_engine=None) -> None:
    eng = target_engine if target_engine is not None else get_engine()
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
