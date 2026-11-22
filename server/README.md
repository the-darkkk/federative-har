# Central Federated Learning Server (`server/`)

FastAPI aggregation server, SQLAlchemy ORM persistence layer, and round state coordinator for Federated Human Activity Recognition (HAR).

---

## 1. Quickstart

Run the server locally using Uvicorn:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is accessible at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 2. Configuration & Environment Variables

Settings are controlled via environment variables or loaded from default fallbacks:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `sqlite:///./server/federative_har.db` | Database connection string. Switching to PostgreSQL (`postgresql+psycopg://user:pass@host:5432/dbname`) requires zero code changes. |
| `MIN_CLIENTS` | `3` | Minimum valid client updates required to trigger immediate FedAvg aggregation. |
| `ROUND_TIMEOUT_SEC` | `30.0` | Maximum wait time in seconds before an active round aggregates with available submissions (>= 1). |
| `IN_DIM` | `128` | Model input dimension (window magnitude sequence length). |
| `HID_DIM` | `64` | Hidden layer width. |
| `OUT_DIM` | `6` | Number of activity target classes. |
| `WEIGHTS_COUNT` | `8646` | Total float32 parameter count (34,584 bytes). |
| `INIT_SEED` | `42` | Seed for deterministic bootstrap initialization of model `v0`. |
| `STORAGE_DIR` | `server/storage` | File system path for saving binary model weight checkpoints (`model_vN.bin`). |
| `GLOBAL_TEST_PATH` | `data/processed/global_test.npz` | Holdout validation dataset used for server-side evaluation. |

---

## 3. REST API Specifications (`/api/v1`)

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status, current model version, and database backend. | `200` |
| `GET` | `/api/v1/model/latest` | Metadata and validation accuracy for the active global model. | `200` |
| `GET` | `/api/v1/model/latest/weights` | Raw binary `float32` weight stream (`application/octet-stream`) with `X-Model-Version` and `X-Weights-Sha256` headers. | `200` |
| `POST` | `/api/v1/model/submit` | Submit client model update and telemetry via `multipart/form-data`. | `200` (accepted / duplicate)<br>`409` (stale model conflict)<br>`422` (invalid/corrupted weights buffer) |
| `GET` | `/api/v1/round/status` | Current active round target version, accepted client count, threshold, and remaining timeout. | `200` |
| `GET` | `/api/v1/models/history` | Chronological list of all model checkpoints and server evaluation metrics. | `200` |
| `GET` | `/api/v1/telemetry` | Recent client submission telemetry logs (`limit` parameter, default 50). | `200` |

---

## 4. Client Submission Contract (`POST /api/v1/model/submit`)

Multipart form data fields:
- `client_id` (`string`, required): Unique identifier for the client device.
- `base_version` (`integer`, required): The global model version from which local training started.
- `sample_count` (`integer`, required): Number of training windows used in local update.
- `local_loss` (`float`, required): Average Cross-Entropy loss on local client data.
- `local_accuracy` (`float`, optional): Local validation accuracy if computed on device.
- `platform` (`string`, required): Client OS / environment (e.g. `dummy_pc`, `android`, `ios`, `steam_deck`).
- `input_mode` (`string`, required): Data source mode (`dataset` or `sensor`).
- `train_time_ms` (`float`, optional): Wall-clock training duration in milliseconds.
- `weights` (`binary file`, required): Exactly 8,646 `float32` values (34,584 bytes), matching C tensor layout `[W1, b1, W2, b2]`.

### Response Codes:
- `200 OK`: Submission accepted (`status="accepted"`) or recognized duplicate (`status="duplicate"`).
- `409 Conflict`: `base_version != current_version`. Returns error message and latest model metadata for client synchronization.
- `422 Unprocessable Entity`: Byte length does not equal 34,584 bytes or buffer contains `NaN`/`Inf` values.
