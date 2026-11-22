# Reference Python Dummy Client (`client/dummy/`)

A desktop reference client that simulates mobile device execution using the C99 compute core (`federative_har.dll` / `libfederative_har.so`) and communicating via the server REST API.

---

## 1. Start the Aggregation Server

In a terminal, start the central server:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Verify that the server is up by opening the interactive API docs:
- `http://localhost:8000/docs`

---

## 2. Run Dummy Clients to Complete a Federated Round

By default, the server requires `MIN_CLIENTS = 3` submissions to trigger FedAvg aggregation. Open three separate terminals and run each client with its corresponding non-IID data shard:

### Client 0 (Terminal 1):
```bash
python client/dummy/dummy_client.py --dataset data/processed/non_iid/client_0.npz --client-id dummy_client_0
```

### Client 1 (Terminal 2):
```bash
python client/dummy/dummy_client.py --dataset data/processed/non_iid/client_1.npz --client-id dummy_client_1
```

### Client 2 (Terminal 3):
```bash
python client/dummy/dummy_client.py --dataset data/processed/non_iid/client_2.npz --client-id dummy_client_2
```

---

## 3. Verify Aggregation & Metrics

Upon the 3rd submission:
1. The server performs FedAvg aggregation on-the-fly and evaluates validation accuracy on `global_test.npz`.
2. Inspect `GET /api/v1/model/latest` in Swagger UI (`http://localhost:8000/docs`):
   - `version` advances from `0` to `1`.
   - `participants_count` equals `3`.
   - `val_accuracy` reflects server-side holdout performance.
3. Inspect `GET /api/v1/telemetry` to view submission metrics and training runtimes.
