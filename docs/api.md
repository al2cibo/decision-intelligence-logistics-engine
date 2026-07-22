# API Reference

## Running the API

Make sure dependencies are installed, then start the server:

```bash
PYTHONPATH=src uvicorn api.app:app --reload
```

The server will be available at `http://localhost:8000`.

### Interactive docs (Swagger UI)

Open `http://localhost:8000/docs` in your browser. FastAPI generates a full interactive interface where you can explore all endpoints, inspect request/response schemas, and send requests directly.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/forecast` | Per-destination demand forecasting |
| `POST` | `/optimize` | Multi-period min-cost flow optimization |
| `POST` | `/plan` | Full pipeline: forecast → optimize in one call |

Optional `holding_cost` may be supplied on each destination in `/optimize` and
`/plan`. When present, the multi-period optimizer includes inventory holding cost
in the objective, and responses expose `transportation_cost` and `holding_cost`
alongside `total_cost`.

## Examples

### `/health`

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### `/plan`

```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "demand_history": [
      {"date":"2026-06-01","destination_id":"D1","demand":100},
      {"date":"2026-06-02","destination_id":"D1","demand":105},
      {"date":"2026-06-03","destination_id":"D1","demand":110},
      {"date":"2026-06-04","destination_id":"D1","demand":115},
      {"date":"2026-06-05","destination_id":"D1","demand":120},
      {"date":"2026-06-06","destination_id":"D1","demand":125},
      {"date":"2026-06-07","destination_id":"D1","demand":130},
      {"date":"2026-06-08","destination_id":"D1","demand":135},
      {"date":"2026-06-09","destination_id":"D1","demand":140},
      {"date":"2026-06-10","destination_id":"D1","demand":145},
      {"date":"2026-06-01","destination_id":"D2","demand":50},
      {"date":"2026-06-02","destination_id":"D2","demand":55},
      {"date":"2026-06-03","destination_id":"D2","demand":60},
      {"date":"2026-06-04","destination_id":"D2","demand":65},
      {"date":"2026-06-05","destination_id":"D2","demand":70},
      {"date":"2026-06-06","destination_id":"D2","demand":75},
      {"date":"2026-06-07","destination_id":"D2","demand":80},
      {"date":"2026-06-08","destination_id":"D2","demand":85},
      {"date":"2026-06-09","destination_id":"D2","demand":90},
      {"date":"2026-06-10","destination_id":"D2","demand":95}
    ],
    "origins": [
      {"origin_id":"O1","daily_capacity":200},
      {"origin_id":"O2","daily_capacity":200}
    ],
    "lanes": [
      {"origin_id":"O1","destination_id":"D1","unit_cost":1},
      {"origin_id":"O1","destination_id":"D2","unit_cost":10},
      {"origin_id":"O2","destination_id":"D1","unit_cost":10},
      {"origin_id":"O2","destination_id":"D2","unit_cost":1}
    ],
    "destinations": [
      {"destination_id":"D1", "holding_cost": 1.0},
      {"destination_id":"D2", "holding_cost": 0.5}
    ],
    "model_names": ["naive_forecaster"],
    "train_ratio": 0.8,
    "selection_metric": "wape",
    "max_workers": 1,
    "minimum_history_length": 10,
    "random_seed": 42,
    "model_params": {},
    "initial_inventory": {}
  }'
```

## Example Output

The `/plan` endpoint executes the complete decision pipeline:

```text
Historical Demand
        ↓
Forecasting
        ↓
Demand Forecast
        ↓
Network Optimization
        ↓
Shipment Plan
```

Example response:

```json
{
  "forecast": {
    "successful": [
      {
        "destination_id": "D1",
        "best_model": "naive_forecaster",
        "forecast": 140,
        "wape": 0.034
      },
      {
        "destination_id": "D2",
        "best_model": "naive_forecaster",
        "forecast": 90,
        "wape": 0.053
      }
    ]
  },
  "optimization": {
    "total_cost": 230,
    "transportation_cost": 230,
    "holding_cost": 0,
    "flows": [
      {
        "origin_id": "O1",
        "destination_id": "D1",
        "flow": 140
      },
      {
        "origin_id": "O2",
        "destination_id": "D2",
        "flow": 90
      }
    ]
  }
}
```

In this example:

* Destination **D1** receives a forecast demand of **140 units**
* Destination **D2** receives a forecast demand of **90 units**
* The optimizer routes each destination through its lowest-cost origin
* All demand is satisfied while minimizing transportation cost
* The resulting shipment plan has a total logistics cost of **230**

This demonstrates the complete decision workflow: demand forecasting, model selection, forecast extraction,
and cost-optimal network planning in a single API call.
