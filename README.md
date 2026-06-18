# Decision Intelligence Logistics Engine

An end-to-end decision system for logistics planning that combines demand forecasting, stochastic simulation, and network optimization.

The project is designed to showcase production-oriented applied science and engineering skills at the intersection of:

- Operations Research
- Machine Learning
- Data Engineering
- MLOps
- API-based deployment

## Project Goal

Build a scalable logistics decision engine that can:

1. Generate or ingest historical shipment and demand data
2. Forecast future demand — independently per destination
3. Simulate uncertain logistics scenarios
4. Optimize origin-destination flows under capacity and cost constraints
5. Expose the full pipeline through an API

This repository reflects how real-world planning systems are built: not only with mathematical models, but also with robust data pipelines, modular software design, and deployable services.

---

## Latest Release: v1.0

Validated through:
- 167 automated tests
- Reproducibility checks
- Optimization consistency tests
- Experiment infrastructure verification

See: [docs/reports/V1_0_FUNCTIONALITY_TEST_REPORT.md](docs/reports/V1_0_FUNCTIONALITY_TEST_REPORT.md)

---

## Architecture

Three-layer pipeline: **Data → Forecasting → Optimization**.

- Per-destination local model architecture (each destination independently trained and selected)
- Multi-period min-cost flow optimizer with inventory tracking
- FastAPI serving layer wiring both engines

See [docs/architecture.md](docs/architecture.md) for the full system diagram, LP formulation, and component breakdown.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11+ |
| DataFrames | Polars |
| Optimization | OR-Tools (GLOP, CBC) |
| Statistical Models | statsmodels (ETS, ARIMA) |
| Metrics | scikit-learn |
| Parallelism | joblib |
| Numerics | NumPy |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Testing | pytest, Hypothesis (property-based testing) |
| API | FastAPI, Uvicorn |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/<your-username>/decision-intelligence-logistics-engine.git
cd decision-intelligence-logistics-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### Analysis Environment (notebooks-related dependencies)
pip install -r requirements-analysis.txt

# Run the full pipeline demo
python scripts/example_end_to_end_pipeline.py

# Run tests
python -m pytest tests/ -v
```

---

## API

Start the server:

```bash
PYTHONPATH=src uvicorn api.app:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/forecast` | Per-destination demand forecasting |
| `POST` | `/optimize` | Multi-period min-cost flow optimization |
| `POST` | `/plan` | Full pipeline: forecast → optimize in one call |

See [docs/api.md](docs/api.md) for endpoint details, request schemas, and example requests/responses.

---

## Testing

```bash
python -m pytest tests/ -v
# 167 passed
```

Key correctness properties verified:
- Data isolation between destinations
- Temporal split correctness (no future leakage)
- Row-order independence
- Model selection minimality with tiebreaking
- Fault tolerance completeness
- Determinism across executions
- Pipeline protocol conformance

---

## Planned Features

- [x] FastAPI endpoints for end-to-end execution (`/forecast`, `/optimize`, `/plan`)
- [ ] Stochastic simulation layer implementation (interface defined via `SimulationInterface`)
- [ ] MLflow experiment tracking
- [ ] Docker support
- [ ] ML model integration (LightGBM, XGBoost, Prophet)
- [ ] Hierarchical forecasting
- [ ] Performance benchmarking
- [ ] Visualization config support (show/save via YAML)

---

## Author

**Christian Piermarini**
Applied Scientist / Operations Research / Machine Learning
