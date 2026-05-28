
### Project Structure Overview

The project is organized into three main components, each with a distinct role in the system architecture.

### 1. Core Package (`src/`)

The `src/` directory contains the core logic of the system and is structured as an internal Python library.

It includes:
- data processing and pipeline logic  
- forecasting models  
- optimization models  
- utility functions  

This layer is designed to be modular and reusable. It does not contain executable scripts; instead, it exposes functions and classes that can be imported and used by other parts of the system.

#### Data Layer (`src/data/`)

Handles ingestion, validation, and processing of logistics data:
- `Reader` — loads parquet files (demand_history, origins, destinations, lanes) from a configurable path
- `InputData` — dataclass bundling the four DataFrames
- `DataProcessor` — orchestrates per-dataset processors
- `BaseProcessor` — shared validation (non-empty, no nulls, required columns)
- Specialized processors: `DemandProcessor` (dedup + sort), `LanesProcessor`, `DestinationsProcessor`, `OriginsProcessor`
- `generate_synthetic_logistics_data()` — creates realistic synthetic data with weekly seasonality, trend, promotions, and noise

#### Forecasting Layer (`src/forecasting/`)

Demand prediction with multiple models and automatic selection:
- `BaseForecaster` — ABC defining the `fit()`, `predict()`, `name` interface
- Models: NaiveForecaster, SeasonalForecaster, RollingWindowForecaster, ETSForecaster, ARIMAForecaster
- `ForecastingPipeline` — runs all models sequentially with optional train/test split
- `Evaluator` — computes MAE, MSE, RMSE, MAPE, WAPE
- `ModelSelector` — picks best model by configurable metric (default WAPE)
- `ForecastExtractor` — extracts forecast column and aggregates demand per destination

#### Optimization Layer (`src/optimization/`)

Minimum-cost transportation LP using OR-Tools (GLOP/CBC):
- Decision variables: flow per origin-destination lane
- Constraints: demand satisfaction (≥), capacity limits (≤)
- Pre-solve feasibility checks: unreachable destinations, insufficient capacity
- Returns `OptimizationResult` with flows DataFrame and total cost

#### Postprocessing (`src/postprocessing/`)

- `MetricsSummary` — collects per-model metrics, normalizes schema, exports CSV
- `VisualizationEngine` — matplotlib time series plots (actuals vs predicted)

#### Configuration (`src/utils/`)

- YAML-based config with `DataConfig` and `ForecastingConfig` dataclasses
- `get_project_root()` for path resolution

---

### 2. Scripts (`scripts/`)

The `scripts/` directory contains executable entry points used to run specific workflows.

Typical use cases include:
- running the data pipeline  
- generating synthetic data  
- training models  
- executing optimization jobs  

Scripts act as orchestrators: they import and call functions from the core package (`src/`) but do not implement core logic themselves.

The main entry point is `example_end_to_end_pipeline.py`, which executes the full flow:
```
Reader → DataProcessor → ForecastingPipeline → Evaluator → MetricsSummary
  → ModelSelector → ForecastExtractor → Optimizer → Flow decisions
```

---

### 3. API Layer (`src/api/`)

The API layer exposes the system through HTTP endpoints using FastAPI.

It serves as a service interface that allows external users or systems to interact with the core logic. For example, it can expose endpoints for forecasting, optimization, or querying results.

Unlike scripts, the API runs as a persistent service.

---
