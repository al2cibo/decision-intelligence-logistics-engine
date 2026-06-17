
### Project Structure Overview

The project is organized into four main areas: a core Python library (`src/`), batch
experiment infrastructure (`experiments/`), executable entry points (`scripts/`), and an
HTTP API (`src/api/`). For a full technical reference — module maps, LP formulation,
metric formulas — see the `.claude/` directory.

---

### 1. Core Library (`src/`)

`src/` is an installable Python package importable as top-level modules (`data`,
`forecasting`, `optimization`, `api`, `utils`). It contains no executable scripts.

#### Data Layer (`src/data/`)

Handles ingestion, validation, and processing of logistics data:

- `Reader` — loads 4 parquet files (`demand_history`, `origins`, `destinations`, `lanes`)
  from a configurable directory into an `InputData` dataclass.
- `DataProcessor` — orchestrates per-table processors (dedup + sort + validation).
- Individual processors: `DemandProcessor`, `LanesProcessor`, `DestinationsProcessor`,
  `OriginsProcessor` — each calls shared validation functions from
  `processing/validation.py` (`validate_non_empty`, `validate_no_nulls`,
  `validate_columns`).
- `generate_synthetic_logistics_data()` — creates realistic synthetic logistics data
  with weekly seasonality, trend, promotions, and noise.

#### Forecasting Layer (`src/forecasting/`)

Per-destination multi-model demand forecasting with automatic model selection:

- `config.py` — **stdlib-only**. Owns `PerDestinationConfig` (the full set of
  per-destination runtime parameters), `KNOWN_METRICS`, and
  `_validate_per_destination_config`. No dependency on any other project module.
- `models/` — 5 concrete forecasters, all subclassing `BaseForecaster`:
  `NaiveForecaster`, `SeasonalForecaster`, `RollingWindowForecaster`,
  `ETSForecaster` (Holt-Winters via statsmodels), `SARIMAXForecaster`.
- `registry/` — `ModelRegistry` (name → factory callable) and `create_default_registry()`
  which registers all 5 models.
- `evaluation/` — `Evaluator` (computes MAE, MSE, RMSE, MAPE, WAPE), `ModelSelector`
  (picks best model by metric from a list of `(name, metrics)` tuples),
  `PerDestinationModelSelector` (wraps `ModelSelector` with per-destination validation,
  returns frozen `SelectionResult`).
- `results/` — frozen dataclasses `ForecastResult` (per-model outcome for one
  destination) and `TimePeriod`.
- `pipeline/` — the core orchestration:
  - `PerDestinationForecastingPipeline` — fits/evaluates all models per destination
    (joblib parallelism, per-destination fault tolerance, deterministic seeding via
    `abs(hash(destination_id)) ^ random_seed`). Produces `AggregatedForecastingResult`
    with `.successful` / `.failed` and `.export_forecasts()` (extracts the selected
    model's test-period forecasts as `[destination_id, date, demand]` for the optimizer).
  - `create_forecasting_pipeline(config: PerDestinationConfig)` — factory that
    validates model names against the registry and builds the pipeline.
  - `ForecastingPipelineProtocol` — `@runtime_checkable Protocol` for structural typing.

#### Optimization Layer (`src/optimization/`)

Minimum-cost multi-period transportation LP using OR-Tools (GLOP/CBC):

- `MultiPeriodOptimizer.solve(...)` — builds and solves the LP. Decision variables:
  `flow[origin, destination, period]` and `inv[destination, period]`. Constraints:
  inventory balance per destination/period, capacity per origin/period. Optional holding
  costs added to objective if `destinations_df` has a `holding_cost` column.
- Pre-solve validation: structural checks → cost/capacity checks → variable-count guard
  (`MAX_VARIABLES = 1_000_000`) → feasibility checks.
- Returns `MultiPeriodResult(flows, inventory, total_cost, transportation_cost,
  holding_cost)`.
- Implementation is split into a `multi_period/` package:
  `optimizer.py`, `validation.py`, `preprocessing.py`, `model_builder.py`,
  `solution_extractor.py`, `result.py`.
- Shared `validation.py` at `src/optimization/` level — reused across modules.

#### Configuration (`src/utils/`)

- `config.py` — **stdlib-only**. Owns application-level config: `DataConfig`,
  `ForecastingConfig` (top-level metric/ratio knobs), `Config` (root dataclass),
  `load_config(project_root, config_path)`. Does not know about `PerDestinationConfig`
  or any forecasting internals.
- `system_paths.py` — `get_project_root()`.

---

### 2. Experiments (`experiments/`)

Batch infrastructure for running named experiment configs against a versioned dataset,
saving results for analysis:

- `experiment_config.py` — `ExperimentConfig` dataclass + `load_experiment_config()`.
  Imports `PerDestinationConfig` from `forecasting.config`.
- `run_experiment.py` — runs one experiment config end-to-end (forecast → optimize)
  and saves 5 artifacts: `metrics.json`, `forecasts.parquet`, `flows.parquet`,
  `inventory.parquet`, `config.yaml`.
- `run_all.py` — runs all 3 experiment configs and prints a summary table.
- `configs/` — 3 YAML experiment definitions:
  `baseline_naive.yaml`, `baseline_global_ets.yaml`, `model_selection.yaml`.
- `datasets/synthetic_v1/` — versioned input data (4 parquet files, committed to git).
- `results/` — gitignored; populated at runtime.

Usage:
```bash
PYTHONPATH=src python experiments/run_experiment.py experiments/configs/model_selection.yaml
PYTHONPATH=src python experiments/run_all.py
```

---

### 3. Scripts (`scripts/`)

Executable entry points that orchestrate the core library. No core logic implemented here.

- `example_end_to_end_pipeline.py` — CLI demo: ingest → forecast → optimize, using
  `data/synthetic2/`. Constructs `PerDestinationConfig` from `forecasting.config`
  directly (not from YAML).
- `generate_data.py` — regenerates the `data/synthetic2/` parquet files.
- `serve.py` — launches the FastAPI app via uvicorn with reload.

---

### 4. API Layer (`src/api/`)

FastAPI service exposing the pipeline over HTTP. Runs as a persistent process; no
filesystem dependency (all I/O is JSON in, JSON out).

- `GET /health` — liveness check.
- `POST /forecast` — per-destination forecasting on historical demand data.
- `POST /optimize` — multi-period transportation LP given a demand time series.
- `POST /plan` — full pipeline: forecast → extract demand → optimize, in one call.

Backed by `LogisticsAPI` (the only concrete `APIInterface`). Pydantic request/response
models in `models.py`. All `ValueError` → HTTP 422; all other exceptions → HTTP 500.

---

### End-to-end data flow

```
Parquet files (data/ or experiments/datasets/)
  → Reader → InputData
  → DataProcessor → clean InputData
  → PerDestinationForecastingPipeline.run() → AggregatedForecastingResult
  → .export_forecasts() → [destination_id, date, demand] DataFrame
  → MultiPeriodOptimizer.solve() → MultiPeriodResult (flows, inventory, total_cost)
```
