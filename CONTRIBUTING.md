# Contributing to Decision Intelligence Logistics Engine

Thank you for your interest in contributing. This document covers everything you need to get started: setting up your environment, running tests, formatting code, and submitting a pull request.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Fork and Clone](#fork-and-clone)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Running Tests](#running-tests)
6. [Code Formatting](#code-formatting)
7. [Branching and Commits](#branching-and-commits)
8. [Opening a Pull Request](#opening-a-pull-request)
9. [Reporting Issues](#reporting-issues)

---

## Prerequisites

- Python **3.11 or higher**
- `git`
- A virtual environment tool (`venv` is fine)

---

## Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/decision-intelligence-logistics-engine.git
cd decision-intelligence-logistics-engine
```

3. Add the upstream remote so you can stay in sync:

```bash
git remote add upstream https://github.com/chripiermarini/decision-intelligence-logistics-engine.git
```

---

## Environment Setup

Create and activate a virtual environment, then install all dependencies (core + dev):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pip install -r requirements.txt
```

The `.[dev]` extras install `pytest`, `hypothesis`, and `black`. The `requirements.txt` adds the API dependencies (`fastapi`, `uvicorn`) and any other runtime packages.

---

## Project Structure

```
src/
  api/            # FastAPI app and LogisticsAPI facade
  forecasting/    # Per-destination forecasting pipeline and models
  optimization/   # LP formulation and OR-Tools solver
  simulation/     # Stochastic scenario generation
  utils/          # Shared helpers
experiments/      # 2x2 factorial experiment infra: configs, runners (run_experiment.py,
                  # run_all.py), naive baseline heuristic, realized-cost evaluator,
                  # versioned datasets (datasets/synthetic_v1, synthetic_v2)
tests/
  data/           # Tests for data ingestion/processing
  forecasting/    # Unit + integration tests for the forecasting pipeline
                  # (mirrors src/forecasting/'s module layout: models/, evaluation/,
                  # pipeline/, registry/, results/)
  optimization/   # Tests for the optimization module
  experiments/    # Tests for the experiments module (naive heuristic, etc.)
  utils/          # Tests for shared utilities
scripts/          # Runnable end-to-end examples
configs/          # YAML configuration files
```

---

## Running Tests

All tests live under `tests/` and are run with `pytest`. The project uses both standard unit tests and property-based tests via `hypothesis`.

**Run the full suite:**

```bash
python -m pytest tests/ -v
```

Or via the Makefile shortcut:

```bash
make test
```

**Run a specific module:**

```bash
python -m pytest tests/optimization/ -v
```

**What to check before opening a PR:**

- All existing tests pass.
- New behaviour is covered by at least one test.
- Property-based tests (if any) are not flaky — run them a couple of times if unsure.

---

## Running Experiments

The `experiments/` module runs the 2×2 factorial design (naive vs. DILE forecasting ×
naive vs. DILE optimization) against the versioned `experiments/datasets/synthetic_v2/`
dataset. `PYTHONPATH=src` is required since these scripts import from `src/` directly:

```bash
# Run all four scenarios (B00, B01, B10, B11) and print a consolidated summary
PYTHONPATH=src python experiments/run_all.py

# Run a single scenario
PYTHONPATH=src python experiments/run_experiment.py experiments/configs/B11_dile_forecast_dile_opt.yaml
```

Do not overwrite `experiments/datasets/synthetic_v1/` or `synthetic_v2/` — they are
versioned fixtures for reproducible results; regenerate only via
`scripts/generate_data.py` if you specifically intend to change the paper dataset.

---

## Code Formatting

This project uses [Black](https://black.readthedocs.io/) with a line length of **88** and a target of Python 3.11 (configured in `pyproject.toml`).

**Format all source files before committing:**

```bash
python -m black src/ tests/ scripts/
```

Or via the Makefile:

```bash
make format
```

**Check formatting without modifying files** (useful in CI or for a quick pre-commit sanity check):

```bash
python -m black --check src/ tests/ scripts/
```

Or:

```bash
make lint
```

> Pull requests that fail the `black --check` step will be asked to reformat before merging. Running `make format` locally is the easiest way to avoid this.

---

## Branching and Commits

**Branch naming:**

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-description>` | `feat/add-exponential-smoothing` |
| Bug fix | `fix/<short-description>` | `fix/optimizer-infeasible-edge-case` |
| Refactor | `refactor/<short-description>` | `refactor/decouple-optimize-from-forecast-config` |
| Docs | `docs/<short-description>` | `docs/update-api-readme` |
| Tests | `test/<short-description>` | `test/hypothesis-coverage-forecaster` |

**Commits:**

- Use the [Conventional Commits](https://www.conventionalcommits.org/) style: `type(scope): short description`.
- Keep commits focused — one logical change per commit.
- Write the subject line in the imperative mood: *"add exponential smoothing"*, not *"added"* or *"adding"*.

Examples:

```
feat(forecasting): add exponential smoothing model
fix(optimization): handle infeasible LP when demand exceeds capacity
refactor(api): make PerDestinationConfig optional for optimize-only usage
test(forecasting): add hypothesis tests for pipeline edge cases
```

---

## Opening a Pull Request

1. Push your branch to your fork:

```bash
git push origin feat/your-feature
```

2. Open a PR against `chripiermarini:main` on GitHub.

3. In the PR description, include:
   - What problem it solves (reference the issue with `Closes #N` if applicable).
   - A brief summary of the approach.
   - Any trade-offs or alternatives you considered.

4. Before requesting review, confirm locally:

```bash
make format   # or: python -m black src/ tests/ scripts/
make lint     # should report "All done! ✨"
make test     # all tests must pass
```

Pull requests are automatically checked through GitHub Actions. 
PRs must pass all formatting and test checks before they can be merged.
---

## Reporting Issues

Open a GitHub Issue and include:

- A short, descriptive title.
- Steps to reproduce (for bugs) or a clear description of the desired behaviour (for features).
- Any relevant tracebacks, logs, or code snippets.
- Your Python version (`python --version`) and OS.

For questions or discussion, prefer opening an Issue over a direct message so the conversation stays visible to all contributors.
