.PHONY: test run lint format clean

test:
	python -m pytest tests/ -v

run:
	python scripts/example_end_to_end_pipeline.py

lint:
	python -m black --check src/ tests/ scripts/

format:
	python -m black src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
