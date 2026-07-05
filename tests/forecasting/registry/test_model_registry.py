"""Unit tests for ModelRegistry."""

import pytest

from forecasting.registry.model_registry import ModelRegistry
from forecasting.models.base_forecaster import BaseForecaster
import polars as pl


class DummyForecaster(BaseForecaster):
    """A minimal forecaster for testing the registry."""

    def __init__(self, param_a: int = 1):
        self.param_a = param_a
        self.state = []

    @property
    def name(self) -> str:
        return "dummy"

    def fit(self, df: pl.DataFrame) -> None:
        self.state.append("fitted")

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class AnotherForecaster(BaseForecaster):
    """A second minimal forecaster for testing."""

    @property
    def name(self) -> str:
        return "another"

    def fit(self, df: pl.DataFrame) -> None:
        pass

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class TestModelRegistryRegister:
    def test_register_single_model(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        assert "dummy" in registry

    def test_register_overwrites_duplicate(self):
        registry = ModelRegistry()
        registry.register("model", DummyForecaster)
        registry.register("model", AnotherForecaster)
        instance = registry.create("model")
        assert instance.name == "another"

    def test_register_multiple_models(self):
        registry = ModelRegistry()
        registry.register("a", DummyForecaster)
        registry.register("b", AnotherForecaster)
        assert len(registry) == 2


class TestModelRegistryCreate:
    def test_create_returns_instance(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        instance = registry.create("dummy")
        assert isinstance(instance, BaseForecaster)
        assert instance.name == "dummy"

    def test_create_passes_kwargs(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        instance = registry.create("dummy", param_a=42)
        assert instance.param_a == 42

    def test_create_raises_keyerror_for_unknown_name(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        with pytest.raises(KeyError) as exc_info:
            registry.create("nonexistent")
        # Message should list the requested name and available models
        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "dummy" in msg

    def test_create_returns_independent_instances(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        a = registry.create("dummy")
        b = registry.create("dummy")
        a.state.append("mutated")
        assert b.state == []


class TestModelRegistryListModels:
    def test_list_models_empty(self):
        registry = ModelRegistry()
        assert registry.list_models() == []

    def test_list_models_preserves_insertion_order(self):
        registry = ModelRegistry()
        registry.register("c", DummyForecaster)
        registry.register("a", AnotherForecaster)
        registry.register("b", DummyForecaster)
        assert registry.list_models() == ["c", "a", "b"]

    def test_list_models_after_overwrite_preserves_position(self):
        """Overwriting a key in a dict preserves its position in Python 3.7+."""
        registry = ModelRegistry()
        registry.register("first", DummyForecaster)
        registry.register("second", AnotherForecaster)
        registry.register("first", AnotherForecaster)
        # 'first' keeps its original position
        assert registry.list_models() == ["first", "second"]


class TestModelRegistryContainsAndLen:
    def test_contains_true(self):
        registry = ModelRegistry()
        registry.register("dummy", DummyForecaster)
        assert "dummy" in registry

    def test_contains_false(self):
        registry = ModelRegistry()
        assert "dummy" not in registry

    def test_len_empty(self):
        registry = ModelRegistry()
        assert len(registry) == 0

    def test_len_after_registrations(self):
        registry = ModelRegistry()
        registry.register("a", DummyForecaster)
        registry.register("b", AnotherForecaster)
        assert len(registry) == 2

    def test_len_after_overwrite(self):
        registry = ModelRegistry()
        registry.register("a", DummyForecaster)
        registry.register("a", AnotherForecaster)
        assert len(registry) == 1
