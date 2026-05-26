"""Optimization layer: minimum-cost transportation LP solver."""

from .optimizer import Optimizer, OptimizationResult
from .multi_period_optimizer import MultiPeriodOptimizer
from .multi_period_result import MultiPeriodResult
from .optimizer_interface import OptimizerInterface
from .validation import (
    validate_not_empty,
    validate_columns,
    check_unreachable_destinations,
    check_capacity_feasibility,
    validate_non_negative_costs,
    validate_positive_capacities,
    validate_origins_in_lanes,
)

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "MultiPeriodOptimizer",
    "MultiPeriodResult",
    "OptimizerInterface",
    "validate_not_empty",
    "validate_columns",
    "check_unreachable_destinations",
    "check_capacity_feasibility",
    "validate_non_negative_costs",
    "validate_positive_capacities",
    "validate_origins_in_lanes",
]
