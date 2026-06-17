"""Optimization layer: multi-period transportation LP solver."""

from .optimizer import MultiPeriodOptimizer
from .result import MultiPeriodResult
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
    "MultiPeriodOptimizer",
    "MultiPeriodResult",
    "validate_not_empty",
    "validate_columns",
    "check_unreachable_destinations",
    "check_capacity_feasibility",
    "validate_non_negative_costs",
    "validate_positive_capacities",
    "validate_origins_in_lanes",
]
