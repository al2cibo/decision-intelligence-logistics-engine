"""Abstract base class defining the API layer contract.

This module provides the ``APIInterface`` ABC that future API implementations
must satisfy. It declares abstract methods for forecasting and optimization
operations with full type annotations.
"""

from abc import ABC, abstractmethod

import polars as pl

from optimization.optimizer import OptimizationResult


class APIInterface(ABC):
    """Abstract interface for the API layer.

    Subclasses must implement ``forecast`` and ``optimize`` to provide
    concrete API behaviour for the logistics engine.

    Methods
    -------
    forecast(input_data)
        Generate demand forecasts from historical data.
    optimize(demand, origins, lanes)
        Solve a minimum-cost transportation problem.
    """

    @abstractmethod
    def forecast(self, input_data: pl.DataFrame) -> pl.DataFrame:
        """Generate demand forecasts from historical input data.

        Parameters
        ----------
        input_data : pl.DataFrame
            Historical demand data used as input for the forecasting model.

        Returns
        -------
        pl.DataFrame
            Forecasted demand values.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        ...

    @abstractmethod
    def optimize(
        self,
        demand: pl.DataFrame,
        origins: pl.DataFrame,
        lanes: pl.DataFrame,
    ) -> OptimizationResult:
        """Solve a minimum-cost transportation optimization problem.

        Parameters
        ----------
        demand : pl.DataFrame
            Demand data specifying required quantities per destination.
        origins : pl.DataFrame
            Origin (warehouse/supplier) data with capacity constraints.
        lanes : pl.DataFrame
            Available shipping lanes with unit costs connecting origins
            to destinations.

        Returns
        -------
        OptimizationResult
            The optimization solution containing flows and total cost.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        ...
