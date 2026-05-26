"""Abstract base class and result type for the simulation layer.

This module defines the contract that all simulation implementations must
satisfy. It provides a ``SimulationResult`` dataclass for returning outputs
and a ``SimulationInterface`` ABC that enforces the ``run`` method signature.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Attributes
    ----------
    metrics : dict[str, float]
        Summary metrics produced by the simulation (e.g. service level,
        total cost, fill rate).
    trace : pl.DataFrame
        Detailed trace of the simulation execution, with one row per
        simulated event or time step.
    """

    metrics: dict[str, float]
    trace: pl.DataFrame


class SimulationInterface(ABC):
    """Abstract interface for the simulation layer.

    Subclasses must implement the ``run`` method to execute a simulation
    against a given scenario. Attempting to instantiate a subclass that
    does not implement all abstract methods will raise ``TypeError``.

    Examples
    --------
    >>> class MySimulation(SimulationInterface):
    ...     def run(self, scenario, **params):
    ...         return SimulationResult(metrics={}, trace=scenario)
    >>> sim = MySimulation()
    """

    @abstractmethod
    def run(self, scenario: pl.DataFrame, **params: object) -> SimulationResult:
        """Execute a simulation run on the given scenario.

        Parameters
        ----------
        scenario : pl.DataFrame
            Input scenario data describing the conditions to simulate
            (e.g. demand patterns, network topology, disruption events).
        **params : object
            Additional keyword arguments controlling simulation behaviour
            (e.g. number of replications, random seed, time horizon).

        Returns
        -------
        SimulationResult
            A result object containing summary metrics and a detailed
            execution trace.

        Raises
        ------
        ValueError
            If the scenario DataFrame is missing required columns or
            contains invalid data for the simulation.
        """
        ...
