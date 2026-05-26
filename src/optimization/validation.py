"""Shared validation functions for transportation optimization.

This module provides reusable input validation logic used by both the
single-period :class:`Optimizer` and the :class:`MultiPeriodOptimizer`.
Each function raises ``ValueError`` with a descriptive message when
validation fails.
"""

import polars as pl


def validate_not_empty(**named_dfs: pl.DataFrame) -> None:
    """Raise ValueError if any named DataFrame is empty.

    Parameters
    ----------
    **named_dfs : pl.DataFrame
        Keyword arguments mapping descriptive names to DataFrames.
        The name is used in the error message when a DataFrame is empty.

    Raises
    ------
    ValueError
        If any provided DataFrame has zero rows.

    Examples
    --------
    >>> validate_not_empty(demand=demand_df, origins=origins_df)
    """
    for name, df in named_dfs.items():
        if df.is_empty():
            raise ValueError(name)


def validate_columns(
    df: pl.DataFrame,
    required: set[str],
    df_name: str,
    *,
    message_template: str | None = None,
) -> None:
    """Raise ValueError if a DataFrame is missing required columns.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate.
    required : set[str]
        Set of column names that must be present.
    df_name : str
        Descriptive name used in the error message.
    message_template : str | None, optional
        Custom error message template with ``{df_name}``, ``{missing}``, and
        ``{required}`` placeholders.  When ``None`` (default), uses the
        standard format.

    Raises
    ------
    ValueError
        If any required columns are absent from the DataFrame.
    """
    missing = required - set(df.columns)
    if missing:
        if message_template is not None:
            raise ValueError(
                message_template.format(
                    df_name=df_name,
                    missing=sorted(missing),
                    required=sorted(required),
                )
            )
        raise ValueError(
            f"{df_name} missing columns {sorted(missing)}. "
            f"Expected: {sorted(required)}"
        )


def check_unreachable_destinations(
    demand_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    *,
    demand_col: str = "destination_id",
    lanes_col: str = "destination_id",
    message_template: str = "Unreachable destinations (no lane available): {unreachable}",
) -> None:
    """Raise ValueError if demanded destinations have no serving lane.

    Parameters
    ----------
    demand_df : pl.DataFrame
        DataFrame containing destination demand data.
    lanes_df : pl.DataFrame
        DataFrame containing lane definitions.
    demand_col : str, optional
        Column name for destination IDs in the demand DataFrame
        (default ``"destination_id"``).
    lanes_col : str, optional
        Column name for destination IDs in the lanes DataFrame
        (default ``"destination_id"``).
    message_template : str, optional
        Error message template with ``{unreachable}`` placeholder
        (default uses single-period optimizer format).

    Raises
    ------
    ValueError
        If any demanded destination has no lane serving it.
    """
    demanded_ids = set(demand_df[demand_col].unique().to_list())
    lane_dest_ids = set(lanes_df[lanes_col].unique().to_list())
    unreachable = sorted(demanded_ids - lane_dest_ids)
    if unreachable:
        raise ValueError(message_template.format(unreachable=unreachable))


def check_capacity_feasibility(
    total_demand: float,
    total_capacity: float,
    *,
    message_template: str = (
        "Insufficient total daily_capacity. "
        "Total demand: {total_demand}, "
        "total daily_capacity: {total_capacity}, "
        "shortfall: {shortfall}"
    ),
) -> None:
    """Raise ValueError if total capacity is less than total demand.

    Parameters
    ----------
    total_demand : float
        Sum of all demand values.
    total_capacity : float
        Sum of all available capacity.
    message_template : str, optional
        Error message template with ``{total_demand}``, ``{total_capacity}``,
        and ``{shortfall}`` placeholders.

    Raises
    ------
    ValueError
        If total capacity is strictly less than total demand.
    """
    if total_capacity < total_demand:
        shortfall = total_demand - total_capacity
        raise ValueError(
            message_template.format(
                total_demand=total_demand,
                total_capacity=total_capacity,
                shortfall=shortfall,
            )
        )


def validate_non_negative_costs(
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame | None = None,
) -> None:
    """Raise ValueError if any cost column contains negative values.

    Checks ``unit_cost`` in the lanes DataFrame and, if provided,
    ``holding_cost`` in the destinations DataFrame.

    Parameters
    ----------
    lanes_df : pl.DataFrame
        DataFrame with an optional ``unit_cost`` column.
    destinations_df : pl.DataFrame | None, optional
        DataFrame with an optional ``holding_cost`` column (default ``None``).

    Raises
    ------
    ValueError
        If any ``unit_cost`` or ``holding_cost`` value is negative.
    """
    if "unit_cost" in lanes_df.columns:
        negative_costs = lanes_df.filter(pl.col("unit_cost") < 0)
        if not negative_costs.is_empty():
            invalid_rows = negative_costs.select(
                "origin_id", "destination_id", "unit_cost"
            ).to_dicts()
            raise ValueError(
                f"Negative unit_cost values found: {invalid_rows}"
            )

    if destinations_df is not None and "holding_cost" in destinations_df.columns:
        negative_holding = destinations_df.filter(pl.col("holding_cost") < 0)
        if not negative_holding.is_empty():
            invalid_rows = negative_holding.select(
                "destination_id", "holding_cost"
            ).to_dicts()
            raise ValueError(
                f"Negative holding_cost values found: {invalid_rows}"
            )


def validate_positive_capacities(origins_df: pl.DataFrame) -> None:
    """Raise ValueError if any origin has non-positive daily_capacity.

    Parameters
    ----------
    origins_df : pl.DataFrame
        DataFrame with a ``daily_capacity`` column.

    Raises
    ------
    ValueError
        If any ``daily_capacity`` value is zero or negative.
    """
    if "daily_capacity" in origins_df.columns:
        invalid = origins_df.filter(pl.col("daily_capacity") <= 0)
        if not invalid.is_empty():
            invalid_rows = invalid.select(
                "origin_id", "daily_capacity"
            ).to_dicts()
            raise ValueError(
                f"Non-positive daily_capacity values found: {invalid_rows}"
            )


def validate_origins_in_lanes(
    origins_df: pl.DataFrame, lanes_df: pl.DataFrame
) -> None:
    """Raise ValueError if lanes reference origins not in origins_df.

    Parameters
    ----------
    origins_df : pl.DataFrame
        DataFrame with an ``origin_id`` column listing valid origins.
    lanes_df : pl.DataFrame
        DataFrame with an ``origin_id`` column referencing origins.

    Raises
    ------
    ValueError
        If any origin referenced in lanes is not present in origins_df.
    """
    origin_ids = set(origins_df["origin_id"].to_list())
    lane_origin_ids = set(lanes_df["origin_id"].to_list())
    missing_origins = sorted(lane_origin_ids - origin_ids)
    if missing_origins:
        raise ValueError(
            f"Origins referenced in lanes but missing from origins_df: "
            f"{missing_origins}"
        )
