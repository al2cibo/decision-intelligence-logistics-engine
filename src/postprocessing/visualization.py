"""Visualization utilities for logistics forecasting results."""

import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path


class VisualizationEngine:
    """Engine for producing time-series visualizations of forecast results.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing time-series data with at minimum a
        ``destination_id`` column, a ``date`` column, and one or more
        value columns for actuals and predictions.
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df

    def produce_timeseries_plots(
        self,
        target_destination: str,
        actuals_col_name: str,
        predicted_col_name: str,
        save_fig_location: Path = None,
        show: bool = False,
    ):
        """Produce a time-series plot comparing actuals to predictions.

        Parameters
        ----------
        target_destination : str
            The destination identifier to filter the data on.
        actuals_col_name : str
            Column name containing actual demand values.
        predicted_col_name : str
            Column name containing predicted demand values.
        save_fig_location : Path, optional
            Directory path where the figure will be saved. If ``None``,
            the figure is not saved to disk.
        show : bool, optional
            Whether to display the plot interactively. Defaults to
            ``False`` so that automated pipelines do not block.

        Raises
        ------
        ValueError
            If no data is found for the specified destination.
        """
        destination_df = self.df.filter(
            pl.col("destination_id") == target_destination
        ).sort("date")

        if destination_df.height == 0:
            raise ValueError(f"No data found for destination {target_destination}")

        dates = destination_df["date"].to_numpy()
        actuals = destination_df[actuals_col_name].to_numpy()
        preds = destination_df[predicted_col_name].to_numpy()

        fig, plot_ax = plt.subplots(figsize=(12, 5))

        plot_ax.plot(dates, actuals, label="Actuals")
        plot_ax.plot(dates, preds, color="red", label=predicted_col_name)

        plot_ax.set_xlabel("Date")
        plot_ax.set_ylabel("Demand")
        plot_ax.set_title(
            f"Actual vs Predicted ({actuals_col_name}) — Destination {target_destination}"
        )

        plot_ax.legend()

        if save_fig_location:
            save_fig_location.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_fig_location
                / f"{actuals_col_name}_{predicted_col_name}_{target_destination}.png"
            )

        if show:
            plt.show()
        else:
            plt.close(fig)
