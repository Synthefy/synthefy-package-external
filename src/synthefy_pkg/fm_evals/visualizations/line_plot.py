"""
Functions for creating line plots for forecast evaluation.
"""

import os
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)


def plot_single_sample(
    eval_sample: SingleEvalSample,
    forecasts: Union[SingleSampleForecast, List[SingleSampleForecast]],
    figsize: tuple[int, int] = (10, 6),
    history_color: str = "blue",
    target_color: str = "green",
    forecast_colors: Optional[List[str]] = None,
    alpha: float = 0.7,
    input_ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot a single sample from evaluation and one or more forecasts.

    Args:
        eval_sample: SingleEvalSample instance
        forecasts: SingleSampleForecast or list of SingleSampleForecast
        figsize: Size of the figure (ignored if ax is provided)
        history_color: Color for history line
        target_color: Color for target line
        forecast_colors: List of colors for forecast lines (if None, uses default matplotlib cycle)
        alpha: Transparency for lines
        ax: Optional matplotlib Axes to plot on. If None, a new figure and axis are created.

    Returns:
        matplotlib Axes object with the plot

    Raises:
        ValueError: If the sample_id does not match across all inputs
    """
    # Ensure forecasts is a list
    if not isinstance(forecasts, list):
        forecasts = [forecasts]

    # Check that all sample_ids match
    eval_id = eval_sample.sample_id
    for forecast in forecasts:
        if not np.array_equal(forecast.sample_id, eval_id):
            raise ValueError(
                f"Sample ID mismatch: {forecast.sample_id} != {eval_id}"
            )

    # Create figure and axis if not provided
    if input_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = input_ax

    ax = cast(Axes, ax)

    # Plot history
    ax.plot(
        eval_sample.history_timestamps,
        eval_sample.history_values,
        color=history_color,
        alpha=alpha,
        label="History",
    )

    # Get the last history point for connecting target and forecasts
    last_history_timestamp = eval_sample.history_timestamps[-1]
    last_history_value = eval_sample.history_values[-1]

    # Plot target with connection to history
    continuous_target_timestamps = np.concatenate(
        [[last_history_timestamp], eval_sample.target_timestamps]
    )
    continuous_target_values = np.concatenate(
        [[last_history_value], eval_sample.target_values]
    )
    ax.plot(
        continuous_target_timestamps,
        continuous_target_values,
        color=target_color,
        alpha=alpha,
        label="Target",
    )

    # Plot forecasts with connection to history
    if not isinstance(forecast_colors, list) or not forecast_colors:
        # Use distinct colors for better visibility
        distinct_colors = [
            "red",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        forecast_colors = distinct_colors

    for i, forecast in enumerate(forecasts):
        color = forecast_colors[i % len(forecast_colors)]
        label = (
            f"Forecast ({getattr(forecast, 'model_name', 'model')})"
            if hasattr(forecast, "model_name")
            else f"Forecast {i + 1}"
        )

        # Create continuous line from last history point to forecast
        # Combine last history point with forecast timestamps and values
        continuous_timestamps = np.concatenate(
            [[last_history_timestamp], forecast.timestamps]
        )
        continuous_values = np.concatenate(
            [[last_history_value], forecast.values]
        )

        ax.plot(
            continuous_timestamps,
            continuous_values,
            color=color,
            alpha=alpha,
            label=label,
        )

    # Add labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Convert sample_id to string and split at underscores
    sample_id_str = str(eval_id)
    title_parts = sample_id_str.split("_")
    title_text = "\n".join(title_parts)

    # Add metrics information if available
    metrics_parts = []
    for i, forecast in enumerate(forecasts):
        if hasattr(forecast, "metrics") and forecast.metrics is not None:
            mae = getattr(forecast.metrics, "mae", None)
            mape = getattr(forecast.metrics, "mape", None)

            forecast_label = (
                getattr(forecast, "model_name", f"Forecast {i + 1}")
                if hasattr(forecast, "model_name")
                else f"Forecast {i + 1}"
            )

            metrics_part = f"{forecast_label}: "
            if mae is not None:
                metrics_part += f"MAE={mae:.4f}"
            if mape is not None:
                metrics_part += f", MAPE={mape:.4f}%"

            if mae is not None or mape is not None:
                metrics_parts.append(metrics_part)

    # Combine title with metrics
    if metrics_parts:
        # Insert a newline after every 2 metric parts, otherwise use pipe
        grouped = [
            " | ".join(metrics_parts[i : i + 2])
            for i in range(0, len(metrics_parts), 2)
        ]
        title_text += "\n" + "\n".join(grouped)

    ax.set_title(title_text, fontsize=8)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    return ax


def plot_batch_forecasts(
    eval_batch: EvalBatchFormat,
    forecast_batches,
    pdf_path: str,
    figsize: tuple[int, int] = (10, 6),
    history_color: str = "grey",
    target_color: str = "black",
    forecast_colors: Optional[List[str]] = None,
    alpha: float = 0.7,
):
    """Plot batch forecasts to PDF, handling multivariate batches with covariates.

    Only plots samples where forecast=True (target series), skipping covariates.
    Handles cases where eval_batch has more correlates than forecast_batch
    (e.g., eval_batch has target + covariates, forecast_batch only has target forecasts).
    """
    # Check batch size matches across all forecast batches
    for forecast_batch in forecast_batches:
        if eval_batch.batch_size != forecast_batch.batch_size:
            logger.error(
                f"Batch size mismatch: {eval_batch.batch_size} != {forecast_batch.batch_size}"
            )
            return

    try:
        with PdfPages(pdf_path) as pdf:
            for b_idx in range(eval_batch.batch_size):
                # Find forecast=True samples in eval_batch and match with forecast_batch
                forecast_idx = 0  # Index into forecast_batch correlates
                for c_idx in range(eval_batch.num_correlates):
                    eval_sample = eval_batch[b_idx, c_idx]
                    if not eval_sample.forecast:
                        # Skip covariates (metadata samples)
                        continue

                    # Get corresponding forecast from each forecast batch
                    forecast_samples = []
                    for forecast_batch in forecast_batches:
                        if forecast_idx < forecast_batch.num_correlates:
                            forecast_samples.append(forecast_batch[b_idx, forecast_idx])
                        else:
                            logger.warning(
                                f"No forecast available for batch {b_idx}, correlate {c_idx}"
                            )

                    if not forecast_samples:
                        continue

                    fig, ax = plt.subplots(figsize=figsize)
                    ax = cast(Axes, ax)
                    plot_single_sample(
                        eval_sample,
                        forecast_samples,
                        figsize=figsize,
                        history_color=history_color,
                        target_color=target_color,
                        forecast_colors=forecast_colors,
                        alpha=alpha,
                        input_ax=ax,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    forecast_idx += 1  # Move to next forecast correlate

        plt.close("all")
    except Exception as e:
        logger.error(f"Failed to plot or save PDF {pdf_path}: {e}")
        plt.close("all")
        return


def lineplot_from_df(
    df: pd.DataFrame,
    column_names: list[str],
    ax: Optional[Axes] = None,
    history_color: str = "blue",
    target_color: str = "green",
    forecast_colors: Optional[list[str]] = None,
    alpha: float = 0.7,
) -> Axes:
    """
    Plot ground truth and forecast from a DataFrame as produced by join_as_dfs.

    Args:
        df: DataFrame with columns [timestamps, <col>, <col>_forecast, split]
        column_names: List of column names to plot (ground truth columns)
        ax: Optional matplotlib Axes to plot on. If None, a new figure and axis are created.
        history_color: Color for history line
        target_color: Color for target line
        forecast_colors: List of colors for forecast lines (if None, uses default matplotlib cycle)
        alpha: Transparency for lines

    Returns:
        matplotlib Axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]
    ax = cast(Axes, ax)

    if not isinstance(forecast_colors, list) or not forecast_colors:
        # Use distinct colors for better visibility
        distinct_colors = [
            "red",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        forecast_colors = distinct_colors

    for idx, col in enumerate(column_names):
        # Plot history
        history_mask = df["split"] == "history"
        target_mask = df["split"] == "target"
        ax.plot(
            df.loc[history_mask, "timestamps"],
            df.loc[history_mask, col],
            color=history_color,
            alpha=alpha,
            label=f"{col} History" if idx == 0 else None,
        )
        # Plot target
        ax.plot(
            df.loc[target_mask, "timestamps"],
            df.loc[target_mask, col],
            color=target_color,
            alpha=alpha,
            label=f"{col} Target" if idx == 0 else None,
        )
        # Plot forecast (only for target region)
        forecast_col = f"{col}_forecast"
        if forecast_col in df.columns:
            color = forecast_colors[idx % len(forecast_colors)]
            ax.plot(
                df.loc[target_mask, "timestamps"],
                df.loc[target_mask, forecast_col],
                color=color,
                alpha=alpha,
                linestyle="--",
                label=f"{col} Forecast" if idx == 0 else None,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Line Plot from DataFrame")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    return ax
