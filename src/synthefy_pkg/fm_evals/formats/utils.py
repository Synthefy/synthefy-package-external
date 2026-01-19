"""Utility helpers for fm_evals formats sub-package."""

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
)


def join_as_dfs(
    eval_batch: EvalBatchFormat,
    forecast_output: ForecastOutputFormat,
    split: str | None = None,
    with_metrics: bool = False,
) -> List[pd.DataFrame]:
    """Join an ``EvalBatchFormat`` with a ``ForecastOutputFormat`` and return a list of DataFrames.

    For each batch, concatenate history + target for actuals, and [NaN]*len(history) + forecast for forecast column (if forecasted).
    Only add forecast columns for correlates that were forecasted. Use original column names for actuals.
    Assumes all correlates have the same timestamps.

    Parameters
    ----------
    eval_batch : EvalBatchFormat
        The evaluation batch format
    forecast_output : ForecastOutputFormat
        The forecast output format
    split : str | None, optional
        Split information, by default None
    with_metrics : bool, optional
        Whether to include metrics information in the output DataFrames, by default False.
        When True, includes sample-level metrics (MAE, MAPE) for each correlate and batch-level metrics if available.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames, one for each batch
    """
    if not isinstance(eval_batch, EvalBatchFormat):
        raise TypeError("eval_batch must be an EvalBatchFormat instance")
    if not isinstance(forecast_output, ForecastOutputFormat):
        raise TypeError(
            "forecast_output must be a ForecastOutputFormat instance"
        )
    if (
        eval_batch.batch_size != forecast_output.batch_size
        or eval_batch.num_correlates != forecast_output.num_correlates
    ):
        raise ValueError(
            "eval_batch and forecast_output must have the same shape (B, NC)"
        )

    result_dfs: List[pd.DataFrame] = []

    for b in range(eval_batch.batch_size):
        columns = {}
        split_col = None
        timestamps_ref = None
        metrics_data = {} if with_metrics else None

        for nc in range(eval_batch.num_correlates):
            eval_sample = eval_batch[b, nc]  # type: ignore[index]
            forecast_sample = forecast_output[b, nc]  # type: ignore[index]

            eval_df = eval_sample.to_df()  # type: ignore[attr-defined]
            # Split into history and target
            if "split" in eval_df.columns:
                history_mask = eval_df["split"] == "history"
                target_mask = eval_df["split"] == "target"
                history = eval_df[history_mask]
                target = eval_df[target_mask]
            else:
                # If no split, treat all as target
                history = eval_df.iloc[0:0]
                target = eval_df

            # Get timestamps and assert all correlates match
            timestamps = pd.concat(
                [history["timestamps"], target["timestamps"]]
            ).to_numpy()
            if timestamps_ref is None:
                timestamps_ref = timestamps
            else:
                assert np.array_equal(timestamps, timestamps_ref), (
                    "Timestamps do not match across correlates!"
                )

            # Get actuals
            actuals = pd.concat(
                [history["values"], target["values"]]
            ).to_numpy()
            col_name = (
                getattr(eval_sample, "column_name", None)  # type: ignore[attr-defined]
                or f"correlate_{nc}"
            )
            columns[col_name] = actuals

            # Check if this correlate was forecasted
            is_forecasted = getattr(
                eval_sample, "forecast", True
            )  # default True for backward compatibility
            if is_forecasted:
                # Get forecast: pad with NaN for history, then forecast values for target
                fcast_values = forecast_sample.to_df()["values"].to_numpy()  # type: ignore[attr-defined]
                forecasts = np.concatenate(
                    [np.full(len(history), np.nan), fcast_values]
                )
                forecast_col = f"{col_name}_forecast"
                columns[forecast_col] = forecasts

                # Add sample-level metrics if requested and available
                if (
                    with_metrics
                    and metrics_data is not None
                    and (
                        not isinstance(forecast_sample, list)
                        and hasattr(forecast_sample, "metrics")
                        and forecast_sample.metrics is not None
                    )
                ):
                    metrics_data[f"{col_name}_mae"] = (
                        forecast_sample.metrics.mae
                    )
                    metrics_data[f"{col_name}_mape"] = (
                        forecast_sample.metrics.mape
                    )

            # Save split column if present (from eval_df)
            if split_col is None and "split" in eval_df.columns:
                split_col = pd.concat(
                    [history["split"], target["split"]]
                ).to_numpy()

        # Build DataFrame
        df = pd.DataFrame({"timestamps": timestamps_ref, **columns})
        if split_col is not None:
            df["split"] = split_col

        # Add batch-level metrics if requested and available
        if (
            with_metrics
            and getattr(forecast_output, "metrics", None) is not None
        ):
            df["batch_mae"] = forecast_output.metrics.mae  # type: ignore[attr-defined]
            df["batch_mape"] = forecast_output.metrics.mape  # type: ignore[attr-defined]

        # Add sample-level metrics as additional columns if requested
        if with_metrics and metrics_data is not None:
            for key, value in metrics_data.items():
                df[key] = value

        result_dfs.append(df)

    return result_dfs


__all__ = ["join_as_dfs"]
