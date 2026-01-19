from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)


def fill_nan_values(array: np.ndarray) -> np.ndarray:
    # Replace NaN values with mean of non-NaN values
    nan_mask = np.isnan(array)
    if np.any(nan_mask):
        # Check if all values are NaN
        if np.all(nan_mask):
            logger.debug("All values are NaN, filling with zeros")
            array = np.zeros_like(array)
        else:
            logger.debug("Filling NaN values with mean of non-NaN values")
            # Calculate mean of non-NaN values
            mean_value = np.nanmean(array)
            # Fill NaN values with the mean
            array = np.where(nan_mask, mean_value, array)
    return array


def truncate_eval_batch(batch: EvalBatchFormat, max_context_length: int):
    for i in range(batch.batch_size):
        for j in range(batch.num_correlates):
            if len(batch[i, j].history_values) > max_context_length:
                batch[i, j].history_values = batch[i, j].history_values[
                    -max_context_length:
                ]
                batch[i, j].history_timestamps = batch[i, j].history_timestamps[
                    -max_context_length:
                ]
                # batch[i, j].target_values = batch[i, j].target_values[-max_context_length:]
    return batch


def split_eval_batch(
    batch: EvalBatchFormat, num_target_rows: int
) -> EvalBatchFormat:
    if num_target_rows <= 0:
        raise ValueError("num_target_rows must be a positive integer")

    samples_out: list[list[SingleEvalSample]] = []

    for b in range(batch.batch_size):
        row_out: list[SingleEvalSample] = []
        for nc in range(batch.num_correlates):
            sample = batch[b, nc]  # type: ignore[index]

            history_length = len(sample.history_timestamps)
            if history_length <= num_target_rows:
                raise ValueError(
                    "Cannot split sample into history and target: "
                    f"history length {history_length} must be greater than num_target_rows {num_target_rows}."
                )

            # New history from the original history (excluding the last num_target_rows)
            new_history_timestamps = sample.history_timestamps[
                :-num_target_rows
            ]
            new_history_values = sample.history_values[:-num_target_rows]

            # New target taken from the tail of the original history
            new_target_timestamps = sample.history_timestamps[-num_target_rows:]
            new_target_values = sample.history_values[-num_target_rows:]

            new_sample = SingleEvalSample(
                sample_id=sample.sample_id,
                history_timestamps=new_history_timestamps,
                history_values=new_history_values,
                target_timestamps=new_target_timestamps,
                target_values=new_target_values,
                forecast=sample.forecast,
                metadata=sample.metadata,
                leak_target=sample.leak_target,
                column_name=getattr(sample, "column_name", None),
            )

            row_out.append(new_sample)
        samples_out.append(row_out)

    return EvalBatchFormat(samples_out)


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect date/timestamp column from DataFrame.

    Checks for common date column names, then tries to parse columns
    to find datetime-like data.

    Args:
        df: DataFrame to analyze

    Returns:
        Column name of detected date column, or None if not found or if
        the DataFrame index is already a DatetimeIndex
    """
    # Check if index is already DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # Signal to use index

    # Check common date column names (case-sensitive priority)
    common_names = [
        "date",
        "timestamp",
        "time",
        "datetime",
        "Date",
        "Timestamp",
        "Time",
        "Datetime",
        "dt",
    ]

    for col in common_names:
        if col in df.columns:
            return col

    # Try to parse columns to find datetime-like data
    for col in df.columns:
        try:
            # Test first 10 rows (or fewer if dataframe is small)
            sample_size = min(10, len(df))
            if sample_size > 0:
                pd.to_datetime(df[col].iloc[:sample_size])
                return col
        except (ValueError, TypeError):
            continue

    return None
