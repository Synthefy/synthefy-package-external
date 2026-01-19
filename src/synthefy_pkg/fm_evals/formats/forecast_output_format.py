"""
A class that represents a forecast output in the eval format.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from synthefy_pkg.fm_evals.formats.metrics import ForecastMetrics


class SingleSampleForecast:
    def __init__(
        self,
        sample_id: Any,
        timestamps: np.ndarray,
        values: np.ndarray,
        model_name: str,
    ):
        self.sample_id = sample_id
        self.timestamps = timestamps
        self.values = values
        self.model_name = model_name  # Useful for plotting and comparisons
        self.metrics: Optional[ForecastMetrics] = (
            None  # Sample-level forecast metrics (set after computation)
        )

        assert len(timestamps) == len(values), (
            "Timestamps and values must have the same length"
        )

    def to_df(self):
        data = {
            "timestamps": self.timestamps,
            "values": self.values,
        }
        n_rows = len(self.timestamps)
        # If sample_id is array-like (but not a string), split into multiple columns and broadcast
        if isinstance(
            self.sample_id, (tuple, list, np.ndarray)
        ) and not isinstance(self.sample_id, str):
            for idx, val in enumerate(self.sample_id):
                data[f"sample_id_{idx}"] = np.full(n_rows, val)
        else:
            data["sample_id"] = np.full(n_rows, self.sample_id)
        df = pd.DataFrame(data)
        return df

    def __str__(self):
        return (
            f"SingleSampleForecast(sample_id={self.sample_id}, "
            f"timestamps.shape={self.timestamps.shape}, "
            f"values.shape={self.values.shape}, "
            f"model_name='{self.model_name}')"
        )


class ForecastOutputFormat:
    """
    A class that represents a forecast output in the eval format.

    We define a strict type for this so that downstream code can always be compatible with the output format.
    """

    def __init__(self, forecasts: list[list[SingleSampleForecast]]):
        # Logical assertions for nested list structure
        assert isinstance(forecasts, list) and len(forecasts) > 0, (
            "Forecasts must be a non-empty list of lists"
        )
        num_correlates = len(forecasts[0])
        assert num_correlates > 0, "Each batch must have at least one correlate"
        for row in forecasts:
            assert isinstance(row, list), (
                "Each batch must be a list of SingleSampleForecasts"
            )
            assert len(row) == num_correlates, (
                "All batches must have the same number of correlates"
            )
            # t_shape = row[0].timestamps.shape
            # v_shape = row[0].values.shape

            # # Check that all forecasts in a row have consistent shapes
            # for forecast in row:
            #     # All timestamps in a batch must have same shape
            #     assert forecast.timestamps.shape == t_shape, (
            #         "All timestamps in a batch must have the same shape"
            #     )
            #     # All values in a batch must have same shape
            #     assert forecast.values.shape == v_shape, (
            #         "All values in a batch must have the same shape"
            #     )
            #     # Each forecast's timestamps and values must match
            #     assert forecast.timestamps.shape == forecast.values.shape, (
            #         "Timestamps and values must have the same shape in each forecast"
            #     )
        self.forecasts = forecasts  # [B][NC]
        self.batch_size = len(forecasts)
        self.num_correlates = num_correlates
        self.metrics: Optional[ForecastMetrics] = (
            None  # Batch-level forecast metrics (set after computation)
        )

    @classmethod
    def from_arrays(
        cls,
        sample_ids: np.ndarray,
        timestamps: np.ndarray,
        values: np.ndarray,
        model_name: str = "",
    ):
        """
        Construct ForecastOutputFormat from np.ndarrays.
        sample_ids: [B, NC]
        timestamps: [B, NC, T]
        values: [B, NC, T]
        """
        # Add back original assertions
        assert len(timestamps.shape) == 3, "Timestamps must be [B, NC, T]"
        assert len(values.shape) == 3, "Values must be [B, NC, T]"
        assert timestamps.shape == values.shape, (
            "Timestamps and values must have the same shape"
        )
        assert sample_ids.shape[0] == timestamps.shape[0], (
            "Sample IDs and timestamps must have the same batch size"
        )
        assert sample_ids.shape[1] == timestamps.shape[1], (
            "Sample IDs and timestamps must have the same number of correlates"
        )
        batch_size = timestamps.shape[0]
        assert batch_size == values.shape[0] == sample_ids.shape[0], (
            "Batch size of timestamps, values, and sample IDs must match"
        )
        num_correlates = timestamps.shape[1]
        assert num_correlates == values.shape[1] == sample_ids.shape[1], (
            "Timestamps, and values must have the same number of correlates"
        )
        B, NC, T = timestamps.shape
        forecasts = []
        for b in range(B):
            row = []
            for nc in range(NC):
                row.append(
                    SingleSampleForecast(
                        sample_id=sample_ids[b, nc],
                        timestamps=timestamps[b, nc],
                        values=values[b, nc],
                        model_name=model_name,
                    )
                )
            forecasts.append(row)
        obj = cls(forecasts)
        obj.batch_size = B
        obj.num_correlates = NC
        return obj

    def to_arrays(self, targets_only: bool = True):
        sample_ids = np.array(
            [
                [
                    forecast.sample_id
                    for forecast in row
                    if len(forecast.timestamps) > 0 or not targets_only
                ]
                for row in self.forecasts
            ]
        )
        timestamps = np.array(
            [
                [
                    forecast.timestamps
                    for forecast in row
                    if len(forecast.timestamps) > 0 or not targets_only
                ]
                for row in self.forecasts
            ]
        )
        values = np.array(
            [
                [
                    forecast.values
                    for forecast in row
                    if len(forecast.timestamps) > 0 or not targets_only
                ]
                for row in self.forecasts
            ]
        )
        return sample_ids, timestamps, values

    def to_dfs(self):
        dfs = []
        for row in self.forecasts:
            dfs.append([forecast.to_df() for forecast in row])
        return dfs

    def __getitem__(self, idx: int | tuple[int, int] | slice):
        if isinstance(idx, tuple) and len(idx) == 2:
            b, nc = idx
            return self.forecasts[b][nc]
        return self.forecasts[idx]

    def __str__(self):
        return (
            f"ForecastOutputFormat(batch_size={self.batch_size}, "
            f"num_correlates={self.num_correlates}, "
            f"forecasts=[...])"
        )
