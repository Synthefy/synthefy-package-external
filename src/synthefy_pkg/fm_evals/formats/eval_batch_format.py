"""
A class that represents a batch of data in the eval format.
"""

import datetime
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from pandas.tseries.frequencies import to_offset


class SingleEvalSample:
    def __init__(
        self,
        sample_id: Any,
        history_timestamps: np.ndarray,
        history_values: np.ndarray,
        target_timestamps: np.ndarray,
        target_values: np.ndarray,
        forecast: bool = True,
        metadata: bool = False,
        leak_target: bool = False,
        column_name: Optional[str] = None,
    ):
        self.sample_id = sample_id
        self.history_timestamps = history_timestamps
        self.history_values = history_values
        self.target_timestamps = target_timestamps
        self.target_values = target_values
        self.forecast = forecast  # Maybe don't forecast a particular target, but use as a correlate.
        self.metadata = metadata
        self.leak_target = leak_target
        self.column_name = column_name
        # All should be 1D arrays of length T (or history_T, target_T)
        assert history_timestamps.shape == history_values.shape, (
            "History timestamps and values must have the same shape"
        )
        assert target_timestamps.shape == target_values.shape, (
            "Target timestamps and values must have the same shape"
        )

    def to_arrays(self):
        return (
            self.sample_id,
            self.history_timestamps,
            self.history_values,
            self.target_timestamps,
            self.target_values,
        )

    def to_df(self):
        # Concatenate history and target arrays
        timestamps = np.concatenate(
            [self.history_timestamps, self.target_timestamps]
        )
        values = np.concatenate([self.history_values, self.target_values])
        split = np.array(
            ["history"] * len(self.history_timestamps)
            + ["target"] * len(self.target_timestamps)
        )
        data = {
            "timestamps": timestamps,
            "values": values,
            "split": split,
        }
        if self.column_name is not None:
            data["column_name"] = np.full(len(timestamps), self.column_name)
        n_rows = len(timestamps)
        if hasattr(self, "sample_id"):
            sample_id = self.sample_id
            if isinstance(
                sample_id, (tuple, list, np.ndarray)
            ) and not isinstance(sample_id, str):
                for idx, val in enumerate(sample_id):
                    data[f"sample_id_{idx}"] = np.full(n_rows, val)
            else:
                data["sample_id"] = np.full(n_rows, sample_id)
        df = pd.DataFrame(data)
        return df

    def __str__(self):
        lines = []
        lines.append("SingleEvalSample:")
        lines.append(f"  sample_id: {self.sample_id}")
        if len(self.history_timestamps) > 0:
            lines.append(
                f"  history: {self.history_timestamps[0]} to {self.history_timestamps[-1]} ({len(self.history_timestamps)} steps)"
            )
        else:
            lines.append("  history: (empty)")
        if len(self.target_timestamps) > 0:
            lines.append(
                f"  target: {self.target_timestamps[0]} to {self.target_timestamps[-1]} ({len(self.target_timestamps)} steps)"
            )
        else:
            lines.append("  target: (empty)")
        lines.append(f"  history_values shape: {self.history_values.shape}")
        lines.append(f"  target_values shape: {self.target_values.shape}")
        lines.append(f"  forecast: {self.forecast}")
        lines.append(f"  metadata: {self.metadata}")
        lines.append(f"  leak_target: {self.leak_target}")
        if self.column_name is not None:
            lines.append(f"  column_name: {self.column_name}")
        return "\n".join(lines)


class EvalBatchFormat:
    """
    A class that represents a batch of data in the eval format.

    Although this is essentially a glorified dictionary, forcing everything that crosses the dataloader / forecaster barrier to be in this format
    makes it easier to implement new forecasters and dataloaders.
    """

    def __init__(
        self,
        samples: List[List[SingleEvalSample]],
        target_cols: List[str] = [],
        metadata_cols: List[str] = [],
        leak_cols: List[str] = [],
        sample_id_cols: List[str] = [],
        timestamp_col: Optional[str] = None,
    ):
        assert isinstance(samples, list) and len(samples) > 0, (
            "Samples must be a non-empty list of lists"
        )
        num_correlates = len(samples[0])
        assert num_correlates > 0, "Each batch must have at least one correlate"
        for i, row in enumerate(samples):
            assert isinstance(row, list), (
                "Each batch must be a list of SingleEvalSample"
            )
            assert len(row) == num_correlates, (
                f"All batches must have the same number of correlates. Batch {i} has {len(row)} correlates, we expected {num_correlates}"
            )
            ht_shape = row[0].history_timestamps.shape
            hv_shape = row[0].history_values.shape
            tt_shape = row[0].target_timestamps.shape
            tv_shape = row[0].target_values.shape
            for sample in row:
                assert sample.history_timestamps.shape == ht_shape, (
                    f"All history timestamps in a correlate set must have the same shape. Sample {sample.sample_id} has shape {sample.history_timestamps.shape}, we expected {ht_shape}"
                )
                assert sample.history_values.shape == hv_shape, (
                    f"All history values in a correlate set must have the same shape. Sample {sample.sample_id} has shape {sample.history_values.shape}, we expected {hv_shape}"
                )
                assert sample.target_timestamps.shape == tt_shape, (
                    f"All target timestamps in a correlate set must have the same shape. Sample {sample.sample_id} has shape {sample.target_timestamps.shape}, we expected {tt_shape}"
                )
                assert sample.target_values.shape == tv_shape, (
                    f"All target values in a correlate set must have the same shape. Sample {sample.sample_id} has shape {sample.target_values.shape}, we expected {tv_shape}"
                )
        self.samples = samples  # [B][NC]
        self.batch_size = len(samples)
        self.num_correlates = num_correlates
        self.target_cols = target_cols
        self.metadata_cols = metadata_cols
        self.leak_cols = leak_cols
        self.sample_id_cols = sample_id_cols
        self.timestamp_col = timestamp_col

        self.history_length = (
            samples[0][0].history_timestamps.shape[0]
            if self.batch_size > 0 and self.num_correlates > 0
            else 0
        )
        self.target_length = (
            samples[0][0].target_timestamps.shape[0]
            if self.batch_size > 0 and self.num_correlates > 0
            else 0
        )

    @classmethod
    def from_dfs(
        cls,
        dfs: List[pd.DataFrame],
        timestamp_col: str,
        cutoff_date: Optional[str] = None,
        num_target_rows: Optional[int] = None,
        target_cols: List[str] = [],
        metadata_cols: List[str] = [],
        leak_cols: List[str] = [],
        sample_id_cols: List[str] = [],
        forecast_window: Optional[Union[str, int]] = None,
        stride: Optional[Union[str, int]] = None,
    ):
        """
        Construct EvalBatchFormat from a list of pandas DataFrames.

        This method creates evaluation batches from multiple DataFrames, supporting both
        single-window and backtesting scenarios. The method automatically determines the
        appropriate processing mode based on the provided parameters.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of pandas DataFrames to process. Each DataFrame should contain time series
            data with a timestamp column and various feature columns.
        timestamp_col : str
            Name of the column containing timestamps. This column will be converted to
            datetime format and used for splitting data into history and target periods.
        cutoff_date : Optional[str], default=None
            Date string (e.g., "2023-01-01") to split data into history (≤ cutoff) and
            target (> cutoff) periods. Mutually exclusive with num_target_rows.
        num_target_rows : Optional[int], default=None
            Number of rows to use as target period (taken from the end of each DataFrame).
            Mutually exclusive with cutoff_date.
        target_cols : List[str], default=[]
            Column names to be used as forecast targets. These columns will have
            forecast=True in the resulting SingleEvalSample objects.
        metadata_cols : List[str], default=[]
            Column names to be used as metadata/correlates. These columns will have
            metadata=True in the resulting SingleEvalSample objects.
        leak_cols : List[str], default=[]
            Column names that are allowed to leak target information. Must be a subset
            of metadata_cols. These columns will have leak_target=True.
        sample_id_cols : List[str], default=[]
            Column names to use for creating unique sample IDs. Values from these columns
            will be concatenated with the column name to form sample IDs.
        forecast_window : Optional[Union[str, int]], default=None
            For backtesting: the size of the forecast window. If str, interpreted as
            pandas time offset (e.g., "7D" for 7 days). If int, interpreted as number
            of rows. Must be provided together with stride.
        stride : Optional[Union[str, int]], default=None
            For backtesting: the step size between consecutive forecasts. If str,
            interpreted as pandas time offset. If int, interpreted as number of rows.
            Must be provided together with forecast_window.

        Returns
        -------
        EvalBatchFormat or None
            An EvalBatchFormat object containing the processed samples, or None if no
            valid windows could be created.

        Raises
        ------
        ValueError
            - If both cutoff_date and num_target_rows are provided
            - If neither cutoff_date nor num_target_rows are provided
            - If forecast_window is provided without stride or vice versa
            - If no history rows are found for the given cutoff_date
            - If type mismatches occur (e.g., string vs int for forecast_window/stride)

        Notes
        -----
        The method supports four processing modes:
        1. Single window by date: cutoff_date provided, no forecast_window/stride
        2. Single window by rows: num_target_rows provided, no forecast_window/stride
        3. Backtesting by date: cutoff_date + forecast_window/stride (all strings)
        4. Backtesting by rows: num_target_rows + forecast_window/stride (all integers)

        Each DataFrame is processed independently, and the results are combined into
        a single EvalBatchFormat object. The timestamp column is automatically converted
        to datetime format and sorted chronologically.

        Examples
        --------
        # Single window by date
        batch = EvalBatchFormat.from_dfs(
            dfs=[df1, df2],
            timestamp_col="date",
            cutoff_date="2023-06-01",
            target_cols=["sales"],
            metadata_cols=["temperature", "holiday"]
        )

        # Backtesting by date
        batch = EvalBatchFormat.from_dfs(
            dfs=[df1],
            timestamp_col="date",
            cutoff_date="2023-01-01",
            target_cols=["sales"],
            forecast_window="7D",
            stride="1D"
        )
        """

        # Decide if we're doing backtesting or not.
        # We are doing backtesting if we have a forecast_window and stride.
        backtesting = False
        if forecast_window is not None or stride is not None:
            if forecast_window is None or stride is None:
                raise ValueError(
                    "Forecast Window and Stride must be provided together"
                )
            backtesting = True

        # Decide if we're splitting dataframes by date or by rows
        if cutoff_date is not None:
            if num_target_rows is not None:
                raise ValueError(
                    "Only one of cutoff_date or num_target_rows can be provided"
                )
            split_by_date = True
        else:
            if num_target_rows is None:
                raise ValueError(
                    "Either cutoff_date or num_target_rows must be provided"
                )
            split_by_date = False

        # Ensure timestamp_col is datetime in all DataFrames
        for i, df in enumerate(dfs):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            dfs[i] = df.sort_values(by=timestamp_col).reset_index(drop=True)

        # For each dataframe, dispatch to the appropriate helper method
        eval_batch = []
        for df in dfs:
            if backtesting:
                if split_by_date:
                    assert isinstance(cutoff_date, str), (
                        "cutoff_date must be a string"
                    )
                    assert isinstance(forecast_window, str), (
                        "forecast_window must be a string"
                    )
                    assert isinstance(stride, str), "stride must be a string"
                    eval_batch += cls.from_df_backtesting_by_date(
                        df,
                        timestamp_col,
                        cutoff_date,
                        target_cols,
                        metadata_cols,
                        leak_cols,
                        sample_id_cols,
                        forecast_window,
                        stride,
                    )
                else:
                    assert isinstance(num_target_rows, int), (
                        "num_target_rows must be an integer"
                    )
                    assert isinstance(forecast_window, int), (
                        "forecast_window must be an integer"
                    )
                    assert isinstance(stride, int), "stride must be an integer"
                    eval_batch += cls.from_df_backtesting_by_rows(
                        df,
                        timestamp_col,
                        num_target_rows,
                        target_cols,
                        metadata_cols,
                        leak_cols,
                        sample_id_cols,
                        forecast_window,
                        stride,
                    )
            else:
                if split_by_date:
                    assert isinstance(cutoff_date, str), (
                        "cutoff_date must be a string"
                    )
                    eval_batch += cls.from_df_by_date(
                        df,
                        timestamp_col,
                        cutoff_date,
                        target_cols,
                        metadata_cols,
                        leak_cols,
                        sample_id_cols,
                    )
                else:
                    assert isinstance(num_target_rows, int), (
                        "num_target_rows must be an integer"
                    )
                    eval_batch += cls.from_df_by_rows(
                        df,
                        timestamp_col,
                        num_target_rows,
                        target_cols,
                        metadata_cols,
                        leak_cols,
                        sample_id_cols,
                    )

        if len(eval_batch) == 0:
            logger.error(
                "No valid windows could be created from the provided dataframes. "
                "Please check your inputs and try again."
            )
            return None

        return cls(
            eval_batch,
            target_cols,
            metadata_cols,
            leak_cols,
            sample_id_cols,
            timestamp_col,
        )

    @classmethod
    def from_df_backtesting_by_date(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        cutoff_date: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        sample_id_cols: List[str],
        forecast_window: str,
        stride: str,
    ) -> List[List[SingleEvalSample]]:
        # Identify a timestamp if it exists and convert cutoff_date to a timestamp
        tz = None
        if df[timestamp_col].dt.tz is not None:
            tz = df[timestamp_col].dt.tz
        cutoff_timestamp = pd.Timestamp(cutoff_date, tz=tz)

        # Slice the dataframe into history and target based on timestamp
        windows = []
        while True:
            # Slice the dataframe into history and target based on timestamp
            # We select only targets that are within cutoff_timestamp + forecast_window
            history_df = df[df[timestamp_col] <= cutoff_timestamp]
            target_df = df[
                (df[timestamp_col] > cutoff_timestamp)
                & (
                    df[timestamp_col]
                    <= cutoff_timestamp + pd.Timedelta(forecast_window)
                )
            ]

            if len(history_df) == 0:
                raise ValueError(
                    f"No history rows found for the given cutoff_date: {cutoff_date}. "
                    "Please check your inputs and try again."
                )

            # We are done if there are no target rows left
            if len(target_df) == 0:
                break

            # Split the dataframe into correlates (one SingleEvalSample per correlate)
            windows.append(
                cls.split_df_to_correlates(
                    history_df,
                    target_df,
                    timestamp_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                    sample_id_cols,
                )
            )

            # Move the cutoff timestamp forward by the stride
            cutoff_timestamp += pd.Timedelta(stride)

        return windows

    @classmethod
    def from_df_backtesting_by_rows(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        num_target_rows: int,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        sample_id_cols: List[str],
        forecast_window: int,
        stride: int,
    ) -> List[List[SingleEvalSample]]:
        windows = []

        cutoff_idx = len(df) - num_target_rows
        while True:
            if cutoff_idx + forecast_window > len(df):
                break

            history_df = df.iloc[:cutoff_idx]
            target_df = df.iloc[cutoff_idx : cutoff_idx + forecast_window]
            windows.append(
                cls.split_df_to_correlates(
                    history_df,
                    target_df,
                    timestamp_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                    sample_id_cols,
                )
            )
            cutoff_idx += stride

        return windows

    @classmethod
    def from_df_by_date(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        cutoff_date: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        sample_id_cols: List[str],
    ) -> List[List[SingleEvalSample]]:
        # Identify a timestamp if it exists and convert cutoff_date to a timestamp
        tz = None
        if df[timestamp_col].dt.tz is not None:
            tz = df[timestamp_col].dt.tz
        cutoff_timestamp = pd.Timestamp(cutoff_date, tz=tz)

        # Slice the dataframe into history and target based on timestamp
        history_mask = df[timestamp_col] <= cutoff_timestamp
        target_mask = df[timestamp_col] > cutoff_timestamp

        if len(df[history_mask]) == 0:
            raise ValueError(
                f"No history rows found for the given cutoff_date: {cutoff_date}. "
                "Please check your inputs and try again."
            )

        if len(df[target_mask]) == 0:
            return []

        # Split the dataframe into correlates (one SingleEvalSample per correlate)
        return [
            cls.split_df_to_correlates(
                df[history_mask],
                df[target_mask],
                timestamp_col,
                target_cols,
                metadata_cols,
                leak_cols,
                sample_id_cols,
            )
        ]

    @classmethod
    def from_df_by_rows(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        num_target_rows: int,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        sample_id_cols: List[str],
    ) -> List[List[SingleEvalSample]]:
        if num_target_rows <= 0:
            raise ValueError("num_target_rows must be a positive integer")
        elif num_target_rows >= len(df):
            raise ValueError(
                "num_target_rows must be less than the number of rows in the dataframe"
            )
        else:
            return [
                cls.split_df_to_correlates(
                    df.iloc[:-num_target_rows],
                    df.iloc[-num_target_rows:],
                    timestamp_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                    sample_id_cols,
                )
            ]

    @staticmethod
    def split_df_to_correlates(
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        timestamp_col: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        sample_id_cols: List[str],
    ) -> List[SingleEvalSample]:
        # We only allow leaking for metadata columns.
        assert np.all([x in metadata_cols for x in leak_cols]), (
            "All leak columns must be metadata columns. Leaking is only allowed for metadata columns."
        )

        all_cols = target_cols + [
            x for x in metadata_cols if x not in target_cols
        ]  # Preserve ordering

        correlates = []
        for col in all_cols:
            history_timestamps = history_df[timestamp_col].values.astype(
                "datetime64[ns]"
            )
            history_values = history_df[col].values.astype(np.float64)
            target_timestamps = target_df[timestamp_col].values.astype(
                "datetime64[ns]"
            )
            target_values = target_df[col].values.astype(np.float64)

            sample_id = np.array(
                [
                    "_".join(
                        [
                            history_df[id_col].values.astype(str)[0]
                            for id_col in sample_id_cols
                        ]
                        + [col]
                    )
                ]
            )

            sample = SingleEvalSample(
                sample_id=sample_id,
                history_timestamps=history_timestamps,
                history_values=history_values,
                target_timestamps=target_timestamps,
                target_values=target_values,
                forecast=col in target_cols,
                metadata=col in metadata_cols,
                leak_target=col in leak_cols if leak_cols else False,
                column_name=col,
            )
            correlates.append(sample)
        return correlates

    def to_dfs(self):
        dfs = []
        for row in self.samples:
            dfs.append([sample.to_df() for sample in row])
        return dfs

    @classmethod
    def reconstruct_wide_format(
        cls, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Reconstruct wide-format DataFrames from a DataFrame with nested sensor data.

        Args:
            df: DataFrame with columns: sample_id, history_timestamps, history_values,
                target_timestamps, target_values

        Returns:
            tuple of (history_df, target_df, combined_df):
            - history_df: DataFrame with history data, columns are sensor IDs
            - target_df: DataFrame with target data, columns are sensor IDs
            - combined_df: DataFrame with all data combined, columns are sensor IDs
        """
        # Initialize DataFrames
        history_dfs = []
        target_dfs = []
        combined_dfs = []

        history_dfs_current = []
        target_dfs_current = []
        combined_dfs_current = []

        current_idx_id = None

        def update_stacked_dfs(
            history_dfs_current,
            target_dfs_current,
            combined_dfs_current,
            history_timestamps,
            target_timestamps,
            idx_id,
        ):
            if len(history_dfs_current) == 0:
                return None, None, None

            # Pad dataframes to same length with NaN values
            # This is safe because the downstream analysis pipeline handles NaN values appropriately:
            # - Statistical features check for NaN before processing
            # - Causality analysis has interpolation options for NaN values
            # - Decomposition methods drop NaN values before analysis
            def pad_dataframes_to_same_length(dfs):
                if not dfs:
                    return dfs

                # Find the maximum length among all dataframes
                max_length = max(len(df) for df in dfs)

                # If all dataframes are already the same length, no padding needed
                if all(len(df) == max_length for df in dfs):
                    return dfs

                # Pad each dataframe to the maximum length
                padded_dfs = []
                for df in dfs:
                    if len(df) < max_length:
                        # Create additional rows with NaN values
                        # Use the same index structure as the original dataframe
                        new_index = pd.RangeIndex(
                            start=len(df), stop=max_length
                        )
                        padding_rows = pd.DataFrame(
                            np.nan, index=new_index, columns=df.columns
                        )
                        padded_df = pd.concat([df, padding_rows], axis=0)
                        padded_dfs.append(padded_df)
                    else:
                        padded_dfs.append(df)

                return padded_dfs

            # Pad all dataframe lists to ensure consistent lengths
            history_dfs_current = pad_dataframes_to_same_length(
                history_dfs_current
            )
            target_dfs_current = pad_dataframes_to_same_length(
                target_dfs_current
            )
            combined_dfs_current = pad_dataframes_to_same_length(
                combined_dfs_current
            )

            stacked_history = pd.concat(history_dfs_current, axis=1)
            stacked_target = pd.concat(target_dfs_current, axis=1)
            stacked_combined = pd.concat(combined_dfs_current, axis=1)

            # Create timestamps that match the actual dataframe lengths
            # Since the dataframes are concatenated from multiple sensors, we need to create
            # timestamps that represent the entire batch, not just the last row's timestamps

            def create_timestamps_for_dataframe(
                df_length, base_timestamps=None
            ):
                """Create timestamps for a dataframe of given length."""
                if df_length == 0:
                    return np.array([])

                if base_timestamps is not None and len(base_timestamps) > 0:
                    # Use the base timestamps as a reference for frequency
                    if len(base_timestamps) >= 2:
                        # Calculate interval from base timestamps
                        if isinstance(
                            base_timestamps[0], (np.datetime64, pd.Timestamp)
                        ):
                            intervals = np.diff(base_timestamps)
                            median_interval = np.median(intervals)
                            median_delta_us = int(
                                median_interval.astype(
                                    "timedelta64[us]"
                                ).astype("int64")
                            )
                            start_time = base_timestamps[0]
                            timestamps = [
                                start_time
                                + pd.Timedelta(microseconds=median_delta_us * i)
                                for i in range(df_length)
                            ]
                        else:
                            intervals = np.diff(base_timestamps)
                            median_interval = np.median(intervals)
                            start_time = base_timestamps[0]
                            timestamps = [
                                start_time + median_interval * i
                                for i in range(df_length)
                            ]
                    else:
                        # Single timestamp, use default interval
                        start_time = base_timestamps[0]
                        if isinstance(
                            start_time, (np.datetime64, pd.Timestamp)
                        ):
                            timestamps = [
                                start_time + pd.Timedelta(seconds=i)
                                for i in range(df_length)
                            ]
                        else:
                            timestamps = [
                                start_time + i for i in range(df_length)
                            ]
                else:
                    # No base timestamps, create default ones
                    timestamps = np.arange(df_length)

                return np.array(timestamps)

            # Create timestamps for each dataframe
            history_timestamps_for_df = create_timestamps_for_dataframe(
                len(stacked_history), history_timestamps
            )
            target_timestamps_for_df = create_timestamps_for_dataframe(
                len(stacked_target), target_timestamps
            )

            stacked_history["timestamp"] = history_timestamps_for_df
            stacked_history["idx"] = np.full(len(stacked_history), idx_id)
            stacked_target["timestamp"] = target_timestamps_for_df
            stacked_target["idx"] = np.full(len(stacked_target), idx_id)

            # For combined dataframe, concatenate the timestamps
            combined_timestamps = np.concatenate(
                [history_timestamps_for_df, target_timestamps_for_df]
            )
            stacked_combined["timestamp"] = combined_timestamps
            stacked_combined["idx"] = np.full(len(stacked_combined), idx_id)
            return stacked_history, stacked_target, stacked_combined

        for idx, row in df.iterrows():
            # Extract sensor ID (remove brackets and quotes)
            sensor_id = str(row["sample_id"]).strip("[]").strip("'")
            # Get timestamps and values
            history_timestamps = row["history_timestamps"]
            history_values = row["history_values"]
            target_timestamps = row["target_timestamps"]
            target_values = row["target_values"]

            idx_id = sensor_id.split("_")[0]
            # assert that the idx_id is of the format b[0-9]
            assert re.match(r"b[0-9]+", idx_id), (
                "idx_id must be of the format b[0-9]+"
            )

            if current_idx_id is None:
                current_idx_id = idx_id
            else:
                if idx_id != current_idx_id:
                    # stack together all the dataframees with the same idx
                    stacked_history, stacked_target, stacked_combined = (
                        update_stacked_dfs(
                            history_dfs_current,
                            target_dfs_current,
                            combined_dfs_current,
                            history_timestamps,
                            target_timestamps,
                            current_idx_id,
                        )
                    )
                    (
                        history_dfs_current,
                        target_dfs_current,
                        combined_dfs_current,
                    ) = [], [], []
                    history_dfs.append(stacked_history)
                    target_dfs.append(stacked_target)
                    combined_dfs.append(stacked_combined)
                    current_idx_id = idx_id
            sensor_id = "_".join(sensor_id.split("_")[1:])

            # Create history DataFrame for this sensor
            history_df = pd.DataFrame({sensor_id: history_values})
            history_dfs_current.append(history_df)

            # Create target DataFrame for this sensor
            target_df = pd.DataFrame({sensor_id: target_values})
            target_dfs_current.append(target_df)

            # Create combined DataFrame for this sensor
            combined_values = np.concatenate([history_values, target_values])
            combined_df = pd.DataFrame({sensor_id: combined_values})
            combined_dfs_current.append(combined_df)

        stacked_history, stacked_target, stacked_combined = update_stacked_dfs(
            history_dfs_current,
            target_dfs_current,
            combined_dfs_current,
            history_timestamps,
            target_timestamps,
            current_idx_id,
        )
        if stacked_history is not None:
            history_dfs.append(stacked_history)
            target_dfs.append(stacked_target)
            combined_dfs.append(stacked_combined)

        # Note: We don't need global padding here because the analysis pipeline
        # handles different group lengths appropriately. Global padding was causing
        # "Insufficient data for feature extraction" errors by creating too many NaN values.
        # The decomposition analysis concatenation issue should be handled at the
        # analysis level, not at the data reconstruction level.

        # Concatenate all DataFrames by columns
        history_df = pd.concat(history_dfs, axis=0)
        target_df = pd.concat(target_dfs, axis=0)
        combined_df = pd.concat(combined_dfs, axis=0)
        return history_df, target_df, combined_df

    @classmethod
    def from_arrays(
        cls,
        sample_ids: np.ndarray,
        history_timestamps: np.ndarray,
        history_values: np.ndarray,
        target_timestamps: np.ndarray,
        target_values: np.ndarray,
        forecast: Union[bool, np.ndarray] = True,
        metadata: Union[bool, np.ndarray] = True,
        leak_target: Union[bool, np.ndarray] = False,
        column_name: Optional[str] = None,
    ):
        """
        Construct EvalBatchFormat from np.ndarrays.
        sample_ids: [B, NC]
        history_timestamps: [B, NC, T]
        history_values: [B, NC, T]
        target_timestamps: [B, NC, T]
        target_values: [B, NC, T]
        """
        assert len(history_timestamps.shape) == 3, (
            "History timestamps must be [B, NC, T]"
        )
        assert len(history_values.shape) == 3, (
            "History values must be [B, NC, T]"
        )
        assert len(target_timestamps.shape) == 3, (
            "Target timestamps must be [B, NC, T]"
        )
        assert len(target_values.shape) == 3, "Target values must be [B, NC, T]"
        assert history_timestamps.shape == history_values.shape, (
            "History timestamps and values must have the same shape"
        )
        assert target_timestamps.shape == target_values.shape, (
            "Target timestamps and values must have the same shape"
        )
        assert sample_ids.shape[0] == history_timestamps.shape[0], (
            "Sample IDs and history timestamps must have the same batch size"
        )
        assert sample_ids.shape[1] == history_timestamps.shape[1], (
            "Sample IDs and history timestamps must have the same number of correlates"
        )

        if isinstance(forecast, bool):
            forecast = np.full(sample_ids.shape, forecast)
        else:
            # Make sure the forecast array is the same shape as the sample_ids array
            assert forecast.shape == sample_ids.shape[0:2], (
                "Forecast must be the same shape as sample_ids (B, NC)"
            )

        if isinstance(metadata, bool):
            metadata = np.full(sample_ids.shape, metadata)
        else:
            # Make sure the metadata array is the same shape as the sample_ids array
            assert metadata.shape == sample_ids.shape[0:2], (
                "Metadata must be the same shape as sample_ids (B, NC)"
            )

        if isinstance(leak_target, bool):
            leak_target = np.full(sample_ids.shape, leak_target)
        else:
            # Make sure the leak_target array is the same shape as the sample_ids array
            assert leak_target.shape == sample_ids.shape[0:2], (
                "Leak target must be the same shape as sample_ids (B, NC)"
            )

        B, NC, T_hist = history_timestamps.shape
        _, _, T_tgt = target_timestamps.shape
        samples = []
        for b in range(B):
            row = []
            for nc in range(NC):
                row.append(
                    SingleEvalSample(
                        sample_id=sample_ids[b, nc],
                        history_timestamps=history_timestamps[b, nc],
                        history_values=history_values[b, nc],
                        target_timestamps=target_timestamps[b, nc],
                        target_values=target_values[b, nc],
                        forecast=forecast[b, nc],
                        metadata=metadata[b, nc],
                        leak_target=leak_target[b, nc],
                        column_name=column_name,
                    )
                )
            samples.append(row)
        obj = cls(samples)
        obj.batch_size = B
        obj.num_correlates = NC
        obj.history_length = T_hist
        obj.target_length = T_tgt
        return obj

    def to_arrays(self, targets_only: bool = True):
        sample_ids = np.array(
            [
                [
                    sample.sample_id
                    for sample in row
                    if (sample.forecast or not targets_only)
                ]
                for row in self.samples
            ]
        )
        history_timestamps = np.array(
            [
                [
                    sample.history_timestamps
                    for sample in row
                    if (sample.forecast or not targets_only)
                ]
                for row in self.samples
            ]
        )
        history_values = np.array(
            [
                [
                    sample.history_values
                    for sample in row
                    if (sample.forecast or not targets_only)
                ]
                for row in self.samples
            ]
        )
        target_timestamps = np.array(
            [
                [
                    sample.target_timestamps
                    for sample in row
                    if (sample.forecast or not targets_only)
                ]
                for row in self.samples
            ]
        )
        target_values = np.array(
            [
                [
                    sample.target_values
                    for sample in row
                    if (sample.forecast or not targets_only)
                ]
                for row in self.samples
            ]
        )
        return (
            sample_ids,
            history_timestamps,
            history_values,
            target_timestamps,
            target_values,
        )

    def __getitem__(self, idx: tuple[int, int]) -> SingleEvalSample:
        if isinstance(idx, tuple) and len(idx) == 2:
            b, nc = idx
            return self.samples[b][nc]
        raise TypeError("EvalBatchFormat only supports indexing as batch[i, j]")

    def __str__(self):
        lines = []
        num_samples = len(self.samples)
        total_timeseries = sum(len(row) for row in self.samples)
        lines.append("EvalBatchFormat:")
        lines.append(f"  Number of samples: {num_samples}")
        lines.append(f"  Total time series: {total_timeseries}")
        lines.append("  (Note: history/target lengths may vary across samples)")
        if num_samples > 0 and len(self.samples[0]) > 0:
            sample = self.samples[0][0]
            lines.append("  First sample preview:")
            lines.append("    " + "\n    ".join(sample.__str__().splitlines()))
        return "\n".join(lines)
