import glob
import os
import random
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow.ipc import RecordBatchStreamReader

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class GIFTEvalUnivariateDataloader(BaseEvalDataloader):
    """
    Dataloader for GIFT univariate dataset.
    subdata paths:
    """

    def __init__(
        self,
        data_dir: str,
        forecast_length: int,
        history_length: Optional[int] = None,
        random_ordering: bool = False,
    ):
        self.data_dir = data_dir
        self.forecast_length = forecast_length
        self.max_history_length = history_length
        self.random_ordering = random_ordering

        self.arrow_files = self._collect_arrow_files()
        self.df = self._load_arrow_files_as_df(self.arrow_files)

    def _collect_arrow_files(self) -> list[str]:
        return glob.glob(os.path.join(self.data_dir, "*.arrow"))

    def _load_arrow_files_as_df(self, arrow_files: list[str]) -> pd.DataFrame:
        dfs = []
        for arrow_file in arrow_files:
            with open(arrow_file, "rb") as f:
                reader = RecordBatchStreamReader(f)
                table = reader.read_all()
            dfs.append(table.to_pandas())
        if self.random_ordering:
            random.shuffle(dfs)
        return pd.concat(dfs)

    def _create_timestamp_array(
        self, start_date: str, frequency: str, length: int
    ) -> np.ndarray:
        if isinstance(start_date, np.ndarray):
            start_date = start_date[0]
        return pd.date_range(
            start=start_date, periods=length, freq=frequency
        ).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for i in range(len(self.df)):
            sample_row = self.df.iloc[i]

            target_array = np.array(sample_row["target"])

            if isinstance(target_array[0], np.ndarray):
                # We have nc > 1, need to extract nested arrays
                target_array = np.array([np.array(x) for x in target_array])
            else:
                # We have nc = 1, need to reshape to (1, T)
                target_array = target_array.reshape(1, -1)

            # Collect some lengths for later
            nc = target_array.shape[0]
            target_length = target_array.shape[1]
            if target_length < self.forecast_length:
                raise ValueError(
                    f"Length of entire target array ({target_length}) is less than forecast length ({self.forecast_length})"
                )
            if target_length - self.forecast_length <= 0:
                raise ValueError(
                    f"No available history length for time series of length {target_length}"
                )

            # Create a timestamp array
            timestamp_array = self._create_timestamp_array(
                sample_row["start"], sample_row["freq"], target_length
            )

            # Repeat the timestamp array for each channel
            if nc > 1:
                timestamp_array = np.concatenate(
                    [timestamp_array.reshape(1, -1)] * nc, axis=0
                )
            else:
                timestamp_array = timestamp_array.reshape(1, -1)

            assert timestamp_array.shape == target_array.shape

            # Slice out forecast portion of all channels
            target_timestamps = timestamp_array[:, -self.forecast_length :]
            target_values = target_array[:, -self.forecast_length :]

            # Slice out history portion of all channels
            if self.max_history_length is not None:
                # Use up to max_history_length, but allow less if the time series is shorter
                available_history_length = min(
                    self.max_history_length,
                    target_length - self.forecast_length,
                )
                history_timestamps = timestamp_array[
                    :,
                    -(
                        available_history_length + self.forecast_length
                    ) : -self.forecast_length,
                ]
                history_values = target_array[
                    :,
                    -(
                        available_history_length + self.forecast_length
                    ) : -self.forecast_length,
                ]
            else:
                history_timestamps = timestamp_array[:, : -self.forecast_length]
                history_values = target_array[:, : -self.forecast_length]

            yield EvalBatchFormat.from_arrays(
                sample_ids=np.full((1, nc), sample_row["item_id"]),
                history_timestamps=history_timestamps.reshape(1, nc, -1),
                history_values=history_values.reshape(1, nc, -1),
                target_timestamps=target_timestamps.reshape(1, nc, -1),
                target_values=target_values.reshape(1, nc, -1),
            )
