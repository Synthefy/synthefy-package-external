import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class CGMDataloader(BaseEvalDataloader):
    """
    Dataloader for CGM Dubosson data.

    This dataloader handles multivariate time series data with continuous glucose monitoring
    and related physiological parameters. The data includes glucose levels, insulin doses,
    calories, heart rate, breathing rate, posture, activity, HRV, and core temperature.
    Glucose level (gl) is used as the target variable for forecasting.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/cgm/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single CGM Dubosson data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert time column to datetime
        df["time"] = pd.to_datetime(df["time"])

        # Sort by timestamp
        df = df.sort_values(by="time").reset_index(drop=True)

        # Rename the timestamp column to a consistent name for downstream processing
        df = df.rename(columns={"time": "timestamp"})

        return df

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            # Define target and metadata columns
            target_cols = ["gl"]  # Glucose level is the target
            metadata_cols = [
                "fast_insulin",
                "slow_insulin",
                "calories",
                "balance",
                "quality",
                "HR",
                "BR",
                "Posture",
                "Activity",
                "HRV",
                "CoreTemp",
            ]

            # 5 days of 5-minute data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=576,  # 2 days of 5-minute data (2 * 24 * 12)
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                sample_id_cols=["id"],  # Include patient ID in sample ID
                forecast_window=288,  # 1 day of 5-minute data (24 * 12)
                stride=288,
            )
            if eval_batch is not None:
                yield eval_batch
