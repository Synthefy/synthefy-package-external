import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class TACDataloader(BaseEvalDataloader):
    """
    Dataloader for TAC (Total Activity Count) data.

    This dataloader handles multivariate time series data with TAC values and
    accelerometer data (x, y, z coordinates). The TAC value is used as the target
    variable for forecasting, while accelerometer data serves as metadata.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/tac/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single TAC data file."""
        # Load only the columns we need: timestamp, tac, x, y, z
        df = pd.read_csv(file_path, usecols=["timestamp", "tac", "x", "y", "z"])

        # Convert timestamp column to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Sort by timestamp
        df = df.sort_values(by="timestamp").reset_index(drop=True)

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
            target_cols = ["tac"]  # TAC is the target variable
            metadata_cols = ["x", "y", "z"]  # Accelerometer data as metadata

            # 30-min data over 20-ish hours
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=10,  # forecast 5
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],  # No leak columns
                sample_id_cols=[],
                # no striding
            )

            if eval_batch is not None:
                yield eval_batch
