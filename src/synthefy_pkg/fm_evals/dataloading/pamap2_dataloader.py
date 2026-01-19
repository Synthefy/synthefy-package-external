import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class Pamap2Dataloader(BaseEvalDataloader):
    """
    Dataloader for PAMAP2 dataset.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/pamap2/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess a CSV file from S3.

        Args:
            file_path: S3 path to the CSV file

        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(file_path)

        # Only keep rows that have a value in the heart_rate column
        # drops sampling frequency to 9Hz
        df = df.dropna(subset=["heart_rate"])

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Sort by timestamp to ensure chronological order
        df = df.sort_values("timestamp").reset_index(drop=True)

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
            target_cols = ["heart_rate"]

            # All sensor data as metadata
            metadata_cols = (
                [f"hand_{i}" for i in range(1, 14)]
                + [f"chest_{i}" for i in range(1, 14)]
                + [f"ankle_{i}" for i in range(1, 14)]
                + ["activityID"]
            )

            # sampling frequency: 9Hz
            # Use 70% of data for history, 30% for forecasting
            total_rows = len(df)
            history_rows = int(0.7 * total_rows)
            target_rows = total_rows - history_rows

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=target_rows,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=["activityID"],
                sample_id_cols=[],
            )

            if eval_batch is not None:
                yield eval_batch
