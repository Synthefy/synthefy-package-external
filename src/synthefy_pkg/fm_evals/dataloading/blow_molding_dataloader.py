import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class BlowMoldingDataloader(BaseEvalDataloader):
    """
    Dataloader for Blow Molding market data.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/blow_molding/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single blow molding data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

        # Sort by timestamp
        df = df.sort_values(by="Date").reset_index(drop=True)

        # Rename the timestamp column to a consistent name for downstream processing
        df = df.rename(columns={"Date": "timestamp"})

        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=["object"]).columns
        for col in numeric_columns:
            if col != "timestamp":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how="all")

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

            target_cols = ["Domestic Market Blow Molding, Low Price"]

            all_columns = list(df.columns)
            metadata_cols = [
                col
                for col in all_columns
                if col not in ["timestamp"] + target_cols
            ]

            # 23 years of monthly data (276)
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=24,  # 2 years of monthly data
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                sample_id_cols=[],
                forecast_window=6,  # 6 months forecast window
                stride=6,
            )

            if eval_batch is not None:
                yield eval_batch
