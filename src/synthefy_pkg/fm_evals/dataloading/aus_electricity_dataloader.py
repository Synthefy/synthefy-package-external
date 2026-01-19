import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class AusElectricityDataloader(BaseEvalDataloader):
    """
    Dataloader for Australian electricity data.

    This dataloader handles multivariate time series data with electricity information
    from Queensland (QLD) and New South Wales (NSW). The data includes price, demand,
    generation, availability, and other electricity market metrics.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/aus_electricity/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Australian electricity data file."""
        # Load the CSV file with semicolon delimiter
        df = pd.read_csv(file_path, delimiter=";")

        timestamp_col = df.columns[0]
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], format="%d/%m/%Y %H:%M", dayfirst=True
        )
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        df = df.rename(columns={timestamp_col: "Time-ending"})

        df = df.set_index("Time-ending")

        # Resample from half-hourly to weekly data - pick first observation of each week
        weekly_df = df.resample("W").first()

        # Reset index to get Time-ending back as a column
        weekly_df = weekly_df.reset_index()

        weekly_df = weekly_df.iloc[:, :11]

        return weekly_df

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

            target_cols = [df.columns[1]]  # Second column is the price column

            metadata_cols = list(
                df.columns[2:]
            )  # All columns after the price column

            # 11 years of weekly data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Time-ending",
                num_target_rows=192,  # 3 year forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=24,  # 6 months
                stride=24,
            )
            if eval_batch is not None:
                yield eval_batch
