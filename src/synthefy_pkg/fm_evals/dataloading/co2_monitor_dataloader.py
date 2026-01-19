import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class CO2MonitorDataloader(BaseEvalDataloader):
    """
    Dataloader for CO2 Monitor data from SML (Smart Living Lab).

    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/co2_monitor/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Create datetime from Date and Time columns
        df["datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str)
        )
        df = df.drop(columns=["Date", "Time"])
        df = df.sort_values("datetime").reset_index(drop=True)

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

            co2_cols = [
                col for col in df.columns if col.startswith("Carbon dioxide")
            ]
            if not co2_cols:
                print(f"Warning: No CO2 column found in {file_path}")
                continue

            target_cols = co2_cols
            metadata_cols = [
                col
                for col in df.columns
                if col not in co2_cols and col != "datetime"
            ]

            # Calculate cutoff date as last week
            end_date = df["datetime"].max()
            cutoff_date = end_date - pd.Timedelta(days=7)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                cutoff_date=cutoff_date_str,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=["Day of the week"],
                sample_id_cols=[],
                forecast_window="1D",  # 1 day
                stride="1D",  # 1 day
            )
            if eval_batch is not None:
                yield eval_batch
