import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class MTARidershipDataloader(BaseEvalDataloader):
    """
    Dataloader for MTA (Metropolitan Transportation Authority) ridership data.
    """

    def __init__(self, config=None, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/mta_ridership/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single MTA ridership data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.rename(
            columns={"Subways: Total Estimated Ridership": "Subway Ridership"}
        )
        df = df.sort_values("Date").reset_index(drop=True)

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

            target_cols = ["Subway Ridership"]
            metadata_cols = [
                col
                for col in df.columns
                if col not in ["Date", "Year", "Subway Ridership"]
            ]

            # daily data for a year
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Date",
                num_target_rows=120,  # 4 months of target data
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=60,  # 2 month stride
                stride=60,  # 2 month stride
            )
            if eval_batch is not None:
                yield eval_batch
