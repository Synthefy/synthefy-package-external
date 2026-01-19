import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import (
    encode_discrete_metadata,
    list_s3_files,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class BeijingEmbassyDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = (
            self._collect_files()
        )  # this is currently just one file w 5 years of data
        random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""

        path = "s3://synthefy-fm-eval-datasets/beijing_embassy/"
        csv_files = list_s3_files(path, file_extension=".csv")

        return csv_files

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single CSV file."""
        df = pd.read_csv(file_path)

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Sort by timestamp
        df = df.sort_values(by="date").reset_index(drop=True)

        return df

    def __len__(self):
        assert self.csv_files is not None, "csv_files is None"
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        # Define columns
        target_cols = ["pollution"]
        metadata_cols = [
            "dew",
            "temp",
            "press",
            "wnd_dir",
            "wnd_spd",
            "snow",
            "rain",
        ]

        assert self.csv_files is not None, "csv_files is None"
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = self._load_data(file_path)
            df = encode_discrete_metadata(df, columns=["wnd_dir"])

            if len(df) == 0:
                continue

            total_rows = len(df)
            target_rows = total_rows - int(
                0.8 * total_rows
            )  # 4 years history, 1 year forecast

            # hourly data, so week-long stride
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="date",
                num_target_rows=target_rows,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # hourly data, so week-long stride
                stride=168,
            )

            if eval_batch is not None:
                yield eval_batch
