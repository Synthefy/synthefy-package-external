import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class DailyElectricityDataloader(BaseEvalDataloader):
    def __init__(self, dataset_name: str, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files(dataset_name)
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self, dataset_name: str):
        path = (
            f"s3://synthefy-fm-eval-datasets/daily_electricity/{dataset_name}/"
        )
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)

        return df

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            if "Demand (MWh)" not in df.columns:
                continue
            exclude_cols = ["Timestamp", "Demand (MWh)"]
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

            if len(metadata_cols) == 0:
                continue

            # daily data for a year
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Timestamp",
                num_target_rows=90,  # 3 months
                target_cols=["Demand (MWh)"],
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=30,  # 1 month
                stride=30,
            )
            if eval_batch is not None:
                yield eval_batch
