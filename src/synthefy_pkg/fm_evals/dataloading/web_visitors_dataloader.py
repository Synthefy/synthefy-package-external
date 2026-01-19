import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class WebVisitorsDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/website_visitors/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Convert comma-separated numeric columns to float
        numeric_cols = [
            "Page.Loads",
            "Unique.Visits",
            "First.Time.Visits",
            "Returning.Visits",
        ]
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert to numeric, handling any errors
                df[col] = df[col].astype(str).str.replace(",", "").astype(float)

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

            exclude_cols = ["Date", "Page.Loads", "Unique.Visits", "Row", "Day"]
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

            if len(metadata_cols) == 0:
                continue

            # daily data ~ 2168 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Date",
                num_target_rows=672,  # 4 weeks
                target_cols=["Page.Loads"],
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # 1 week
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
