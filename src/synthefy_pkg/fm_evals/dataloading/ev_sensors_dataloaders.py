import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class EVSensorsDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/ev-sensors/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["datetime"] = pd.to_datetime(df["datetime"])

        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.set_index("datetime")
        df = df.reset_index()

        for col in df.columns:
            if col != "datetime" and df[col].dtype == "object":
                # Replace 'P0MR' and other non-numeric strings with 0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_cols = ["SOC"]
            metadata_cols = [
                col for col in df.columns if col not in ["datetime", "SOC"]
            ]

            # hourly data for 5 years ~ 43,800 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                num_target_rows=17520,  # 2 years
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # a week
                stride=168,
            )

            if eval_batch is not None:
                yield eval_batch
