import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class MdsMicrogridDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/mds_microgrid/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601")
        except ValueError:
            try:
                df["Timestamp"] = pd.to_datetime(
                    df["Timestamp"], format="mixed"
                )
            except ValueError:
                df["Timestamp"] = pd.to_datetime(
                    df["Timestamp"], errors="coerce"
                )
                df = df.dropna(subset=["Timestamp"])

        df = df.sort_values("Timestamp").reset_index(drop=True)

        df = df.set_index("Timestamp")

        df = df.reset_index()

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_cols = ["Battery_Active_Power"]
            metadata_cols = [
                col
                for col in df.columns
                if col not in ["Timestamp", "Battery_Active_Power"]
            ]

            # a month of 10s data ~ 259,000 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Timestamp",
                num_target_rows=65000,  # 1 week
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=200,
                stride=200,
            )

            if eval_batch is not None:
                yield eval_batch
