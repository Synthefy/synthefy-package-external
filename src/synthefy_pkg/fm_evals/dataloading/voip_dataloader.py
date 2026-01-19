import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class VoIPDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/voip/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        # create simulated time points starting from a random time and updated hourly
        start_time = pd.to_datetime("2020-01-01") + pd.Timedelta(
            days=np.random.randint(0, 1500)
        )
        df["Time"] = pd.date_range(start=start_time, periods=len(df), freq="H")

        # df["Time"] = pd.to_datetime(df["Time"])

        df = df.sort_values("Time").reset_index(drop=True)

        df = df.set_index("Time")

        df = df.reset_index()

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_cols = ["MOS"]
            metadata_cols = [
                col for col in df.columns if col not in ["Time", "MOS"]
            ]

            # Set num_target_rows to 30% of total rows
            num_target_rows = int(len(df) * 0.3)

            # Set forecast_window and stride based on num_target_rows
            if num_target_rows > 200:
                forecast_window = 200
                stride = 200
            else:
                forecast_window = num_target_rows
                stride = num_target_rows

            # a month of 10s data ~ 259,000 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Time",
                num_target_rows=num_target_rows,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=forecast_window,
                stride=stride,
            )

            if eval_batch is not None:
                yield eval_batch
