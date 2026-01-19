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


class BlueBikesDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/blue_bikes/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        discrete_columns = []
        if "day_of_week" in df.columns:
            discrete_columns.append("day_of_week")
        if "weather_condition" in df.columns:
            discrete_columns.append("weather_condition")

        if discrete_columns:
            df = encode_discrete_metadata(df, discrete_columns)

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

            exclude_cols = [
                "datetime",
                "count",
                "casual_riders_count",
                "member_riders_count",
            ]
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

            if len(metadata_cols) == 0:
                continue

            # hourly data for 2.5 years ~ 21,000 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                num_target_rows=4500,  # ~6 months
                target_cols=["count"],
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # 1 week
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
