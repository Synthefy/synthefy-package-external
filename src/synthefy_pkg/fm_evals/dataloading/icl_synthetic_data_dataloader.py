import random
from pathlib import Path

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class ICLSyntheticDataDataloader(BaseEvalDataloader):
    def __init__(self, data_location: str, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.data_location = data_location
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        csv_files = sorted([f for f in Path(self.data_location).glob("*.csv")])
        return csv_files

    def _convert_columns_to_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw csvs have columns for year month day hour and minute.
        Convert these to timestamps.
        """
        df["timestamp"] = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute"]]
        )
        df = df.drop(columns=["year", "month", "day", "hour", "minute"])
        return df

    def __len__(self):
        return len(self.csv_files)

    def __iter__(self):
        for file in self.csv_files:
            df = pd.read_csv(file)
            df = self._convert_columns_to_timestamps(df)

            non_timestamp_cols = [
                col for col in df.columns if col != "timestamp"
            ]

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=128,
                target_cols=non_timestamp_cols,
                metadata_cols=non_timestamp_cols,
            )
            yield eval_batch
