import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class SyntheticMediumLagDataloader(BaseEvalDataloader):
    def __init__(
        self,
        data_location: str,
        num_target_rows: int = 230,
        max_length: Optional[int] = None,
        random_ordering: bool = False,
    ):
        self.random_ordering = random_ordering
        self.data_location = data_location
        self.num_target_rows = num_target_rows
        self.max_length = max_length
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        csv_files = sorted([f for f in Path(self.data_location).glob("*.csv")])
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_location}"
            )
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

    def _create_timestamp_array(
        self, start_date: str, frequency: str, length: int
    ) -> np.ndarray:
        return pd.date_range(
            start=start_date, periods=length, freq=frequency
        ).to_numpy()

    def __len__(self):
        if self.max_length is not None:
            return min(self.max_length, len(self.csv_files))
        return len(self.csv_files)

    def __iter__(self):
        for file in self.csv_files:
            df = pd.read_csv(file)

            # remove columns
            columns_to_remove = []
            for col in df.columns:
                is_feature_col = "feat" in col
                is_target_col = "target" in col
                if is_feature_col or is_target_col:
                    continue
                columns_to_remove.append(col)

            # create fake timestamps
            df["timestamp"] = self._create_timestamp_array(
                start_date="2020-01-01", frequency="1H", length=len(df)
            )

            non_timestamp_cols = [
                col
                for col in df.columns
                if col != "timestamp" and col not in columns_to_remove
            ]
            target_cols = [non_timestamp_cols[-1]]
            metadata_cols = [
                non_timestamp_cols[i]
                for i in range(len(non_timestamp_cols) - 1)
            ]
            leak_cols = metadata_cols

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=self.num_target_rows,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=leak_cols,
            )
            yield eval_batch
