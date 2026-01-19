import random

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class TrafficPEMSDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.data_location = "s3://synthefy-fm-eval-datasets/traffic_PeMS/"
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        csv_files = list_s3_files(self.data_location, file_extension=".csv")
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

    def __len__(self):
        return len(self.csv_files)

    def __iter__(self):
        for file in self.csv_files:
            df = pd.read_csv(file)
            df = self._convert_columns_to_timestamps(df)

            non_timestamp_cols = [
                col for col in df.columns if col != "timestamp"
            ]
            # rename last non_timestamp_col to target_col
            df.rename(
                columns={non_timestamp_cols[-1]: "target_col"}, inplace=True
            )
            non_timestamp_cols[-1] = "target_col"

            # rename all other columns sequentially
            for i, col in enumerate(non_timestamp_cols[:-1]):
                df.rename(columns={col: f"metadata_col_{i}"}, inplace=True)
                non_timestamp_cols[i] = f"metadata_col_{i}"

            target_cols = [non_timestamp_cols[-1]]
            metadata_cols = [
                non_timestamp_cols[i]
                for i in range(len(non_timestamp_cols) - 1)
            ]

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=68,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
            )
            yield eval_batch
