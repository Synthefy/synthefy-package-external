import random
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


class TaiwanAQDataloader(BaseEvalDataloader):
    """
    Dataloader for Taiwan air quality data.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/taiwan_aq/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Taiwan air quality data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df = df.sort_values("date").reset_index(drop=True)

        columns_to_remove = [
            "sitename",
            "county",
            "longitude",
            "latitude",
            "siteid",
        ]
        df = df.drop(
            columns=[col for col in columns_to_remove if col in df.columns]
        )

        # handle missing values
        df = df.replace("-", pd.NA)
        numeric_cols = df.select_dtypes(include=["object"]).columns
        for col in numeric_cols:
            if col != "date" and col not in ["pollutant", "status"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode discrete metadata columns
        discrete_cols = ["pollutant", "status"]
        df = encode_discrete_metadata(
            df, [col for col in discrete_cols if col in df.columns]
        )

        return df

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_cols = ["aqi"]

            metadata_cols = [
                "pollutant",
                "status",
                "so2",
                "co",
                "o3",
                "o3_8hr",
                "pm10",
                "pm2.5",
                "no2",
                "nox",
                "no",
                "windspeed",
                "winddirec",
                "unit",
                "co_8hr",
                "pm2.5_avg",
                "pm10_avg",
                "so2_avg",
            ]

            # Filter to only include columns that exist in the dataframe
            metadata_cols = [col for col in metadata_cols if col in df.columns]

            # a year and a month of hourly data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="date",
                num_target_rows=6480,  # 3+ month history, 9 months forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # 1 week of hourly data
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
