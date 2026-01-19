import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class OpenAQDataloader(BaseEvalDataloader):
    """
    Dataloader for OpenAQ air quality data.

    This dataloader handles multivariate time series data with air quality measurements
    including PM1, PM2.5, relative humidity, temperature, and other parameters.
    PM2.5 is used as the target variable, while other parameters serve as covariates.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/open_aq/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single OpenAQ data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)
        df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"])
        relevant_parameters = [
            "pm1",
            "pm25",
            "relativehumidity",
            "temperature",
            "um003",
        ]
        df = df[df["parameter"].isin(relevant_parameters)]

        # Pivot the data to wide format (one column per parameter)
        df_wide = df.pivot_table(
            index=["datetimeUtc", "location_id"],
            columns="parameter",
            values="value",
            aggfunc="first",
        ).reset_index()

        df_wide.columns.name = None
        df_wide = df_wide.sort_values("datetimeUtc").reset_index(drop=True)
        df_wide = df_wide.rename(columns={"datetimeUtc": "datetime"})

        return df_wide

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

            # Define target and metadata columns
            target_cols = ["pm25"]  # PM2.5 is the target
            metadata_cols = [
                "pm1",
                "relativehumidity",
                "temperature",
                "um003",
            ]

            # 3.5 months of hourly data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                num_target_rows=720,  # a month of hourly data
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                sample_id_cols=[
                    "location_id"
                ],  # Include location_id in sample ID
                forecast_window=168,  # a week of hourly data
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
