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


class BeijingAQDataloader(BaseEvalDataloader):
    """
    Dataloader for Beijing Air Quality (PRSA_Data_Aotizhongxin_20130301-20170228.csv) data.

    This dataloader handles multivariate time series data with air quality measurements
    including PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dew point, rain,
    wind direction, and wind speed. PM2.5 is used as the target variable, while other
    parameters serve as covariates. Wind direction is encoded as discrete metadata.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/beijing_aq/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Beijing AQ data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Create datetime from year, month, day, hour columns
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

        # Drop the original date/time columns
        df = df.drop(columns=["No", "year", "month", "day", "hour"])

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        # Handle missing values - forward fill for time series data
        df = df.ffill().bfill()

        # Encode discrete metadata (wind direction)
        df = encode_discrete_metadata(df, ["wd"])

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

            # Define target and metadata columns
            target_cols = ["PM2.5"]  # PM2.5 is the target
            metadata_cols = [
                "PM10",
                "SO2",
                "NO2",
                "CO",
                "O3",
                "TEMP",
                "PRES",
                "DEWP",
                "RAIN",
                "wd",  # Now encoded as discrete metadata
                "WSPM",
            ]

            # hourly data for 4 years
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                num_target_rows=8760,  # 1 years of hourly data (365 * 24)
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],  # No leak columns for now
                sample_id_cols=["station"],  # Include station in sample ID
                forecast_window=168,  # hourly data, so week-long stride
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
