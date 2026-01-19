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


class MNInterstateDataloader(BaseEvalDataloader):
    """
    Dataloader for MN Interstate Traffic Volume data.

    This dataloader handles multivariate time series data with traffic volume information
    from the Metro Interstate Traffic Volume dataset. The data includes hourly traffic volume
    along with weather conditions, temperature, and holiday information.

    Target variable: traffic_volume
    Metadata variables: temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, holiday
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/mn_interstate/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single MN Interstate data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert date_time column to datetime
        df["date_time"] = pd.to_datetime(df["date_time"])

        # Sort by timestamp
        df = df.sort_values(by="date_time").reset_index(drop=True)

        # Rename the timestamp column to a consistent name for downstream processing
        df = df.rename(columns={"date_time": "timestamp"})

        # Encode categorical variables
        categorical_cols = ["holiday", "weather_main", "weather_description"]
        df = encode_discrete_metadata(df, categorical_cols)

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
            target_cols = ["traffic_volume"]  # Traffic volume is the target
            metadata_cols = [
                "temp",
                "rain_1h",
                "snow_1h",
                "clouds_all",
                "weather_main",  # Now encoded as discrete metadata
                "weather_description",  # Now encoded as discrete metadata
                "holiday",  # Now encoded as discrete metadata
            ]

            # hourly data for 1 year
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=4464,  # forecast 6 months of hourly data
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=["holiday"],
                forecast_window=168,  # hourly data, so week-long stride
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
