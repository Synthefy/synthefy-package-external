import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class TetuanPowerDataloader(BaseEvalDataloader):
    """
    Dataloader for Tetuan City power consumption data.

    This dataloader handles multivariate time series data with power consumption
    from three different zones in Tetuan City. The data includes temperature,
    humidity, wind speed, and diffuse flows as metadata, with zone power consumption
    as the target variable.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/tetuan_power/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Tetuan power data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values("DateTime").reset_index(drop=True)

        # Downsample to hourly (take every 6 rows)
        df = df.iloc[::6].reset_index(drop=True)

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

            target_cols = ["Zone Power Consumption"]

            metadata_cols = [
                "Temperature",
                "Humidity",
                "Wind Speed",
                "general diffuse flows",
                "diffuse flows",
            ]

            # hourly data for about 1 year
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="DateTime",
                num_target_rows=6480,  # 3 months history, 9 months forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # 1 week of hourly data
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
