import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class GasSensorDataloader(BaseEvalDataloader):
    """
    Dataloader for gas sensor data.

    This dataloader handles multivariate time series data from gas sensors.
    The data includes gas concentrations (methane, ethylene, CO) and sensor readings
    from 16 different sensor channels. Time is measured in seconds.
    """

    def __init__(self, random_ordering: bool = False):
        """
        Initialize the gas sensor dataloader.
        """
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the S3 data location."""
        s3_path = "s3://synthefy-fm-eval-datasets/gas_sensor/"
        csv_files = list_s3_files(s3_path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single gas sensor data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        df = df.dropna()
        reference_date = pd.Timestamp("2023-01-01 00:00:00")
        df["timestamp"] = reference_date + pd.to_timedelta(
            df["Time (seconds)"], unit="s"
        )

        # Downsample the data to reduce frequency
        # Take every 100th row to reduce from 10ms to 1 second intervals
        df = df.iloc[::100].reset_index(drop=True)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def _get_column_configuration(self, df: pd.DataFrame, file_path: str):
        """
        Determine target and metadata columns based on the file type.
        """
        if "methane" in file_path.lower():
            target_cols = ["Methane conc (ppm)"]
        elif "co" in file_path.lower():
            target_cols = ["CO conc (ppm)"]
        else:
            target_cols = [col for col in df.columns if "conc" in col.lower()]

        # All sensor columns and ethylene are metadata
        metadata_cols = [
            col
            for col in df.columns
            if col.startswith("Sensor_") or "Ethylene" in col
        ]

        return target_cols, metadata_cols

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

            target_cols, metadata_cols = self._get_column_configuration(
                df, file_path
            )

            # data every second for 12 hours
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=2000,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=200,
                stride=200,
            )
            if eval_batch is not None:
                yield eval_batch
