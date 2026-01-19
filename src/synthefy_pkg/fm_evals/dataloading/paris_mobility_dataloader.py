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


class ParisMobilityDataloader(BaseEvalDataloader):
    """
    Dataloader for Paris mobility data with weather information.

    This dataloader handles hourly mobility data from different regions in Paris
    combined with daily weather data. The target column is 'Commune Montreuil'
    and includes weather features like temperature, humidity, precipitation, etc.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/paris_mobility/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Paris mobility data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Hour"])
        df = df.drop(["Date", "Hour"], axis=1)

        # Remove the 'Total' column
        if "Total" in df.columns:
            df = df.drop("Total", axis=1)

        # Convert sunrise and sunset times to hour of day (numeric)
        if "sunrise" in df.columns:
            df["sunrise_hour"] = (
                pd.to_datetime(df["sunrise"]).dt.hour
                + pd.to_datetime(df["sunrise"]).dt.minute / 60.0
            )
            df = df.drop("sunrise", axis=1)

        if "sunset" in df.columns:
            df["sunset_hour"] = (
                pd.to_datetime(df["sunset"]).dt.hour
                + pd.to_datetime(df["sunset"]).dt.minute / 60.0
            )
            df = df.drop("sunset", axis=1)

        # Hardcode discrete/categorical columns for encoding
        discrete_columns = ["preciptype", "conditions", "severerisk"]

        # Encode discrete metadata columns
        if discrete_columns:
            df = encode_discrete_metadata(df, discrete_columns)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

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

            # Target column is 'Commune Montreuil'
            target_cols = ["Commune Montreuil"]
            metadata_cols = [
                col
                for col in df.columns
                if col not in ["timestamp"] + target_cols
            ]

            df.rename(
                columns={
                    col: f"IRIS_{i}"
                    for i, col in enumerate(metadata_cols)
                    if col.find("IRIS") != -1
                },
                inplace=True,
            )
            metadata_cols = [
                "IRIS_" + str(i)
                for i in range(len(metadata_cols))
                if metadata_cols[i].find("IRIS") != -1
            ] + [
                metadata_cols[i]
                for i in range(len(metadata_cols))
                if metadata_cols[i].find("IRIS") == -1
            ]

            print(metadata_cols)

            # hourly data for about 4.5 months
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=1116,  # 1.5 months of hourly data
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # a week (168 hours)
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
