import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class NYElectricityDataloader(BaseEvalDataloader):
    """
    Dataloader for New York electricity data.
    """

    def __init__(self, config=None, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        self.required_metadata_cols = [
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "precipitation (mm)",
            "cloud_cover (%)",
        ]
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        path = "s3://synthefy-fm-eval-datasets/ny_electricity2025/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            target_cols = ["Demand (MWh)"]
            metadata_cols = [
                "temperature_2m (°C)",
                "relative_humidity_2m (%)",
                "precipitation (mm)",
                "cloud_cover (%)",
            ]
            metadata_cols = [col for col in metadata_cols if col in df.columns]

            # hourly data for a month
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=336,  # 2 weeks
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # 1 week
                stride=168,
            )

            if eval_batch is not None:
                yield eval_batch
