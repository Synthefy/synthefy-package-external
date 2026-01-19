import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class PastaSalesDataloader(BaseEvalDataloader):
    """
    Dataloader for pasta sales data.
    """

    def __init__(self, config=None, random_ordering: bool = False):
        """
        Initialize the pasta sales dataloader.
        """
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the S3 data location."""
        # S3 path where the pasta sales CSV files are stored
        path = "s3://synthefy-fm-eval-datasets/pasta_sales/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single pasta sales data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").reset_index(drop=True)

        brand1_qty_cols = [
            col for col in df.columns if col.startswith("QTY_B1_")
        ]
        target_col = brand1_qty_cols[0]
        df = df.rename(columns={target_col: "Brand 1 Sales"})

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

            target_cols = ["Brand 1 Sales"]
            metadata_cols = [
                col for col in df.columns if col not in ["DATE"] + target_cols
            ]
            leak_cols = [col for col in df.columns if col.startswith("PROMO_")]

            # 5 years of daily data (~1800)
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="DATE",
                num_target_rows=365,  # 1 year forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=leak_cols,  # Promotion columns can leak target information
                forecast_window=30,  # 1 month forecast window
                stride=30,
            )

            if eval_batch is not None:
                yield eval_batch
