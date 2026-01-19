import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class ERCOTLoadDataloader(BaseEvalDataloader):
    """
    Dataloader for ERCOT (Electric Reliability Council of Texas) load data.

    This dataloader handles multivariate time series data with load information
    from different regions in Texas. The data includes hourly load values for
    8 regions plus the total ERCOT load.
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""

        path = "s3://synthefy-fm-eval-datasets/ercot_load/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single ERCOT load data file."""
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Verify COAST is the second column (index 1)
        if df.columns[1] != "COAST":
            raise ValueError(
                f"Expected COAST as second column, but found: {df.columns[1]}"
            )

        # Use first column as timestamp column regardless of its name
        timestamp_col = df.columns[0]

        # Convert timestamp column - handle "24:00" format
        def parse_timestamp(ts_str):
            if "24:00" in ts_str:
                # Replace "24:00" with "00:00" and add one day
                ts_str = ts_str.replace("24:00", "00:00")
                dt = pd.to_datetime(ts_str)
                return dt + pd.Timedelta(days=1)
            else:
                return pd.to_datetime(ts_str)

        df[timestamp_col] = df[timestamp_col].apply(parse_timestamp)

        # Process all columns except the first (timestamp) and last (ERCOT)
        # Columns 1 to -2 (excluding the last column)
        for i in range(1, len(df.columns) - 1):
            col_name = df.columns[i]
            df[col_name] = (
                df[col_name].astype(str).str.replace(",", "").astype(float)
            )

        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        # Rename the timestamp column to a consistent name for downstream processing
        df = df.rename(columns={timestamp_col: "Hour Ending"})

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

            # Use column positions instead of names
            # COAST is the second column (index 1)
            target_cols = [df.columns[1]]  # COAST

            # Metadata columns are columns 2 to -2 (excluding the last ERCOT column)
            metadata_cols = list(df.columns[2:-1])

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="Hour Ending",
                num_target_rows=6576,  # 3 months history, 9 months forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=168,  # a week
                stride=168,
            )
            if eval_batch is not None:
                yield eval_batch
