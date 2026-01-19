import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


# bavaria 175296 rows
# germany 175296 rows
# flood 3010 rows
class CausalRiversDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/causalrivers/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            # Get the 2nd column as target (index 1)
            target_col = df.columns[1]

            # Exclude datetime (1st column) and target (2nd column)
            exclude_cols = [df.columns[0], target_col]
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

            if len(metadata_cols) == 0:
                continue

            if "germany" in file_path.lower() or "bavaria" in file_path.lower():
                num_target_rows = 87648  # roughly 50%
            else:
                num_target_rows = 1000  # roughly 40%

            # 15-minute data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="datetime",
                num_target_rows=num_target_rows,
                target_cols=[target_col],
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=192,  # 2 days
                stride=192,
            )
            if eval_batch is not None:
                yield eval_batch
