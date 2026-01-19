from pathlib import Path
from typing import Iterator

import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class WikipediaDataloader(BaseEvalDataloader):
    def __init__(self):
        self.csv_files = self._collect_files()

    def _collect_files(self):
        """Collect and sort all CSV files from the three Wikipedia subfolders."""
        paths = [
            "s3://synthefy-fm-eval-datasets/wikipedia/wiki_1/",
            "s3://synthefy-fm-eval-datasets/wikipedia/wiki_2/",
            "s3://synthefy-fm-eval-datasets/wikipedia/wiki_3/",
        ]
        all_csv_files = []
        for path in paths:
            csv_files = list_s3_files(path, file_extension=".csv")
            logger.info(f"adding files from {path} with {len(csv_files)}")
            all_csv_files.extend(csv_files)
        return all_csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        timestamp_col = df.columns[0]
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y-%m-%d")
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        df = df.set_index("timestamp")

        # Data is already daily, no resampling needed
        # Reset index to get Time-ending back as a column
        df = df.reset_index()

        return df

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) < 400:
                continue

            target_cols = [df.columns[1]]  # Second column is the target column

            metadata_cols = list(df.columns[2:])

            # print(len(metadata_cols))
            # TODO: when renaming to generic names, we should do this with a flag
            df.rename(columns={target_cols[0]: "target"}, inplace=True)
            df.rename(
                columns={
                    metadata_cols[i]: f"metadata_col_{i}"
                    for i in range(len(metadata_cols))
                },
                inplace=True,
            )
            metadata_cols = [
                f"metadata_col_{i}" for i in range(len(metadata_cols))
            ]
            target_cols = ["target"]

            # Daily data for 1.5 year ~550 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=200,
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
            )
            if eval_batch is not None:
                yield eval_batch
