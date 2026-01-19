import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class ECLDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = "s3://synthefy-fm-eval-datasets/ecl/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, low_memory=False)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        df = df.set_index("datetime")

        df = df.reset_index()

        # Convert European decimal format (comma as decimal separator) to float
        for col in df.columns:
            if col != "datetime" and df[col].dtype == "object":
                # Check if the column contains any values with comma decimal separators
                has_comma_decimals = df[col].str.contains(r",", na=False).any()
                if has_comma_decimals:
                    # Replace comma with dot and convert to float, handling NaN values
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", "."), errors="coerce"
                    )
                else:
                    # Try to convert to numeric anyway in case it's already in correct format
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_cols = ["MT_001"]
            metadata_cols = [
                col for col in df.columns if col not in ["datetime", "MT_001"]
            ]

            # because the dataframe is so large, subdivide the dataframe into 1024 length chunks, and subsample the columns to a random subset of 32
            df_chunks = [df.iloc[i : i + 1024] for i in range(0, len(df), 1024)]
            if self.random_ordering:
                random.shuffle(df_chunks)
            for chunk in df_chunks:
                sampled_cols = random.sample(metadata_cols, 50)
                subchunk = chunk[["datetime"] + sampled_cols + target_cols]
                renamed_metadata_cols = [
                    f"metadata_col_{i}" for i in range(len(sampled_cols))
                ]
                # rename metadata cols to generic names
                subchunk.rename(
                    columns={
                        col: f"metadata_col_{i}"
                        for i, col in enumerate(sampled_cols)
                    },
                    inplace=True,
                )

                eval_batch = EvalBatchFormat.from_dfs(
                    dfs=[subchunk],
                    timestamp_col="datetime",
                    num_target_rows=1024,
                    target_cols=target_cols,
                    metadata_cols=renamed_metadata_cols,
                    leak_cols=[],
                    forecast_window=192,  # 2 days
                    stride=192,
                )

                if eval_batch is not None:
                    yield eval_batch

            # # 15-minute data for 3 years ~ 105,000 rows
            # eval_batch = EvalBatchFormat.from_dfs(
            #     dfs=[df],
            #     timestamp_col="datetime",
            #     num_target_rows=245280,  # 1 year forecast
            #     target_cols=target_cols,
            #     metadata_cols=metadata_cols,
            #     leak_cols=[],
            #     forecast_window=192,  # 2 days
            #     stride=192,
            # )

            # if eval_batch is not None:
            #     yield eval_batch
