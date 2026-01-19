import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class FredMdDataloader(BaseEvalDataloader):
    def __init__(self, dataset_name: str, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.dataset_name = dataset_name
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        path = f"s3://synthefy-fm-eval-datasets/fred_md/{self.dataset_name}/"
        csv_files = list_s3_files(path, file_extension=".csv")
        return csv_files

    def _get_target_col_from_filename(self) -> str:
        if self.dataset_name == "fred_md1":
            target_col = "RPI"
        elif self.dataset_name == "fred_md2":
            target_col = "UNRATE"
        elif self.dataset_name == "fred_md3":
            target_col = "HOUST"
        elif self.dataset_name == "fred_md4":
            target_col = "UMCSENTx"
        elif self.dataset_name == "fred_md5":
            target_col = "TOTRESNS"
        elif self.dataset_name == "fred_md6":
            target_col = "FEDFUNDS"
        elif self.dataset_name == "fred_md7":
            target_col = "CPIAUCSL"
        elif self.dataset_name == "fred_md8":
            target_col = "S&P 500"
        else:
            target_col = "error"

        return target_col

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["sasdate"] = pd.to_datetime(df["sasdate"])
        df = df.sort_values("sasdate").reset_index(drop=True)

        return df

    def __len__(self) -> int:
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for file_path in self.csv_files:
            df = self._load_and_preprocess_data(file_path)

            if len(df) == 0:
                continue

            target_col = self._get_target_col_from_filename()
            if target_col == "error":
                target_col = df.columns[1]

            exclude_cols = ["sasdate", target_col]
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

            if len(metadata_cols) == 0:
                continue

            # monthly data 1959-2024 ~ 800 rows
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="sasdate",
                num_target_rows=360,  # 30 years
                target_cols=[target_col],
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=60,  # 5 years
                stride=60,
            )
            if eval_batch is not None:
                yield eval_batch
