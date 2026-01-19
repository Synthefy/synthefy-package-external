import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class CursorTabsDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_files = list_s3_files(
            "s3://synthefy-fm-eval-datasets/cursor-tabs/"
        )
        if self.random_ordering:
            random.shuffle(self.csv_files)
        self.dfs = [pd.read_csv(file) for file in self.csv_files]

    def __len__(self) -> int:
        return len(self.dfs)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for df in self.dfs:
            groups = [
                group for _, group in df.groupby("User ID") if len(group) > 14
            ]

            metadata_cols = [
                "Chat Suggested Lines Added",
                "Chat Suggested Lines Deleted",
                "Chat Accepted Lines Added",
                "Chat Accepted Lines Deleted",
                "Chat Total Applies",
                "Chat Total Accepts",
                "Chat Total Rejects",
            ]

            eval_batch = EvalBatchFormat.from_dfs(
                dfs=groups,
                timestamp_col="Date",
                num_target_rows=7,
                target_cols=["Tabs Accepted"],
                metadata_cols=metadata_cols,
                sample_id_cols=["Email"],
            )

            if eval_batch is None:
                continue

            yield eval_batch
