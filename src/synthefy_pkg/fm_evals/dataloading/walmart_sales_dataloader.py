import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import list_s3_files
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class WalmartSalesDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.df = pd.read_csv(
            "s3://synthefy-fm-eval-datasets/walmart-sales/walmart_sales.csv"
        )
        self.groups = [
            group for _, group in self.df.groupby("Store") if len(group) > 30
        ]
        if self.random_ordering:
            random.shuffle(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for group in self.groups:
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[group],
                timestamp_col="Date",
                num_target_rows=30,
                target_cols=["Weekly_Sales"],
                metadata_cols=[
                    "Holiday_Flag",
                    "Temperature",
                    "Fuel_Price",
                    "CPI",
                    "Unemployment",
                ],
                sample_id_cols=["Store"],
            )

            if eval_batch is None:
                continue

            yield eval_batch


if __name__ == "__main__":
    dl = WalmartSalesDataloader()
    for eval_batch in dl:
        print(eval_batch)
