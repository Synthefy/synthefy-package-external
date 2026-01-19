import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat

DATASET_INFO = {
    "energy": {
        "s3_uri": "s3://synthefy-fm-eval-datasets/gpt-synthetic/energy.csv",
        "timestamp": "timestamp",
        "group": ["city"],
        "target": ["energy_production"],
        "metadata": ["temperature", "humidity"],
    },
    "manufacturing": {
        "s3_uri": "s3://synthefy-fm-eval-datasets/gpt-synthetic/manufacturing.csv",
        "timestamp": "timestamp",
        "group": ["machine"],
        "target": ["failure_risk"],
        "metadata": ["ambient_temp", "vibration", "load_factor"],
    },
    "retail": {
        "s3_uri": "s3://synthefy-fm-eval-datasets/gpt-synthetic/retail.csv",
        "timestamp": "timestamp",
        "group": ["product"],
        "target": ["sales"],
        "metadata": ["ad_spend", "tv_show", "f1_event"],
    },
    "supply_chain": {
        "s3_uri": "s3://synthefy-fm-eval-datasets/gpt-synthetic/supply_chain.csv",
        "timestamp": "Date",
        "group": ["Region"],
        "target": ["DailyOrderCount"],
        "metadata": ["Promotion", "Holiday", "Temperature", "FuelPrice"],
    },
    "traffic": {
        "s3_uri": "s3://synthefy-fm-eval-datasets/gpt-synthetic/traffic.csv",
        "timestamp": "timestamp",
        "group": ["location"],
        "target": ["traffic_count"],
        "metadata": ["rainfall_mm", "holiday"],
    },
}


class GPTSyntheticDataloader(BaseEvalDataloader):
    def __init__(self, dataset_name: str, random_ordering: bool = False):
        self.random_ordering = random_ordering
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Dataset {dataset_name} is not valid. Valid datasets are: {list(DATASET_INFO.keys())}"
            )

        self.dataset_info = DATASET_INFO[dataset_name]
        self.df = pd.read_csv(self.dataset_info["s3_uri"])
        self.groups = self._get_groups(self.df, self.dataset_info["group"])
        if self.random_ordering:
            random.shuffle(self.groups)

    def _get_groups(
        self, df: pd.DataFrame, group_cols: list[str]
    ) -> list[pd.DataFrame]:
        return [group for _, group in df.groupby(group_cols)]

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for group in self.groups:
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[group],
                timestamp_col=self.dataset_info["timestamp"],
                cutoff_date="2024-01-01",
                target_cols=self.dataset_info["target"],
                metadata_cols=self.dataset_info["metadata"],
                sample_id_cols=self.dataset_info["group"],
                forecast_window="30D",
                stride="30D",
            )

            if eval_batch is None:
                raise ValueError(
                    f"No valid backtesting windows were created for group {group}"
                )

            yield eval_batch
