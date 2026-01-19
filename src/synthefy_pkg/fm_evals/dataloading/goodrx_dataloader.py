import random
from typing import Iterator

import pandas as pd

from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat

FORECAST_FOR_SINGLE_PHARMACY = True
FORECAST_PHARMACY = (
    "a2d6ddd9262d5d9b0e259e7d21a451b59540b37beae9c9cb767acebc7f4eec62"
)
DROP_GROUPS_WITH_FEWER_THAN_10_ROWS = False


class GoodRxDataloader:
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.df = self._load_data_df()
        self.groups = self._group_and_filter_groups(self.df)
        if self.random_ordering:
            random.shuffle(self.groups)

    def _load_data_df(self) -> pd.DataFrame:
        path = "s3://goodrx-poc-datasets/GoodRxData/processed_all_pharm_bucket_weekly_agg_all.parquet"
        df = pd.read_parquet(path)

        df = df.sort_values("week_start")

        if DROP_GROUPS_WITH_FEWER_THAN_10_ROWS:
            company_counts = df["parent_company_hash"].value_counts()
            companies_to_keep = company_counts[company_counts >= 10].index
            df = df[df["parent_company_hash"].isin(companies_to_keep)]

        return df

    def _group_and_filter_groups(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        company_counts = df["parent_company_hash"].value_counts()
        companies_to_keep = company_counts[company_counts >= 10].index
        df = df[df["parent_company_hash"].isin(companies_to_keep)]

        if FORECAST_FOR_SINGLE_PHARMACY:
            df = df[df["parent_company_hash"] == FORECAST_PHARMACY]

        groups = []
        for _, group in df.groupby(
            ["parent_company_hash", "drug_id", "days_supply_bucket"]
        ):
            if len(group) >= 20:
                groups.append(group)

        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for group in self.groups:
            to_return = EvalBatchFormat.from_dfs(
                dfs=[group],
                timestamp_col="week_start",
                cutoff_date="2025-05-01",
                target_cols=["net_transactions"],
                metadata_cols=[
                    "unit_ingredient_cost",
                    "is_holiday",
                    "holiday_weight",
                ],
                leak_cols=[
                    "unit_ingredient_cost",
                    "is_holiday",
                    "holiday_weight",
                ],
                forecast_window="2W",
                stride="2W",
                sample_id_cols=[
                    "parent_company_hash",
                    "drug_id",
                    "days_supply_bucket",
                ],
            )
            if to_return is None:
                continue
            else:
                yield to_return
