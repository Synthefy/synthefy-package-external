import pandas as pd

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class ComplexSeasonalTimeseriesDataloader(BaseEvalDataloader):
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        self.csv_path = "/home/synthefy/data/nonlinear_seasonal_time_series.csv"
        self.df = pd.read_csv(self.csv_path)

    def __len__(self):
        return 1

    def __iter__(self):
        to_ret = EvalBatchFormat.from_dfs(
            dfs=[self.df],
            num_target_rows=90,
            timestamp_col="time",
            target_cols=["value"],
            metadata_cols=["seasonality", "trend", "seasonal_period"],
        )

        if not to_ret:
            raise ValueError("No data loaded")

        yield to_ret


if __name__ == "__main__":
    dataloader = ComplexSeasonalTimeseriesDataloader()
    for batch in dataloader:
        print(batch)
        break
