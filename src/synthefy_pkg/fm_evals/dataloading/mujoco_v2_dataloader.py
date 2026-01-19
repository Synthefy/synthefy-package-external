import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import (
    add_noise,
    list_s3_files,
    lowpass_filter,
    resample_df,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class MujocoV2Dataloader(BaseEvalDataloader):
    """
    Dataloader for Mujoco Halfcheetah Medium V2 data.

    This dataloader handles the Mujoco Halfcheetah Medium V2 dataset.
    This is an RL domain, from a policy that is either expert or partially trained
    """

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        if not hasattr(self, "paths"):
            self.paths = []
            raise ValueError("paths must be set in the subclass")
        if not hasattr(self, "target_col"):
            self.target_col = []
            raise ValueError("target_col must be set in the subclass")
        if not hasattr(self, "keep_cols"):
            self.keep_cols = []
            raise ValueError("keep_cols must be set in the subclass")
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        paths = self.paths
        all_csv_files = []
        for path in paths:
            new_files = list_s3_files(path, file_extension=".csv")
            logger.info("adding files from", path, len(new_files))
            all_csv_files.extend(new_files)
        csv_files = all_csv_files
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Mujoco V2 data file."""
        # Load the CSV file with semicolon delimiter
        df = pd.read_csv(file_path, delimiter=",")

        timestamp_col = df.columns[-1]

        # Parse timestamps with error handling to filter out malformed data
        try:
            # First try with the expected format
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col],
                format="%Y-%m-%d %H:%M:%S",
                dayfirst=True,
                errors="coerce",
            )
        except ValueError:
            # If that fails, try more flexible parsing
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col], errors="coerce"
            )

        # Remove rows with invalid timestamps (NaT values)
        initial_rows = len(df)
        df = df.dropna(subset=[timestamp_col])
        dropped_rows = initial_rows - len(df)
        logger.info(
            f"Dropped {dropped_rows} rows with invalid timestamps from {file_path} with rows {len(df)}"
        )

        if dropped_rows > 0:
            logger.info(
                f"Warning: Dropped {dropped_rows} rows with invalid timestamps from {file_path}"
            )

        if len(df) == 0:
            logger.info(
                f"Error: No valid data remaining after timestamp filtering in {file_path}"
            )
            return pd.DataFrame()

        df = df.sort_values(timestamp_col).reset_index(drop=True)
        df = df.rename(columns={timestamp_col: "timestamp"})

        # add noise (an example: add 10% relative Gaussian noise to obs_* columns)

        # # Down sampling (example: resample every 5 minutes; or stride sampling every 10 rows)
        # df = resample_df(
        #     df,
        #     time_col="timestamp",
        #     freq="5T",
        #     agg="mean",
        #     upsample_interpolate="linear",
        #     keep_non_numeric="first",
        # )

        # low-pass filter: below is an example: moving-average window=5
        obs_cols = [c for c in df.columns if c.startswith("obs_")]
        print("obs_cols", obs_cols)
        df = lowpass_filter(df, columns=obs_cols, method="gaussian", window=20)

        # add noise
        obs_cols = [c for c in df.columns if c.startswith("obs_")]
        df = add_noise(
            df,
            columns=obs_cols,
            dist="gaussian",
            level=0.05,
            mode="relative",
            random_state=42,
        )

        # Don't set timestamp as index - keep it as a column for EvalBatchFormat
        # df = df.set_index("timestamp")
        logger.info(
            f"Loaded {len(df)} rows from {file_path} with cols {len(df.columns)}"
        )
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(
            f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )

        return df

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = self._load_and_preprocess_data(file_path)

            if len(df) < 500:
                # drop short trajectories
                continue

            target_cols = [
                str(df.columns[self.target_col])
            ]  # The last observation is the target column

            metadata_cols = [df.columns[c] for c in self.keep_cols]

            # 11 years of weekly data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=128,  # 3 year forecast
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=128,
                stride=128,
            )
            if eval_batch is not None:
                yield eval_batch


class MujocoHalfCheetahV2Dataloader(MujocoV2Dataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_col = 16
        self.keep_cols = list(range(16)) + list(range(17, 23))
        self.paths = [
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-random-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-expert-v2/",
        ]
        super().__init__(random_ordering=random_ordering)


class MujocoAntV2Dataloader(MujocoV2Dataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_col = 26
        self.keep_cols = list(range(26)) + list(range(27 + 84, 27 + 84 + 9))
        self.paths = [
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-random-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-expert-v2/",
        ]
        super().__init__(random_ordering=random_ordering)


class MujocoHopperV2Dataloader(MujocoV2Dataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_col = 10
        self.keep_cols = list(range(10)) + list(range(11, 11 + 3))
        self.paths = [
            # "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-random-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-expert-v2/",
        ]
        super().__init__(random_ordering=random_ordering)


class MujocoWalker2dV2Dataloader(MujocoV2Dataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_col = 16
        self.keep_cols = list(range(16)) + list(range(17, 17 + 5))
        self.paths = [
            # "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-random-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-replay-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-expert-v2/",
            "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-expert-v2/",
        ]
        super().__init__(random_ordering=random_ordering)
