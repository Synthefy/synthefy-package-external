import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.dataloader_utils import (
    add_noise,
    list_s3_files,
    lowpass_filter,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class KITTIDataloader:
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target_1", "target_2"]
        self.keep_cols = [
            "track_1_x",
            "track_1_y",
            "track_2_x",
            "track_2_y",
            "track_3_x",
            "track_3_y",
            "track_4_x",
            "track_4_y",
        ]
        self.timestamp_col = "date"
        self.paths = ["s3://synthefy-fm-eval-datasets/KITTI/"]
        self.noise_level = 0.0
        self.resample_freq = None
        self.lowpass_window = 0
        self.random_ordering = random_ordering

        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self):
        """Collect and sort all CSV files in the data location."""
        paths = self.paths
        all_csv_files = []
        for path in paths:
            new_files = list_s3_files(path, file_extension=".csv")
            logger.info(f"adding files from {path} with {len(new_files)}")
            all_csv_files.extend(new_files)
        csv_files = all_csv_files
        return csv_files

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = df = pd.read_csv(file_path, delimiter=",")
            # low-pass filter: below is an example: moving-average window=5
            if self.lowpass_window > 0:
                obs_cols = [
                    c
                    for c in df.columns
                    if c in self.keep_cols + self.target_cols
                ]
                df = lowpass_filter(
                    df,
                    columns=obs_cols,
                    method="moving_average",
                    window=self.lowpass_window,
                )

            # add noise (an example: add 10% relative Gaussian noise to obs_* columns)
            if self.noise_level > 0:
                obs_cols = [
                    c
                    for c in df.columns
                    if c in self.keep_cols + self.target_cols
                ]
                df = add_noise(
                    df,
                    columns=obs_cols,
                    dist="gaussian",
                    level=self.noise_level,
                    mode="relative",
                    random_state=42,
                )

            if len(df) < 50:
                # drop short trajectories
                continue

            target_cols = (
                self.target_cols
            )  # The last observation is the target column

            df = df.drop(columns=["Unnamed: 0"])

            metadata_cols = [
                self.keep_cols[i] for i in range(len(self.keep_cols))
            ]
            print("before renaming", df.columns)
            df.rename(
                columns={
                    col: f"track_{(i // 2)}_{'x' if i % 2 == 0 else 'y'}"
                    for i, col in enumerate(df.columns)
                    if col not in ["timestamp", "target_1", "target_2"]
                },
                inplace=True,
            )

            HIGHEST_NUM_TRACKS = 10
            # append nan columns if len(df.columns) < 9
            if len(df.columns) < HIGHEST_NUM_TRACKS:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            np.nan,
                            index=df.index,
                            columns=[
                                f"track_{(i // 2)}_{'x' if i % 2 == 0 else 'y'}"
                                for i in range(
                                    len(df.columns), HIGHEST_NUM_TRACKS
                                )
                            ],
                        ),
                    ],
                    axis=1,
                )
            print("after renaming", df.columns)
            timestamp_col = "timestamp"
            # create random timestamps starting from a random date at 1s interval
            df[timestamp_col] = pd.date_range(
                start=pd.to_datetime("2025-01-01")
                + pd.Timedelta(seconds=np.random.randint(0, 3600 * 24 * 365)),
                periods=len(df),
                freq="1s",
            )

            # 11 years of weekly data
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col="timestamp",
                num_target_rows=48,  # trajectories may be only 100
                target_cols=target_cols,
                metadata_cols=metadata_cols,
                leak_cols=[],
                forecast_window=48,
                stride=48,
            )
            if eval_batch is not None:
                yield eval_batch

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return len(self.csv_files)
