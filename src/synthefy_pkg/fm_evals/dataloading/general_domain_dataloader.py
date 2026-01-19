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


class GeneralizedDomainDataloader(BaseEvalDataloader):
    """
    Dataloader for a generalized domain (vision, language, video, etc.).

    This dataloader handles the generalized domain dataset.
    This is a domain that is either expert or partially trained
    """

    def __init__(self, random_ordering: bool = False):
        if not hasattr(self, "timestamp_col"):
            self.timestamp_col = None
        if not hasattr(self, "noise_level"):
            self.noise_level = 0.0
        if not hasattr(self, "resample_freq"):
            self.resample_freq = None
        if not hasattr(self, "lowpass_window"):
            self.lowpass_window = 0
        self.random_ordering = random_ordering
        if not hasattr(self, "paths"):
            self.paths = []
            raise ValueError("paths must be set in the subclass")
        if not hasattr(self, "target_cols"):
            self.target_cols = []
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
            logger.info(f"adding files from {path} with {len(new_files)}")
            all_csv_files.extend(new_files)
        csv_files = all_csv_files
        return csv_files

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single Mujoco V2 data file."""
        # Load the CSV file with semicolon delimiter
        df = pd.read_csv(file_path, delimiter=",")

        timestamp_col = (
            "timestamp" if self.timestamp_col is None else self.timestamp_col
        )

        logger.info(f"Columns: {df.columns}")

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
        except KeyError:
            logger.warning(
                f"Timestamp column {timestamp_col} not found in {self.timestamp_col} and df.columns {df.columns}"
            )
        finally:
            # create synthetic timestamps randomly, using either daily, hourly, or minutely or secondly
            frequency = np.random.choice(["D", "H", "T", "S"])
            # choose a random start date between 2020-01-01 and 2025-01-01
            start_date = pd.to_datetime("2005-01-01") + pd.Timedelta(
                days=np.random.randint(0, 365 * 5)
            )
            df["timestamp"] = pd.date_range(
                start=start_date, periods=len(df), freq=frequency
            )
            timestamp_col = "timestamp"

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

        if self.resample_freq is not None:
            # Down sampling (example: resample every 5 minutes; or stride sampling every 10 rows)
            df = resample_df(
                df,
                time_col="timestamp",
                freq=self.resample_freq,
                agg="mean",
                upsample_interpolate="linear",
                keep_non_numeric="first",
            )

        # low-pass filter: below is an example: moving-average window=5
        if self.lowpass_window > 0:
            obs_cols = [
                c for c in df.columns if c in self.keep_cols + self.target_cols
            ]
            df = lowpass_filter(
                df,
                columns=obs_cols,
                method="gaussian",
                window=self.lowpass_window,
            )

        # add noise (an example: add 10% relative Gaussian noise to obs_* columns)
        if self.noise_level > 0:
            obs_cols = [
                c for c in df.columns if c in self.keep_cols + self.target_cols
            ]
            df = add_noise(
                df,
                columns=obs_cols,
                dist="gaussian",
                level=self.noise_level,
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

            target_cols = (
                self.target_cols
            )  # The last observation is the target column

            metadata_cols = self.keep_cols

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


class SpriteworldDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = [
            "Target_inst0_feature0",
            "Target_inst1_feature1",
            "Target_inst2_feature2",
            "Target_inst3_feature3",
        ]
        self.keep_cols = [
            "Action_inst0_feature0",
            "Action_inst1_feature1",
            "Ball0_feature0",
            "Ball0_feature1",
            "Ball0_feature2",
            "Ball0_feature3",
            "Ball0_feature4",
            "Ball1_feature0",
            "Ball1_feature1",
            "Ball1_feature2",
            "Ball1_feature3",
            "Ball1_feature4",
            "Control_inst0_feature0",
            "Control_inst1_feature1",
            "Control_inst2_feature2",
            "Control_inst3_feature3",
            "Poly3vert0form_feature0",
            "Poly3vert0form_feature1",
            "Poly3vert0form_feature2",
            "Poly3vert0form_feature3",
            "Poly3vert0form_feature4",
            "Poly3vert0form_feature5",
            "Poly3vert0form_feature6",
            "Poly3vert0form_feature7",
            "Poly3vert1form_feature0",
            "Poly3vert1form_feature1",
            "Poly3vert1form_feature2",
            "Poly3vert1form_feature3",
            "Poly3vert1form_feature4",
            "Poly3vert1form_feature5",
            "Poly3vert1form_feature6",
            "Poly3vert1form_feature7",
            "Poly3vert2form_feature0",
            "Poly3vert2form_feature1",
            "Poly3vert2form_feature2",
            "Poly3vert2form_feature3",
            "Poly3vert2form_feature4",
            "Poly3vert2form_feature5",
            "Poly3vert2form_feature6",
            "Poly3vert2form_feature7",
            "Poly4vert3form_feature0",
            "Poly4vert3form_feature1",
            "Poly4vert3form_feature2",
            "Poly4vert3form_feature3",
            "Poly4vert3form_feature4",
            "Poly4vert3form_feature5",
            "Poly4vert3form_feature6",
            "Poly4vert3form_feature7",
            "Poly4vert4form_feature0",
            "Poly4vert4form_feature1",
            "Poly4vert4form_feature2",
            "Poly4vert4form_feature3",
            "Poly4vert4form_feature4",
            "Poly4vert4form_feature5",
            "Poly4vert4form_feature6",
            "Poly4vert4form_feature7",
            "Poly4vert5form_feature0",
            "Poly4vert5form_feature1",
            "Poly4vert5form_feature2",
            "Poly4vert5form_feature3",
            "Poly4vert5form_feature4",
            "Poly4vert5form_feature5",
            "Poly4vert5form_feature6",
            "Poly4vert5form_feature7",
        ]
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/spriteworld/group_000{id}/"
            for id in range(10)
        ] + [
            f"s3://synthefy-fm-eval-datasets/spriteworld/group_00{id}/"
            for id in range(10, 20)
        ]
        super().__init__(random_ordering=random_ordering)


class CIFAR100Dataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["feature_11"]
        self.keep_cols = [f"feature_{i}" for i in range(11)]
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/cifar100_timeseries_csvs/folder_{id}/"
            for id in range(1, 21)
        ]
        self.noise_level = 0.05
        self.lowpass_window = 3
        super().__init__(random_ordering=random_ordering)


class OpenWebTextDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.timestamp_col = None
        self.target_cols = ["feature_19"]
        self.keep_cols = [f"feature_{i}" for i in range(19)]
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/openwebtext_timeseries_csvs/folder_{id}/"
            for id in range(80, 100)
        ]
        self.noise_level = 0.05
        self.lowpass_window = 5
        super().__init__(random_ordering=random_ordering)


class DynamicDataDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.timestamp_col = "Unnamed: 0"
        self.target_cols = ["target"]
        self.keep_cols = [f"obs{i}" for i in range(1, 23)]
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/dynamic_data_csvs/group_{id}/"
            for id in range(1, 211)
        ]
        super().__init__(random_ordering=random_ordering)


class SCM_large_convlag_synin_sDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target"]
        self.keep_cols = [f"column_{id}" for id in range(1, 46)]
        self.timestamp_col = "timestamp"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/synthetic_scm/large_convlag_synin_s/csv/batch_folder_00000{id}/"
            for id in range(1, 10)
        ] + [
            "s3://synthefy-fm-eval-datasets/synthetic_scm/large_convlag_synin_s/csv/batch_folder_000010/"
        ]
        super().__init__(random_ordering=random_ordering)


class SCM_medium_convlag_synin_sDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target"]
        self.keep_cols = [f"column_{id}" for id in range(1, 31)]
        self.timestamp_col = "timestamp"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/synthetic_scm/medium_convlag_synin_s/csv/batch_folder_00000{id}/"
            for id in range(1, 10)
        ] + [
            "s3://synthefy-fm-eval-datasets/synthetic_scm/medium_convlag_synin_s/csv/batch_folder_000010/"
        ]
        super().__init__(random_ordering=random_ordering)


class SCM_medium_obslag_synin_sDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target"]
        self.keep_cols = [f"column_{id}" for id in range(1, 31)]
        self.timestamp_col = "timestamp"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/synthetic_scm/medium_obslag_synin_s/csv/batch_folder_00000{id}/"
            for id in range(1, 10)
        ] + [
            "s3://synthefy-fm-eval-datasets/synthetic_scm/medium_obslag_synin_s/csv/batch_folder_000010/"
        ]
        super().__init__(random_ordering=random_ordering)


class SCM_tiny_convlag_synin_nsDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target"]
        self.keep_cols = [f"column_{id}" for id in range(1, 21)]
        self.timestamp_col = "timestamp"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/synthetic_scm/tiny_convlag_synin_ns/csv/batch_folder_00000{id}/"
            for id in range(1, 10)
        ] + [
            "s3://synthefy-fm-eval-datasets/synthetic_scm/tiny_convlag_synin_ns/csv/batch_folder_000010/"
        ]
        super().__init__(random_ordering=random_ordering)


class SCM_tiny_obslag_synin_nsDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target"]
        self.keep_cols = [f"column_{id}" for id in range(1, 21)]
        self.timestamp_col = "timestamp"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/synthetic_scm/tiny_obslag_synin_ns/csv/batch_folder_00000{id}/"
            for id in range(1, 10)
        ] + [
            "s3://synthefy-fm-eval-datasets/synthetic_scm/tiny_obslag_synin_ns/csv/batch_folder_000010/"
        ]
        super().__init__(random_ordering=random_ordering)


class NasdaqTraderDataloader(GeneralizedDomainDataloader):
    def __init__(self, random_ordering: bool = False):
        self.target_cols = ["target_4"]
        self.keep_cols = []
        for id in range(1, 5):
            self.keep_cols.append(f"Open_{id}")
            self.keep_cols.append(f"High_{id}")
            self.keep_cols.append(f"Low_{id}")
            self.keep_cols.append(f"Volume_{id}")
            if id != 5:
                self.keep_cols.append(f"target_{id}")
        self.timestamp_col = "date"
        self.paths = [
            f"s3://synthefy-fm-eval-datasets/stock_nasdaqtrader/group00{id}/"
            for id in range(1, 3)
        ]
        super().__init__(random_ordering=random_ordering)
