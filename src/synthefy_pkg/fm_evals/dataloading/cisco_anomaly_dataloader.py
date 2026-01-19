"""Dataloader for Cisco anomaly detection datasets.

Each dataset contains time series data with anomalies labeled.
Used for fine-tuning and evaluating Chronos models on anomaly detection.
"""

from typing import Iterator

import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class CiscoAnomalyDataloader(BaseEvalDataloader):
    """Base dataloader for Cisco anomaly detection datasets.

    Args:
        random_ordering: Whether to randomize file order
        use_first_half: If True, use first 50% of data (for fine-tuning).
                       If False, use last 50% (for inference/evaluation).
        history_length: History window length in rows (default: 192 = 48h at 15-min intervals)
        prediction_length: Prediction length in rows (default: 1 = 1 step)
        stride: Stride between windows (default: 1)
    """

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        super().__init__(random_ordering)
        self.use_first_half = use_first_half
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.stride = stride

        # Set in subclasses
        if not hasattr(self, "dataset_name"):
            self.dataset_name = None

        if self.dataset_name is None:
            raise ValueError("dataset_name must be set in subclass")

        # Use S3 path for the CSV file
        self.csv_file = f"s3://synthefy-fm-eval-datasets/cisco_anomaly/{self.dataset_name}.csv"

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the CSV file."""
        logger.info(f"Loading {self.csv_file}")
        df = pd.read_csv(self.csv_file)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter to first or last 50%
        total_rows = len(df)
        if self.use_first_half:
            half_end = total_rows // 2
            df = df.iloc[:half_end].copy()
            logger.info(
                f"Using first 50%: {len(df)} rows (from {total_rows} total)"
            )
        else:
            half_start = total_rows // 2
            df = df.iloc[half_start:].copy()
            logger.info(
                f"Using last 50%: {len(df)} rows (from {total_rows} total)"
            )

        # Remove rows with invalid timestamps
        initial_rows = len(df)
        df = df.dropna(subset=["timestamp"])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(
                f"Dropped {dropped_rows} rows with invalid timestamps"
            )

        if len(df) == 0:
            logger.error("No valid data remaining after filtering")
            return pd.DataFrame()

        logger.info(
            f"Loaded {len(df)} rows with timestamp range: "
            f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        )

        return df

    def __len__(self) -> int:
        """Return 1 (single file per dataset)."""
        return 1

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat object for this dataset."""
        df = self._load_and_preprocess_data()

        if len(df) == 0:
            return

        # Use avgrtt as target column
        target_cols = ["avgrtt"]
        metadata_cols = []

        # Create eval batch with sliding windows
        # Each window has history_length history and prediction_length target
        eval_batch = EvalBatchFormat.from_dfs(
            dfs=[df],
            timestamp_col="timestamp",
            num_target_rows=self.prediction_length,
            target_cols=target_cols,
            metadata_cols=metadata_cols,
            leak_cols=[],
            forecast_window=self.prediction_length,
            stride=self.stride,
        )

        if eval_batch is not None:
            yield eval_batch


class PeriodicAnomalyTimeSeries2Dataloader(CiscoAnomalyDataloader):
    """Dataloader for periodic_anomaly_time_series_2 dataset."""

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        self.dataset_name = "periodic_anomaly_time_series_2"
        super().__init__(
            random_ordering=random_ordering,
            use_first_half=use_first_half,
            history_length=history_length,
            prediction_length=prediction_length,
            stride=stride,
        )


class Multimodal5Dataloader(CiscoAnomalyDataloader):
    """Dataloader for multimodal_5 dataset."""

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        self.dataset_name = "multimodal_5"
        super().__init__(
            random_ordering=random_ordering,
            use_first_half=use_first_half,
            history_length=history_length,
            prediction_length=prediction_length,
            stride=stride,
        )


class LongDurationAnomalyTimeSeries59Dataloader(CiscoAnomalyDataloader):
    """Dataloader for long_duration_anomaly_time_series_59 dataset."""

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        self.dataset_name = "long_duration_anomaly_time_series_59"
        super().__init__(
            random_ordering=random_ordering,
            use_first_half=use_first_half,
            history_length=history_length,
            prediction_length=prediction_length,
            stride=stride,
        )


class HighConfidenceAnomalyTimeSeries3Dataloader(CiscoAnomalyDataloader):
    """Dataloader for high_confidence_anomaly_time_series_3 dataset."""

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        self.dataset_name = "high_confidence_anomaly_time_series_3"
        super().__init__(
            random_ordering=random_ordering,
            use_first_half=use_first_half,
            history_length=history_length,
            prediction_length=prediction_length,
            stride=stride,
        )


class SpreadAnomalyTimeSeries9Dataloader(CiscoAnomalyDataloader):
    """Dataloader for spread_anomaly_time_series_9 dataset."""

    def __init__(
        self,
        random_ordering: bool = False,
        use_first_half: bool = False,
        history_length: int = 192,
        prediction_length: int = 1,
        stride: int = 1,
    ):
        self.dataset_name = "spread_anomaly_time_series_9"
        super().__init__(
            random_ordering=random_ordering,
            use_first_half=use_first_half,
            history_length=history_length,
            prediction_length=prediction_length,
            stride=stride,
        )
