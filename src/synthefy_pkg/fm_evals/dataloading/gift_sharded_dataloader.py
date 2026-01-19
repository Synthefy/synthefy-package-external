"""Dataloader for sharded GIFT-eval datasets.

This dataloader reads from the sharded dataset format used for fine-tuning,
making it compatible with eval.py for model evaluation.

The sharded format consists of:
- index.sqlite: SQLite database with sample metadata
- history_shards/: tar files containing history arrays
- target_shards/: tar files containing target arrays
"""

import random
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from synthefy_dataset_utils.sharded_dataset_reader import ShardedDatasetReader

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat


class GIFTShardedDataloader(BaseEvalDataloader):
    """
    Dataloader for sharded GIFT-eval datasets.

    Similar interface to GIFTEvalUnivariateDataloader but reads from
    the columnar tar-sharded format instead of arrow files.
    """

    def __init__(
        self,
        data_path: str,
        forecast_length: Optional[int] = None,
        history_length: Optional[int] = None,
        filter_datasets: Optional[List[str]] = None,
        random_ordering: bool = False,
        limit: Optional[int] = None,
    ):
        """
        Initialize the sharded GIFT dataloader.

        Args:
            data_path: Path to the sharded dataset root (containing index.sqlite)
            forecast_length: Forecast length to use. If None, uses target array length.
            history_length: Maximum history length (truncate from start if longer)
            filter_datasets: Optional list of dataset names to include (from metadata)
            random_ordering: Whether to shuffle samples
            limit: Optional limit on number of samples to load
        """
        self.data_path = data_path
        self.forecast_length = forecast_length
        self.max_history_length = history_length
        self.filter_datasets = filter_datasets
        self.random_ordering = random_ordering
        self.limit = limit

        logger.info(f"Loading sharded dataset from: {data_path}")
        self.reader = ShardedDatasetReader(data_path)
        self.sample_ids = self._collect_sample_ids()
        logger.info(f"Loaded {len(self.sample_ids)} samples")

    def _collect_sample_ids(self) -> List[str]:
        """Collect sample IDs, optionally filtered by dataset name."""
        if self.filter_datasets:
            if len(self.filter_datasets) == 1:
                where_clause = "json_extract(meta_json, '$.dataset') = ?"
                params = (self.filter_datasets[0],)
            else:
                placeholders = ",".join("?" * len(self.filter_datasets))
                where_clause = (
                    f"json_extract(meta_json, '$.dataset') IN ({placeholders})"
                )
                params = tuple(self.filter_datasets)

            sample_ids = self.reader.query_ids(
                where_clause=where_clause, params=params
            )
        else:
            sample_ids = self.reader.query_ids()

        if self.random_ordering:
            random.shuffle(sample_ids)

        if self.limit:
            sample_ids = sample_ids[: self.limit]

        return sample_ids

    def _create_timestamp_array(
        self, start_date: Optional[str], frequency: Optional[str], length: int
    ) -> np.ndarray:
        """Create timestamp array from metadata fields.

        Same implementation as GIFTEvalUnivariateDataloader.
        """
        if start_date and frequency:
            if isinstance(start_date, np.ndarray):
                start_date = start_date[0]
            try:
                return pd.date_range(
                    start=start_date, periods=length, freq=frequency
                ).to_numpy()
            except Exception:
                pass

        # Fallback: generate synthetic timestamps (hourly from 2020-01-01)
        return pd.date_range(
            start="2020-01-01", periods=length, freq="h"
        ).to_numpy()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for sample_id in self.sample_ids:
            try:
                yield self._load_sample(sample_id)
            except Exception as e:
                logger.warning(f"Skipping sample {sample_id}: {e}")
                continue

    def _load_sample(self, sample_id: str) -> EvalBatchFormat:
        """Load a single sample and convert to EvalBatchFormat.

        Similar logic to GIFTEvalUnivariateDataloader.__iter__ but adapted
        for the sharded format where history and target are separate arrays.
        """
        # Load history and target arrays
        arrays, meta = self.reader.get_sample(
            sample_id, fields=["history", "target"]
        )

        history_array = arrays.get("history")
        target_array = arrays.get("target")

        if history_array is None or target_array is None:
            raise ValueError("Missing history or target array")

        # Ensure arrays are 2D: (nc, T) - same pattern as gift_eval_dataloader
        if history_array.ndim == 1:
            history_array = history_array.reshape(1, -1)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, -1)

        # Collect dimensions
        nc = history_array.shape[0]  # Number of channels
        history_length = history_array.shape[1]
        target_length = target_array.shape[1]

        # Determine actual forecast length (similar to gift_eval_dataloader validation)
        actual_forecast_length = self.forecast_length or target_length
        if target_length < actual_forecast_length:
            raise ValueError(
                f"Length of target array ({target_length}) is less than "
                f"forecast length ({actual_forecast_length})"
            )

        # Slice target to forecast length (take first N values)
        target_values = target_array[:, :actual_forecast_length]

        # Create timestamp array for the combined length
        total_length = history_length + actual_forecast_length
        timestamp_array = self._create_timestamp_array(
            meta.get("start"), meta.get("freq"), total_length
        )

        # Repeat the timestamp array for each channel (same as gift_eval_dataloader)
        if nc > 1:
            timestamp_array = np.concatenate(
                [timestamp_array.reshape(1, -1)] * nc, axis=0
            )
        else:
            timestamp_array = timestamp_array.reshape(1, -1)

        # Slice out history portion - same logic as gift_eval_dataloader
        if self.max_history_length is not None:
            # Use up to max_history_length, truncating from the start
            available_history_length = min(
                self.max_history_length, history_length
            )
            history_timestamps = timestamp_array[:, :available_history_length]
            history_values = history_array[:, -available_history_length:]
        else:
            history_timestamps = timestamp_array[:, :history_length]
            history_values = history_array

        # Target timestamps are the remaining portion
        target_timestamps = timestamp_array[:, -actual_forecast_length:]

        return EvalBatchFormat.from_arrays(
            sample_ids=np.full((1, nc), sample_id),
            history_timestamps=history_timestamps.reshape(1, nc, -1),
            history_values=history_values.reshape(1, nc, -1),
            target_timestamps=target_timestamps.reshape(1, nc, -1),
            target_values=target_values.reshape(1, nc, -1),
        )

    def close(self):
        """Close the reader and free resources."""
        self.reader.close()
