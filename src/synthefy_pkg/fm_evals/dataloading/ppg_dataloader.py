"""
PPG (Blood Volume Pulse) Dataloader for evaluating fine-tuned Chronos models.

This dataloader loads PPG data from sharded datasets created during fine-tuning
and converts them to EvalBatchFormat for evaluation with the forecasting API.
"""

import os
import random
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)


class PPGShardedDataloader(BaseEvalDataloader):
    """
    Simplified PPG dataloader that directly reads from sharded datasets.

    This is optimized for quick evaluation of fine-tuned models.
    Supports loading covariates (e.g., ECG) as correlates for multivariate evaluation.
    """

    def __init__(
        self,
        sharded_dataset_path: str,
        max_samples: Optional[int] = 100,
        random_ordering: bool = False,
        covariate_fields: Optional[List[str]] = None,
    ):
        """
        Initialize the dataloader.

        Args:
            sharded_dataset_path: Path to sharded dataset (train or val).
            max_samples: Maximum samples to evaluate (None for all).
            random_ordering: Whether to shuffle samples.
            covariate_fields: List of covariate field names to load (e.g., ["ECG"]).
                              These will be included as correlates with forecast=False.
        """
        self.random_ordering = random_ordering
        self.sharded_dataset_path = os.path.expanduser(sharded_dataset_path)
        self.max_samples = max_samples
        self.covariate_fields = covariate_fields or []

        # Import and load
        from synthefy_dataset_utils.sharded_dataset_reader import (
            ShardedDatasetReader,
        )

        self.reader = ShardedDatasetReader(self.sharded_dataset_path)
        self.sample_ids = self.reader.query_ids()

        if max_samples:
            self.sample_ids = self.sample_ids[:max_samples]

        if self.random_ordering:
            random.shuffle(self.sample_ids)

        # Build list of fields to load
        self.fields_to_load = ["history", "target"]
        for cov in self.covariate_fields:
            self.fields_to_load.append(cov)
            self.fields_to_load.append(f"{cov}_future")

        logger.info(
            f"PPGShardedDataloader initialized: {len(self.sample_ids)} samples "
            f"from {self.sharded_dataset_path}, covariates: {self.covariate_fields}"
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        for sample_id in self.sample_ids:
            arrays, meta = self.reader.get_sample(
                sample_id, fields=self.fields_to_load
            )

            history = arrays.get("history", np.array([]))
            target = arrays.get("target", np.array([]))

            if len(history) == 0 or len(target) == 0:
                continue

            # Create timestamps as datetime64 with proper units (seconds from epoch)
            # PPG data is typically sampled at high frequency, use millisecond resolution
            base_time = np.datetime64("2023-01-01T00:00:00", "ms")
            history_timestamps = np.array(
                [
                    base_time + np.timedelta64(i * 10, "ms")
                    for i in range(len(history))
                ]
            )
            target_timestamps = np.array(
                [
                    base_time + np.timedelta64((len(history) + i) * 10, "ms")
                    for i in range(len(target))
                ]
            )

            # Build list of correlates: first BVP (target), then covariates
            correlates = []

            # BVP target sample
            bvp_sample = SingleEvalSample(
                sample_id=sample_id,
                history_timestamps=history_timestamps,
                history_values=history.astype(np.float32),
                target_timestamps=target_timestamps,
                target_values=target.astype(np.float32),
                forecast=True,
                metadata=False,
                leak_target=False,
                column_name="BVP",
            )
            correlates.append(bvp_sample)

            # Add covariate correlates - for now, include all leaked covariates
            for cov_name in self.covariate_fields:
                cov_history = arrays.get(cov_name, np.array([]))
                cov_future = arrays.get(f"{cov_name}_future", np.array([]))

                if len(cov_history) == 0:
                    logger.error(
                        f"Covariate {cov_name} history not found in sample {sample_id}"
                    )
                    raise ValueError(
                        f"Covariate {cov_name} history not found in sample {sample_id}"
                    )

                # Use zeros for future if not available
                if len(cov_future) == 0:
                    cov_future = np.zeros(len(target), dtype=np.float32)

                cov_sample = SingleEvalSample(
                    sample_id=f"{sample_id}_{cov_name}",
                    history_timestamps=history_timestamps,
                    history_values=cov_history.astype(np.float32),
                    target_timestamps=target_timestamps,
                    target_values=cov_future.astype(np.float32),
                    forecast=False,  # Covariate - not a forecast target
                    metadata=True,  # This is metadata/correlate
                    leak_target=True,  # Covariates can be "leaked" (known during forecast)
                    column_name=cov_name,
                )
                correlates.append(cov_sample)

            yield EvalBatchFormat(
                samples=[correlates],
                target_cols=["BVP"],
                metadata_cols=self.covariate_fields,
                leak_cols=self.covariate_fields,
            )

    def close(self):
        """Close the reader."""
        if hasattr(self, "reader"):
            self.reader.close()
