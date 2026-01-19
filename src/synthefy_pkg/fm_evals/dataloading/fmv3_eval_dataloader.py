"""Eval specific dataloader for FMV3 Formats."""

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.v3_sharded_dataloader import V3ShardedDataloader
from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.preprocessing.fmv2_preprocess import (
    NORM_RANGES,
    TIMESTAMPS_FEATURES,
    inverse_time_vector,
)


class FMV3EvalDataloader(BaseEvalDataloader):
    def __init__(self, config: Configuration):
        self.config = config
        self.v3_dataloader = V3ShardedDataloader(config)
        self.test_dataloader = self.v3_dataloader.test_dataloader()

        self.ts_start_idx = config.dataset_config.continuous_start_idx
        self.ts_end_idx = config.dataset_config.continuous_end_idx
        self.forecast_length = config.dataset_config.forecast_length
        self.history_length = (
            config.dataset_config.time_series_length
            - config.dataset_config.forecast_length
        )

        self.timestamp_start_idx = config.dataset_config.timestamp_start_idx
        self.timestamp_end_idx = config.dataset_config.timestamp_end_idx

        self.timestamp_vector_dimension = len(TIMESTAMPS_FEATURES)

    def _batch_to_eval_batch_format(self, batch) -> EvalBatchFormat:
        """
        Reshapes a batch from the V3ShardedDataloader format to the eval format.
        """
        sfm_input_vector = batch["timeseries"]
        batch_size, num_correlates, vector_dim = sfm_input_vector.shape

        value_data = sfm_input_vector[:, :, self.ts_start_idx : self.ts_end_idx]
        history_data = value_data[:, :, : self.history_length]
        target_data = value_data[:, :, self.history_length :]

        timestamp_slice = sfm_input_vector[
            :, :, self.timestamp_start_idx : self.timestamp_end_idx
        ]
        assert timestamp_slice.shape == (
            batch_size,
            num_correlates,
            (self.history_length + self.forecast_length)
            * self.timestamp_vector_dimension,
        ), "Timestamp slice must be [B, NC, T]"

        # Get the timestamps back from the timestamp slice
        timestamp_slice = timestamp_slice.reshape(
            batch_size, num_correlates, -1, self.timestamp_vector_dimension
        )
        timestamp_slice = timestamp_slice.reshape(
            batch_size * num_correlates, -1, self.timestamp_vector_dimension
        )
        timestamp_slice = inverse_time_vector(timestamp_slice)
        timestamp_slice = timestamp_slice.reshape(
            batch_size,
            num_correlates,
            self.history_length + self.forecast_length,
        )

        history_timestamps = timestamp_slice[:, :, : self.history_length]
        target_timestamps = timestamp_slice[:, :, self.history_length :]

        return EvalBatchFormat.from_arrays(
            sample_ids=batch["sample_ids"],
            history_timestamps=history_timestamps,
            history_values=history_data,
            target_timestamps=target_timestamps,
            target_values=target_data,
        )

    def __len__(self):
        return len(self.test_dataloader)

    def __iter__(self):
        for batch in self.test_dataloader:
            yield self._batch_to_eval_batch_format(batch)
