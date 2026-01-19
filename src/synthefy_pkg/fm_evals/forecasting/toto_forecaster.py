import os
import sys
from typing import Any, Dict

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.fm_evals.forecasting.base_forecaster import BaseForecaster
from synthefy_pkg.fm_evals.forecasting.utils import fill_nan_values
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


class TotoForecaster(BaseForecaster):
    """ToTo forecaster that calls a remote ToTo API.

    Configuration:
    - Reads base URL from environment variable `TOTO_SERVER_URL`.
      Falls back to `http://localhost:8000` if not set.
    """

    def __init__(self, num_samples: int = 10):
        super().__init__("ToTo Multivariate Forecaster")

        # Base URL for FastAPI server from env var
        env_url = os.getenv("MODEL_TT_SERVER_URL", "http://localhost:8000")
        self.base_url = env_url.rstrip("/")

        self.num_samples = num_samples  # For probabilistic forecasts

        # Perform server health check
        try:
            logger.info("Checking server health...")
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            logger.info(
                f"Server health: {resp.json().get('status', 'UNKNOWN')}"
            )
        except Exception as e:
            logger.error(f"Failed to check server health: {e}")
            logger.error("Are you sure the server is running?")
            sys.exit(1)

    def _validate_timestamps(self, history_timestamps: np.ndarray) -> None:
        for b in range(history_timestamps.shape[0]):
            for nc in range(history_timestamps.shape[1]):
                correlate_timestamps = history_timestamps[b, nc]

                # Calculate time deltas between consecutive timestamps
                time_deltas = np.diff(correlate_timestamps)

                # Check if all deltas are equal (within floating point precision)
                if not np.allclose(
                    time_deltas.astype(np.float64),
                    time_deltas[0].astype(np.float64),
                ):
                    logger.warning(
                        f"tt assumes evenly spaced timestamps. "
                        f"Found varying time deltas in batch {b}: {time_deltas}. "
                        f"This may lead to incorrect results, but since tt "
                        f"actually ignores time deltas right now, we'll continue."
                    )

    def _validate_target_lengths(self, target_timestamps: np.ndarray) -> None:
        # Use target timestamps to make sure that across all batches
        # and all correlates, the target length is the same
        for b in range(target_timestamps.shape[0]):
            for nc in range(target_timestamps.shape[1]):
                target_timestamps_correlate = target_timestamps[b, nc]
                if (
                    target_timestamps_correlate.shape[0]
                    != target_timestamps[0, 0].shape[0]
                ):
                    raise ValueError(
                        f"tt requires all correlates to have the same target length. "
                        f"Correlate {nc} has different target length than correlate 0."
                    )

    def _convert_timestamps_to_seconds(
        self, timestamps: np.ndarray
    ) -> np.ndarray:
        # input: [B, NC, T] -> output: [B, NC, T]

        # Convert timestamps to seconds since epoch
        return timestamps.astype("datetime64[s]").astype(np.float64)

    def _get_time_delta_seconds(
        self, timestamps_in_seconds: np.ndarray
    ) -> np.ndarray:
        # input: [B, NC, T] -> output: [B, NC, T-1]
        # Just use the last time delta for each correlate
        return np.diff(timestamps_in_seconds, axis=2)[:, :, -1]

    def fit(self, batch: EvalBatchFormat) -> bool:
        # make sure we have enough history
        for b in range(batch.batch_size):
            for nc in range(batch.num_correlates):
                if len(batch[b, nc].history_values) < 2:
                    logger.error(
                        f"tt requires at least 2 history values to infer time deltas. Found {len(batch[b, nc].history_values)} history values for correlate {nc} in batch {b}."
                    )
                    return False
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        # We can't forecast in batches due to inhomogenous shapes when backtesting
        # So, we need to forecast one sample at a time
        # Toto is fast, so I'm not worried about performance atm.
        B = batch.batch_size
        batch_sample_ids = []
        batch_timestamps = []
        batch_values = []
        for b in range(B):
            sample = EvalBatchFormat([batch.samples[b]])
            (
                sample_ids,
                history_timestamps,
                history_values,
                target_timestamps,
                target_values,
            ) = sample.to_arrays(targets_only=False)

            nc = sample.num_correlates

            # For ToTo, all correlates share a single set of timestamps
            self._validate_timestamps(history_timestamps)

            # All target lengths musst match
            self._validate_target_lengths(target_timestamps)

            # Fill NaN values with mean of non-NaN values
            history_values = fill_nan_values(history_values)

            # ToTo requires timestamps in seconds and time delta in seconds
            # Although these are not being used by toto right now
            history_timestamps_seconds = self._convert_timestamps_to_seconds(
                history_timestamps
            )
            time_delta_seconds = self._get_time_delta_seconds(
                history_timestamps_seconds
            )

            prediction_length = target_timestamps.shape[2]

            payload: Dict[str, Any] = {
                "input_series": history_values.astype(np.float32).tolist(),
                "timestamp_seconds": history_timestamps_seconds.astype(
                    np.float32
                ).tolist(),
                "time_interval_seconds": time_delta_seconds.astype(
                    np.float32
                ).tolist(),
                "prediction_length": int(prediction_length),
                "num_samples": int(self.num_samples),
                "samples_per_batch": int(1),
            }
            resp = requests.post(
                f"{self.base_url}/forecast", json=payload, timeout=60
            )
            resp.raise_for_status()
            result_json = resp.json()
            api_result = np.asarray(result_json["result"], dtype=np.float32)  # type: ignore[index]

            batch_sample_ids.append(sample_ids)
            batch_timestamps.append(target_timestamps)
            batch_values.append(api_result)

        batch_sample_ids = np.concatenate(batch_sample_ids, axis=0)
        batch_timestamps = np.concatenate(batch_timestamps, axis=0)
        batch_values = np.concatenate(batch_values, axis=0)

        forecast_output = ForecastOutputFormat.from_arrays(
            sample_ids=batch_sample_ids,
            timestamps=batch_timestamps,
            values=batch_values,
            model_name=self.name,
        )

        # If a correlate is not forecasted, we set the forecast to an empty array
        for b in range(forecast_output.batch_size):
            for nc in range(forecast_output.num_correlates):
                if not batch[b, nc].forecast:
                    forecast_output.forecasts[b][nc] = SingleSampleForecast(
                        sample_id=batch_sample_ids[b, nc],
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float64),
                        model_name=self.name,
                    )

        return forecast_output


class TotoUnivariateForecaster(TotoForecaster):
    def __init__(self, num_samples: int = 10):
        super().__init__(num_samples)

    def _predict_single_sample(self, sample):
        """Predict a single sample using Toto API"""
        # Reshape to [1, 1, T] format for API
        history_values = sample.history_values.reshape(1, 1, -1)
        history_timestamps = sample.history_timestamps.reshape(1, 1, -1)
        target_timestamps = sample.target_timestamps.reshape(1, 1, -1)

        # Validate timestamps (adapted from TotoForecaster)
        time_deltas = np.diff(sample.history_timestamps)
        if not np.allclose(
            time_deltas.astype(np.float64), time_deltas[0].astype(np.float64)
        ):
            logger.warning(
                f"tt assumes evenly spaced timestamps. "
                f"Found varying time deltas: {time_deltas}. "
                f"This may lead to incorrect results, but since tt "
                f"actually ignores time deltas right now, we'll continue."
            )

        # Fill NaN values
        history_values = fill_nan_values(history_values)

        # Convert timestamps (adapted from TotoForecaster)
        history_timestamps_seconds = history_timestamps.astype(
            "datetime64[s]"
        ).astype(np.float64)
        time_delta_seconds = np.diff(history_timestamps_seconds, axis=2)[
            :, :, -1
        ]

        prediction_length = target_timestamps.shape[2]

        # Call API with batch_size=1
        payload: Dict[str, Any] = {
            "input_series": history_values.astype(np.float32).tolist(),
            "timestamp_seconds": history_timestamps_seconds.astype(
                np.float32
            ).tolist(),
            "time_interval_seconds": time_delta_seconds.astype(
                np.float32
            ).tolist(),
            "prediction_length": int(prediction_length),
            "num_samples": int(self.num_samples),
            "samples_per_batch": int(1),
        }
        resp = requests.post(
            f"{self.base_url}/forecast", json=payload, timeout=60
        )
        resp.raise_for_status()
        api_result = np.asarray(resp.json()["result"], dtype=np.float32)  # type: ignore[index]
        return api_result[0, 0, :]  # Extract prediction for single sample

    def fit(self, batch: EvalBatchFormat) -> bool:
        # make sure we have enough history
        for b in range(batch.batch_size):
            for nc in range(batch.num_correlates):
                if len(batch[b, nc].history_values) < 2:
                    logger.error(
                        f"tt requires at least 2 history values to infer time deltas. Found {len(batch[b, nc].history_values)} history values for correlate {nc} in batch {b}."
                    )
                    return False
        return True

    def _predict(self, batch: EvalBatchFormat) -> ForecastOutputFormat:
        B = batch.batch_size
        NC = batch.num_correlates

        forecasts = []
        for i in tqdm(range(B), desc="Predicting to"):
            row = []
            for j in tqdm(
                range(NC), desc="Predicting to"
            ):
                sample = batch[i, j]

                if not sample.forecast:
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=np.array([]),
                            values=np.array([], dtype=np.float32),
                            model_name=self.name,
                        )
                    )
                    continue

                try:
                    pred_values = self._predict_single_sample(sample)
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=sample.target_timestamps,
                            values=pred_values,
                            model_name=self.name,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error predicting sample {i}, correlate {j}: {e}"
                    )
                    pred_values = np.full(
                        len(sample.target_timestamps), np.nan, dtype=np.float32
                    )
                    row.append(
                        SingleSampleForecast(
                            sample_id=sample.sample_id,
                            timestamps=sample.target_timestamps,
                            values=pred_values,
                            model_name=self.name,
                        )
                    )

            forecasts.append(row)

        return ForecastOutputFormat(forecasts)

    def predict_for_training(
        self, batch: EvalBatchFormat, num_samples: int = 10
    ) -> ForecastOutputFormat:
        (
            sample_ids,
            history_timestamps,
            history_values,
            target_timestamps,
            target_values,
        ) = batch.to_arrays(targets_only=False)

        # For ToTo, all correlates share a single set of timestamps
        self._validate_timestamps(history_timestamps)

        # All target lengths musst match
        self._validate_target_lengths(target_timestamps)

        # Fill NaN values with mean of non-NaN values
        history_values = fill_nan_values(history_values)

        # ToTo requires timestamps in seconds and time delta in seconds
        # Although these are not being used by toto right now
        history_timestamps_seconds = self._convert_timestamps_to_seconds(
            history_timestamps
        )
        time_delta_seconds = self._get_time_delta_seconds(
            history_timestamps_seconds
        )

        prediction_length = target_timestamps.shape[2]

        payload: Dict[str, Any] = {
            "input_series": history_values.astype(np.float32).tolist(),
            "timestamp_seconds": history_timestamps_seconds.astype(
                np.float32
            ).tolist(),
            "time_interval_seconds": time_delta_seconds.astype(
                np.float32
            ).tolist(),
            "prediction_length": int(prediction_length),
            "num_samples": int(num_samples),
            "samples_per_batch": int(batch.batch_size),
        }
        resp = requests.post(
            f"{self.base_url}/forecast", json=payload, timeout=60
        )
        resp.raise_for_status()
        api_result = np.asarray(resp.json()["result"], dtype=np.float32)  # type: ignore[index]

        forecast_output = ForecastOutputFormat.from_arrays(
            sample_ids=sample_ids,
            timestamps=target_timestamps,
            values=api_result,
            model_name=self.name,
        )

        # If a correlate is not forecasted, we set the forecast to an empty array
        for b in range(forecast_output.batch_size):
            for nc in range(forecast_output.num_correlates):
                if not batch[b, nc].forecast:
                    forecast_output.forecasts[b][nc] = SingleSampleForecast(
                        sample_id=sample_ids[b, nc],
                        timestamps=np.array([], dtype=np.float64),
                        values=np.array([], dtype=np.float64),
                        model_name=self.name,
                    )

        return forecast_output
