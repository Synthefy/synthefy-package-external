import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from synthefy_pkg.app.config import ForecastSettings
from synthefy_pkg.app.data_models import (
    ForecastRequest,
    ForecastResponse,
    InfoContainer,
    SynthefyRequest,
    SynthefyResponse,
    TimeStamps,
)
from synthefy_pkg.app.utils.api_utils import (
    array_to_timeseries,
    check_scale_by_metadata_used,
    convert_list_to_isoformat,
    create_synthefy_response_from_other_types,
    generate_ts_summary,
    metadata_to_dataframe,
)
from synthefy_pkg.experiments.forecast_experiment import ForecastExperiment
from synthefy_pkg.preprocessing.preprocess import DataPreprocessor
from synthefy_pkg.utils.scaling_utils import (
    inverse_transform_discrete,
    load_continuous_scalers,
    load_discrete_encoders,
    load_timeseries_scalers,
    transform_using_scaler,
)

COMPILE = True


def shift_to_zero_out_forecast_length(
    request: SynthefyRequest,
    forecast_length: int,
) -> SynthefyRequest:
    """
    Shifts the data from [forecast_length:] to the front and zeros out the last forecast_length entries.
    This is used for true forecasting, where we want to predict the future values, and not just mask the last forecast_length entries.
    The downstream forecasting code masks/removes the last forecast_length entries, so fill in None/missing values for the forecasted part.

    Args:
        request: SynthefyRequest object
    Returns:
        Modified SynthefyRequest object
    """
    shifted_request = request.model_copy(deep=True)

    for window in shifted_request.windows:
        shift_start = forecast_length

        # Shift timeseries data
        for ts in window.timeseries_data:
            forecast_values = ts.values[shift_start:]
            ts.values = forecast_values + [None] * forecast_length

        # Shift timestamps if they exist
        if window.timestamps and window.timestamps.values:
            # Convert timestamps to pandas Timestamp objects if they're integers/floats
            if isinstance(window.timestamps.values[0], (int, float)):
                window.timestamps.values = [
                    (
                        pd.Timestamp.fromtimestamp(
                            ts / 1000
                        )  # Convert from milliseconds to seconds
                        if ts > 1e10  # Check if timestamp is in milliseconds
                        else pd.Timestamp.fromtimestamp(ts)
                    )
                    for ts in window.timestamps.values
                ]

            forecast_timestamps = convert_list_to_isoformat(
                window.timestamps.values[shift_start:]
            )

            # Convert last two timestamps to calculate time delta
            last_ts = pd.Timestamp(forecast_timestamps[-1])
            second_last_ts = pd.Timestamp(forecast_timestamps[-2])
            real_time_delta = last_ts - second_last_ts  # type: ignore

            padding_timestamps = []
            current_ts = last_ts
            for _ in range(forecast_length):
                current_ts += real_time_delta  # type: ignore
                padding_timestamps.append(current_ts)
            window.timestamps.values = convert_list_to_isoformat(
                [pd.Timestamp(ts) for ts in forecast_timestamps]
                + padding_timestamps  # type: ignore
            )

        # Shift metadata if it exists
        if window.metadata:
            # Shift discrete conditions if they exist
            if window.metadata.discrete_conditions:
                for discrete in window.metadata.discrete_conditions:
                    forecast_values = discrete.values[shift_start:]
                    discrete.values = forecast_values + [None] * forecast_length

            # Shift continuous conditions if they exist
            if window.metadata.continuous_conditions:
                for continuous in window.metadata.continuous_conditions:
                    forecast_values = continuous.values[shift_start:]
                    continuous.values = (
                        forecast_values + [None] * forecast_length
                    )

    return shifted_request


class ForecastService:
    dataset_name: str
    settings: ForecastSettings
    experiment: ForecastExperiment
    forecast_length: int
    preprocess_config: Dict[str, Any]
    encoders: Optional[Dict[str, Any]] = None
    saved_scalers: Dict[str, Any]

    def __init__(self, dataset_name: str, settings: ForecastSettings):
        logger.info(f"Initializing ForecastService with settings: {settings}")
        self.settings = settings
        self.dataset_name = dataset_name

        if self.settings.model_to_use == "sfmv2":
            self.experiment = ForecastExperiment(
                self.settings.forecast_config_path
            )
        else:
            raise ValueError(
                f"Invalid model to use: {self.settings.model_to_use}"
            )
        self.forecast_length = (
            self.experiment.configuration.dataset_config.forecast_length
        )

        with open(self.settings.preprocess_config_path, "r") as f:
            self.preprocess_config = yaml.safe_load(f)

        if len(self.preprocess_config["timestamps_col"]) > 1:
            raise ValueError(
                f"Only zero or one timestamp column is supported - currently: {self.preprocess_config['timestamps_col']}"
            )

        self.encoders = load_discrete_encoders(self.dataset_name)
        self.saved_scalers = {
            "continuous": load_continuous_scalers(self.dataset_name),
            "timeseries": load_timeseries_scalers(self.dataset_name),
        }
        self.validate_scalers()

    def validate_scalers(self):
        if check_scale_by_metadata_used(
            self.saved_scalers["timeseries"]
        ) or check_scale_by_metadata_used(self.saved_scalers["continuous"]):
            raise ValueError(
                "Forecast does not support scalers by discrete metadata"
            )

    async def forecast(
        self,
        request: SynthefyRequest,
        streaming: bool = False,
        true_forecast_with_shifting: bool = False,
        suffix_label: str = "_synthetic",
    ) -> SynthefyResponse:
        """
        inputs:
            request: SynthefyRequest object indicating user's request for forecasting
        outputs:
            forecast_response: SynthefyResponse object
        description:
            This forecasts the last pred_len steps of the timeseries, conditioned on all data before the last pred_len steps.
        """
        if true_forecast_with_shifting:
            request = shift_to_zero_out_forecast_length(
                request, self.forecast_length
            )

        logger.info("Forecasting for request")
        forecast_responses: List[ForecastResponse] = []
        for window_idx in request.selected_windows.window_indices:
            for _ in range(request.n_forecast_windows):
                forecast_request = ForecastRequest(
                    past_metadata=request.windows[window_idx].metadata,
                    past_timeseries=request.windows[window_idx].timeseries_data,
                    past_timestamps=request.windows[window_idx].timestamps,
                    text=request.text,
                )
                x_axis_values, batch = self._preprocess_request(
                    forecast_request
                )
                logger.info("Preprocessed request")
                timeseries_whole = self._forecast(
                    batch
                )  # timeseries_whole.shape = (channels, window_size)

                # do an np isclose on the [:forecast length]
                try:
                    assert np.allclose(
                        timeseries_whole[:, : -self.forecast_length],
                        batch["timeseries_full"][0, :, : -self.forecast_length]
                        .cpu()  # Move tensor to CPU first
                        .detach()
                        .numpy(),
                    )
                except Exception:
                    raise Exception("Non forecasting timestamps are not close")
                logger.info("Done with forecasting")

                decoded_discrete_cols, decoded_discrete_windows = (
                    inverse_transform_discrete(
                        batch["discrete_label_embedding"].cpu(),
                        dataset_name=self.dataset_name,
                    )
                )
                timeseries_whole = transform_using_scaler(
                    windows=np.expand_dims(timeseries_whole, axis=0),
                    timeseries_or_continuous="timeseries",
                    original_discrete_windows=decoded_discrete_windows,
                    original_discrete_colnames=decoded_discrete_cols,
                    dataset_name=self.dataset_name,
                    inverse_transform=True,
                )[0]

                forecast_response = await self.convert_to_forecast_response(
                    x_axis_values=x_axis_values,
                    timeseries_preds=timeseries_whole,
                    request=forecast_request,
                    streaming=streaming,
                    suffix_label=suffix_label,
                )
                logger.info("Done with converting to forecast response")
                forecast_responses.append(forecast_response)

        return await create_synthefy_response_from_other_types(
            self.dataset_name,
            forecast_responses,
            streaming=streaming,
            text=request.text,
        )

    async def convert_to_forecast_response(
        self,
        x_axis_values: TimeStamps,
        timeseries_preds: np.ndarray,
        request: ForecastRequest,
        info_containers: Optional[List[InfoContainer]] = None,
        streaming: bool = False,
        suffix_label: str = "_synthetic",
    ) -> ForecastResponse:
        """
        inputs:
            # timeseries_preds: np.ndarray (num_channels, window_size)
            timeseries_preds: np.ndarray (num_channels, features, window_size)
            suffix_label: Default to "_synthetic"
        outputs:
            forecast_response: ForecastResponse object
        """

        # Axes are swapped because array_to_timeseries expects (window_size, num_channels)
        # (num_channels, window_size) -> (window_size, num_channels)
        timeseries_data = array_to_timeseries(
            timeseries_preds.swapaxes(0, 1),
            channel_names=[
                f"{timeseries.name}{suffix_label}"
                for timeseries in request.past_timeseries
            ],
        )

        # only care about the forecasted part - use the request for the previous
        for timeseries in timeseries_data:
            timeseries.values = timeseries.values[-self.forecast_length :]

        # insert the [0:window_size - forecast_length] to get the full window size
        for idx, timeseries in enumerate(request.past_timeseries):
            timeseries_data[idx].values = (
                timeseries.values[: -self.forecast_length]
                + timeseries_data[idx].values
            )

        if self.settings.show_gt_forecast_timeseries and not streaming:
            timeseries_data += request.past_timeseries

        pred_len = (
            self.experiment.configuration.dataset_config.forecast_length + 1
        )
        start_of_forecast_timestamp = x_axis_values.values[-pred_len]

        text = await generate_ts_summary(
            query=request.text,
            org_ts=[ts.dict() for ts in request.past_timeseries],
            out_ts=[ts.dict() for ts in timeseries_data],
        )

        return ForecastResponse(
            x_axis=x_axis_values,
            timeseries_data=timeseries_data,
            metadata=request.past_metadata,
            info_containers=info_containers,
            start_of_forecast_timestamp=TimeStamps(
                name=x_axis_values.name, values=[start_of_forecast_timestamp]
            ),
            text=text,
        )

    def _preprocess_request(
        self, request: ForecastRequest
    ) -> Tuple[TimeStamps, dict]:
        """
        inputs:
            request: ForecastRequest object
        outputs:
            x_axis_values: TimeStamps object
            batch: dict
        description:
            This function converts the request dictionary to a ForecastRequest object.
            It raises an exception if the format is incorrect or necessary fields are missing.
            It also preprocesses the request to make it easier to use for the model.
        """
        # TODO: Throw an error or a warning if the history is too long or too short, which is related to `SynthefyForecastingModelV1.pred_len`
        # `pred_len` is the number of steps to predict into the future.
        df = metadata_to_dataframe(
            metadata=request.past_metadata,
            timeseries=request.past_timeseries,
            timestamps=request.past_timestamps,
        )

        x_axis_name = (
            request.past_timestamps.name if request.past_timestamps else "index"
        )
        # Standard Preprocessing
        preprocessor = DataPreprocessor(self.settings.preprocess_config_path)
        if not preprocessor.use_label_col_as_discrete_metadata:
            preprocessor.group_labels_cols = []

        preprocessor.discrete_cols = list(preprocessor.discrete_cols) + list(
            preprocessor.group_labels_cols
        )
        # Explicitly set these to empty; we need timeseries, discrete, continuous and scaled_timestamps conditions.
        preprocessor.group_labels_cols = []
        # preprocessor.group_labels_scalers = {}  # can we remove this altogether?

        preprocessor.process_data(
            df,
            saved_scalers=self.saved_scalers,
            saved_encoders=self.encoders,
            save_files_on=False,
        )
        (
            timeseries_full,
            discrete_conditions,
            continuous_conditions,
            timestamp_embeddings,
        ) = (
            preprocessor.windows_data_dict["timeseries"]["windows"],
            preprocessor.windows_data_dict["discrete"]["windows"],
            preprocessor.windows_data_dict["continuous"]["windows"],
            preprocessor.windows_data_dict["timestamp_conditions"]["windows"],
        )

        # Note 1: These data are not embeddings. They are raw data.
        # Note 2: (batch, window_size, channels) -> (batch, channels, window_size)
        # Note 3: The API does not currently support batched requests
        # However, the model and the preprocessing pipeline are designed to handle batches.
        # Therefore, a batch of size 1 is used.
        # We are following the conventions of `forecast_via_decoders()`
        batch = {
            "timeseries_full": np.swapaxes(timeseries_full, 1, 2),
            "discrete_label_embedding": discrete_conditions,
            "continuous_label_embedding": continuous_conditions,
            "timestamp_label_embedding": timestamp_embeddings,
        }

        x_axis_values = TimeStamps(
            name=x_axis_name,
            values=list(df[request.past_timestamps.name].values)
            if request.past_timestamps
            else list(range(len(df))),
            # safer than df.index in case index is jumbled
        )
        # convert x_axis_values to isoformat for JSON serialization
        if request.past_timestamps:
            x_axis_values.values = (
                convert_list_to_isoformat(x_axis_values.values)
                if not isinstance(request.past_timestamps.values[0], int)
                and not isinstance(request.past_timestamps.values[0], float)
                else [int(i) for i in x_axis_values.values]
            )

        return x_axis_values, batch

    def _forecast(self, batch: dict) -> np.ndarray:
        """
        Run the search on the metadata
        inputs: request - ForecastRequest object
        outputs: timeseries - concatenated history and forecast
        """
        return self.experiment.forecast_one_window(
            model_checkpoint_path=self.settings.forecast_model_path, batch=batch
        )
