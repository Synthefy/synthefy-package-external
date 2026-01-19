from typing import List

from loguru import logger

from synthefy_pkg.app.config import ViewSettings
from synthefy_pkg.app.data_models import (
    MetaData,
    SynthefyRequest,
    SynthefyResponse,
    SynthefyTimeSeriesWindow,
    TimeStamps,
)
from synthefy_pkg.app.utils.api_utils import (
    array_to_continuous,
    array_to_discrete,
    array_to_timeseries,
    convert_list_to_isoformat,
)
from synthefy_pkg.ts_search.ts_view import TimeSeriesView

COMPILE = True


class ViewService:
    settings: ViewSettings
    viewer: TimeSeriesView

    def __init__(self, settings: ViewSettings):
        logger.info("Initializing ViewService")
        self.settings = settings
        self.viewer = TimeSeriesView(self.settings.preprocess_config_path)
        self.window_prefix = settings.window_naming_config

        logger.info("Initialized ViewService")

    def get_time_series_view(self, request: SynthefyRequest) -> SynthefyResponse:
        """
        inputs:
            request: SynthefyRequest object indicating user's request for viewing time series data
        outputs:
            view_response: SynthefyResponse object
        description:
            It then fetches the data according to the post request and returns the response.
        """
        logger.info("Validating request")
        request = self._validate_request(request)  # error handling

        # TODO: add an option to use metadata/timestamps ranges instead of text
        logger.info("Validated request - viewing time series")

        processed_arrays_dict = self.viewer.view_ts(
            request.text, request.n_view_windows
        )
        logger.info("Done with viewing time series - converting to response")
        top_timestamps_windows = processed_arrays_dict.pop("timestamp")
        top_timeseries_windows = processed_arrays_dict.pop("timeseries")
        top_continuous_windows = processed_arrays_dict.pop("continuous")
        top_discrete_windows = processed_arrays_dict.pop("discrete")

        x_axis_name = (
            self.viewer.timestamps_col[0] if self.viewer.timestamps_col else "index"
        )

        synthefy_windows: List[SynthefyTimeSeriesWindow] = []

        for window_idx in range(len(top_timeseries_windows)):
            timestamps = TimeStamps(
                name=x_axis_name,
                values=convert_list_to_isoformat(
                    top_timestamps_windows[window_idx].reshape(-1)
                ),
            )
            timeseries_list = array_to_timeseries(
                top_timeseries_windows[window_idx],
                channel_names=self.viewer.timeseries_cols,
            )

            continuous_list = array_to_continuous(
                top_continuous_windows[window_idx],
                self.viewer.continuous_cols,
            )
            discrete_list = array_to_discrete(
                top_discrete_windows[window_idx],
                self.viewer.original_discrete_cols,
            )

            synthefy_windows.append(
                SynthefyTimeSeriesWindow(
                    id=window_idx,
                    name=f"{self.window_prefix} {window_idx}",
                    timestamps=timestamps,
                    timeseries_data=timeseries_list,
                    metadata=MetaData(
                        discrete_conditions=discrete_list,
                        continuous_conditions=continuous_list,
                    ),
                    text="Example of View Text",
                )
            )

        logger.info("Done with converting to response")

        # TODO - return a parsed version of the text so the user can see if the text was extracted correctly
        return SynthefyResponse(
            windows=synthefy_windows,
            anomaly_timestamps=[],
            forecast_timestamps=[],
            combined_text=f"Results for '{request.text}'",
        )

    def _validate_request(self, request: SynthefyRequest) -> SynthefyRequest:
        """
        inputs:
            request: SynthefyRequest object
        outputs:
            view_request: SynthefyRequest object
        description:
            This function converts the request dictionary to a SynthefyRequest object.
            It raises an exception if the format is incorrect or necessary fields are missing.
        """

        # Validate the request
        if request.text is None or request.text == "":
            raise ValueError("'text' must be provided.")

        # TODO integrate search set later
        # if request.search_set is not None:
        #     # only take train/val/test - throw exception if anything else is present
        #     if any(x not in ["train", "val", "test"] for x in request.search_set):
        #         raise ValueError(
        #             "Only 'train', 'val', 'test' are allowed in 'search_set'."
        #         )
        # else:
        #     request.search_set = ["train", "val", "test"]

        return request
