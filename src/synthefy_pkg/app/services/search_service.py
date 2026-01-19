from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from synthefy_pkg.app.config import SearchSettings
from synthefy_pkg.app.data_models import (
    MetaData,
    SearchRequest,
    SearchResponse,
    SynthefyRequest,
    SynthefyResponse,
    TimeStamps,
)
from synthefy_pkg.app.utils.api_utils import (
    array_to_continuous,
    array_to_discrete,
    array_to_timeseries,
    check_scale_by_metadata_used,
    convert_list_to_isoformat,
    convert_synthefy_request_to_search_request,
    create_synthefy_response_from_other_types,
    inverse_transform_discrete,
    timeseries_to_dataframe,
)
from synthefy_pkg.ts_search.ts_search import TimeSeriesSearch
from synthefy_pkg.utils.scaling_utils import (
    load_discrete_encoders,
    load_timeseries_scalers,
    transform_using_scaler,
)

COMPILE = True


class SearchService:
    dataset_name: str
    settings: SearchSettings

    def __init__(self, dataset_name: str, settings: SearchSettings):
        self.dataset_name = dataset_name
        self.settings = settings
        self.ts_searcher = TimeSeriesSearch(
            config_path=self.settings.preprocess_config_path
        )

    def validate_scalers(self):
        """
        Fail if `scale_by_metadata` was used in the preprocessing
        """
        timeseries_scalers = load_timeseries_scalers(self.dataset_name)
        if check_scale_by_metadata_used(timeseries_scalers):
            raise ValueError(
                "Search does not support scalers by discrete metadata"
            )

    async def search(self, request: SynthefyRequest) -> SynthefyResponse:
        """
        Parameteres:
            - request (SynthefyRequest): object indicating user's request for search
        Returns:
            - search_response (SynthefyResponse)
        description:
            This function searches the timeseries windows for the query/metadata conditions
        """
        logger.info("Loading scalers and encoders for search")
        # get the encoders/scalers
        # Don't handle search when we have scalers by discrete metadata
        # TODO @Bekzat - temporary change until we make the preprocessing encoding NOT use the scale by metadata
        self.validate_scalers()

        encoders = load_discrete_encoders(self.dataset_name)
        logger.info(
            "Loaded scalers and encoders for search - preprocessing now"
        )
        # convert to SearchRequest
        search_request = convert_synthefy_request_to_search_request(request)

        x_axis_name, query_np, n_closest, search_set = self._preprocess_request(
            search_request
        )

        query_np_scaled = transform_using_scaler(
            windows=np.expand_dims(query_np.copy(), axis=0),
            timeseries_or_continuous="timeseries",
            dataset_name=self.dataset_name,
            inverse_transform=False,
            transpose_timeseries=False,
        )
        logger.info("Preprocessed request - searching now")
        (
            _,
            _,
            closest_windows_timestamps,
            closest_windows,
            closest_continuous_conditions,
            closest_discrete_conditions,
        ) = self.ts_searcher.search(
            query_np_scaled,
            n_closest=n_closest,
            search_set=search_set,
        )
        logger.info(f"found {len(closest_windows)} closest windows")
        logger.info("Done with searching - converting to SearchResponse now")
        response = self.convert_to_search_response(
            x_axis_name,
            query_np,
            search_request.search_metadata,
            search_request.search_timestamps,
            closest_windows,
            closest_windows_timestamps,
            closest_continuous_conditions,
            closest_discrete_conditions,
            encoders,
        )
        logger.info(
            "Done with converting to SearchResponse - converting to SynthefyResponse now"
        )

        return await create_synthefy_response_from_other_types(
            self.dataset_name, [response], text=request.text
        )

    def _preprocess_request(
        self, request: SearchRequest
    ) -> Tuple[str, np.ndarray, int, List[str]]:
        """
        Parameteres:
            - request (SearchRequest): object
        Returns:
            - x_axis_name (str): name of the x-axis
            - query_np (np.ndarray): query time series window (2d of shape (window_size, n_channels)) to search for
            - n_closest (int): number of closest windows to return
            - search_set (List[str])
        description:
            This function converts the request dictionary to a SearchRequest object.
            It raises an exception if the format is incorrect or necessary fields are missing.
            It also preprocesses the request to make it easier to use for the model.
        """
        query_df = timeseries_to_dataframe(
            timeseries=request.search_query, timestamps=None
        )
        query_np = query_df.values
        x_axis_name = (
            request.search_timestamps.name
            if request.search_timestamps.name
            else "index"
        )
        n_closest = request.n_closest
        search_set = request.search_set
        return x_axis_name, query_np, n_closest, search_set

    def _search(self, request: SearchRequest):
        """
        Run the search on the metadata
        inputs: request - SearchRequest object
        outputs: search_response - SearchResponse object
        """
        # TODO
        pass

    def convert_to_search_response(
        self,
        x_axis_name: str,
        query_response: np.ndarray,
        query_metadata: MetaData,
        query_timestamps: TimeStamps,
        closest_windows: np.ndarray,
        closest_windows_timestamps: np.ndarray,
        closest_continuous_conditions: Optional[np.ndarray],
        closest_discrete_conditions: Optional[np.ndarray],
        encoders: Dict[str, Dict[str, Any]],
    ) -> SearchResponse:
        """
        Convert the timeseries prediction to the SearchResponse format
        Parameteres:
            - x_axis_name: str
            - query_response (np.ndarray): The query time series window (2d of shape (window_size, n_channels))
            - closest_windows: np.ndarray
            - closest_windows_timestamps: np.ndarray
            - closest_continuous_conditions: np.ndarray
            - closest_discrete_conditions: np.ndarray
            - encoders: Dict[str, Dict[str, Any]]
        outputs:
            SearchResponse object
        """
        logger.info("Done with searching - converting to response now")

        x_axis_values = (
            convert_list_to_isoformat(query_timestamps.values)
            if not isinstance(query_timestamps.values[0], int)
            and not isinstance(query_timestamps.values[0], float)
            else [int(i) for i in query_timestamps.values]
        )
        x_axis_response = [TimeStamps(name=x_axis_name, values=x_axis_values)]
        # add the query to the respond window
        timeseries_response = [
            array_to_timeseries(
                query_response,
                channel_names=[
                    f"{i}_query"
                    for i in self.ts_searcher.dataset.timeseries_cols
                ],
            )
        ]
        metadata_response = [query_metadata]

        if closest_discrete_conditions is not None:
            decoded_discrete_col_names, decoded_discrete_conditions = (
                inverse_transform_discrete(
                    closest_discrete_conditions, encoders
                )
            )
        else:
            decoded_discrete_col_names, decoded_discrete_conditions = (
                [],
                closest_discrete_conditions,
            )

        closest_windows = transform_using_scaler(
            windows=closest_windows,
            timeseries_or_continuous="timeseries",
            dataset_name=self.dataset_name,
            inverse_transform=True,
            transpose_timeseries=False,
        )
        if closest_continuous_conditions is not None:
            closest_continuous_conditions = transform_using_scaler(
                windows=closest_continuous_conditions,
                timeseries_or_continuous="continuous",
                dataset_name=self.dataset_name,
                inverse_transform=True,
            )

        for window_idx in range(len(closest_windows)):
            # Check if the timestamps are sequential integers starting from 0
            x_axis_values = (
                convert_list_to_isoformat(
                    closest_windows_timestamps[window_idx].reshape(-1)
                )
                if not isinstance(query_timestamps.values[0], int)
                and not isinstance(query_timestamps.values[0], float)
                else [
                    int(i)
                    for i in closest_windows_timestamps[window_idx].reshape(-1)
                ]
            )
            x_axis_response.append(
                TimeStamps(
                    name=x_axis_name,
                    values=x_axis_values,
                )
            )
            timeseries_list = array_to_timeseries(
                closest_windows[window_idx],
                channel_names=self.ts_searcher.dataset.timeseries_cols,
            )

            continuous_list = (
                array_to_continuous(
                    closest_continuous_conditions[window_idx],
                    self.ts_searcher.dataset.continuous_cols,
                )
                if closest_continuous_conditions is not None
                else []
            )

            discrete_list = (
                array_to_discrete(
                    decoded_discrete_conditions[window_idx],
                    decoded_discrete_col_names,
                )
                if decoded_discrete_conditions is not None
                else []
            )

            timeseries_response.append(timeseries_list)
            metadata_response.append(
                MetaData(
                    discrete_conditions=discrete_list,
                    continuous_conditions=continuous_list,
                )
            )

        logger.info(
            f"Done with converting to SearchResponse - len(timeseries_response) = {len(timeseries_response)}, len(metadata_response) = {len(metadata_response)}"
        )
        return SearchResponse(
            x_axis=x_axis_response,
            timeseries_data=timeseries_response,
            metadata=metadata_response,
            text="Example of Search Text",
        )
