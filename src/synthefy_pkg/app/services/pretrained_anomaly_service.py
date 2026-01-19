import heapq

import pandas as pd
import yaml
from loguru import logger

from synthefy_pkg.anomaly_detection.synthefy_anomaly_detector import (
    SynthefyAnomalyDetector,
)
from synthefy_pkg.app.config import PreTrainedAnomalySettings
from synthefy_pkg.app.data_models import (
    SynthefyRequest,
    SynthefyResponse,
    WindowsSelectionOptions,
)
from synthefy_pkg.app.utils.api_utils import (
    convert_discrete_metadata_to_label_tuple,
    convert_list_to_isoformat,
    get_labels_description,
)

COMPILE = True


class PreTrainedAnomalyService:
    dataset_name: str
    settings: PreTrainedAnomalySettings

    def __init__(self, dataset_name: str, settings: PreTrainedAnomalySettings):
        self.settings = settings
        self.dataset_name = dataset_name

        self.window_prefix = settings.window_naming_config.get(
            "pretrained_anomaly_prefix", "Anomaly"
        )
        with open(self.settings.preprocess_config_path, "r") as f:
            preprocess_config = yaml.safe_load(f)
        self.preprocess_config = preprocess_config

    def pretrained_anomaly_detection(
        self, request: SynthefyRequest, streaming: bool = False
    ) -> SynthefyResponse:
        """
        inputs:
            request: PreTrainedAnomalyRequest object with data for anomaly detection
        outputs:
            pretrained_anomaly_response: PretrainedAnomalyResponse object
        description:
            This function runs anomaly detection on the given data using the pretrained model
        """
        request = self._preprocess_request(request)  # error handling
        return self._pretrained_anomaly_detection(request, streaming)

    def _preprocess_request(self, request: SynthefyRequest) -> SynthefyRequest:
        """
        inputs:
            request: PreTrainedAnomalyRequest object
        outputs:
            pretrained_anomaly_request: PreTrainedAnomalyRequest object
        description:
            This function converts the request dictionary to a PreTrainedAnomalyRequest object.
            It raises an exception if the format is incorrect or necessary fields are missing.
            It also preprocesses the request to make it easier to use for the model.
        """
        # TODO: Error handling
        return request

    def _pretrained_anomaly_detection(
        self, request: SynthefyRequest, streaming: bool = False
    ) -> SynthefyResponse:
        """
        inputs:
            request: PreTrainedAnomalyRequest object
        outputs:
            pretrained_anomaly_response: PretrainedAnomalyResponse
        description:
            Passes each time series in the request to the synthefy anomaly detector.
            Retrieves results and sorts the top n_anomalies results based on ranking
            metric returned by SynthefyAnomalyDetector
        """
        k = request.n_anomalies

        anomaly_detector = SynthefyAnomalyDetector(
            threshold=self.settings.anomaly_threshold
        )

        if (
            request.selected_windows.window_type
            != WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
        ):
            raise ValueError(
                "You can only synthesize based on the current view windows."
            )

        # get the selected windows for anomaly detection
        selected_window_indices = request.selected_windows.window_indices
        windows_for_anomaly_detection = [
            request.windows[idx] for idx in selected_window_indices
        ]
        logger.info(
            f"Using {len(windows_for_anomaly_detection)} windows for anomaly detection"
        )

        processed_windows = list(
            map(anomaly_detector.detect, windows_for_anomaly_detection)
        )

        top_k = heapq.nlargest(k, processed_windows, key=lambda x: x[2])
        logger.info(f"Found {len(top_k)} anomalies")
        top_k_sorted = sorted(top_k, key=lambda x: x[2])

        top_k_ts_windows = [x[0] for x in top_k_sorted]
        top_k_anomaly_ts = [x[1] for x in top_k_sorted]

        labels_description = get_labels_description(self.dataset_name)
        # add the window id and name to the windows
        for i in range(len(top_k_ts_windows)):
            top_k_ts_windows[i].id = i
            top_k_ts_windows[i].name = f"{self.window_prefix} {i}"
            top_k_ts_windows[i].text = ""

            # TODO: Need to support the case when with no timestamps.
            # For now raising error if the timestamp col contains indices instead of timestamps
            if (
                isinstance(top_k_ts_windows[i].timestamps.values, list)
                and len(top_k_ts_windows[i].timestamps.values) > 0
                and isinstance(top_k_ts_windows[i].timestamps.values[0], (int, float))
            ):
                raise ValueError(
                    "Timestamps should exist as datetime values, not just indices"
                )
            else:
                # Otherwise, safely convert all to datetime and then to ISO
                top_k_ts_windows[i].timestamps.values = convert_list_to_isoformat(
                    [
                        (
                            pd.to_datetime(ts).tz_convert("UTC")
                            if pd.to_datetime(ts).tzinfo
                            else pd.to_datetime(ts).tz_localize("UTC")
                        )
                        for ts in top_k_ts_windows[i].timestamps.values
                    ]
                )

        for i in range(len(top_k_anomaly_ts)):
            if (
                isinstance(top_k_anomaly_ts[i].values, list)
                and len(top_k_anomaly_ts[i].values) > 0
                and isinstance(top_k_anomaly_ts[i].values[0], (int, float))
            ):
                raise ValueError(
                    "Anomaly timestamps should exist as datetime values, not just indices"
                )
            else:
                top_k_anomaly_ts[i].values = convert_list_to_isoformat(
                    [
                        pd.to_datetime(ts).tz_localize("UTC")
                        for ts in top_k_anomaly_ts[i].values
                    ]
                )

            # Sort anomalies by timestamp
            top_k_anomaly_ts[i].values = sorted(top_k_anomaly_ts[i].values)

        # custom-ish for now since anomaly detection doesnt use the api_utils helper function that all other services use
        for i in range(len(top_k_ts_windows)):
            top_k_ts_windows[i].metadata.discrete_conditions = (
                convert_discrete_metadata_to_label_tuple(
                    top_k_ts_windows[i].metadata.discrete_conditions,
                    labels_description["group_labels_combinations"],
                    self.preprocess_config["group_labels"]["cols"],
                )
                if not streaming
                else top_k_ts_windows[i].metadata.discrete_conditions
            )

        response = SynthefyResponse(
            windows=top_k_ts_windows,
            anomaly_timestamps=top_k_anomaly_ts,
            combined_text=(
                "No Anomalies found" * k
                if len(top_k_anomaly_ts) == 0
                else f"Found {k} Anomalies"
            ),
        )
        return response
