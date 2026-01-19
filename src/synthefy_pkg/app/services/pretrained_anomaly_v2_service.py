import json
from typing import Any, Dict, Union

import pandas as pd
from loguru import logger

from synthefy_pkg.anomaly_detection.synthefy_anomaly_detector_v2 import \
    AnomalyDetector
from synthefy_pkg.app.data_models import (DynamicTimeSeriesData,
                                          PreTrainedAnomalyV2Request,
                                          PreTrainedAnomalyV2Response)

COMPILE = True


class PreTrainedAnomalyV2Service:
    def __init__(self, settings):
        self.settings = settings

    def pretrained_anomaly_detection(
        self, request: Union[PreTrainedAnomalyV2Request, DynamicTimeSeriesData]
    ) -> PreTrainedAnomalyV2Response:
        """
        inputs:
            request: PreTrainedAnomalyRequest object with data for anomaly detection
        outputs:
            pretrained_anomaly_response: PretrainedAnomalyResponse object
        description:
            This function runs anomaly detection on the given data using the pretrained model
        """
        df = self._preprocess_request(request)  # error handling
        return self._pretrained_anomaly_detection(df)

    def _set_anomaly_parameters(
        self,
        preprocess_config: Dict[str, Any],
        request: Union[PreTrainedAnomalyV2Request, DynamicTimeSeriesData],
    ) -> None:
        if isinstance(request, PreTrainedAnomalyV2Request):
            # Use request parameters directly if provided
            self.num_anomalies_limit = request.num_anomalies_limit
            self.min_anomaly_score = request.min_anomaly_score
            self.n_jobs = request.n_jobs
        elif isinstance(request, DynamicTimeSeriesData):
            # Fall back to config values for DynamicTimeSeriesData
            self.num_anomalies_limit = (
                preprocess_config.get("anomaly_detection", {})
                .get("settings", {})
                .get("num_anomalies_limit", 50)
            )
            self.min_anomaly_score = (
                preprocess_config.get("anomaly_detection", {})
                .get("settings", {})
                .get("min_anomaly_score", None)
            )
            self.n_jobs = (
                preprocess_config.get("anomaly_detection", {})
                .get("settings", {})
                .get("n_jobs", -1)
            )

    def _preprocess_request(
        self, request: Union[PreTrainedAnomalyV2Request, DynamicTimeSeriesData]
    ) -> pd.DataFrame:
        """
        Preprocess the request by validating and converting the root dictionary to a DataFrame.

        Args:
            request: Union[PreTrainedAnomalyV2Request, DynamicTimeSeriesData]

        Returns:
            DataFrame converted from the root dictionary

        Raises:
            ValueError: If root is empty or contains invalid data
            KeyError: If required columns are missing
            TypeError: If data types are incorrect
        """
        try:
            # Check if root is empty
            if not request.root:
                raise ValueError("Root dictionary cannot be empty")

            # Convert root dictionary to DataFrame
            df = pd.DataFrame.from_dict(request.root)

            # Load preprocessing config
            with open(self.settings.preprocess_config_path, "r") as f:
                preprocess_config = json.load(f)

            self._set_anomaly_parameters(preprocess_config, request)

            # Basic validation
            if df.empty:
                raise ValueError("Converted DataFrame is empty")

            # Check for required columns (add your specific requirements)
            if preprocess_config["timestamps_col"][0] not in df.columns:
                raise KeyError(
                    f"Timestamps required for anomaly detection, please add: {preprocess_config['timestamps_col']}"
                )

            required_columns = preprocess_config["timeseries"]["cols"]
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise KeyError(f"Missing required columns: {missing_columns}")

            return df

        except Exception as e:
            logger.error(f"Error preprocessing request: {str(e)}")
            raise ValueError(
                f"Failed to preprocess anomaly_v2 request: {str(e)}"
            ) from e

    def _pretrained_anomaly_detection(
        self, df: pd.DataFrame
    ) -> PreTrainedAnomalyV2Response:
        """
        Performs anomaly detection using the pretrained model.

        Args:
            request: The validated anomaly detection request
            df: Preprocessed DataFrame containing the time series data

        Returns:
            PreTrainedAnomalyV2Response containing detected anomalies in the format:
            {kpi_name: {anomaly_type: {group_key: List[AnomalyMetadata]}}}

        Raises:
            ValueError: If anomaly detection fails
        """
        try:
            anomaly_detector = AnomalyDetector(
                config_source=self.settings.preprocess_config_path,
            )

            results, concurrent_results = anomaly_detector.detect_anomalies(
                df=df,
                num_anomalies_limit=self.num_anomalies_limit,
                min_anomaly_score=self.min_anomaly_score,
                n_jobs=self.n_jobs,
            )

            # Validate results structure matches expected response format
            if not isinstance(results, dict):
                raise ValueError("Anomaly detection results must be a dictionary")

            # Create and validate response
            response = PreTrainedAnomalyV2Response(
                results=results, concurrent_results=concurrent_results
            )
            return response

        except Exception as e:
            logger.error(f"Error during anomaly detection: {str(e)}")
            raise ValueError(f"Anomaly detection failed: {str(e)}") from e
