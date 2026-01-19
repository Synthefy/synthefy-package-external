from typing import List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
import logging

from synthefy_pkg.anomaly_detection.anomaly_detector import AnomalyDetector
from synthefy_pkg.app.data_models import SynthefyTimeSeriesWindow, TimeStamps

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled=True
logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)  

COMPILE = True

class SynthefyAnomalyDetector(AnomalyDetector):
    def __init__(self, threshold: float = 3.0, interval_width: float = 0.95):
        self.threshold = threshold
        self.interval_width = interval_width

    def _fit_and_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        inputs:
            data: pd.DataFrame
        outputs:
            anomalies: pd.DataFrame
        description:
            Fits a prophet model to the input data and detects anomalies based
            on magnitude of residual. Returns a dataframe of anomaly timestamps
            with original input information.
        """
        model = Prophet(interval_width=self.interval_width)
        model.fit(data)

        # predict with the fitted model
        forecast = model.predict(data)
        forecast.reset_index(inplace=True)
        data.reset_index(inplace=True)

        # replace ds and y columns
        forecast["ds"] = data["ds"]
        forecast["y"] = data["y"]

        # Place anomaly indicators
        forecast["anomaly"] = 0

        forecast["residuals"] = np.abs(forecast["y"] - forecast["yhat"])

        residuals_std = forecast["residuals"].std()

        forecast["anomaly"] = (forecast["residuals"] > (self.threshold * residuals_std))

        # Extract anomalies as a separate dataframe
        anomalies = forecast[forecast["anomaly"] == 1].copy()

        return anomalies

    def _get_synthefy_timeseries_window(
        self, window: SynthefyTimeSeriesWindow, anomalies_by_channel: List[pd.DataFrame]
    ) -> Tuple[SynthefyTimeSeriesWindow, TimeStamps]:
        """
        inputs:
            window: SynthefyTimeSeriesWindow
            anomalies_by_channel: List[pd.DataFrame]
        outputs:
            winodw: SynthefyTimeSeriesWindow
        description:
            Populates a SynthefyTimeSeriesWindow object with all the original information
            from the input window as well as anomaly timestamps from anomalies_by_channel.
        TODO: Probably some kind of failure if no anomalies are detected.
        """
        # Compute the union of ds column
        ds_union = set()

        for df in anomalies_by_channel:
            ds_union.update(df["ds"])

        ds_union_list = list(ds_union)

        # Set the anomalous timestamps
        window_ts = TimeStamps(name="timestamps", values=sorted(ds_union_list))

        return window, window_ts
    
    def _threshold_anomalies_based_on_zscore(self, anomalies_by_channel: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        inputs:
            anomalies_by_channel: List[pd.DataFrame]
        outputs:
            anomalies_by_channel: List[pd.DataFrame]
        description:
            Thresholds anomalies based on modified z-score.
        """

        thresholded_anomalies_by_channel = []
        for df in anomalies_by_channel:
            residuals_median = df["residuals"].median()
            mad = np.median(np.abs(df["residuals"] - residuals_median))

            if mad == 0:
                # Case 1: If all residuals are 0, there are no anomalies
                if residuals_median == 0:
                    df["anomaly"] = False
                # Case 2: If all residuals are equal but non-zero, everything might be anomalous
                else:
                    df["anomaly"] = True
                df["modified_zscore"] = 0  # or np.nan to indicate special case
            else:
                # Normal case: calculate modified z-score
                df["modified_zscore"] = 0.6745 * (df["residuals"] - residuals_median) / mad
                df["anomaly"] = df["modified_zscore"].abs() > self.threshold

            df = df[df["anomaly"] == 1]
            thresholded_anomalies_by_channel.append(df)

        return thresholded_anomalies_by_channel

    def _get_ranking_metric(self, anomalies_by_channel: List[pd.DataFrame]) -> float:
        """
        inputs:
            anomalies_by_channel: List[pd.DataFrame]
        outputs:
            max_residual: float
        description:
            Returns a metric used for ranking a particular set of anomalies per channel
            among other time series windows. In this case, uses the max across channels
            and timestamps.
        """
        max_residual = float("-inf")

        for df in anomalies_by_channel:
            max_residual = max(max_residual, df["residuals"].max())

        return max_residual

    def detect(
        self, window: SynthefyTimeSeriesWindow
    ) -> Tuple[SynthefyTimeSeriesWindow, TimeStamps, float]:
        """
        inputs:
            window: SynthefyTimeSeriesWindow
        outputs:
           Tuple[SynthefyTimeSeriesWindow, float]
        description:
            Uses Prophet to detect anomalies in input window. Returns the
            original input window with anomaly timestamps as well as a real
            number to use for ranking this anomaly among others.
        """
        # Convert input SynthefyTimeSeriesWindow into a list of channel_dfs
        channel_dfs, channel_names = self._window_to_df(window)

        # fit model to the data and collect anomalies
        anomalies_by_channel = list(map(self._fit_and_predict, channel_dfs))

        # threshold anomalies based on modified z-score
        anomalies_by_channel = self._threshold_anomalies_based_on_zscore(anomalies_by_channel)

        output_window, output_window_ts = self._get_synthefy_timeseries_window(
            window, anomalies_by_channel
        )
        ranking_metric = self._get_ranking_metric(anomalies_by_channel)

        return output_window, output_window_ts, ranking_metric
