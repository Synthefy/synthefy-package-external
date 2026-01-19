"""Abstraction/parent class for anomaly detector, regardless of what model it uses (prophet vs synthefy, etc)."""

from typing import List, Tuple

import pandas as pd

from synthefy_pkg.app.data_models import SynthefyTimeSeriesWindow

COMPILE = True


class AnomalyDetector:
    """
    A parent class defining what anomaly detectors should do.
    Maybe this should be a protocol instead-- we should decide how to type this.
    """

    def detect(
        self, data: SynthefyTimeSeriesWindow
    ) -> Tuple[SynthefyTimeSeriesWindow, float]:
        """
        inputs:
            data: SynthefyTimeSeriesWindow - Window to do anomaly detection on
        outputs:
            Tuple[SynthefyTimeSeriesWindow, float] - A window with anomaly information
            and a real number to use for ranking.
        description:
            Performs anomaly detection on a single SynthefyTimeSeriesWindow. Also returned
            is a real number used for ranking the potential anomalies returned in this
            SynthefyTimeSeriesWindow
        """
        raise NotImplementedError("detect method must be implemented in subclass.")

    def _window_to_df(
        self, window: SynthefyTimeSeriesWindow
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        inputs:
            window: SynthefyTimeSeriesWindow
        outputs:
            Tuple[List[pd.DataFrame], List[str]]
        description:
            Given a SynthefyTimeSeriesWindow, returns each channel as a dataframe consisting
            of 'ds' and 'y' columns, along with paired column names.
            NOTE: Metadata in the input window is ignored.
        """
        ds = (
            pd.Series(window.timestamps.values)
            if window.timestamps
            else pd.Series(list(range(len(window.timeseries_data[0].values))))
        )

        # Remove the timezone from ds
        ds = pd.to_datetime(ds, errors="coerce")
        ds = ds.dt.tz_localize(None)

        channel_dfs = []
        channel_names = []

        for channel_ts in window.timeseries_data:
            channel_values = channel_ts.values

            df = pd.DataFrame({"ds": ds, "y": channel_values})

            channel_dfs.append(df)
            channel_names.append(channel_ts.name)

        return channel_dfs, channel_names
