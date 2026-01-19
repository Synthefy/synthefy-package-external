"""Tests for synthefy anomaly detector."""

import os

import pytest

from synthefy_pkg.anomaly_detection.synthefy_anomaly_detector import (
    SynthefyAnomalyDetector,
)
from synthefy_pkg.app.data_models import (
    SynthefyRequest,
    SynthefyTimeSeriesWindow,
    TimeStamps,
)


@pytest.fixture
def synthefy_timeseries_window() -> SynthefyTimeSeriesWindow:
    """Loads a synthefy timeseries window from the test json."""
    with open(
        os.path.join(
            os.environ.get("SYNTHEFY_PACKAGE_BASE"),
            "src/synthefy_pkg/app/tests/test_jsons/dispatch_twamp_one_month_request.json",
        ),
        "r",
    ) as f:
        json_data = f.read()

    # I know its deprecated but I need it to work now
    synthefy_request = SynthefyRequest.parse_raw(json_data)
    return synthefy_request.windows[0]


@pytest.mark.usefixtures("synthefy_timeseries_window")
class TestSynthefyAnomalyDetection:
    def test_window_to_df(self, synthefy_timeseries_window):
        anomaly_detector = SynthefyAnomalyDetector()

        channel_dfs, channel_names = anomaly_detector._window_to_df(
            synthefy_timeseries_window
        )

        assert len(channel_dfs) == 5

        for channel_df in channel_dfs:
            assert "ds" in channel_df.columns and "y" in channel_df.columns

    def test_fit_and_predict(self, synthefy_timeseries_window):
        anomaly_detector = SynthefyAnomalyDetector()

        channel_dfs, channel_names = anomaly_detector._window_to_df(
            synthefy_timeseries_window
        )

        for channel_df in channel_dfs:
            anomaly_df = anomaly_detector._fit_and_predict(channel_df)

            if len(anomaly_df) > 0:
                assert (anomaly_df["anomaly"] == 1).all()

    def test_end_to_end(self, synthefy_timeseries_window):
        anomaly_detector = SynthefyAnomalyDetector()

        output_window, output_window_ts, ranking_metric = anomaly_detector.detect(
            synthefy_timeseries_window
        )

        assert isinstance(ranking_metric, float)
        assert isinstance(output_window_ts, TimeStamps)
        assert isinstance(output_window, SynthefyTimeSeriesWindow)
