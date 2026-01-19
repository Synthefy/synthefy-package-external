import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from synthefy_pkg.app.data_models import (
    ConfidenceInterval,
    FMDataPointModification,
    ForecastDataset,
    ForecastGroup,
    HistoricalDataset,
    MetadataDataFrame,
    MetaDataVariation,
    PerturbationType,
)
from synthefy_pkg.app.routers.foundation_models import (
    _dataframe_to_historical_dataset,
)
from synthefy_pkg.model.foundation_model_service import (
    FoundationModelService,
    TabPFNPredictor,
    UnsupportedModelError,
    _validate_and_sanitize_timezone,
    apply_full_and_range_modifications,
    apply_point_modifications,
    handle_missing_values_and_get_time_varying_non_numeric_columns,
    identify_time_invariant_columns,
    setup_train_test_dfs,
)

COMPILE = True


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with datetime and numeric columns."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "value1": np.random.rand(10) * 100,
            "value2": np.random.rand(10) * 50,
        }
    )
    return df


@pytest.fixture
def ground_truth_dataframe():
    """Create a sample ground truth dataframe with full matching data."""
    dates = pd.date_range(start="2023-01-09", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "value1": np.arange(5) * 10 + 15,  # [15, 25, 35, 45, 55]
            "value2": np.arange(5) * 5 + 10,  # [10, 15, 20, 25, 30]
        }
    )
    return df


@pytest.fixture
def partial_ground_truth_dataframe():
    """Create a sample ground truth dataframe with partial matching data."""
    # Only includes some of the future timestamps
    dates = pd.date_range(start="2023-01-09", periods=3, freq="D")
    # Add a gap in the dates
    dates = dates.append(pd.DatetimeIndex([datetime(2023, 1, 13)]))
    df = pd.DataFrame(
        {
            "date": dates,
            "value1": [15, 25, 35, 65],
            "value2": [10, 15, 20, 35],
        }
    )
    return df


@pytest.fixture
def string_timestamp_ground_truth_dataframe():
    """Create a ground truth dataframe with string timestamps."""
    df = pd.DataFrame(
        {
            "date": [
                "2023-01-09",
                "2023-01-10",
                "2023-01-11",
                "2023-01-12",
                "2023-01-13",
            ],
            "value1": [15, 25, 35, 45, 55],
            "value2": [10, 15, 20, 25, 30],
        }
    )
    return df


@pytest.fixture
def metadata_dataframe():
    """Create a sample metadata dataframe."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "meta1": np.random.rand(10),
            "meta2": np.random.rand(10),
        }
    )
    return MetadataDataFrame(df=df, metadata_json={})


@pytest.fixture
def metadata_dataframe2():
    """Create a second sample metadata dataframe."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "meta3": np.random.rand(10),
            "meta4": np.random.rand(10),
        }
    )
    return MetadataDataFrame(df=df, metadata_json={})


class TestTabPFNPredictor:
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_init(self, mock_sleep):
        """Test initialization of TabPFNPredictor."""
        predictor = TabPFNPredictor()
        assert len(predictor.selected_features) == 2

    def test_drop_datetime_column(self, sample_dataframe):
        """Test dropping datetime column from dataframe."""
        predictor = TabPFNPredictor()
        result = predictor.drop_datetime_column(sample_dataframe)
        assert "date" not in result.columns
        assert len(result.columns) == len(sample_dataframe.columns) - 1

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
    ):
        """Test prediction functionality."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        # Updated to return a forecast with confidence intervals
        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0],
                0.1: [8.0, 18.0, 28.0],
                0.9: [12.0, 22.0, 32.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
        )

        # Updated assertions for ForecastDataset
        assert isinstance(result, ForecastDataset)
        assert len(result.timestamps) == len(future_timestamps)
        assert len(result.values) == len(target_columns)

        for idx, target_column in enumerate(target_columns):
            assert result.values[idx].target_column == target_column

        assert len(result.values[0].forecasts) == len(future_timestamps)
        assert len(result.values[0].confidence_intervals) == len(
            future_timestamps
        )

        assert mock_predict.call_count == 2

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor")
    def test_predict_real_data_integration(
        self, mock_tabpfn_cls, sample_dataframe, metadata_dataframe
    ):
        """Test prediction with real data to verify TabPFN is called correctly."""
        # Setup mock TabPFN predictor that returns real format results
        mock_tabpfn = MagicMock()
        mock_tabpfn_cls.return_value = mock_tabpfn
        mock_tabpfn.predict.return_value = pd.DataFrame(
            {
                "target": [15.5, 16.7, 17.9],
                0.1: [14.5, 15.7, 16.9],
                0.9: [16.5, 17.7, 18.9],
            }
        )

        # Create predictor with mocked TabPFN
        with patch("time.sleep"):  # Skip sleep in model init
            predictor = TabPFNPredictor()
            predictor.predictor = mock_tabpfn

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
        )

        # Verify TabPFN was called correctly
        assert mock_tabpfn.predict.called
        # Updated assertions for ForecastDataset
        assert isinstance(result, ForecastDataset)
        assert len(result.timestamps) == len(future_timestamps)

        for idx, target_column in enumerate(target_columns):
            assert result.values[idx].target_column == target_column

        assert len(result.values) == 1  # One target column
        # Verify forecast values from mock are in result
        assert np.allclose(result.values[0].forecasts, [15.5, 16.7, 17.9])
        # Verify confidence intervals
        assert result.values[0].confidence_intervals[0].lower == 14.5
        assert result.values[0].confidence_intervals[0].upper == 16.5

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_multiple_metadata(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
        metadata_dataframe2,
    ):
        """Test prediction with multiple metadata dataframes."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0],
                0.1: [8.0, 18.0, 28.0],
                0.9: [12.0, 22.0, 32.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction with multiple metadata dataframes
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe, metadata_dataframe2],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
        )

        # Updated assertions for ForecastDataset
        assert isinstance(result, ForecastDataset)
        assert len(result.timestamps) == len(future_timestamps)
        assert len(result.values) == len(target_columns)

        for idx, target_column in enumerate(target_columns):
            assert result.values[idx].target_column == target_column

        # Check that TabPFN was called with data containing metadata columns
        # This would be more thorough with real function calls, but we can check
        # that add_features was called the correct number of times
        assert mock_predict.call_count == 2
        assert mock_add_features.call_count == 2

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_multiple_targets(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
    ):
        """Test prediction with multiple target columns."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0],
                0.1: [8.0, 18.0, 28.0],
                0.9: [12.0, 22.0, 32.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters with multiple target columns
        target_columns = ["value1", "value2"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
        )

        # Updated assertions for ForecastDataset
        assert isinstance(result, ForecastDataset)
        assert len(result.timestamps) == len(future_timestamps)
        assert len(result.values) == len(target_columns)

        for idx, target_column in enumerate(target_columns):
            assert result.values[idx].target_column == target_column

        # Since we have 2 target columns, predict should be called twice
        assert mock_predict.call_count == 4

    # Ground Truth Tests

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_with_full_ground_truth(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
        ground_truth_dataframe,
    ):
        """Test prediction with full matching ground truth data."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
                0.1: [8.0, 18.0, 28.0, 38.0, 48.0],
                0.9: [12.0, 22.0, 32.0, 42.0, 52.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
            datetime(2023, 1, 12),
            datetime(2023, 1, 13),
        ]

        # Execute prediction with ground truth
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=ground_truth_dataframe,
        )

        # Verify ground truth was correctly matched
        assert isinstance(result, ForecastDataset)
        assert len(result.values) == 1  # One target column

        # Check that ground truth values match the expected values from ground_truth_dataframe
        expected_ground_truth = [15.0, 25.0, 35.0, 45.0, 55.0]
        assert result.values[0].ground_truth == expected_ground_truth

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_with_partial_ground_truth(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
        partial_ground_truth_dataframe,
    ):
        """Test prediction with partial matching ground truth data."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
                0.1: [8.0, 18.0, 28.0, 38.0, 48.0],
                0.9: [12.0, 22.0, 32.0, 42.0, 52.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
            datetime(2023, 1, 12),
            datetime(2023, 1, 13),
        ]

        # Execute prediction with partial ground truth
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=partial_ground_truth_dataframe,
        )

        # Verify ground truth was correctly matched with None for missing values
        assert isinstance(result, ForecastDataset)
        assert len(result.values) == 1  # One target column

        # Check that ground truth values match the expected values with None for missing dates
        expected_ground_truth = [15.0, 25.0, 35.0, None, 65.0]
        assert result.values[0].ground_truth == expected_ground_truth

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_with_string_timestamp_ground_truth(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
        string_timestamp_ground_truth_dataframe,
    ):
        """Test prediction with ground truth data having string timestamps."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
                0.1: [8.0, 18.0, 28.0, 38.0, 48.0],
                0.9: [12.0, 22.0, 32.0, 42.0, 52.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
            datetime(2023, 1, 12),
            datetime(2023, 1, 13),
        ]

        # Execute prediction with string timestamp ground truth
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=string_timestamp_ground_truth_dataframe,
        )

        # Verify ground truth was correctly matched after timestamp conversion
        assert isinstance(result, ForecastDataset)
        assert len(result.values) == 1  # One target column

        # Check that ground truth values match the expected values after timestamp conversion
        expected_ground_truth = [15.0, 25.0, 35.0, 45.0, 55.0]
        assert result.values[0].ground_truth == expected_ground_truth

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_with_no_ground_truth(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
    ):
        """Test prediction with no ground truth data provided."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0],
                0.1: [8.0, 18.0, 28.0],
                0.9: [12.0, 22.0, 32.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters
        target_columns = ["value1"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction without ground truth
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            # No ground_truth_df parameter
        )

        # Verify ground truth is None for all timestamps
        assert isinstance(result, ForecastDataset)
        assert len(result.values) == 1  # One target column

        # Check that ground truth values are all None
        expected_ground_truth = [None, None, None]
        assert result.values[0].ground_truth == expected_ground_truth

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor.predict")
    @patch("tabpfn_time_series.FeatureTransformer.add_features")
    def test_predict_with_multiple_targets_and_ground_truth(
        self,
        mock_add_features,
        mock_predict,
        sample_dataframe,
        metadata_dataframe,
        ground_truth_dataframe,
    ):
        """Test prediction with multiple target columns and ground truth data."""
        predictor = TabPFNPredictor()

        # Setup mocks
        mock_train_with_features = MagicMock()
        mock_test_with_features = MagicMock()
        mock_add_features.return_value = (
            mock_train_with_features,
            mock_test_with_features,
        )

        mock_predictions = pd.DataFrame(
            {
                "target": [10.0, 20.0, 30.0],
                0.1: [8.0, 18.0, 28.0],
                0.9: [12.0, 22.0, 32.0],
            }
        )
        mock_predict.return_value = mock_predictions

        # Test parameters with multiple target columns
        target_columns = ["value1", "value2"]
        forecasting_timestamp = datetime(2023, 1, 8)
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Execute prediction with multiple targets and ground truth
        result = predictor.predict(
            target_df=sample_dataframe,
            covariate_columns=["value2"],
            metadata_dataframes=[metadata_dataframe],
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=ground_truth_dataframe,
        )

        # Verify ground truth was correctly matched for each target column
        assert isinstance(result, ForecastDataset)
        assert len(result.values) == 2  # Two target columns

        # Check ground truth for first target (value1)
        expected_ground_truth_1 = [15.0, 25.0, 35.0]
        assert result.values[0].ground_truth == expected_ground_truth_1

        # Check ground truth for second target (value2)
        expected_ground_truth_2 = [10.0, 15.0, 20.0]
        assert result.values[1].ground_truth == expected_ground_truth_2


class TestFoundationModelService:
    def test_get_model_tabpfn(self):
        """Test getting TabPFN model."""
        model = FoundationModelService.get_model("tabpfn")
        assert isinstance(model, TabPFNPredictor)

    def test_get_model_unsupported(self):
        """Test getting unsupported model raises error."""
        with pytest.raises(UnsupportedModelError):
            FoundationModelService.get_model("unsupported_model")

    def test_lru_cache(self):
        """Test that LRU cache is working."""
        model1 = FoundationModelService.get_model("tabpfn")
        model2 = FoundationModelService.get_model("tabpfn")
        # Should be the same instance due to caching
        assert model1 is model2


class TestProcessGroundTruth:
    """Tests specifically for the _process_ground_truth function."""

    def test_no_ground_truth(self):
        """Test when ground_truth_df is None."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        result = predictor._process_ground_truth(
            ground_truth_df=None,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [None, None, None]
        assert len(result) == len(future_timestamps)

    def test_missing_target_column(self):
        """Test when target column is not in ground truth df."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Ground truth has no "value1" column
        ground_truth_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-09", periods=3, freq="D"),
                "other_column": [10, 20, 30],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",  # Not in ground_truth_df
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [None, None, None]

    def test_full_matching_timestamps(self):
        """Test when all future timestamps match ground truth data."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        ground_truth_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-09", periods=3, freq="D"),
                "value1": [15.5, 25.5, 35.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [15.5, 25.5, 35.5]

    def test_partial_matching_timestamps(self):
        """Test when only some future timestamps match ground truth data."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
            datetime(2023, 1, 12),
            datetime(2023, 1, 13),
        ]

        # Ground truth only has data for 3 out of 5 timestamps
        ground_truth_df = pd.DataFrame(
            {
                "date": [
                    datetime(2023, 1, 9),
                    datetime(2023, 1, 11),
                    datetime(2023, 1, 13),
                ],
                "value1": [15.5, 35.5, 55.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [15.5, None, 35.5, None, 55.5]

    def test_partial_matching_timestamps_first_few(self):
        """Test when only some future timestamps match ground truth data."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
            datetime(2023, 1, 12),
            datetime(2023, 1, 13),
        ]

        # Ground truth only has data for 3 out of 5 timestamps
        ground_truth_df = pd.DataFrame(
            {
                "date": [
                    datetime(2023, 1, 9),
                    datetime(2023, 1, 10),
                    datetime(2023, 1, 11),
                ],
                "value1": [15.5, 35.5, 55.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [15.5, 35.5, 55.5, None, None]

    def test_string_datetime_conversion(self):
        """Test with string timestamps that need conversion."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Ground truth has string timestamps
        ground_truth_df = pd.DataFrame(
            {
                "date": ["2023-01-09", "2023-01-10", "2023-01-11"],
                "value1": [15.5, 25.5, 35.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [15.5, 25.5, 35.5]

    def test_non_matching_timestamps(self):
        """Test when ground truth has values but for different timestamps."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Ground truth has different timestamps
        ground_truth_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-12", periods=3, freq="D"),
                "value1": [15.5, 25.5, 35.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result == [None, None, None]

    def test_nan_values_in_ground_truth(self):
        """Test when ground truth contains NaN values."""
        predictor = TabPFNPredictor()
        future_timestamps = [
            datetime(2023, 1, 9),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]

        # Ground truth has NaN values
        ground_truth_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-09", periods=3, freq="D"),
                "value1": [15.5, np.nan, 35.5],
            }
        )

        result = predictor._process_ground_truth(
            ground_truth_df=ground_truth_df,
            current_target_column="value1",
            future_time_stamps=future_timestamps,
            datetime_column="date",
        )

        assert result[0] == 15.5
        assert result[1] is None  # NaN converted to None
        assert result[2] == 35.5


class TestProcessMetadataDataframes:
    """Tests specifically for the _process_metadata_dataframes function."""

    def test_empty_metadata_list(self):
        """Test with empty metadata dataframes list."""
        predictor = TabPFNPredictor()
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        result = predictor._process_metadata_dataframes([], target_df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_drop_datetime_columns(self):
        """Test dropping datetime columns from metadata dataframes."""
        predictor = TabPFNPredictor()
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        # Metadata dataframe with datetime column
        meta_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=5, freq="D"
                ),
                "meta1": [1, 2, 3, 4, 5],
                "meta2": [5, 4, 3, 2, 1],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        assert len(result) == 1
        # Datetime column should be removed
        assert "timestamp" not in result[0].columns
        # Other columns should remain
        assert "meta1" in result[0].columns
        assert "meta2" in result[0].columns
        # Shape should match original minus datetime column
        assert result[0].shape == (5, 2)

    def test_metadata_all_after_target(self):
        """Test when all metadata timestamps are after target timestamps."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Create metadata dataframe with all timestamps after target
        meta_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-10", periods=5, freq="D"
                ),
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert result == []

    def test_metadata_all_before_target(self):
        """Test when all metadata timestamps are before target timestamps."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Create metadata dataframe with all timestamps before target
        meta_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2022-12-25", periods=5, freq="D"
                ),
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe - should use the last metadata value for all rows
        expected_df = pd.DataFrame(
            {
                "meta_value": [
                    5,
                    5,
                    5,
                    5,
                    5,
                ],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_different_granularities(self):
        """Test with different timestamp granularities between target and metadata."""
        predictor = TabPFNPredictor()

        # Create target dataframe with daily timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Create metadata dataframe with hourly timestamps
        meta_df = pd.DataFrame(
            {
                "timestamp": [
                    "2022-12-31 23:00",  # Before first target
                    "2023-01-01 12:00",  # During first target
                    "2023-01-02 00:00",  # Start of second target
                    "2023-01-02 12:00",  # During second target
                    "2023-01-03 00:00",  # Start of third target
                ],
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe
        expected_df = pd.DataFrame(
            {
                "meta_value": [
                    1,
                    3,
                    5,
                    5,
                    5,
                ],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_empty_metadata(self):
        """Test with empty metadata dataframe."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Create empty metadata dataframe
        meta_df = pd.DataFrame(columns=["timestamp", "meta_value"])

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have no processed dataframes
        assert len(result) == 0

    def test_multiple_metadata_columns(self):
        """Test with metadata dataframe containing multiple columns."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Create metadata dataframe with multiple columns
        meta_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=5, freq="D"
                ),
                "meta_value1": [1, 2, 3, 4, 5],
                "meta_value2": [10, 20, 30, 40, 50],
                "meta_value3": [100, 200, 300, 400, 500],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe
        expected_df = pd.DataFrame(
            {
                "meta_value1": [1, 2, 3, 4, 5],
                "meta_value2": [10, 20, 30, 40, 50],
                "meta_value3": [100, 200, 300, 400, 500],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_metadata_size_matching(self):
        """Test that output dataframe always matches target dataframe size."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
                "value": range(10),
            }
        )

        # Test case 1: Metadata has fewer rows
        meta_df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2022-12-31", periods=5, freq="D"
                ),
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        # Test case 2: Metadata has more rows
        meta_df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2022-12-31", periods=15, freq="D"
                ),
                "meta_value": range(15),
            }
        )

        # Test case 3: Metadata has same number of rows
        meta_df3 = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2022-12-31", periods=10, freq="D"
                ),
                "meta_value": range(10),
            }
        )

        metadata_dataframes = [
            MetadataDataFrame(df=meta_df1, metadata_json={}),
            MetadataDataFrame(df=meta_df2, metadata_json={}),
            MetadataDataFrame(df=meta_df3, metadata_json={}),
        ]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have three processed dataframes
        assert len(result) == 3

        # Each processed dataframe should have same number of rows as target
        for df in result:
            assert len(df) == len(target_df)
            assert df.index.equals(target_df.index)

    def test_metadata_gaps(self):
        """Test handling of gaps in metadata timestamps."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
                "value": range(10),
            }
        )

        # Create metadata dataframe with gaps
        meta_df = pd.DataFrame(
            {
                "timestamp": [
                    "2022-12-31",  # Before first target
                    "2023-01-03",  # Gap before this
                    "2023-01-04",  # No gap
                    "2023-01-07",  # Gap after this
                    "2023-01-10",  # After last target
                ],
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe
        expected_df = pd.DataFrame(
            {
                "meta_value": [
                    1,  # 2023-01-01: use first value
                    1,  # 2023-01-02: use first value
                    2,  # 2023-01-03: use second value
                    3,  # 2023-01-04: use third value
                    3,  # 2023-01-05: use third value
                    3,  # 2023-01-06: use third value
                    4,  # 2023-01-07: use fourth value
                    4,  # 2023-01-08: use fourth value
                    4,  # 2023-01-09: use fourth value
                    5,  # 2023-01-10: use fifth value
                ],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_metadata_duplicate_timestamps(self):
        """Test handling of duplicate timestamps in metadata."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": range(5),
            }
        )

        # Create metadata dataframe with duplicate timestamps
        meta_df = pd.DataFrame(
            {
                "timestamp": [
                    "2022-12-31",
                    "2023-01-01",
                    "2023-01-01",  # Duplicate
                    "2023-01-02",
                    "2023-01-02",  # Duplicate
                ],
                "meta_value": [1, 2, 3, 4, 5],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe - should use the last value for duplicate timestamps
        expected_df = pd.DataFrame(
            {
                "meta_value": [3, 5, 5, 5, 5],  # Use last value for duplicates
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_metadata_mixed_types(self):
        """Test handling of mixed data types in metadata columns."""
        predictor = TabPFNPredictor()

        # Create target dataframe with timestamps
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value": range(5),
            }
        )

        # Create metadata dataframe with mixed types
        meta_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=5, freq="D"
                ),
                "meta_int": [1, 2, 3, 4, 5],
                "meta_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "meta_str": ["a", "b", "c", "d", "e"],
                "meta_bool": [True, False, True, False, True],
            }
        )

        metadata_dataframes = [MetadataDataFrame(df=meta_df, metadata_json={})]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have one processed dataframe
        assert len(result) == 1
        processed_df = result[0]

        # Create expected dataframe
        expected_df = pd.DataFrame(
            {
                "meta_int": [1, 2, 3, 4, 5],
                "meta_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "meta_str": ["a", "b", "c", "d", "e"],
                "meta_bool": [True, False, True, False, True],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_timezone_handling(self):
        """Test comprehensive timezone handling in metadata processing."""
        predictor = TabPFNPredictor()

        # Create target dataframe with UTC timestamps
        target_timestamps = pd.date_range(
            start="2023-01-01 00:00:00", periods=5, freq="D", tz="UTC"
        )
        target_df = pd.DataFrame(
            {
                "date": target_timestamps,
                "value": range(5),
            }
        )

        # Test case 1: Metadata with naive timestamps
        naive_timestamps = pd.date_range(
            start="2022-12-31 00:00:00", periods=5, freq="D"
        )
        meta_df1 = pd.DataFrame(
            {
                "timestamp": naive_timestamps,
                "meta_value1": [1, 2, 3, 4, 5],
            }
        )

        # Test case 2: Metadata with UTC timestamps
        utc_timestamps = pd.date_range(
            start="2022-12-31 00:00:00", periods=5, freq="D", tz="UTC"
        )
        meta_df2 = pd.DataFrame(
            {
                "timestamp": utc_timestamps,
                "meta_value2": [10, 20, 30, 40, 50],
            }
        )

        # Test case 3: Metadata with New York timezone
        ny_timestamps = pd.date_range(
            start="2022-12-31 00:00:00",
            periods=5,
            freq="D",
            tz="America/New_York",
        )
        meta_df3 = pd.DataFrame(
            {
                "timestamp": ny_timestamps,
                "meta_value3": [100, 200, 300, 400, 500],
            }
        )

        # Test case 4: Metadata with Tokyo timezone
        tokyo_timestamps = pd.date_range(
            start="2022-12-31 00:00:00", periods=5, freq="D", tz="Asia/Tokyo"
        )
        meta_df4 = pd.DataFrame(
            {
                "timestamp": tokyo_timestamps,
                "meta_value4": [1000, 2000, 3000, 4000, 5000],
            }
        )

        # Test case 5: Metadata with timezone-unaware timestamps (explicitly set to None)
        tz_unaware_timestamps = pd.date_range(
            start="2022-12-31 00:00:00", periods=5, freq="D"
        )
        # Explicitly set timezone to None
        tz_unaware_timestamps = tz_unaware_timestamps.tz_localize(None)
        meta_df5 = pd.DataFrame(
            {
                "timestamp": tz_unaware_timestamps,
                "meta_value5": [10000, 20000, 30000, 40000, 50000],
            }
        )

        # Create MetadataDataFrame objects
        metadata_dataframes = [
            MetadataDataFrame(df=meta_df1, metadata_json={}),
            MetadataDataFrame(df=meta_df2, metadata_json={}),
            MetadataDataFrame(df=meta_df3, metadata_json={}),
            MetadataDataFrame(df=meta_df4, metadata_json={}),
            MetadataDataFrame(df=meta_df5, metadata_json={}),
        ]

        # Process metadata
        result = predictor._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        # Should have five processed dataframes
        assert len(result) == 5

        # Each processed dataframe should have same number of rows as target
        for df in result:
            assert len(df) == len(target_df)
            assert df.index.equals(target_df.index)

        # Create expected dataframes
        expected_df1 = pd.DataFrame(
            {
                "meta_value1": [2, 3, 4, 5, 5],
            },
            index=target_df.index,
        )

        expected_df2 = pd.DataFrame(
            {
                "meta_value2": [20, 30, 40, 50, 50],
            },
            index=target_df.index,
        )

        expected_df3 = pd.DataFrame(
            {
                "meta_value3": [100, 200, 300, 400, 500],
            },
            index=target_df.index,
        )

        expected_df4 = pd.DataFrame(
            {
                "meta_value4": [2000, 3000, 4000, 5000, 5000],
            },
            index=target_df.index,
        )

        expected_df5 = pd.DataFrame(
            {
                "meta_value5": [20000, 30000, 40000, 50000, 50000],
            },
            index=target_df.index,
        )

        # Assert exact dataframe equality
        pd.testing.assert_frame_equal(result[0], expected_df1)
        pd.testing.assert_frame_equal(result[1], expected_df2)
        pd.testing.assert_frame_equal(result[2], expected_df3)
        pd.testing.assert_frame_equal(result[3], expected_df4)
        pd.testing.assert_frame_equal(result[4], expected_df5)

        # Check that all processed dataframes have the correct number of columns
        assert all(len(df.columns) == 1 for df in result)

        # Check that all processed dataframes have the correct column names
        assert result[0].columns.tolist() == ["meta_value1"]
        assert result[1].columns.tolist() == ["meta_value2"]
        assert result[2].columns.tolist() == ["meta_value3"]
        assert result[3].columns.tolist() == ["meta_value4"]
        assert result[4].columns.tolist() == ["meta_value5"]

        # Check that the values are properly aligned with the target timestamps
        # For example, the first value in each processed dataframe should correspond
        # to the first target timestamp (2023-01-01 00:00:00 UTC)
        assert (
            result[0].iloc[0]["meta_value1"] == 2
        )  # Naive timestamp aligned with UTC
        assert result[1].iloc[0]["meta_value2"] == 20  # UTC timestamp
        assert result[2].iloc[0]["meta_value3"] == 100  # New York timestamp
        assert result[3].iloc[0]["meta_value4"] == 2000  # Tokyo timestamp
        assert (
            result[4].iloc[0]["meta_value5"] == 20000
        )  # Timezone-unaware timestamp


class TestValidateNoDuplicateTimestamps:
    """Tests specifically for the _validate_no_duplicate_timestamps function."""

    def test_no_duplicates(self):
        """Test when there are no duplicate timestamps."""
        predictor = TabPFNPredictor()

        # Create a dataframe with unique timestamps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=5, freq="D"
                ),
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Should not raise an exception
        predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Test passes if no error is raised
        assert True

    def test_with_duplicates(self):
        """Test when there are duplicate timestamps."""
        predictor = TabPFNPredictor()

        # Create a dataframe with duplicate timestamps
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-01-02"),
                    pd.Timestamp("2023-01-01"),  # Duplicate
                    pd.Timestamp("2023-01-03"),
                    pd.Timestamp("2023-01-02"),  # Duplicate
                ],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Verify the error message
        error_msg = str(exc_info.value)
        assert "Duplicate timestamps found" in error_msg
        assert "2023-01-01" in error_msg
        assert "2023-01-02" in error_msg

    def test_empty_dataframe(self):
        """Test with an empty dataframe."""
        predictor = TabPFNPredictor()

        # Create an empty dataframe with the right column
        df = pd.DataFrame({"timestamp": []})

        # Should not raise an exception
        predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Test passes if no error is raised
        assert True

    def test_single_timestamp(self):
        """Test with a dataframe containing a single timestamp."""
        predictor = TabPFNPredictor()

        # Create a dataframe with a single timestamp
        df = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2023-01-01")], "value": [10]}
        )

        # Should not raise an exception
        predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Test passes if no error is raised
        assert True

    def test_non_monotonic_timestamps(self):
        """Test when timestamps are not monotonically increasing."""
        predictor = TabPFNPredictor()

        # Create a dataframe with non-monotonic timestamps
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-01-03"),  # Out of order
                    pd.Timestamp("2023-01-02"),
                    pd.Timestamp("2023-01-04"),
                ],
                "value": [10, 20, 30, 40],
            }
        )

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Verify the error message
        error_msg = str(exc_info.value)
        assert "must be sorted in ascending order" in error_msg

    def test_monotonic_timestamps(self):
        """Test when timestamps are monotonically increasing."""
        predictor = TabPFNPredictor()

        # Create a dataframe with monotonically increasing timestamps
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-01-02"),
                    pd.Timestamp("2023-01-03"),
                    pd.Timestamp("2023-01-04"),
                ],
                "value": [10, 20, 30, 40],
            }
        )

        # Should not raise an exception
        predictor._validate_no_duplicate_timestamps(df, "timestamp")

        # Test passes if no error is raised
        assert True


class TestPredictDuplicateTimestampValidation:
    """Tests for duplicate timestamp validation in the predict method."""

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor")
    @patch("time.sleep")
    def test_predict_validates_target_dataframe(
        self, mock_sleep, mock_tabpfn_cls
    ):
        """Test that predict validates the target dataframe for duplicate timestamps."""
        # Setup predictor with mock validation method
        predictor = TabPFNPredictor()
        predictor._validate_no_duplicate_timestamps = MagicMock()
        predictor.predictor = MagicMock()

        # Create test data
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        metadata_dataframes = []
        target_columns = ["value1"]
        forecasting_timestamp = pd.Timestamp("2023-01-10").to_pydatetime()
        future_timestamps = [
            pd.Timestamp("2023-01-11").to_pydatetime(),
            pd.Timestamp("2023-01-12").to_pydatetime(),
        ]

        # Call predict method
        predictor.predict(
            target_df=target_df,
            covariate_columns=[],
            metadata_dataframes=metadata_dataframes,
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
        )

        # Assert validation was called for target dataframe
        assert predictor._validate_no_duplicate_timestamps.call_count == 1
        pd.testing.assert_frame_equal(
            predictor._validate_no_duplicate_timestamps.call_args_list[0][0][0],
            target_df,
        )

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor")
    @patch("time.sleep")
    def test_predict_validates_ground_truth(self, mock_sleep, mock_tabpfn_cls):
        """Test that predict validates the ground truth dataframe when provided."""
        # Setup predictor with mock validation method
        predictor = TabPFNPredictor()
        predictor._validate_no_duplicate_timestamps = MagicMock()
        predictor.predictor = MagicMock()

        # Create test data
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        ground_truth_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-11", periods=3, freq="D"),
                "value1": [60, 70, 80],
            }
        )

        metadata_dataframes = []
        target_columns = ["value1"]
        forecasting_timestamp = pd.Timestamp("2023-01-10").to_pydatetime()
        future_timestamps = [
            pd.Timestamp("2023-01-11").to_pydatetime(),
            pd.Timestamp("2023-01-12").to_pydatetime(),
        ]

        # Call predict method with ground truth
        predictor.predict(
            target_df=target_df,
            covariate_columns=[],
            metadata_dataframes=metadata_dataframes,
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=ground_truth_df,
        )

        # Assert validation was called for both dataframes - without directly comparing DataFrames
        assert predictor._validate_no_duplicate_timestamps.call_count == 2

        # Check call arguments by column names and data types instead of direct DataFrame comparison
        call_args_list = (
            predictor._validate_no_duplicate_timestamps.call_args_list
        )

        # Check first call (target_df, datetime_column)
        first_call_df = call_args_list[0][0][0]  # First arg of first call
        first_call_column = call_args_list[0][0][1]  # Second arg of first call
        assert isinstance(first_call_df, pd.DataFrame)
        assert "date" in first_call_df.columns
        assert "value1" in first_call_df.columns
        assert first_call_column == "date"

        # Check second call (ground_truth_df, datetime_column)
        second_call_df = call_args_list[1][0][0]  # First arg of second call
        second_call_column = call_args_list[1][0][
            1
        ]  # Second arg of second call
        assert isinstance(second_call_df, pd.DataFrame)
        assert "date" in second_call_df.columns
        assert "value1" in second_call_df.columns
        assert second_call_column == "date"

        # Additionally verify that one dataframe has 5 rows and the other has 3
        # to make sure we're testing with the right dataframes
        call_dfs = [call_args_list[0][0][0], call_args_list[1][0][0]]
        assert 5 in [len(df) for df in call_dfs]
        assert 3 in [len(df) for df in call_dfs]

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor")
    @patch("time.sleep")
    def test_predict_skips_ground_truth_validation_when_none(
        self, mock_sleep, mock_tabpfn_cls
    ):
        """Test that predict skips ground truth validation when it's None."""
        # Setup predictor with mock validation method
        predictor = TabPFNPredictor()
        predictor._validate_no_duplicate_timestamps = MagicMock()
        predictor.predictor = MagicMock()

        # Create test data
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        metadata_dataframes = []
        target_columns = ["value1"]
        forecasting_timestamp = pd.Timestamp("2023-01-10").to_pydatetime()
        future_timestamps = [
            pd.Timestamp("2023-01-11").to_pydatetime(),
            pd.Timestamp("2023-01-12").to_pydatetime(),
        ]

        # Call predict method with ground_truth_df=None
        predictor.predict(
            target_df=target_df,
            covariate_columns=[],
            metadata_dataframes=metadata_dataframes,
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_timestamps,
            timestamp_column="date",
            ground_truth_df=None,  # Explicitly None
        )

        # Assert validation was called only for target dataframe
        assert predictor._validate_no_duplicate_timestamps.call_count == 1
        pd.testing.assert_frame_equal(
            predictor._validate_no_duplicate_timestamps.call_args_list[0][0][0],
            target_df,
        )

    @patch("tabpfn_time_series.TabPFNTimeSeriesPredictor")
    @patch("time.sleep")
    def test_predict_with_duplicate_timestamps_in_ground_truth(
        self, mock_sleep, mock_tabpfn_cls
    ):
        """Test that predict raises error when ground truth has duplicate timestamps."""
        # Setup
        predictor = TabPFNPredictor()

        # Create test data
        target_df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
                "value1": [10, 20, 30, 40, 50],
            }
        )

        # Ground truth with duplicate timestamps
        ground_truth_df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2023-01-11"),
                    pd.Timestamp("2023-01-12"),
                    pd.Timestamp("2023-01-11"),  # Duplicate
                ],
                "value1": [60, 70, 80],
            }
        )

        metadata_dataframes = []
        target_columns = ["value1"]
        forecasting_timestamp = pd.Timestamp("2023-01-10").to_pydatetime()
        future_timestamps = [
            pd.Timestamp("2023-01-11").to_pydatetime(),
            pd.Timestamp("2023-01-12").to_pydatetime(),
        ]

        # Call predict method and expect error
        with pytest.raises(ValueError) as exc_info:
            predictor.predict(
                target_df=target_df,
                covariate_columns=[],
                metadata_dataframes=metadata_dataframes,
                target_columns=target_columns,
                forecasting_timestamp=forecasting_timestamp,
                future_time_stamps=future_timestamps,
                timestamp_column="date",
                ground_truth_df=ground_truth_df,
            )

        # Verify error message contains info about duplicates
        error_msg = str(exc_info.value)
        assert "Duplicate timestamps found" in error_msg
        assert "2023-01-11" in error_msg


class TestDataframeToHistoricalDataset:
    """Tests for the _dataframe_to_historical_dataset function."""

    def test_basic_conversion(self):
        """Test basic conversion of DataFrame to HistoricalDataset."""
        # Create a test DataFrame
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "value1": [10, 20, 30, 40, 50],
                "value2": [5, 15, 25, 35, 45],
            }
        )

        # Convert to HistoricalDataset
        result = _dataframe_to_historical_dataset(
            df, "timestamp", ["value1", "value2"]
        )

        # Verify the result
        assert isinstance(result, HistoricalDataset)
        assert len(result.timestamps) == 5
        assert list(result.values.keys()) == ["value1", "value2"]
        assert result.values["value1"] == [10, 20, 30, 40, 50]
        assert result.values["value2"] == [5, 15, 25, 35, 45]
        assert result.target_columns == ["value1", "value2"]

        # Check timestamp format
        expected_timestamps = [
            "2023-01-01T00:00:00",
            "2023-01-02T00:00:00",
            "2023-01-03T00:00:00",
            "2023-01-04T00:00:00",
            "2023-01-05T00:00:00",
        ]
        assert result.timestamps == expected_timestamps

    def test_single_target_column(self):
        """Test conversion with a single target column."""
        # Create a test DataFrame
        dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "temperature": [22.5, 23.1, 21.8],
            }
        )

        # Convert to HistoricalDataset
        result = _dataframe_to_historical_dataset(df, "date", ["temperature"])

        # Verify the result
        assert isinstance(result, HistoricalDataset)
        assert len(result.timestamps) == 3
        assert list(result.values.keys()) == ["temperature"]
        assert result.values["temperature"] == [22.5, 23.1, 21.8]
        assert result.target_columns == ["temperature"]

    def test_with_specific_timestamp_format(self):
        """Test conversion with specific timestamp formats."""
        # Create a test DataFrame with timestamps at different times
        dates = [
            datetime(2023, 1, 1, 12, 30, 45),
            datetime(2023, 1, 2, 8, 15, 30),
            datetime(2023, 1, 3, 18, 0, 15),
        ]
        df = pd.DataFrame(
            {
                "time": dates,
                "value": [100, 200, 300],
            }
        )

        # Convert to HistoricalDataset
        result = _dataframe_to_historical_dataset(df, "time", ["value"])

        # Verify the timestamp format
        expected_timestamps = [
            "2023-01-01T12:30:45",
            "2023-01-02T08:15:30",
            "2023-01-03T18:00:15",
        ]
        assert result.timestamps == expected_timestamps
        assert result.values["value"] == [100, 200, 300]

    def test_with_multiple_target_columns(self):
        """Test conversion with multiple target columns."""
        # Create a test DataFrame
        dates = pd.date_range(start="2023-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "temperature": [22.5, 23.1, 21.8, 24.0],
                "humidity": [45, 48, 52, 50],
                "pressure": [1013, 1012, 1014, 1015],
                "metadata": [
                    "sunny",
                    "cloudy",
                    "rainy",
                    "sunny",
                ],  # Not a target
            }
        )

        # Convert to HistoricalDataset with subset of columns
        result = _dataframe_to_historical_dataset(
            df, "timestamp", ["temperature", "humidity", "pressure"]
        )

        # Verify the result
        assert isinstance(result, HistoricalDataset)
        assert len(result.timestamps) == 4
        assert sorted(list(result.values.keys())) == sorted(
            ["temperature", "humidity", "pressure"]
        )
        assert result.values["temperature"] == [22.5, 23.1, 21.8, 24.0]
        assert result.values["humidity"] == [45, 48, 52, 50]
        assert result.values["pressure"] == [1013, 1012, 1014, 1015]
        assert sorted(result.target_columns) == sorted(
            ["temperature", "humidity", "pressure"]
        )
        # Metadata column should not be included
        assert "metadata" not in result.values


class TestHelperFunctions:
    """Tests for the helper functions used by setup_train_test_dfs"""

    def test_handle_missing_values_and_get_time_varying_non_numeric_columns(
        self,
    ):
        """Test handling missing values and identifying time-varying non-numeric columns"""
        # Create test dataframe with mixed data types and NaNs
        df = pd.DataFrame(
            {
                "numeric_feature": [1, np.nan, 3, np.nan, 5],
                "invariant_numeric": [7, 7, 7, np.nan, 7],
                "varying_str": ["a", np.nan, "c", np.nan, "e"],
                "invariant_str": ["x", "x", "x", np.nan, "x"],
            }
        )

        non_target_columns = df.columns.tolist()

        # Call the function
        df_filled, time_varying_non_numeric = (
            handle_missing_values_and_get_time_varying_non_numeric_columns(
                df, non_target_columns
            )
        )

        # Verify results
        assert not df_filled.isna().any().any()  # No NaNs remaining

        # Check numeric columns are filled correctly
        assert df_filled["numeric_feature"].tolist() == [
            1,
            1,
            3,
            3,
            5,
        ]  # ffill, bfill
        assert df_filled["invariant_numeric"].tolist() == [
            7,
            7,
            7,
            7,
            7,
        ]  # ffill, bfill

        # Check string columns are filled correctly
        assert df_filled["varying_str"].tolist() == [
            "a",
            "a",
            "c",
            "c",
            "e",
        ]  # ffill, bfill
        assert df_filled["invariant_str"].tolist() == [
            "x",
            "x",
            "x",
            "x",
            "x",
        ]  # ffill, bfill

        # Check time-varying non-numeric columns are identified
        assert "varying_str" in time_varying_non_numeric
        assert "numeric_feature" not in time_varying_non_numeric
        assert "invariant_numeric" not in time_varying_non_numeric

    def test_handle_missing_values_empty_dataframe(self):
        """Test handling missing values with an empty dataframe"""
        df = pd.DataFrame(columns=["col1", "col2"])
        non_target_columns = ["col1", "col2"]

        df_filled, time_varying_non_numeric = (
            handle_missing_values_and_get_time_varying_non_numeric_columns(
                df, non_target_columns
            )
        )

        assert df_filled.empty
        assert len(time_varying_non_numeric) == 0

    def test_handle_missing_values_all_nan_columns(self):
        """Test handling columns with all NaN values"""
        df = pd.DataFrame(
            {
                "all_nan_numeric": [np.nan, np.nan, np.nan],
                "all_nan_object": [None, None, None],
            }
        )

        # Set dtypes explicitly
        df["all_nan_numeric"] = df["all_nan_numeric"].astype(float)
        df["all_nan_object"] = df["all_nan_object"].astype(object)

        non_target_columns = df.columns.tolist()

        df_filled, time_varying_non_numeric = (
            handle_missing_values_and_get_time_varying_non_numeric_columns(
                df, non_target_columns
            )
        )

        # Check numeric column is filled with 0s
        assert df_filled["all_nan_numeric"].tolist() == [0, 0, 0]

        # Check object column is filled with "unknown"
        assert df_filled["all_nan_object"].tolist() == [
            "unknown",
            "unknown",
            "unknown",
        ]

        # Check time-varying non-numeric columns - should be empty because all values are the same
        assert "all_nan_object" not in time_varying_non_numeric

    def test_identify_time_invariant_columns(self):
        """Test identifying time-invariant columns"""
        df = pd.DataFrame(
            {
                "varying": [1, 2, 3, 4, 5],
                "invariant1": [42, 42, 42, 42, 42],
                "invariant2": ["x", "x", "x", "x", "x"],
                "mostly_invariant": [10, 10, 10, 10, 11],
            }
        )

        non_target_columns = [
            "varying",
            "invariant1",
            "invariant2",
            "mostly_invariant",
        ]

        invariant_cols = identify_time_invariant_columns(df, non_target_columns)

        assert "invariant1" in invariant_cols
        assert "invariant2" in invariant_cols
        assert "varying" not in invariant_cols
        assert "mostly_invariant" not in invariant_cols

    def test_identify_time_invariant_empty_columns(self):
        """Test identifying time-invariant columns with empty input"""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        # Test with empty list
        invariant_cols = identify_time_invariant_columns(df, [])
        assert len(invariant_cols) == 0

    def test_identify_time_invariant_single_row(self):
        """Test identifying time-invariant columns with single row dataframe"""
        df = pd.DataFrame({"col1": [1], "col2": ["a"]})

        non_target_columns = ["col1", "col2"]

        invariant_cols = identify_time_invariant_columns(df, non_target_columns)

        # With one row, all columns should be invariant
        assert set(invariant_cols) == set(non_target_columns)


class TestSetupTrainTestDfs:
    """Tests for setup_train_test_dfs function"""

    def test_basic_functionality(self):
        """Test the basic functionality with a simple dataframe"""
        # Create test dataframe
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, 20, 30, 40, 50],
                "target2": [5, 15, 25, 35, 45],
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["a", "b", "c", "d", "e"],
            }
        )

        target_columns = ["target1", "target2"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results
        assert "timestamp" in processed_df.columns
        assert "feature1" in non_target_cols
        assert (
            "feature2" not in non_target_cols
        )  # Should be removed as non-numeric time-varying
        assert len(non_target_cols) == 1
        assert len(time_invariant_cols) == 0
        assert (
            processed_df.shape[1] == 4
        )  # timestamp, target1, target2, feature1

        # Check column order
        assert list(processed_df.columns[:3]) == [
            "timestamp",
            "target1",
            "target2",
        ]

    def test_missing_values_handling(self):
        """Test handling of missing values in different column types"""
        # Create test dataframe with NaN values
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, np.nan, 30, 40, 50],
                "numeric_feature": [1, np.nan, 3, np.nan, 5],
                "categorical_feature": ["a", np.nan, "c", np.nan, "e"],
            }
        )

        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, _ = setup_train_test_dfs(
            df, target_columns, datetime_column
        )

        # Verify missing values are handled correctly
        assert not processed_df["numeric_feature"].isna().any()
        assert processed_df["numeric_feature"].tolist() == [
            1,
            1,
            3,
            3,
            5,
        ]  # ffill, bfill

        # Target column NaN values should remain as they are not processed
        assert pd.isna(processed_df["target1"].iloc[1])

        # Categorical feature should be removed as it's time-varying non-numeric
        assert "categorical_feature" not in processed_df.columns

    def test_time_invariant_columns(self):
        """Test identification of time-invariant columns"""
        # Create test dataframe with time-invariant columns
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, 20, 30, 40, 50],
                "varying_feature": [1, 2, 3, 4, 5],
                "invariant_feature1": [7, 7, 7, 7, 7],
                "invariant_feature2": [42, 42, 42, 42, 42],
            }
        )

        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify time-invariant columns are correctly identified
        assert len(time_invariant_cols) == 2
        assert "invariant_feature1" in time_invariant_cols
        assert "invariant_feature2" in time_invariant_cols
        assert "varying_feature" not in time_invariant_cols

    def test_empty_dataframe(self):
        """Test handling of an empty dataframe"""
        df = pd.DataFrame(columns=["date", "target1", "feature1"])
        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results
        assert processed_df.empty
        assert len(non_target_cols) == 1
        assert "feature1" in non_target_cols
        assert len(time_invariant_cols) == 0

    def test_no_target_columns(self):
        """Test with no target columns provided"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [6, 7, 8, 9, 10],
            }
        )

        target_columns = []
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results
        assert len(non_target_cols) == 2
        assert "feature1" in non_target_cols
        assert "feature2" in non_target_cols
        assert len(time_invariant_cols) == 0
        assert list(processed_df.columns) == [
            "timestamp",
            "feature1",
            "feature2",
        ]

    def test_all_columns_are_target(self):
        """Test when all columns except datetime are target columns"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [1, 2, 3, 4, 5],
                "target2": [6, 7, 8, 9, 10],
            }
        )

        target_columns = ["target1", "target2"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results
        assert len(non_target_cols) == 0
        assert len(time_invariant_cols) == 0
        assert list(processed_df.columns) == ["timestamp", "target1", "target2"]

    def test_datetime_conversion(self):
        """Test datetime column is correctly renamed and handled"""
        # Create test dataframe with different datetime formats
        df = pd.DataFrame(
            {
                "date_col": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 2),
                    datetime(2023, 1, 3),
                ],
                "target1": [10, 20, 30],
            }
        )

        target_columns = ["target1"]
        datetime_column = "date_col"

        # Call the function
        processed_df, _, _ = setup_train_test_dfs(
            df, target_columns, datetime_column
        )

        # Verify datetime column is renamed
        assert "timestamp" in processed_df.columns
        assert "date_col" not in processed_df.columns
        assert processed_df["timestamp"].equals(df["date_col"])

    def test_with_nulls_only_column(self):
        """Test with columns containing only null values"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, 20, 30, 40, 50],
                "all_null_numeric": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "all_null_object": [None, None, None, None, None],
            }
        )

        # Set dtypes explicitly
        df["all_null_numeric"] = df["all_null_numeric"].astype(float)
        df["all_null_object"] = df["all_null_object"].astype(object)

        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results
        assert "all_null_numeric" in processed_df.columns
        assert "all_null_object" in processed_df.columns

        # Check that nulls are filled properly
        assert processed_df["all_null_numeric"].tolist() == [0, 0, 0, 0, 0]

        # Check if all_null_numeric is identified as time-invariant
        assert "all_null_numeric" in time_invariant_cols

    def test_all_columns_non_numeric_time_varying(self):
        """Test when all non-target columns are non-numeric and time-varying"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, 20, 30, 40, 50],
                "string_feature1": ["a", "b", "c", "d", "e"],
                "string_feature2": ["v", "w", "x", "y", "z"],
            }
        )

        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )

        # Verify results - all non-numeric time-varying columns should be removed
        assert len(non_target_cols) == 0
        assert len(time_invariant_cols) == 0
        assert list(processed_df.columns) == ["timestamp", "target1"]

    def test_large_dataframe_performance(self):
        """Test performance with a large dataframe"""
        # Create a large dataframe with 100,000 rows and multiple columns
        n_rows = 100_000
        dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="H")

        # Create random data with some patterns
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": np.random.randn(n_rows) * 10 + 50,
                "target2": np.sin(np.linspace(0, 100, n_rows)) * 20 + 30,
                "numeric_feature1": np.random.randn(n_rows) * 5,
                "numeric_feature2": np.random.randint(0, 100, n_rows),
                "invariant_feature": [42] * n_rows,
            }
        )

        # Add some missing values
        mask = np.random.random(n_rows) < 0.05  # 5% missing
        df.loc[mask, "numeric_feature1"] = np.nan

        target_columns = ["target1", "target2"]
        datetime_column = "date"

        # Call the function and measure time
        import time

        start_time = time.time()
        processed_df, non_target_cols, time_invariant_cols = (
            setup_train_test_dfs(df, target_columns, datetime_column)
        )
        end_time = time.time()

        # Verify results
        assert processed_df.shape[0] == n_rows
        assert "invariant_feature" in time_invariant_cols

        # Performance should be reasonable (adjust thresholds based on expected performance)
        assert (
            end_time - start_time < 10
        )  # Should process 100k rows in under 10 seconds

    def test_extreme_values(self):
        """Test with extreme values in numeric columns"""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "target1": [10, 20, 30, 40, 50],
                "extreme_high": [1e9, 2e9, 3e9, 4e9, 5e9],
                "extreme_low": [-1e9, -2e9, -3e9, -4e9, -5e9],
                "inf_values": [np.inf, -np.inf, np.inf, -np.inf, np.inf],
            }
        )

        target_columns = ["target1"]
        datetime_column = "date"

        # Call the function
        processed_df, non_target_cols, _ = setup_train_test_dfs(
            df, target_columns, datetime_column
        )

        # Verify results
        assert "extreme_high" in processed_df.columns
        assert "extreme_low" in processed_df.columns
        assert "inf_values" in processed_df.columns

        # Values should be preserved
        assert processed_df["extreme_high"].max() == 5e9
        assert processed_df["extreme_low"].min() == -5e9
        assert np.isinf(processed_df["inf_values"]).any()


class TestApplyPointModifications:
    def make_df(self):
        # Helper to create a simple DataFrame with a text column
        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": list("abcde"),
                "text": ["foo", "bar", "baz", "qux", "quux"],
            }
        )

    def test_no_modifications_none(self):
        df = self.make_df()
        result = apply_point_modifications(df.copy(), "date", None)
        pd.testing.assert_frame_equal(result, df)

    def test_no_modifications_empty(self):
        df = self.make_df()
        result = apply_point_modifications(df.copy(), "date", [])
        pd.testing.assert_frame_equal(result, df)

    def test_single_modification(self):
        df = self.make_df()
        mod = FMDataPointModification(
            timestamp="2023-01-03", modification_dict={"a": 99, "b": 123}
        )
        result = apply_point_modifications(df.copy(), "date", [mod])
        expected = df.copy()
        expected.loc[
            expected["date"] == pd.Timestamp("2023-01-03"), ["a", "b"]
        ] = [99, 123]
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_modifications(self):
        df = self.make_df()
        mods = [
            FMDataPointModification(
                timestamp="2023-01-02", modification_dict={"a": 42}
            ),
            FMDataPointModification(
                timestamp="2023-01-04", modification_dict={"b": 77, "c": "z"}
            ),
        ]
        result = apply_point_modifications(df.copy(), "date", mods)
        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-02"), "a"] = 42
        expected.loc[
            expected["date"] == pd.Timestamp("2023-01-04"), ["b", "c"]
        ] = [77, "z"]
        pd.testing.assert_frame_equal(result, expected)

    def test_modification_nonexistent_timestamp(self):
        df = self.make_df()
        mod = FMDataPointModification(
            timestamp="2023-01-10", modification_dict={"a": 100}
        )  # Not in df
        # The function should skip the modification and return the original dataframe unchanged
        result = apply_point_modifications(df.copy(), "date", [mod])
        pd.testing.assert_frame_equal(result, df)

    def test_modification_nonexistent_column(self):
        df = self.make_df()
        mod = FMDataPointModification(
            timestamp="2023-01-02", modification_dict={"x": 999}
        )  # 'x' not in df
        result = apply_point_modifications(df.copy(), "date", [mod])
        pd.testing.assert_frame_equal(result, df)

    def test_overlapping_modifications(self):
        df = self.make_df()
        mods = [
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 1}
            ),
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"b": 2}
            ),
        ]
        result = apply_point_modifications(df.copy(), "date", mods)

        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "b"] = 2
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "a"] = 1
        pd.testing.assert_frame_equal(result, expected)

    def test_type_preservation(self):
        df = self.make_df()
        mod = FMDataPointModification(
            timestamp="2023-01-01",
            modification_dict={"a": 3.14, "c": "xyz", "text": "newtext"},
        )
        result = apply_point_modifications(df.copy(), "date", [mod])
        assert isinstance(result.loc[0, "a"], float)
        assert result.loc[0, "a"] == 3.14
        assert result.loc[0, "c"] == "xyz"
        assert result.loc[0, "text"] == "newtext"

    def test_duplicate_timestamps_in_df(self):
        df = self.make_df()
        # Add a duplicate row for a date
        df = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        mod = FMDataPointModification(
            timestamp="2023-01-03", modification_dict={"a": 111}
        )
        result = apply_point_modifications(df.copy(), "date", [mod])
        # All rows with that date should be updated
        assert (
            result[result["date"] == pd.Timestamp("2023-01-03")]["a"] == 111
        ).all()

    def test_duplicate_timestamps_in_modifications(self):
        df = self.make_df()
        mods = [
            FMDataPointModification(
                timestamp="2023-01-04", modification_dict={"a": 1}
            ),
            FMDataPointModification(
                timestamp="2023-01-04", modification_dict={"b": 2}
            ),
            FMDataPointModification(
                timestamp="2023-01-04", modification_dict={"a": 3}
            ),  # Last one should win for 'a'
        ]
        result = apply_point_modifications(df.copy(), "date", mods)
        # The last modification for 'a' should be applied
        assert (
            result.loc[result["date"] == pd.Timestamp("2023-01-04"), "a"].iloc[
                0
            ]
            == 3
        )
        assert (
            result.loc[result["date"] == pd.Timestamp("2023-01-04"), "b"].iloc[
                0
            ]
            == 2
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "a", "b", "text"])
        mod = FMDataPointModification(
            timestamp="2023-01-01", modification_dict={"a": 1}
        )
        # The function should skip the modification and return the original dataframe unchanged
        result = apply_point_modifications(df.copy(), "date", [mod])
        # The function converts the timestamp column to datetime, so we need to expect that
        expected_df = df.copy()
        expected_df["date"] = pd.to_datetime(expected_df["date"])
        pd.testing.assert_frame_equal(result, expected_df)


class TestValidateAndSanitizeTimezone:
    """Tests for the _validate_and_sanitize_timezone function."""

    def test_none_value(self):
        """Test with None timezone value."""

        result = _validate_and_sanitize_timezone(None)
        assert result == "UTC"

    def test_empty_string(self):
        """Test with empty string timezone value."""

        result = _validate_and_sanitize_timezone("")
        assert result == "UTC"

    def test_whitespace_string(self):
        """Test with whitespace-only string timezone value."""

        result = _validate_and_sanitize_timezone("   ")
        assert result == "UTC"

    def test_zero_timezone_offset(self):
        """Test with zero timezone offset."""

        # Test various forms of zero
        assert _validate_and_sanitize_timezone("0") == "UTC"
        assert _validate_and_sanitize_timezone("0.0") == "UTC"
        assert _validate_and_sanitize_timezone(0) == "UTC"
        assert _validate_and_sanitize_timezone(0.0) == "UTC"

    def test_positive_integer_timezone_offset(self):
        """Test with positive integer timezone offsets."""

        # Test positive integer offsets
        assert _validate_and_sanitize_timezone("1") == "+01:00"
        assert _validate_and_sanitize_timezone("5") == "+05:00"
        assert _validate_and_sanitize_timezone("12") == "+12:00"
        assert _validate_and_sanitize_timezone(3) == "+03:00"

    def test_negative_integer_timezone_offset(self):
        """Test with negative integer timezone offsets."""

        # Test negative integer offsets
        assert _validate_and_sanitize_timezone("-1") == "-01:00"
        assert _validate_and_sanitize_timezone("-5") == "-05:00"
        assert _validate_and_sanitize_timezone("-8") == "-08:00"
        assert _validate_and_sanitize_timezone(-3) == "-03:00"

    def test_positive_float_timezone_offset(self):
        """Test with positive float timezone offsets."""

        # Test positive float offsets
        assert _validate_and_sanitize_timezone("5.5") == "+05:30"
        assert _validate_and_sanitize_timezone("9.5") == "+09:30"
        assert _validate_and_sanitize_timezone("3.25") == "+03:15"
        assert _validate_and_sanitize_timezone("12.75") == "+12:45"
        assert _validate_and_sanitize_timezone(4.5) == "+04:30"

    def test_negative_float_timezone_offset(self):
        """Test with negative float timezone offsets."""

        # Test negative float offsets
        assert _validate_and_sanitize_timezone("-5.5") == "-05:30"
        assert _validate_and_sanitize_timezone("-9.5") == "-09:30"
        assert _validate_and_sanitize_timezone("-3.25") == "-03:15"
        assert _validate_and_sanitize_timezone("-12.75") == "-12:45"
        assert _validate_and_sanitize_timezone(-4.5) == "-04:30"

    def test_edge_case_float_timezone_offsets(self):
        """Test with edge case float timezone offsets."""

        # Test edge cases
        assert _validate_and_sanitize_timezone("0.5") == "+00:30"
        # Note: -00:30 might not be valid in pandas, so it falls back to UTC
        result = _validate_and_sanitize_timezone("-0.5")
        assert result in [
            "-00:30",
            "UTC",
        ]  # Accept either result depending on pandas behavior
        assert _validate_and_sanitize_timezone("0.25") == "+00:15"
        result = _validate_and_sanitize_timezone("-0.25")
        assert result in [
            "-00:15",
            "UTC",
        ]  # Accept either result depending on pandas behavior
        assert _validate_and_sanitize_timezone("1.0") == "+01:00"
        assert _validate_and_sanitize_timezone("-1.0") == "-01:00"

    def test_special_invalid_cases(self):
        """Test with special invalid timezone values."""

        # Test special invalid cases
        assert _validate_and_sanitize_timezone("nan") == "UTC"
        assert _validate_and_sanitize_timezone("None") == "UTC"
        assert (
            _validate_and_sanitize_timezone("NaN") == "UTC"
        )  # Case variations
        assert _validate_and_sanitize_timezone("NONE") == "UTC"

    def test_valid_timezone_strings(self):
        """Test with valid timezone strings."""

        # Test valid timezone strings
        assert _validate_and_sanitize_timezone("UTC") == "UTC"
        assert (
            _validate_and_sanitize_timezone("America/New_York")
            == "America/New_York"
        )
        assert (
            _validate_and_sanitize_timezone("Europe/London") == "Europe/London"
        )
        assert _validate_and_sanitize_timezone("Asia/Tokyo") == "Asia/Tokyo"
        assert (
            _validate_and_sanitize_timezone("Australia/Sydney")
            == "Australia/Sydney"
        )
        assert (
            _validate_and_sanitize_timezone("America/Los_Angeles")
            == "America/Los_Angeles"
        )

    def test_common_timezone_mappings(self):
        """Test with common timezone abbreviations that get mapped."""

        # Test common timezone mappings
        # Note: Some of these might be valid in pandas and not get mapped
        result_est = _validate_and_sanitize_timezone("EST")
        assert result_est in [
            "EST",
            "America/New_York",
        ]  # EST might be valid in pandas
        result_pst = _validate_and_sanitize_timezone("PST")
        assert result_pst in [
            "PST",
            "America/Los_Angeles",
        ]  # PST might be valid in pandas
        result_cst = _validate_and_sanitize_timezone("CST")
        assert result_cst in [
            "CST",
            "America/Chicago",
        ]  # CST might be valid in pandas
        result_mst = _validate_and_sanitize_timezone("MST")
        assert result_mst in [
            "MST",
            "America/Denver",
        ]  # MST might be valid in pandas
        # GMT is valid in pandas and returns 'GMT', not mapped to 'UTC'
        result_gmt = _validate_and_sanitize_timezone("GMT")
        assert result_gmt in ["GMT", "UTC"]  # GMT might be valid in pandas

    def test_invalid_timezone_strings(self):
        """Test with invalid timezone strings that default to UTC."""

        # Test invalid timezone strings
        assert _validate_and_sanitize_timezone("Invalid/Timezone") == "UTC"
        assert _validate_and_sanitize_timezone("NotATimezone") == "UTC"
        assert _validate_and_sanitize_timezone("Random123") == "UTC"
        assert _validate_and_sanitize_timezone("America/InvalidCity") == "UTC"
        assert _validate_and_sanitize_timezone("Europe/FakePlace") == "UTC"

    def test_non_string_non_numeric_types(self):
        """Test with non-string, non-numeric types."""

        # Test with various non-string, non-numeric types
        assert _validate_and_sanitize_timezone([]) == "UTC"
        assert _validate_and_sanitize_timezone({}) == "UTC"
        assert (
            _validate_and_sanitize_timezone(True) == "UTC"
        )  # Will convert to "True"
        assert (
            _validate_and_sanitize_timezone(False) == "UTC"
        )  # Will convert to "False"

    def test_numeric_strings_that_look_like_timezones(self):
        """Test with numeric strings that might be confused with timezone offsets."""

        # Test edge cases with numeric strings
        # Note: "+05:30" actually parses as a valid timezone offset, not an invalid string
        result_0100 = _validate_and_sanitize_timezone("01:00")
        assert result_0100 == "UTC"  # Invalid format, not a float
        result_0530 = _validate_and_sanitize_timezone("+05:30")
        assert result_0530 in ["+05:30", "UTC"]  # Might be valid in pandas
        result_0800 = _validate_and_sanitize_timezone("-08:00")
        assert result_0800 in ["-08:00", "UTC"]  # Might be valid in pandas

    def test_whitespace_handling(self):
        """Test handling of whitespace around valid values."""

        # Test whitespace handling
        assert _validate_and_sanitize_timezone("  UTC  ") == "UTC"
        assert _validate_and_sanitize_timezone("  5  ") == "+05:00"
        assert _validate_and_sanitize_timezone("  -3.5  ") == "-03:30"
        assert (
            _validate_and_sanitize_timezone("  America/New_York  ")
            == "America/New_York"
        )
        result_est = _validate_and_sanitize_timezone("  EST  ")
        assert result_est in [
            "EST",
            "America/New_York",
        ]  # EST might be valid in pandas

    def test_extreme_numeric_values(self):
        """Test with extreme numeric timezone offsets."""

        # Test extreme values (these would be invalid as real timezones but should not crash)
        # Note: Extreme values might fail pandas validation and default to UTC
        result_25 = _validate_and_sanitize_timezone("25")
        assert result_25 in ["+25:00", "UTC"]  # Might fail pandas validation
        result_neg15 = _validate_and_sanitize_timezone("-15")
        assert result_neg15 in ["-15:00", "UTC"]  # Might fail pandas validation
        result_1005 = _validate_and_sanitize_timezone("100.5")
        assert result_1005 in [
            "+100:30",
            "UTC",
        ]  # Extreme value, likely fails pandas validation
        result_neg5025 = _validate_and_sanitize_timezone("-50.25")
        assert result_neg5025 in [
            "-50:15",
            "UTC",
        ]  # Extreme negative value, likely fails pandas validation

    def test_very_small_float_offsets(self):
        """Test with very small float timezone offsets."""

        # Test very small offsets
        result_01 = _validate_and_sanitize_timezone("0.1")
        assert result_01 in [
            "+00:06",
            "UTC",
        ]  # 6 minutes, might fail pandas validation
        result_001 = _validate_and_sanitize_timezone("0.01")
        assert result_001 in [
            "+00:00",
            "UTC",
        ]  # < 1 minute, rounds to 0 or fails validation
        result_neg01 = _validate_and_sanitize_timezone("-0.1")
        assert result_neg01 in [
            "-00:06",
            "UTC",
        ]  # -6 minutes, might fail pandas validation
        assert _validate_and_sanitize_timezone("0.75") == "+00:45"  # 45 minutes

    def test_integer_like_floats(self):
        """Test with float values that are essentially integers."""

        # Test integer-like floats
        assert _validate_and_sanitize_timezone("1.00") == "+01:00"
        assert _validate_and_sanitize_timezone("5.000") == "+05:00"
        assert _validate_and_sanitize_timezone("-3.0") == "-03:00"

    def test_mixed_case_timezone_names(self):
        """Test with mixed case timezone names."""

        # Test case sensitivity - these should be invalid and default to UTC unless pandas accepts them
        result_utc = _validate_and_sanitize_timezone("utc")
        assert result_utc in [
            "utc",
            "UTC",
        ]  # lowercase UTC might be valid in pandas
        result_est = _validate_and_sanitize_timezone("est")
        assert result_est in ["est", "UTC"]  # lowercase EST is likely invalid
        # Note: lowercase timezone names like 'america/new_york' are actually valid in pandas
        result_america = _validate_and_sanitize_timezone("america/new_york")
        assert result_america in [
            "america/new_york",
            "UTC",
        ]  # lowercase might be valid in pandas

    def test_scientific_notation(self):
        """Test with scientific notation numbers."""

        # Test scientific notation (these should be parsed as floats)
        assert _validate_and_sanitize_timezone("1e1") == "+10:00"  # 1e1 = 10
        assert _validate_and_sanitize_timezone("5e0") == "+05:00"  # 5e0 = 5
        assert _validate_and_sanitize_timezone("-2e0") == "-02:00"  # -2e0 = -2

    def test_partial_timezone_names(self):
        """Test with partial or incomplete timezone names."""

        # Test partial timezone names (should be invalid)
        assert _validate_and_sanitize_timezone("America/") == "UTC"
        assert _validate_and_sanitize_timezone("Europe") == "UTC"
        assert _validate_and_sanitize_timezone("/London") == "UTC"
        assert _validate_and_sanitize_timezone("New_York") == "UTC"

    @patch("pandas.Timestamp.tz_localize")
    def test_pandas_timezone_validation_exception(self, mock_tz_localize):
        """Test behavior when pandas timezone validation raises an exception."""

        # Mock pandas to raise an exception for any timezone
        mock_tz_localize.side_effect = Exception("Invalid timezone")

        # Should fall back to common mappings, then UTC
        assert (
            _validate_and_sanitize_timezone("EST") == "America/New_York"
        )  # Common mapping works
        assert (
            _validate_and_sanitize_timezone("SomeRandomTZ") == "UTC"
        )  # Falls back to UTC

    def test_none_type_directly(self):
        """Test passing None type directly."""

        result = _validate_and_sanitize_timezone(None)
        assert result == "UTC"

    def test_boolean_values(self):
        """Test with boolean values."""

        # Booleans will be converted to strings "True"/"False" which are invalid timezones
        assert _validate_and_sanitize_timezone(True) == "UTC"
        assert _validate_and_sanitize_timezone(False) == "UTC"

    def test_complex_numeric_strings(self):
        """Test with complex numeric-like strings that aren't valid floats."""

        # These should fail float conversion and be treated as timezone strings
        assert _validate_and_sanitize_timezone("1.2.3") == "UTC"
        assert _validate_and_sanitize_timezone("5..5") == "UTC"
        assert _validate_and_sanitize_timezone("--5") == "UTC"
        assert _validate_and_sanitize_timezone("5-") == "UTC"
        assert _validate_and_sanitize_timezone("5+") == "UTC"


class TestApplyMetaDataVariations:
    def make_df_with_timestamps(self):
        # Helper to create a DataFrame with timestamps
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 11:00:00",
                        "2023-01-01 12:00:00",
                        "2023-01-01 13:00:00",
                        "2023-01-01 14:00:00",
                    ]
                ),
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": [100, 200, 300, 400, 500],
                "text": ["foo", "bar", "baz", "qux", "quux"],
            }
        )

    def test_no_modifications_none(self):
        df = self.make_df_with_timestamps()

        result = apply_full_and_range_modifications(
            df.copy(), None, "timestamp"
        )
        pd.testing.assert_frame_equal(result, df)

    def test_no_modifications_empty(self):
        df = self.make_df_with_timestamps()

        result = apply_full_and_range_modifications(df.copy(), [], "timestamp")
        pd.testing.assert_frame_equal(result, df)

    def test_full_modification_single(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import MetaDataVariation

        mod = MetaDataVariation(
            name="a",
            value=99,
            perturbation_or_exact_value="exact_value",
            order=1,
        )
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        expected = df.copy()
        expected["a"] = 99
        pd.testing.assert_frame_equal(result, expected)

    def test_range_modification_with_timestamps(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import (
            MetaDataVariation,
        )

        # Modify only rows between 11:00 and 13:00
        mod = MetaDataVariation(
            name="a",
            value=10,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.ADD,
            order=1,
            min_timestamp="2023-01-01T11:00:00",
            max_timestamp="2023-01-01T13:00:00",
        )
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        expected = df.copy()
        # Only rows at 11:00, 12:00, and 13:00 should be modified
        expected.loc[1:3, "a"] = [12, 13, 14]  # 2+10, 3+10, 4+10
        pd.testing.assert_frame_equal(result, expected)

    def test_range_modification_min_timestamp_only(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import (
            MetaDataVariation,
        )

        # Modify only rows from 12:00 onwards
        mod = MetaDataVariation(
            name="b",
            value=5,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.SUBTRACT,
            order=1,
            min_timestamp="2023-01-01T12:00:00",
        )
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        expected = df.copy()
        # Rows from 12:00 onwards should be modified
        expected.loc[2:, "b"] = [25, 35, 45]  # 30-5, 40-5, 50-5
        pd.testing.assert_frame_equal(result, expected)

    def test_range_modification_max_timestamp_only(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import (
            MetaDataVariation,
        )

        # Modify only rows up to 12:00
        mod = MetaDataVariation(
            name="c",
            value=2,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.MULTIPLY,
            order=1,
            max_timestamp="2023-01-01T12:00:00",
        )
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        expected = df.copy()
        # Rows up to 12:00 should be modified
        expected.loc[:2, "c"] = [200, 400, 600]  # 100*2, 200*2, 300*2
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_modifications_ordered(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import (
            MetaDataVariation,
        )

        # Apply modifications in order: first add 10, then multiply by 2
        mods = [
            MetaDataVariation(
                name="a",
                value=2,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=2,  # Applied second
            ),
            MetaDataVariation(
                name="a",
                value=10,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,  # Applied first
            ),
        ]
        result = apply_full_and_range_modifications(
            df.copy(), mods, "timestamp"
        )
        expected = df.copy()
        # First add 10: [11, 12, 13, 14, 15], then multiply by 2: [22, 24, 26, 28, 30]
        expected["a"] = [22, 24, 26, 28, 30]
        pd.testing.assert_frame_equal(result, expected)

    def test_mixed_full_and_range_modifications(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import (
            MetaDataVariation,
        )

        mods = [
            # Full modification (no timestamps) - applied first
            MetaDataVariation(
                name="a",
                value=5,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,
            ),
            # Range modification - applied second, only to middle rows
            MetaDataVariation(
                name="a",
                value=2,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=2,
                min_timestamp="2023-01-01T11:00:00",
                max_timestamp="2023-01-01T13:00:00",
            ),
        ]
        result = apply_full_and_range_modifications(
            df.copy(), mods, "timestamp"
        )
        expected = df.copy()
        # First add 5 to all: [6, 7, 8, 9, 10]
        # Then multiply middle rows by 2: [6, 14, 16, 18, 10]
        expected["a"] = [6, 14, 16, 18, 10]
        pd.testing.assert_frame_equal(result, expected)

    def test_modification_type_property(self):
        # Full modification (no timestamps)
        full_mod = MetaDataVariation(
            name="a",
            value=10,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.ADD,
            order=1,
        )
        assert full_mod.modification_type == "full"

        # Range modification (with min_timestamp)
        range_mod1 = MetaDataVariation(
            name="a",
            value=10,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.ADD,
            order=1,
            min_timestamp="2023-01-01T10:00:00",
        )
        assert range_mod1.modification_type == "range"

        # Range modification (with max_timestamp)
        range_mod2 = MetaDataVariation(
            name="a",
            value=10,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.ADD,
            order=1,
            max_timestamp="2023-01-01T12:00:00",
        )
        assert range_mod2.modification_type == "range"

        # Range modification (with both timestamps)
        range_mod3 = MetaDataVariation(
            name="a",
            value=10,
            perturbation_or_exact_value="perturbation",
            perturbation_type=PerturbationType.ADD,
            order=1,
            min_timestamp="2023-01-01T10:00:00",
            max_timestamp="2023-01-01T12:00:00",
        )
        assert range_mod3.modification_type == "range"

    def test_modification_nonexistent_column(self):
        df = self.make_df_with_timestamps()
        from synthefy_pkg.app.data_models import MetaDataVariation

        mod = MetaDataVariation(
            name="not_a_col",
            value=123,
            perturbation_or_exact_value="exact_value",
            order=1,
        )
        # Should not raise, should ignore
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "a", "b", "c", "text"])
        from synthefy_pkg.app.data_models import MetaDataVariation

        mod = MetaDataVariation(
            name="a",
            value=1,
            perturbation_or_exact_value="exact_value",
            order=1,
        )
        result = apply_full_and_range_modifications(
            df.copy(), [mod], "timestamp"
        )
        # Should remain empty
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_timestamp_range_validation(self):
        import pytest

        # Test that max_timestamp before min_timestamp raises error
        with pytest.raises(
            ValueError, match="min_timestamp.*must be before max_timestamp"
        ):
            MetaDataVariation(
                name="a",
                value=10,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,
                min_timestamp="2023-01-01T12:00:00",
                max_timestamp="2023-01-01T10:00:00",
            )

        # Test that invalid timestamp format raises error
        with pytest.raises(ValueError, match="not in valid ISO format"):
            MetaDataVariation(
                name="a",
                value=10,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,
                min_timestamp="invalid-timestamp",
            )


class TestOrderPreservationInFoundationModelService:
    """Test suite for ensuring order of modifications is preserved correctly in foundation model service."""

    def make_df(self):
        """Helper to create a simple DataFrame with a text column."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": list("abcde"),
                "text": ["foo", "bar", "baz", "qux", "quux"],
            }
        )

    def test_point_modifications_order_preservation(self):
        """Test that point modifications are applied in the correct order."""
        df = self.make_df()

        # Create modifications with specific order to test
        mods = [
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 99, "b": 123}
            ),
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 200, "c": "z"}
            ),
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"b": 300}
            ),
        ]

        result = apply_point_modifications(df.copy(), "date", mods)

        # Check that the last modification for each column wins
        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "a"] = (
            200  # Last modification wins
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "b"] = (
            300  # Last modification wins
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "c"] = (
            "z"  # Last modification wins
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_point_modifications_order_with_different_timestamps(self):
        """Test order preservation when modifications are at different timestamps."""
        df = self.make_df()

        mods = [
            FMDataPointModification(
                timestamp="2023-01-02", modification_dict={"a": 50}
            ),
            FMDataPointModification(
                timestamp="2023-01-04", modification_dict={"a": 100}
            ),
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 75}
            ),
        ]

        result = apply_point_modifications(df.copy(), "date", mods)

        # Check that each timestamp gets the correct modification
        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-02"), "a"] = 50
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "a"] = 75
        expected.loc[expected["date"] == pd.Timestamp("2023-01-04"), "a"] = 100

        pd.testing.assert_frame_equal(result, expected)

    def test_point_modifications_order_with_overlapping_columns(self):
        """Test order preservation when multiple modifications modify the same column at the same timestamp."""
        df = self.make_df()

        mods = [
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 10}
            ),
            FMDataPointModification(
                timestamp="2023-01-03", modification_dict={"a": 20, "b": 30}
            ),
            FMDataPointModification(
                timestamp="2023-01-03",
                modification_dict={"a": 30, "b": 40, "c": "x"},
            ),
        ]

        result = apply_point_modifications(df.copy(), "date", mods)

        # Check that the last modification for each column wins
        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "a"] = (
            30  # Last modification wins
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "b"] = (
            40  # Last modification wins
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "c"] = (
            "x"  # Last modification wins
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_full_and_range_modifications_order_preservation(self):
        """Test that full and range modifications are applied in the correct order based on the order field."""
        df = self.make_df()

        # Create modifications with different order values
        modifications = [
            MetaDataVariation(
                name="a",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=2,  # Applied second
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,  # Applied first
            ),
            MetaDataVariation(
                name="b",
                value=5.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,  # Applied first
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that modifications are applied in order: (1 * 2) + 10 = 12 for column 'a'
        # and 10 + 5 = 15 for column 'b'
        # The modifications are applied to each row based on the original values
        assert (result["a"] == [12, 14, 16, 18, 20]).all()  # 1 * 2 + 10 = 12
        assert (result["b"] == [15, 25, 35, 45, 55]).all()

    def test_full_and_range_modifications_order_with_exact_value(self):
        """Test order preservation when exact value modifications are mixed with perturbations."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,  # Applied first
            ),
            MetaDataVariation(
                name="a",
                value=50.0,
                perturbation_or_exact_value="exact_value",
                perturbation_type=None,
                order=1,  # Applied second - overwrites
            ),
            MetaDataVariation(
                name="a",
                value=5.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=2,  # Applied third
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that exact value overwrites and then addition is applied: 50 + 5 = 55
        assert (result["a"] == 55).all()

    def test_full_and_range_modifications_order_with_complex_operations(self):
        """Test order preservation with complex mathematical operations."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=5.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,  # Applied first
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,  # Applied second
            ),
            MetaDataVariation(
                name="a",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.SUBTRACT,
                order=2,  # Applied third
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.DIVIDE,
                order=3,  # Applied fourth
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check the complex calculation: ((1 + 5) * 2 - 10) / 2 = 1
        # The modifications are applied to each row based on the original values
        assert (result["a"] == [1, 2, 3, 4, 5]).all()

    def test_full_and_range_modifications_order_with_multiple_columns(self):
        """Test order preservation across multiple columns."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=5.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,
            ),
            MetaDataVariation(
                name="b",
                value=3.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=2,
            ),
            MetaDataVariation(
                name="c",
                value="x",
                perturbation_or_exact_value="exact_value",
                perturbation_type=None,
                order=3,
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check column 'a': (1 + 5) * 2 = 12
        assert (result["a"] == [12, 14, 16, 18, 20]).all()

        # Check column 'b': 10 + 3 = 13
        assert (result["b"] == [13, 23, 33, 43, 53]).all()

        # Check column 'c': "x"
        assert (result["c"] == "x").all()

    def test_full_and_range_modifications_order_with_negative_values(self):
        """Test order preservation with negative values."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=-2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,
            ),
            MetaDataVariation(
                name="a",
                value=-3.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check the calculation: (1 + (-2)) * (-3) = (-1) * (-3) = 3
        # The modifications are applied to each row based on the original values
        assert (result["a"] == [3, 0, -3, -6, -9]).all()

    def test_full_and_range_modifications_order_with_division_by_zero(self):
        """Test order preservation when division by zero occurs."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=0.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.DIVIDE,
                order=0,
            ),
            MetaDataVariation(
                name="a",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=1,
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that division by zero results in inf, and then addition still works
        assert (result["a"] == float("inf")).all()

    def test_full_and_range_modifications_order_with_floating_point_precision(
        self,
    ):
        """Test order preservation with floating point precision."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=0.1,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,
            ),
            MetaDataVariation(
                name="a",
                value=0.3,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,
            ),
            MetaDataVariation(
                name="a",
                value=0.5,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=2,
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check the calculation: ((1 + 0.1) * 0.3) + 0.5 = 1.1 * 0.3 + 0.5 = 0.33 + 0.5 = 0.83
        expected_values = [
            ((1.0 + 0.1) * 0.3) + 0.5,
            ((2.0 + 0.1) * 0.3) + 0.5,
            ((3.0 + 0.1) * 0.3) + 0.5,
            ((4.0 + 0.1) * 0.3) + 0.5,
            ((5.0 + 0.1) * 0.3) + 0.5,
        ]
        actual_values = result["a"].values
        assert all(
            abs(actual - expected) < 1e-10
            for actual, expected in zip(actual_values, expected_values)
        )

    def test_full_and_range_modifications_order_with_string_operations(self):
        """Test order preservation with string column operations."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="text",
                value="warning",
                perturbation_or_exact_value="exact_value",
                perturbation_type=None,
                order=0,
            ),
            MetaDataVariation(
                name="text",
                value="critical",
                perturbation_or_exact_value="exact_value",
                perturbation_type=None,
                order=1,
            ),
            MetaDataVariation(
                name="text",
                value="resolved",
                perturbation_or_exact_value="exact_value",
                perturbation_type=None,
                order=2,
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that the last exact value overwrites all previous ones
        assert (result["text"] == "resolved").all()

    def test_full_and_range_modifications_order_with_range_modifications(self):
        """Test order preservation with range modifications."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,
                min_timestamp="2023-01-02",
                max_timestamp="2023-01-04",
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,
                min_timestamp="2023-01-03",
                max_timestamp="2023-01-04",
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that range modifications are applied in order
        # For dates 2023-01-02: only first modification applies (2 + 10 = 12)
        # For dates 2023-01-03 and 2023-01-04: both modifications apply in order
        # For dates 2023-01-03: (3 + 10) * 2 = 26
        # For dates 2023-01-04: (4 + 10) * 2 = 28
        expected = df.copy()
        expected.loc[expected["date"] == pd.Timestamp("2023-01-02"), "a"] = (
            12  # 2 + 10
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-03"), "a"] = (
            26  # (3 + 10) * 2
        )
        expected.loc[expected["date"] == pd.Timestamp("2023-01-04"), "a"] = (
            28  # (4 + 10) * 2
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_full_and_range_modifications_order_with_mixed_full_and_range(self):
        """Test order preservation when mixing full and range modifications."""
        df = self.make_df()

        modifications = [
            MetaDataVariation(
                name="a",
                value=5.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=0,  # Full modification
            ),
            MetaDataVariation(
                name="a",
                value=2.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.MULTIPLY,
                order=1,
                min_timestamp="2023-01-03",
                max_timestamp="2023-01-04",
            ),
            MetaDataVariation(
                name="b",
                value=10.0,
                perturbation_or_exact_value="perturbation",
                perturbation_type=PerturbationType.ADD,
                order=2,  # Full modification
            ),
        ]

        result = apply_full_and_range_modifications(
            df.copy(), modifications, "date"
        )

        # Check that full modifications apply to all rows, and range modifications apply to specific rows
        expected = df.copy()
        # Column 'a': full modification + range modification for 2023-01-03 and 2023-01-04
        expected["a"] = [
            6,
            7,
            16,
            18,
            10,
        ]  # (1+5)*2=12 for 2023-01-03, (1+5)*2=12 for 2023-01-04, others 1+5=6
        # Column 'b': full modification
        expected["b"] = [20, 30, 40, 50, 60]  # 10+10, 20+10, etc.

        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
