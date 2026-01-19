import copy
import itertools
import json
import os
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from synthefy_pkg.app.config import (
    ForecastSettings,
    PreprocessSettings,
    SynthesisSettings,
)
from synthefy_pkg.app.data_models import (
    ConstraintType,
    DynamicTimeSeriesData,
    ForecastResponse,
    MetaData,
    MetaDataRange,
    MetaDataVariation,
    OneContinuousMetaData,
    OneContinuousMetaDataRange,
    OneDiscreteMetaData,
    OneDiscreteMetaDataRange,
    OneTimeSeries,
    PerturbationType,
    PreTrainedAnomalyResponse,
    SearchRequest,
    SearchResponse,
    SelectedAction,
    SelectedWindows,
    SynthefyRequest,
    SynthefyResponse,
    SynthefyTimeSeriesWindow,
    SynthesisResponse,
    TimeStamps,
    WindowFilters,
    WindowsSelectionOptions,
)
from synthefy_pkg.app.utils.api_utils import (
    apply_metadata_variations,
    array_to_continuous,
    array_to_discrete,
    array_to_timeseries,
    cleanup_local_directories,
    cleanup_tmp_dir,
    convert_discrete_metadata_range_to_label_tuple_range,
    convert_discrete_metadata_to_label_tuple,
    convert_dynamic_time_series_data_to_synthefy_request,
    convert_label_tuple_to_discrete_metadata,
    convert_response_to_synthefy_window_and_text,
    convert_synthefy_response_to_dynamic_time_series_data,
    create_synthefy_response_from_other_types,
    create_window_name_from_group_labels,
    delete_gt_real_timeseries_windows,
    delete_s3_objects,
    detect_time_frequency,
    extract_col_names,
    filter_window_dataframe_by_window_filters,
    filter_window_dataframe_continuous_conditions,
    filter_window_dataframe_discrete_conditions,
    filter_window_dataframe_group_labels,
    format_timestamp_with_optional_fractional_seconds,
    get_labels_description,
    get_settings,
    get_train_config_file_name,
    get_user_tmp_dir,
    get_window_naming_config,
    handle_file_upload,
    inverse_transform_discrete,
    s3_prefix_exists,
    trim_response_to_forecast_window,
    update_metadata_from_query,
)
from synthefy_pkg.app.utils.llm_utils import MetaDataToParse
from synthefy_pkg.preprocessing.preprocess import EmbeddingEncoder


@pytest.fixture
def synthefy_request():
    with open(
        os.path.join(
            str(os.environ.get("SYNTHEFY_PACKAGE_BASE")),
            "src/synthefy_pkg/app/tests/test_jsons/dispatch_twamp_one_month_request.json",
        ),
        "r",
    ) as f:
        synthefy_request_dict = json.load(f)

    return SynthefyRequest(**synthefy_request_dict)


@pytest.fixture
def sample_synthefy_request2():
    return SynthefyRequest(
        windows=[
            SynthefyTimeSeriesWindow(
                timeseries_data=[
                    OneTimeSeries(name="ts1", values=[1.0, 2.0, 3.0]),
                    OneTimeSeries(name="ts2", values=[4.0, 5.0, 6.0]),
                ],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="Label-Tuple-id-category",
                            values=["A-1", "B-2", "C-3"],
                        ),
                        OneDiscreteMetaData(
                            name="status", values=["ok", "ok", "fail"]
                        ),
                        OneDiscreteMetaData(
                            name="device_type", values=["A", "A", "A"]
                        ),
                    ],
                    continuous_conditions=[
                        OneContinuousMetaData(
                            name="temperature", values=[20.0, 21.0, 22.0]
                        ),
                        OneContinuousMetaData(
                            name="pressure", values=[1.0, 1.1, 1.2]
                        ),
                    ],
                ),
                timestamps=TimeStamps(
                    name="time",
                    values=["2023-01-01", "2023-01-02", "2023-01-03"],
                ),
            )
        ],
        selected_windows=SelectedWindows(
            window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
            window_indices=[0],
        ),
        n_anomalies=5,
        n_forecast_windows=1,
        n_synthesis_windows=1,
        n_view_windows=5,
        selected_action=SelectedAction.FORECAST,
        text="Sample forecast request",
        top_n_search_windows=5,
    )


@pytest.mark.parametrize(
    "query, expected_metadata, expected_n_windows",
    [
        (
            "Show forecast for ts1 with status ['ok', 'fail'] and temperature [25.0, 26.0, 27.0]",
            MetaDataToParse(
                discrete_conditions=[
                    OneDiscreteMetaData(name="status", values=["ok", "fail"])
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="temperature", values=[25.0, 26.0, 27.0]
                    )
                ],
                num_examples=1,
            ),
            1,
        ),
        (
            "Synthesize 3 examples with device_type ['B', 'C'] and pressure [1.5, 1.6, 1.7]",
            MetaDataToParse(
                discrete_conditions=[
                    OneDiscreteMetaData(name="device_type", values=["B", "C"])
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="pressure", values=[1.5, 1.6, 1.7]
                    )
                ],
                num_examples=3,
            ),
            3,
        ),
        (
            "Show forecast for ts1 with Label-Tuple-Group ['A-B', 'C-D'] and temperature [25.0, 26.0, 27.0]",
            MetaDataToParse(
                discrete_conditions=[
                    OneDiscreteMetaData(
                        name="Label-Tuple-id-category", values=["A-B", "C-D"]
                    )
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="temperature", values=[25.0, 26.0, 27.0]
                    )
                ],
                num_examples=1,
            ),
            1,
        ),
    ],
)
@patch("synthefy_pkg.app.utils.api_utils.extract_metadata_from_query")
def test_update_metadata_from_query(
    mock_extract_metadata,
    sample_synthefy_request2,
    query,
    expected_metadata,
    expected_n_windows,
):
    sample_synthefy_request2.text = query
    mock_extract_metadata.return_value = expected_metadata

    updated_request = update_metadata_from_query(sample_synthefy_request2)

    assert updated_request.n_forecast_windows == expected_n_windows
    assert updated_request.n_synthesis_windows == expected_n_windows

    for window in updated_request.windows:
        for condition in expected_metadata.discrete_conditions:
            if condition.name.startswith("Label-Tuple-"):
                assert any(
                    dc.name == condition.name
                    and set(dc.values) == set([condition.values[0]])
                    for dc in window.metadata.discrete_conditions
                )
            else:
                assert any(
                    dc.name == condition.name and dc.values == condition.values
                    for dc in window.metadata.discrete_conditions
                )

        for condition in expected_metadata.continuous_conditions:
            assert any(
                cc.name == condition.name
                and cc.values[: len(condition.values)] == condition.values
                for cc in window.metadata.continuous_conditions
            )


def test_extract_col_names(synthefy_request):
    timeseries_cols, continuous_cols, discrete_cols, timestamps_col = (
        extract_col_names(synthefy_request)
    )
    true_timeseries_cols = [
        "counter.drop_mean_fwd",
        "counter.drop_mean_rec",
        "counter.jitter_mean_fwd",
        "counter.jitter_mean_rec",
        "counter.rtt_mean",
    ]
    true_continuous_cols = [
        "distance_to_twamp_reflector_km",
        "counter.rtt_max",
        "counter.rtt_min",
        "counter.jitter_max_fwd",
        "counter.jitter_min_fwd",
        "counter.jitter_max_rec",
        "counter.jitter_min_rec",
        "counter.drop_mean_peak_fwd",
        "counter.drop_mean_peak_rec",
        "counter.tx_pkt",
        "counter.tx_pkt_reflector",
        "counter.rx_pkt",
        "counter.dscp",
        "counter.lost_pkt_fwd",
        "counter.lost_pkt_rec",
        "counter.drop_mean_all",
        "counter.success_rate_all",
        "month_val",
        "day_val",
        "year_val",
    ]
    true_discrete_cols = [
        "Label-Tuple-sto_kng-sran_node_id-sender_dscp",
        "label",
        "client_index",
        "sender_index",
        "client_dscp",
        "connection",
    ]
    true_timestamps_col = ["@timestamp"]
    assert timeseries_cols == true_timeseries_cols
    assert continuous_cols == true_continuous_cols
    assert discrete_cols == true_discrete_cols
    assert timestamps_col == true_timestamps_col


@pytest.fixture
def discrete_df():
    np.random.seed(42)

    # Generate synthetic data with more diverse unique values per column
    num_rows = 1000

    data = pd.DataFrame(
        {
            "Animal": np.random.choice(
                [
                    "cat",
                    "dog",
                    "mouse",
                    "elephant",
                    "lion",
                    "tiger",
                    "bear",
                    "wolf",
                    "fox",
                    "rabbit",
                ],
                size=num_rows,
            ),
            "Color": np.random.choice(
                [
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "white",
                    "purple",
                    "orange",
                    "pink",
                ],
                size=num_rows,
            ),
            "Country": np.random.choice(
                [
                    "USA",
                    "Canada",
                    "Mexico",
                    "Brazil",
                    "UK",
                    "Germany",
                    "France",
                    "China",
                    "India",
                    "Australia",
                    "Japan",
                    "Kazakhstan",
                ],
                size=num_rows,
            ),
            "Product": np.random.choice(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                size=num_rows,
            ),
            "AgeGroup": np.random.choice(
                ["child", "teen", "adult", "senior", "middle-aged"],
                size=num_rows,
            ),
            "Count": np.random.randint(
                1, 100, size=num_rows
            ),  # Discrete integer values
        }
    )
    return data


def test_array_to_timeseries():
    timeseries_window = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    channel_names = ["channel1", "channel2"]

    # Perform inverse transformation
    result = array_to_timeseries(timeseries_window, channel_names)

    assert len(result) == 2
    assert result[0].name == "channel1"
    assert result[1].name == "channel2"
    assert isinstance(result[0], OneTimeSeries)
    assert isinstance(result[1], OneTimeSeries)

    # Assert that the result data is the same as the original
    np.testing.assert_almost_equal(
        np.array(result[0].values, dtype=float),
        timeseries_window[:, 0],
        decimal=5,
    )
    np.testing.assert_almost_equal(
        np.array(result[1].values, dtype=float),
        timeseries_window[:, 1],
        decimal=5,
    )


def test_array_to_continuous():
    continuous_window = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    continuous_col_names = ["feature1", "feature2"]
    result = array_to_continuous(continuous_window, continuous_col_names)

    assert len(result) == 2
    assert result[0].name == "feature1"
    assert result[1].name == "feature2"
    assert isinstance(result[0], OneContinuousMetaData)
    assert isinstance(result[1], OneContinuousMetaData)

    # Assert that the result data is the same as the original
    np.testing.assert_almost_equal(
        np.array(result[0].values, dtype=float),
        continuous_window[:, 0],
        decimal=5,
    )
    np.testing.assert_almost_equal(
        np.array(result[1].values, dtype=float),
        continuous_window[:, 1],
        decimal=5,
    )


@pytest.mark.parametrize(
    "encoder_typesbycols",
    [
        {
            "onehot": ["Animal", "Product", "AgeGroup"],
            "embedding": ["Color", "Country", "Count"],
        },
        {
            "onehot": [
                "Animal",
            ],
            "embedding": ["Product", "AgeGroup", "Color", "Country", "Count"],
        },
        {
            "onehot": ["Color", "Country", "Animal", "Product", "AgeGroup"],
            "embedding": ["Count"],
        },
        {
            "onehot": [
                "Color",
                "Country",
                "Animal",
                "Product",
                "AgeGroup",
                "Count",
            ],
        },
        {
            "embedding": [
                "Color",
                "Country",
                "Animal",
                "Product",
                "AgeGroup",
                "Count",
            ],
        },
    ],
)
def test_inverse_transform_discrete(discrete_df, encoder_typesbycols):
    encoder_instances = {
        "onehot": OneHotEncoder(sparse_output=False),
        "embedding": EmbeddingEncoder(),
    }
    original_data = discrete_df.astype(str)
    cols_order = list(itertools.chain(*encoder_typesbycols.values()))
    windows_original_data = np.lib.stride_tricks.sliding_window_view(
        original_data[cols_order],
        window_shape=(256,),
        axis=0,
    )[::126]
    windows_original_data = np.transpose(windows_original_data, (0, 2, 1))

    encoders = {}
    encoded_bytypedata_list = []
    for encoder_type, encoder_type_cols in encoder_typesbycols.items():
        encoder = encoder_instances[encoder_type]
        encoder.fit(original_data[encoder_type_cols])
        encoders[encoder_type] = encoder
        encoded_bytypedata_list.append(
            encoder.transform(original_data[encoder_type_cols])
        )

    encoded_data = np.concatenate(encoded_bytypedata_list, axis=1)

    windows = np.lib.stride_tricks.sliding_window_view(
        pd.DataFrame(encoded_data), window_shape=(256,), axis=0
    )[::126]

    windows = np.transpose(windows, (0, 2, 1))
    # Perform inverse transformation
    final_col_names, decoded_array = inverse_transform_discrete(
        windows, encoders
    )

    # Assert that the inverse-transformed data is the same as the original
    assert final_col_names == cols_order
    assert np.array_equal(decoded_array, windows_original_data)


def test_inverse_transform_discrete_no_encoders(discrete_df):
    encoder_typesbycols = {}

    original_data = discrete_df.astype(str)
    cols_order = list(itertools.chain(*encoder_typesbycols.values()))
    windows_original_data = np.lib.stride_tricks.sliding_window_view(
        original_data[cols_order],
        window_shape=(256,),
        axis=0,
    )[::126]
    windows_original_data = np.transpose(windows_original_data, (0, 2, 1))

    windows = np.transpose(windows_original_data, (0, 2, 1))
    # Perform inverse transformation
    final_col_names, decoded_array = inverse_transform_discrete(windows, {})

    # Assert that the inverse-transformed data is the same as the original
    assert final_col_names == []
    assert np.array_equal(decoded_array, np.array([]))


def test_array_to_discrete():
    discrete_window = np.array([[1, 2], [3, 4], [5, 6]])
    discrete_col_names = ["discrete1", "discrete2"]
    result = array_to_discrete(discrete_window, discrete_col_names)

    assert len(result) == 2
    assert result[0].name == "discrete1"
    assert result[1].name == "discrete2"
    assert isinstance(result[0], OneDiscreteMetaData)
    assert isinstance(result[1], OneDiscreteMetaData)

    # Check the discrete data matches the original
    np.testing.assert_array_equal(
        np.array([result[0].values, result[1].values]),
        discrete_window.T,  # Transpose because the result list is column-wise
    )


# BELOW are from old dispatcher tests.
@pytest.fixture
def sample_synthefy_request():
    return SynthefyRequest(
        windows=[
            SynthefyTimeSeriesWindow(
                timeseries_data=[
                    OneTimeSeries(name="series1", values=[1.0, 2.0, 3.0])
                ],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="condition1", values=["a", "b", "c"]
                        )
                    ],
                    continuous_conditions=[
                        OneContinuousMetaData(
                            name="condition2", values=[1.0, 2.0, 3.0]
                        )
                    ],
                ),
                timestamps=TimeStamps(
                    name="time",
                    values=["2023-01-01", "2023-01-02", "2023-01-03"],
                ),
            ),
            SynthefyTimeSeriesWindow(
                timeseries_data=[
                    OneTimeSeries(name="series2", values=[2.0, 3.0, 4.0])
                ],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="condition1", values=["b", "c", "d"]
                        )
                    ],
                    continuous_conditions=[
                        OneContinuousMetaData(
                            name="condition2", values=[2.0, 3.0, 4.0]
                        )
                    ],
                ),
                timestamps=TimeStamps(
                    name="time",
                    values=["2023-01-02", "2023-01-03", "2023-01-04"],
                ),
            ),
            SynthefyTimeSeriesWindow(
                timeseries_data=[
                    OneTimeSeries(name="series3", values=[3.0, 4.0, 5.0])
                ],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="condition1", values=["c", "d", "e"]
                        )
                    ],
                    continuous_conditions=[
                        OneContinuousMetaData(
                            name="condition2", values=[3.0, 4.0, 5.0]
                        )
                    ],
                ),
                timestamps=TimeStamps(
                    name="time",
                    values=["2023-01-03", "2023-01-04", "2023-01-05"],
                ),
            ),
        ],
        selected_windows=SelectedWindows(
            window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
            window_indices=[0],
        ),
        n_anomalies=5,
        n_forecast_windows=1,
        n_synthesis_windows=1,
        n_view_windows=5,
        selected_action=SelectedAction.FORECAST,
        text="Sample forecast request",
        top_n_search_windows=5,
    )


@pytest.mark.parametrize(
    "n_forecast_windows, selected_windows",
    [
        (
            1,
            SelectedWindows(
                window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
                window_indices=[0],
            ),
        ),
        (
            2,
            SelectedWindows(
                window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
                window_indices=[1],
            ),
        ),
        (
            2,
            SelectedWindows(
                window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
                window_indices=[0, 2],
            ),
        ),
    ],
)
@pytest.fixture
def sample_synthesis_response():
    return SynthesisResponse(
        x_axis=TimeStamps(
            name="time", values=["2023-01-01", "2023-01-02", "2023-01-03"]
        ),
        timeseries_data=[
            OneTimeSeries(name="series1", values=[1.0, 2.0, 3.0]),
            OneTimeSeries(name="series1_synthetic", values=[1.1, 2.1, 3.1]),
        ],
        metadata=MetaData(
            discrete_conditions=[
                OneDiscreteMetaData(name="condition1", values=["a", "b", "c"])
            ],
            continuous_conditions=[
                OneContinuousMetaData(name="condition2", values=[1.0, 2.0, 3.0])
            ],
        ),
    )


@pytest.fixture
def sample_search_response():
    return SearchResponse(
        x_axis=[
            TimeStamps(
                name="time", values=["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            TimeStamps(
                name="time", values=["2023-01-02", "2023-01-03", "2023-01-04"]
            ),
        ],
        timeseries_data=[
            [OneTimeSeries(name="series1", values=[1.0, 2.0, 3.0])],
            [OneTimeSeries(name="series2", values=[2.0, 3.0, 4.0])],
        ],
        metadata=[
            MetaData(
                discrete_conditions=[
                    OneDiscreteMetaData(
                        name="condition1", values=["a", "b", "c"]
                    ),
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="condition2", values=[1.0, 2.0, 3.0]
                    ),
                ],
            ),
            MetaData(
                discrete_conditions=[
                    OneDiscreteMetaData(
                        name="condition1", values=["b", "c", "d"]
                    ),
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="condition2", values=[2.0, 3.0, 4.0]
                    ),
                ],
            ),
        ],
        text="Sample search response",
    )


@pytest.fixture
def sample_forecast_response():
    return ForecastResponse(
        x_axis=TimeStamps(
            name="time", values=["2023-01-01", "2023-01-02", "2023-01-03"]
        ),
        timeseries_data=[OneTimeSeries(name="series1", values=[1.0, 2.0, 3.0])],
        metadata=MetaData(
            discrete_conditions=[
                OneDiscreteMetaData(name="condition1", values=["a", "b", "c"])
            ],
            continuous_conditions=[
                OneContinuousMetaData(name="condition2", values=[1.0, 2.0, 3.0])
            ],
        ),
        start_of_forecast_timestamp=TimeStamps(
            name="time", values=["2023-01-04"]
        ),
    )


@pytest.fixture
def sample_anomaly_response():
    return PreTrainedAnomalyResponse(
        x_axis=TimeStamps(
            name="time", values=["2023-01-01", "2023-01-02", "2023-01-03"]
        ),
        timeseries_data=[OneTimeSeries(name="series1", values=[1.0, 2.0, 3.0])],
        metadata=MetaData(
            discrete_conditions=[
                OneDiscreteMetaData(name="condition1", values=["a", "b", "c"])
            ],
            continuous_conditions=[
                OneContinuousMetaData(name="condition2", values=[1.0, 2.0, 3.0])
            ],
        ),
        anomaly_timestamps=TimeStamps(
            name="time", values=["2023-01-02", "2023-01-03"]
        ),
    )


class TestConvertResponseToSynthefyWindowAndText:
    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    def test_convert_synthesis_response_to_synthefy_window_and_text(
        self, mock_get_labels, mock_get_window_naming, sample_synthesis_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        window, forecast_timestamp, anomaly_timestamps = (
            convert_response_to_synthefy_window_and_text(
                "dummy_dataset_name", sample_synthesis_response
            )
        )
        assert (
            window.timeseries_data == sample_synthesis_response.timeseries_data
        )
        assert window.metadata == sample_synthesis_response.metadata
        assert window.timestamps == sample_synthesis_response.x_axis
        assert forecast_timestamp == []
        assert anomaly_timestamps == []

    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    def test_convert_forecast_response_to_synthefy_window_and_text(
        self, mock_get_labels, mock_get_window_naming, sample_forecast_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        window, forecast_timestamp, anomaly_timestamps = (
            convert_response_to_synthefy_window_and_text(
                "dummy_dataset_name", sample_forecast_response
            )
        )
        assert window.timestamps == sample_forecast_response.x_axis
        assert (
            window.timeseries_data == sample_forecast_response.timeseries_data
        )
        assert window.metadata == sample_forecast_response.metadata
        assert (
            forecast_timestamp
            == sample_forecast_response.start_of_forecast_timestamp
        )
        assert anomaly_timestamps == []

    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    def test_convert_search_response_to_synthefy_window_and_text(
        self, mock_get_labels, mock_get_window_naming, sample_search_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        windows, forecast_timestamp, anomaly_timestamps = (
            convert_response_to_synthefy_window_and_text(
                "dummy_dataset_name", sample_search_response
            )
        )
        assert len(windows) == len(sample_search_response.x_axis)
        for i in range(len(sample_search_response.x_axis)):
            assert windows[i].id == i
            assert windows[i].timestamps == sample_search_response.x_axis[i]
            assert (
                windows[i].timeseries_data
                == sample_search_response.timeseries_data[i]
            )
            assert windows[i].metadata == sample_search_response.metadata[i]
        assert forecast_timestamp == []
        assert anomaly_timestamps == []


class TestCreateSynthefyResponse:
    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    @pytest.mark.asyncio
    async def test_create_synthefy_response_from_view_and_synthesis_response(
        self, mock_get_labels, mock_get_window_naming, sample_synthesis_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        responses = [sample_synthesis_response, sample_synthesis_response]
        synthefy_response = await create_synthefy_response_from_other_types(
            "dummy_dataset_name", responses
        )
        assert len(synthefy_response.windows) == len(responses)
        assert isinstance(synthefy_response.combined_text, str)
        assert len(synthefy_response.forecast_timestamps) == 0
        assert len(synthefy_response.anomaly_timestamps) == 0

    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    @pytest.mark.asyncio
    async def test_create_synthefy_response_from_search_response(
        self, mock_get_labels, mock_get_window_naming, sample_search_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        responses = [sample_search_response]
        synthefy_response = await create_synthefy_response_from_other_types(
            "dummy_dataset_name", responses
        )
        assert len(synthefy_response.windows) == 2
        assert isinstance(synthefy_response.combined_text, str)
        assert len(synthefy_response.forecast_timestamps) == 0
        assert len(synthefy_response.anomaly_timestamps) == 0

    @patch("synthefy_pkg.app.utils.api_utils.get_window_naming_config")
    @patch("synthefy_pkg.app.utils.api_utils.get_labels_description")
    @pytest.mark.asyncio
    async def test_create_synthefy_response_from_forecast_response(
        self, mock_get_labels, mock_get_window_naming, sample_forecast_response
    ):
        # Mock config returns
        mock_get_window_naming.return_value = {
            "synthesis_prefix": "Synthesis",
            "search_prefix": "Search",
            "forecast_prefix": "Forecast",
        }
        mock_get_labels.return_value = {
            "group_labels_combinations": {},
            "group_label_cols": [],
        }

        responses = [sample_forecast_response, sample_forecast_response]
        synthefy_response = await create_synthefy_response_from_other_types(
            "dummy_dataset_name", responses
        )
        assert len(synthefy_response.windows) == 2
        assert isinstance(synthefy_response.combined_text, str)
        assert len(synthefy_response.forecast_timestamps) == 2
        assert len(synthefy_response.anomaly_timestamps) == 0


class TestDeleteGtRealTimeseriesWindows:
    def test_delete_nothing(self, sample_synthefy_request):
        new_request = delete_gt_real_timeseries_windows(sample_synthefy_request)
        assert new_request == sample_synthefy_request

    def test_delete_synthetic_timeseries(self, sample_synthefy_request):
        original_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        original_request_copy = copy.deepcopy(original_request)  # deep copy

        original_request.windows[0].timeseries_data.append(
            OneTimeSeries(name="to_drop_synthetic", values=[1.0, 2.0, 3.0])
        )  # add data to drop

        new_request = delete_gt_real_timeseries_windows(original_request)
        assert new_request == original_request_copy

    def test_dont_delete_last_window(self, sample_synthefy_request):
        # deep copy to avoid mutating the original request
        original_synthefy_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        for ts in original_synthefy_request.windows[-1].timeseries_data:
            ts.name = "not_to_drop_synthetic"
        new_request = delete_gt_real_timeseries_windows(
            copy.deepcopy(original_synthefy_request)
        )
        assert len(new_request.windows) == len(
            original_synthefy_request.windows
        )

    def test_delete_one_timeseries_from_window(self, sample_synthefy_request):
        # deep copy to avoid mutating the original request
        original_synthefy_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        # make the first window have 2 timeseries, and drop the second one (the first one is real)
        original_synthefy_request.windows[0].timeseries_data.append(
            OneTimeSeries(name="to_drop_synthetic", values=[1.0, 2.0, 3.0])
        )

        new_request = delete_gt_real_timeseries_windows(
            copy.deepcopy(original_synthefy_request)
        )

        assert len(new_request.windows) == len(
            original_synthefy_request.windows
        )
        assert (
            len(new_request.windows[0].timeseries_data)
            == len(original_synthefy_request.windows[0].timeseries_data) - 1
        )
        assert len(new_request.windows[1].timeseries_data) == len(
            original_synthefy_request.windows[1].timeseries_data
        )

    def test_remove_query_suffix(self, sample_synthefy_request):
        # deep copy to avoid mutating the original request
        original_synthefy_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        for window in original_synthefy_request.windows:
            for ts in window.timeseries_data:
                ts.name = f"{ts.name}_query"

        new_request = delete_gt_real_timeseries_windows(
            copy.deepcopy(original_synthefy_request)
        )

        assert len(new_request.windows) == len(
            original_synthefy_request.windows
        )
        for ts in new_request.windows[0].timeseries_data:
            assert not ts.name.endswith("_query")
            assert ts.name in [
                orig_ts.name[:-6]
                for orig_ts in original_synthefy_request.windows[
                    0
                ].timeseries_data
            ]

    def test_remove_synthetic_suffix(self, sample_synthefy_request):
        # deep copy to avoid mutating the original request
        original_synthefy_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        for window in original_synthefy_request.windows:
            for ts in window.timeseries_data:
                ts.name = f"{ts.name}_synthetic"

        new_request = delete_gt_real_timeseries_windows(
            copy.deepcopy(original_synthefy_request)
        )

        assert len(new_request.windows) == len(
            original_synthefy_request.windows
        )
        for ts in new_request.windows[0].timeseries_data:
            assert not ts.name.endswith("_synthetic")
            assert ts.name in [
                orig_ts.name[:-10]
                for orig_ts in original_synthefy_request.windows[
                    0
                ].timeseries_data
            ]

    def test_mixed_query_and_synthetic(self, sample_synthefy_request):
        original_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        original_request.windows[0].timeseries_data[0].name = "series1_query"
        original_request.windows[0].timeseries_data.append(
            OneTimeSeries(name="to_drop_synthetic", values=[1.0, 2.0, 3.0])
        )

        with pytest.raises(
            ValueError,
            match="No timeseries left after dropping -- this should never happen in prod/UI. Please check the code.",
        ):
            delete_gt_real_timeseries_windows(original_request)

    def test_remove_forecast_suffix(self, sample_synthefy_request):
        # deep copy to avoid mutating the original request
        original_synthefy_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        for window in original_synthefy_request.windows:
            for ts in window.timeseries_data:
                ts.name = f"{ts.name}_forecast"

        new_request = delete_gt_real_timeseries_windows(
            copy.deepcopy(original_synthefy_request)
        )

        assert len(new_request.windows) == len(
            original_synthefy_request.windows
        )
        for ts in new_request.windows[0].timeseries_data:
            assert not ts.name.endswith("_forecast")
            assert ts.name in [
                orig_ts.name[:-9]  # -9 for '_forecast'
                for orig_ts in original_synthefy_request.windows[
                    0
                ].timeseries_data
            ]

    def test_mixed_forecast_and_real(self, sample_synthefy_request):
        original_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        # Keep one real timeseries and add a forecast one
        original_request.windows[0].timeseries_data.append(
            OneTimeSeries(name="to_drop_forecast", values=[1.0, 2.0, 3.0])
        )

        new_request = delete_gt_real_timeseries_windows(original_request)

        # Should only keep the real timeseries
        assert len(new_request.windows[0].timeseries_data) == 1
        assert (
            not new_request.windows[0]
            .timeseries_data[0]
            .name.endswith("_forecast")
        )

    def test_mixed_forecast_and_synthetic(self, sample_synthefy_request):
        original_request = SynthefyRequest(
            **sample_synthefy_request.model_dump()
        )
        original_request.windows[0].timeseries_data[0].name = "series1_forecast"
        original_request.windows[0].timeseries_data.append(
            OneTimeSeries(name="to_drop_synthetic", values=[1.0, 2.0, 3.0])
        )

        with pytest.raises(
            ValueError,
            match="No timeseries left after dropping -- this should never happen in prod/UI. Please check the code.",
        ):
            delete_gt_real_timeseries_windows(original_request)


class TestUseLabelTupleWithDiscreteMetadata:
    def test_convert_discrete_metadata_to_label_tuple(self):
        # store_id, store_category are labels
        # store_location is a discrete condition
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1", "2", "2", "1"]),
            OneDiscreteMetaData(
                name="store_category",
                values=["clothes", "watches", "watches", "clothes"],
            ),
            OneDiscreteMetaData(
                name="store_location",
                values=["mall", "standalone", "mall", "suburbs"],
            ),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": ["1-clothes", "2-watches"]
        }

        result = convert_discrete_metadata_to_label_tuple(
            discrete_metadata=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )

        assert len(result) == 2
        assert result[0].name == "Label-Tuple-store_id-store_category"
        assert result[0].values == [
            "1-clothes",
            "2-watches",
            "2-watches",
            "1-clothes",
        ]

        assert "store_id" not in [meta.name for meta in result]
        assert "store_category" not in [meta.name for meta in result]

    def test_empty_group_labels_combinations(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1", "2", "2", "1"]),
            OneDiscreteMetaData(
                name="store_category",
                values=["clothes", "watches", "watches", "clothes"],
            ),
        ]
        group_label_cols = []
        group_labels_combinations = {"a": ["b"]}

        result = convert_discrete_metadata_to_label_tuple(
            discrete_metadata=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == discrete_conditions

        group_label_cols = ["a"]
        group_labels_combinations = {}

        result = convert_discrete_metadata_to_label_tuple(
            discrete_metadata=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == discrete_conditions

    def test_only_group_label_cols(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1", "2", "2", "1"]),
            OneDiscreteMetaData(
                name="store_category",
                values=["clothes", "watches", "watches", "clothes"],
            ),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": ["1-clothes", "2-watches"]
        }

        result = convert_discrete_metadata_to_label_tuple(
            discrete_metadata=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == [
            OneDiscreteMetaData(
                name="Label-Tuple-store_id-store_category",
                values=["1-clothes", "2-watches", "2-watches", "1-clothes"],
            )
        ]

    def test_invalid_combinations(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1", "2", "3", "1"]),
            OneDiscreteMetaData(
                name="store_category",
                values=["clothes", "watches", "electronics", "clothes"],
            ),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": ["1-clothes", "2-watches"]
        }
        with pytest.raises(ValueError, match="Invalid combinations found:"):
            convert_discrete_metadata_to_label_tuple(
                discrete_metadata=discrete_conditions,
                group_labels_combinations=group_labels_combinations,
                group_label_cols=group_label_cols,
            )

    def test_missing_metadata(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1", "2", "2", "1"]),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": ["1-clothes", "2-watches"]
        }

        with pytest.raises(ValueError, match="Missing metadata for columns:"):
            convert_discrete_metadata_to_label_tuple(
                discrete_metadata=discrete_conditions,
                group_labels_combinations=group_labels_combinations,
                group_label_cols=group_label_cols,
            )


class TestUseLabelTupleWithDiscreteMetadataRange:
    def test_convert_discrete_metadata_range_to_label_tuple_range(self):
        # store_id, store_category are labels
        # store_location is a discrete condition
        discrete_conditions = [
            OneDiscreteMetaDataRange(name="store_id", options=["1", "2", "3"]),
            OneDiscreteMetaDataRange(
                name="store_category",
                options=["clothes", "watches", "electronics"],
            ),
            OneDiscreteMetaDataRange(
                name="store_location", options=["mall", "standalone", "suburbs"]
            ),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": [
                "1-clothes",
                "2-watches",
                "3-electronics",
            ]
        }

        result = convert_discrete_metadata_range_to_label_tuple_range(
            discrete_metadata_range=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )

        assert len(result) == 2
        assert result[0].name == "Label-Tuple-store_id-store_category"
        assert result[0].options == ["1-clothes", "2-watches", "3-electronics"]

        assert "store_id" not in [meta.name for meta in result]
        assert "store_category" not in [meta.name for meta in result]

    def test_empty_group_labels_combinations(self):
        discrete_conditions = [
            OneDiscreteMetaDataRange(name="store_id", options=["1", "2", "3"]),
            OneDiscreteMetaDataRange(
                name="store_category",
                options=["clothes", "watches", "electronics"],
            ),
        ]
        group_label_cols = []
        group_labels_combinations = {"a": ["b"]}

        result = convert_discrete_metadata_range_to_label_tuple_range(
            discrete_metadata_range=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == discrete_conditions

        group_label_cols = ["a"]
        group_labels_combinations = {}

        result = convert_discrete_metadata_range_to_label_tuple_range(
            discrete_metadata_range=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == discrete_conditions

    def test_only_group_label_cols(self):
        discrete_conditions = [
            OneDiscreteMetaDataRange(name="store_id", options=["1", "2", "3"]),
            OneDiscreteMetaDataRange(
                name="store_category",
                options=["clothes", "watches", "electronics"],
            ),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": [
                "1-clothes",
                "2-watches",
                "3-electronics",
            ]
        }

        result = convert_discrete_metadata_range_to_label_tuple_range(
            discrete_metadata_range=discrete_conditions,
            group_labels_combinations=group_labels_combinations,
            group_label_cols=group_label_cols,
        )
        assert result == [
            OneDiscreteMetaDataRange(
                name="Label-Tuple-store_id-store_category",
                options=["1-clothes", "2-watches", "3-electronics"],
            )
        ]

    def test_missing_metadata(self):
        discrete_conditions = [
            OneDiscreteMetaDataRange(name="store_id", options=["1", "2", "3"]),
        ]
        group_label_cols = ["store_id", "store_category"]
        group_labels_combinations = {
            "store_id-store_category": [
                "1-clothes",
                "2-watches",
                "3-electronics",
            ]
        }

        with pytest.raises(ValueError, match="Missing metadata for columns:"):
            convert_discrete_metadata_range_to_label_tuple_range(
                discrete_metadata_range=discrete_conditions,
                group_labels_combinations=group_labels_combinations,
                group_label_cols=group_label_cols,
            )


@pytest.fixture
def sample_request_with_label_tuple():
    return SynthefyRequest(
        windows=[
            SynthefyTimeSeriesWindow(
                timeseries_data=[],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="Label-Tuple-store_id-store_category",
                            values=["1-clothes", "2-watches", "1-clothes"],
                        )
                    ],
                    continuous_conditions=[],
                ),
                timestamps=TimeStamps(name="test", values=[1, 2, 3]),
            )
        ],
        selected_windows=SelectedWindows(
            window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
            window_indices=[0],
        ),
        n_anomalies=0,
        n_forecast_windows=0,
        n_synthesis_windows=0,
        n_view_windows=0,
        selected_action=SelectedAction.SYNTHESIS,
        text="",
        top_n_search_windows=0,
    )


@pytest.fixture
def sample_request_without_label_tuple():
    return SynthefyRequest(
        windows=[
            SynthefyTimeSeriesWindow(
                timeseries_data=[],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="store_id", values=["1", "2", "1"]
                        ),
                        OneDiscreteMetaData(
                            name="store_category",
                            values=["clothes", "watches", "clothes"],
                        ),
                    ],
                    continuous_conditions=[],
                ),
                timestamps=TimeStamps(name="test", values=[1, 2, 3]),
            )
        ],
        selected_windows=SelectedWindows(
            window_type=WindowsSelectionOptions.CURRENT_VIEW_WINDOWS,
            window_indices=[0],
        ),
        n_anomalies=0,
        n_forecast_windows=0,
        n_synthesis_windows=0,
        n_view_windows=0,
        selected_action=SelectedAction.SYNTHESIS,
        text="",
        top_n_search_windows=0,
    )


class TestConvertLabelTupleToDiscreteMetadata:
    def test_convert_label_tuple_success(self, sample_request_with_label_tuple):
        request = convert_label_tuple_to_discrete_metadata(
            sample_request_with_label_tuple
        )
        window = request.windows[0]

        assert len(window.metadata.discrete_conditions) == 2
        # Check that the label tuple has been removed
        condition_names = [
            cond.name for cond in window.metadata.discrete_conditions
        ]
        assert "Label-Tuple-store_id-store_category" not in condition_names

        # Check that the original discrete metadata has been restored
        store_id = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_id"
        )
        store_category = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_category"
        )

        assert store_id.values == ["1", "2", "1"]
        assert store_category.values == ["clothes", "watches", "clothes"]

    def test_convert_label_tuple_multiple_windows(
        self, sample_request_with_label_tuple
    ):
        sample_request_with_label_tuple.windows.append(
            SynthefyTimeSeriesWindow(
                timeseries_data=[],
                metadata=MetaData(
                    discrete_conditions=[
                        OneDiscreteMetaData(
                            name="Label-Tuple-store_id-store_category",
                            values=["3-electronics", "4-accessories"],
                        )
                    ],
                    continuous_conditions=[],
                ),
                timestamps=TimeStamps(name="test", values=[1, 2]),
            ),
        )
        modified_request = convert_label_tuple_to_discrete_metadata(
            sample_request_with_label_tuple
        )
        for window in modified_request.windows:
            assert len(window.metadata.discrete_conditions) == 2
            # Check that the label tuple has been removed
            condition_names = [
                cond.name for cond in window.metadata.discrete_conditions
            ]
            assert "Label-Tuple-store_id-store_category" not in condition_names

            # Check that the original discrete metadata has been restored
            store_id = next(
                cond
                for cond in window.metadata.discrete_conditions
                if cond.name == "store_id"
            )
            store_category = next(
                cond
                for cond in window.metadata.discrete_conditions
                if cond.name == "store_category"
            )

            if store_id.values == ["1", "2"]:
                assert store_category.values == ["clothes", "watches"]
            elif store_id.values == ["3", "4"]:
                assert store_category.values == ["electronics", "accessories"]

    def test_no_label_tuple(self, sample_request_without_label_tuple):
        original_request = copy.deepcopy(sample_request_without_label_tuple)
        request = convert_label_tuple_to_discrete_metadata(original_request)
        assert request == original_request
        window = request.windows[0]

        # Ensure no changes have been made
        assert len(window.metadata.discrete_conditions) == 2
        store_id = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_id"
        )
        store_category = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_category"
        )

        assert store_id.values == ["1", "2", "1"]
        assert store_category.values == ["clothes", "watches", "clothes"]

    def test_multiple_label_tuples_in_one_window(
        self, sample_request_with_label_tuple
    ):
        sample_request_with_label_tuple.windows[
            0
        ].metadata.discrete_conditions.append(
            OneDiscreteMetaData(
                name="Label-Tuple-region-department",
                values=["north-electronics", "south-clothing"],
            )
        )
        modified_request = convert_label_tuple_to_discrete_metadata(
            sample_request_with_label_tuple
        )
        window = modified_request.windows[0]

        # Only the first label tuple should be processed and removed
        assert len(window.metadata.discrete_conditions) == 3
        condition_names = [
            cond.name for cond in window.metadata.discrete_conditions
        ]
        assert "Label-Tuple-store_id-store_category" not in condition_names
        assert "Label-Tuple-region-department" in condition_names

        # Check that the original discrete metadata has been restored
        store_id = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_id"
        )
        store_category = next(
            cond
            for cond in window.metadata.discrete_conditions
            if cond.name == "store_category"
        )

        assert store_id.values == ["1", "2", "1"]
        assert store_category.values == ["clothes", "watches", "clothes"]

    def test_invalid_label_tuple_format(self, sample_request_with_label_tuple):
        # Label tuple without enough parts after split
        sample_request_with_label_tuple.windows[0].metadata.discrete_conditions[
            0
        ].name = "Label-Tuple-store_id"

        sample_request_with_label_tuple.windows[0].metadata.discrete_conditions[
            0
        ].values = ["1-clothes", "2-watches", "1-clothes", "3-watches"]

        with pytest.raises(
            ValueError,
            match="Group label cols and values must be the same length.",
        ):
            convert_label_tuple_to_discrete_metadata(
                sample_request_with_label_tuple
            )

    def test_empty_label_tuple_values(self, sample_request_with_label_tuple):
        sample_request_with_label_tuple.windows[0].metadata.discrete_conditions[
            0
        ].values = []

        with pytest.raises(
            ValueError, match="Group label values cannot be empty."
        ):
            convert_label_tuple_to_discrete_metadata(
                sample_request_with_label_tuple
            )


class TestConvertDynamicTimeSeriesDataToSynthefyRequest:
    """Test suite for convert_dynamic_time_series_data_to_synthefy_request function."""

    @pytest.fixture
    def base_dynamic_data(self):
        """Base fixture with valid data for most test cases."""
        return DynamicTimeSeriesData(
            root={
                "timestamp": {
                    "0": "2023-01-01T00:00:00",
                    "1": "2023-01-01T01:00:00",
                    "2": "2023-01-01T02:00:00",
                },
                "temperature": {
                    "0": 20.5,
                    "1": 21.2,
                    "2": 22.1,
                },
                "humidity": {
                    "0": 45.0,
                    "1": 46.5,
                    "2": 47.2,
                },
                "pressure": {
                    "0": 1013.2,
                    "1": 1014.1,
                    "2": 1013.8,
                },
                "sensor_id": {
                    "0": "A123",
                    "1": "A123",
                    "2": "A123",
                },
                "location": {
                    "0": "north",
                    "1": "north",
                    "2": "north",
                },
                "status": {
                    "0": "active",
                    "1": "active",
                    "2": "active",
                },
                "quality_score": {
                    "0": 0.95,
                    "1": 0.97,
                    "2": 0.94,
                },
                "error_rate": {
                    "0": 0.02,
                    "1": 0.01,
                    "2": 0.02,
                },
            }
        )

    @pytest.fixture
    def base_config(self):
        """Base configuration for the conversion function."""
        return {
            "group_label_cols": ["sensor_id"],
            "timeseries_colnames": ["temperature", "humidity", "pressure"],
            "continuous_colnames": ["quality_score", "error_rate"],
            "discrete_colnames": ["location", "status"],
            "timestamps_colname": ["timestamp"],
            "window_size": 3,
            "selected_action": SelectedAction.FORECAST,
        }

    def test_successful_conversion(self, base_dynamic_data, base_config):
        """Test successful conversion with all data types."""
        result = convert_dynamic_time_series_data_to_synthefy_request(
            base_dynamic_data, **base_config
        )

        # Verify basic structure
        assert isinstance(result, SynthefyRequest)
        assert len(result.windows) == 1
        window = result.windows[0]

        # Verify window structure
        assert window.id == 0
        assert window.name == "Window 0"
        assert window.timestamps is not None
        assert window.timestamps.values is not None
        assert len(window.timestamps.values) == base_config["window_size"]

        # Verify timeseries data
        assert len(window.timeseries_data) == len(
            base_config["timeseries_colnames"]
        )
        for ts in window.timeseries_data:
            assert ts.name in base_config["timeseries_colnames"]
            assert len(ts.values) == base_config["window_size"]
            assert all(isinstance(v, float) for v in ts.values)

        # Verify metadata
        assert len(window.metadata.discrete_conditions) == len(
            base_config["discrete_colnames"] + base_config["group_label_cols"]
        )
        assert len(window.metadata.continuous_conditions) == len(
            base_config["continuous_colnames"]
        )

        # Verify selected windows configuration
        assert (
            result.selected_windows.window_type
            == WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
        )
        assert result.selected_windows.window_indices == [0]

    @pytest.mark.parametrize("window_size", [-1, 0, 1000])
    def test_invalid_window_sizes(
        self, base_dynamic_data, base_config, window_size
    ):
        """Test handling of invalid window sizes."""
        modified_config = base_config.copy()
        modified_config["window_size"] = window_size

        with pytest.raises(ValueError):
            convert_dynamic_time_series_data_to_synthefy_request(
                base_dynamic_data, **modified_config
            )

    def test_missing_columns(self, base_dynamic_data, base_config):
        """Test handling of missing columns in the data."""
        column_types = [
            ("timeseries_colnames", "nonexistent_timeseries"),
            ("continuous_colnames", "nonexistent_continuous"),
            ("discrete_colnames", "nonexistent_discrete"),
            ("timestamps_colname", ["nonexistent_timestamp"]),
        ]

        for config_key, nonexistent_col in column_types:
            modified_config = base_config.copy()
            if config_key == "timestamps_colname":
                modified_config[config_key] = nonexistent_col
            else:
                modified_config[config_key] = base_config[config_key] + [
                    nonexistent_col
                ]

            with pytest.raises(ValueError):
                convert_dynamic_time_series_data_to_synthefy_request(
                    base_dynamic_data, **modified_config
                )

    def test_inconsistent_data_lengths(self, base_dynamic_data, base_config):
        """Test handling of inconsistent data lengths."""
        inconsistent_data = DynamicTimeSeriesData(
            root={
                **base_dynamic_data.root,
                "temperature": {  # One value missing
                    "0": 20.5,
                    "1": 21.2,
                },
            }
        )

        with pytest.raises(ValueError):
            convert_dynamic_time_series_data_to_synthefy_request(
                inconsistent_data, **base_config
            )

    @pytest.mark.parametrize(
        "invalid_value,column_type",
        [
            ("not_a_number", "timeseries"),
            ("invalid", "continuous"),
        ],
    )
    def test_invalid_data_types(
        self, base_dynamic_data, base_config, invalid_value, column_type
    ):
        """Test handling of invalid data types."""
        modified_data = copy.deepcopy(base_dynamic_data.root)

        if column_type == "timeseries":
            modified_data["temperature"] = {
                str(i): invalid_value for i in range(base_config["window_size"])
            }
        else:  # continuous
            modified_data["quality_score"] = {
                str(i): invalid_value for i in range(base_config["window_size"])
            }

        invalid_data = DynamicTimeSeriesData(root=modified_data)

        with pytest.raises(ValueError):
            convert_dynamic_time_series_data_to_synthefy_request(
                invalid_data, **base_config
            )

    def test_valid_special_float_values(self, base_dynamic_data, base_config):
        """Test that inf values are accepted."""
        special_values = [float("inf"), float("-inf")]

        for value in special_values:
            modified_data = copy.deepcopy(base_dynamic_data.root)
            # Test in both timeseries and continuous columns
            modified_data["temperature"] = {
                str(i): value for i in range(base_config["window_size"])
            }
            modified_data["quality_score"] = {
                str(i): value for i in range(base_config["window_size"])
            }

            valid_data = DynamicTimeSeriesData(root=modified_data)

            # Should not raise ValueError
            result = convert_dynamic_time_series_data_to_synthefy_request(
                valid_data, **base_config
            )
            assert isinstance(result, SynthefyRequest)

    def test_timestamp_format_handling(self, base_config):
        """Test handling of different timestamp formats."""
        timestamp_formats = [
            {"0": "2023-01-01", "1": "2023-01-02", "2": "2023-01-03"},
            {
                "0": "2023-01-01T00:00:00",
                "1": "2023-01-01T01:00:00",
                "2": "2023-01-01T02:00:00",
            },
            {
                "0": 1672531200,
                "1": 1672534800,
                "2": 1672538400,
            },  # Unix timestamps
            {"0": "01/01/2023", "1": "01/02/2023", "2": "01/03/2023"},
        ]

        for timestamps in timestamp_formats:
            data = DynamicTimeSeriesData(
                root={
                    "timestamp": timestamps,
                    "temperature": {"0": 20.5, "1": 21.2, "2": 22.1},
                    "humidity": {"0": 45.0, "1": 46.5, "2": 47.2},
                    "pressure": {"0": 1013.2, "1": 1014.1, "2": 1013.8},
                    "sensor_id": {"0": "A123", "1": "A123", "2": "A123"},
                    "location": {"0": "north", "1": "north", "2": "north"},
                    "status": {"0": "active", "1": "active", "2": "active"},
                    "quality_score": {"0": 0.95, "1": 0.97, "2": 0.94},
                    "error_rate": {"0": 0.02, "1": 0.01, "2": 0.02},
                }
            )

            result = convert_dynamic_time_series_data_to_synthefy_request(
                data, **base_config
            )

            assert len(result.windows) == 1
            assert result.windows[0].timestamps is not None
            assert result.windows[0].timestamps.values is not None
            assert (
                len(result.windows[0].timestamps.values)
                == base_config["window_size"]
            )
            assert result.windows[0].timestamps.values == list(
                timestamps.values()
            )

    def test_minimal_and_missing_optional_data(
        self, base_dynamic_data, base_config
    ):
        """Test conversion with minimal required data (only timeseries) and various combinations of missing optional data."""
        test_cases = [
            # Case 1: Only required data (timestamps and timeseries)
            {
                "data": {
                    "timestamp": base_dynamic_data.root["timestamp"],
                    "temperature": base_dynamic_data.root["temperature"],
                    "humidity": base_dynamic_data.root["humidity"],
                },
                "config": {
                    "timeseries_colnames": ["temperature", "humidity"],
                    "timestamps_colname": ["timestamp"],
                    "window_size": 3,
                    "selected_action": SelectedAction.ANOMALY_DETECTION,
                    "continuous_colnames": [],
                    "discrete_colnames": [],
                    "group_label_cols": [],
                },
                "expected": {
                    "n_timeseries": 2,
                    "n_continuous": 0,
                    "n_discrete": 0,
                },
            },
            # Case 2: Timeseries + continuous, no discrete
            {
                "data": {
                    "timestamp": base_dynamic_data.root["timestamp"],
                    "temperature": base_dynamic_data.root["temperature"],
                    "quality_score": base_dynamic_data.root["quality_score"],
                },
                "config": {
                    "timeseries_colnames": ["temperature"],
                    "timestamps_colname": ["timestamp"],
                    "window_size": 3,
                    "selected_action": SelectedAction.FORECAST,
                    "continuous_colnames": ["quality_score"],
                    "discrete_colnames": [],
                    "group_label_cols": [],
                },
                "expected": {
                    "n_timeseries": 1,
                    "n_continuous": 1,
                    "n_discrete": 0,
                },
            },
            # Case 3: Timeseries + discrete, no continuous
            {
                "data": {
                    "timestamp": base_dynamic_data.root["timestamp"],
                    "temperature": base_dynamic_data.root["temperature"],
                    "status": base_dynamic_data.root["status"],
                },
                "config": {
                    "timeseries_colnames": ["temperature"],
                    "timestamps_colname": ["timestamp"],
                    "window_size": 3,
                    "selected_action": SelectedAction.FORECAST,
                    "continuous_colnames": [],
                    "discrete_colnames": ["status"],
                    "group_label_cols": [],
                },
                "expected": {
                    "n_timeseries": 1,
                    "n_continuous": 0,
                    "n_discrete": 1,
                },
            },
            # Case 4: Timeseries + group labels only - also test that synthesis does not require timeseries columns data.
            {
                "data": {
                    "timestamp": base_dynamic_data.root["timestamp"],
                    "temperature": base_dynamic_data.root["temperature"],
                    "sensor_id": base_dynamic_data.root["sensor_id"],
                },
                "config": {
                    "timeseries_colnames": ["temperature"],
                    "timestamps_colname": ["timestamp"],
                    "window_size": 3,
                    "selected_action": SelectedAction.SYNTHESIS,
                    "continuous_colnames": [],
                    "discrete_colnames": [],
                    "group_label_cols": ["sensor_id"],
                },
                "expected": {
                    "n_timeseries": 1,
                    "n_continuous": 0,
                    "n_discrete": 1,  # group label becomes discrete
                },
            },
        ]

        for test_case in test_cases:
            data = DynamicTimeSeriesData(root=test_case["data"])
            if (
                test_case["config"]["selected_action"]
                == SelectedAction.SYNTHESIS
            ):
                data.root.pop("temperature")
            result = convert_dynamic_time_series_data_to_synthefy_request(
                data, **test_case["config"]
            )

            # Verify basic structure
            assert isinstance(result, SynthefyRequest)
            assert len(result.windows) == 1
            window = result.windows[0]

            # Verify expected counts of different data types
            assert (
                len(window.timeseries_data)
                == test_case["expected"]["n_timeseries"]
            )
            assert (
                len(window.metadata.continuous_conditions)
                == test_case["expected"]["n_continuous"]
            )
            assert (
                len(window.metadata.discrete_conditions)
                == test_case["expected"]["n_discrete"]
            )

            # Verify timestamps
            assert window.timestamps is not None
            assert window.timestamps.values is not None
            assert (
                len(window.timestamps.values)
                == test_case["config"]["window_size"]
            )
            assert (
                window.timestamps.name
                == test_case["config"]["timestamps_colname"][0]
            )

            # Verify timeseries names match config
            timeseries_names = {ts.name for ts in window.timeseries_data}
            assert timeseries_names == set(
                test_case["config"]["timeseries_colnames"]
            )

            # Verify continuous names match config if present
            if test_case["config"]["continuous_colnames"]:
                continuous_names = {
                    cc.name for cc in window.metadata.continuous_conditions
                }
                assert continuous_names == set(
                    test_case["config"]["continuous_colnames"]
                )

            # Verify discrete names match config if present
            if (
                test_case["config"]["discrete_colnames"]
                or test_case["config"]["group_label_cols"]
            ):
                discrete_names = {
                    dc.name for dc in window.metadata.discrete_conditions
                }
                expected_discrete = set(
                    test_case["config"]["discrete_colnames"]
                    + test_case["config"]["group_label_cols"]
                )
                assert discrete_names == expected_discrete

            if (
                test_case["config"]["selected_action"]
                == SelectedAction.SYNTHESIS
            ):
                # check for -1s in timeseries data
                for ts in window.timeseries_data:
                    assert all(val == -1 for val in ts.values)

    def test_null_values(self, base_dynamic_data, base_config):
        """Test handling of null values in different column types."""
        test_cases = [
            # Test null in timeseries
            {
                "column": "temperature",
                "col_type": "timeseries",
                "null_value": None,
            },
            # Test null in continuous metadata
            {
                "column": "quality_score",
                "col_type": "continuous metadata",
                "null_value": np.nan,
            },
            # Test null in discrete metadata
            {
                "column": "status",
                "col_type": "discrete metadata",
                "null_value": None,
            },
            # Test null in group label
            {
                "column": "sensor_id",
                "col_type": "group label",
                "null_value": None,
            },
            # Test null in timestamp
            {
                "column": "timestamp",
                "col_type": "timestamp",
                "null_value": pd.NA,
            },
        ]

        for test_case in test_cases:
            # Create a copy of the base data
            modified_data = base_dynamic_data.model_copy(deep=True)

            # Insert a null value
            modified_data.root[test_case["column"]]["1"] = test_case[
                "null_value"
            ]

            # Verify that the conversion raises a ValueError with appropriate message
            with pytest.raises(
                ValueError,
                match="contains null values",
            ):
                convert_dynamic_time_series_data_to_synthefy_request(
                    modified_data, **base_config
                )

    def test_mixed_null_values(self, base_dynamic_data, base_config):
        """Test handling of different types of null values (None, np.nan, pd.NA)."""
        null_values = [None, np.nan, pd.NA]

        for null_value in null_values:
            modified_data = base_dynamic_data.model_copy(deep=True)
            modified_data.root["temperature"]["1"] = null_value

            with pytest.raises(
                ValueError,
                match="contains null values",
            ):
                convert_dynamic_time_series_data_to_synthefy_request(
                    modified_data, **base_config
                )

    def test_with_synthesis_constraints(self, base_dynamic_data, base_config):
        """Test conversion with synthesis constraints in the request."""
        # Create a copy of the base data and add constraints as a special key
        modified_data = copy.deepcopy(base_dynamic_data.root)
        # Add constraints directly under _constraints_
        modified_data["_constraints_"] = json.dumps(
            {
                "temperature": {"min": 18.5},
                "humidity": {"max": 60.0},
                "pressure": {"max": 500},
                "_projection_type_": "clipping",
            }
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        # Use synthesis as the selected action
        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        # Convert to SynthefyRequest
        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # Verify constraints were processed correctly
        assert result.synthesis_constraints is not None
        assert len(result.synthesis_constraints.constraints) == 3
        assert (
            result.synthesis_constraints.constraints[0].channel_name
            == "temperature"
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_name
            == ConstraintType.MIN
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_value == 18.5
        )
        assert (
            result.synthesis_constraints.constraints[1].channel_name
            == "humidity"
        )
        assert (
            result.synthesis_constraints.constraints[1].constraint_name
            == ConstraintType.MAX
        )
        assert (
            result.synthesis_constraints.constraints[1].constraint_value == 60.0
        )
        assert (
            result.synthesis_constraints.constraints[2].channel_name
            == "pressure"
        )
        assert (
            result.synthesis_constraints.constraints[2].constraint_name
            == ConstraintType.MAX
        )
        assert (
            result.synthesis_constraints.constraints[2].constraint_value == 500
        )
        assert (
            result.synthesis_constraints.projection_during_synthesis
            == "clipping"
        )

    def test_with_strict_projection_type(self, base_dynamic_data, base_config):
        """Test conversion with 'strict' projection type."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"min": 18.5}, "_projection_type_": "strict"}
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # Verify constraints were properly extracted
        assert result.synthesis_constraints is not None
        assert len(result.synthesis_constraints.constraints) == 1
        assert (
            result.synthesis_constraints.constraints[0].channel_name
            == "temperature"
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_name
            == ConstraintType.MIN
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_value == 18.5
        )

        # Verify projection type
        assert (
            result.synthesis_constraints.projection_during_synthesis == "strict"
        )

    def test_with_invalid_constraint_type(self, base_dynamic_data, base_config):
        """Test handling of invalid constraint types."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"invalid_type": 18.5}}
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # No constraints should be added since the type was invalid
        if result.synthesis_constraints is not None:
            assert len(result.synthesis_constraints.constraints) == 0

    def test_with_invalid_constraint_value(
        self, base_dynamic_data, base_config
    ):
        """Test handling of invalid constraint values."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"min": "not_a_number"}}
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value for temperature:min: not_a_number",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )

    def test_with_empty_constraints(self, base_dynamic_data, base_config):
        """Test handling of empty constraints dictionary."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps({})

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # No constraints should be added
        assert result.synthesis_constraints is None

    def test_constraints_with_non_synthesis_action(
        self, base_dynamic_data, base_config
    ):
        """Test that constraints are still processed even with non-synthesis actions."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"min": 18.5}}
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        # Use forecast as the selected action
        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.FORECAST

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # Constraints should still be processed
        assert result.synthesis_constraints is not None
        assert len(result.synthesis_constraints.constraints) == 1
        assert (
            result.synthesis_constraints.constraints[0].channel_name
            == "temperature"
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_name
            == ConstraintType.MIN
        )
        assert (
            result.synthesis_constraints.constraints[0].constraint_value == 18.5
        )

    def test_constraints_parsing(self):
        # Create a simple dynamic time series data with constraints
        data = {
            "timestamp": {0: "2023-01-01", 1: "2023-01-02", 2: "2023-01-03"},
            "temperature": {0: 20.0, 1: 22.0, 2: 24.0},
            "humidity": {0: 50.0, 1: 55.0, 2: 60.0},
            "status": {0: "normal", 1: "normal", 2: "normal"},
            "_constraints_": json.dumps(
                {
                    "temperature": {"min": 18.5, "max": 30.0},
                    "humidity": {"max": 60.0},
                    "pressure": {"max": 500},
                    "_projection_type_": "clipping",
                }
            ),
        }

        dynamic_data = DynamicTimeSeriesData(root=data)

        # Convert to SynthefyRequest
        request = convert_dynamic_time_series_data_to_synthefy_request(
            dynamic_data,
            group_label_cols=[],
            timeseries_colnames=["temperature", "humidity"],
            continuous_colnames=[],
            discrete_colnames=["status"],
            timestamps_colname=["timestamp"],
            window_size=3,
            selected_action=SelectedAction.SYNTHESIS,
        )

        # Verify constraints were parsed correctly
        assert request.synthesis_constraints is not None

        # Check projection type
        assert (
            request.synthesis_constraints.projection_during_synthesis
            == "clipping"
        )

        # Check constraints list
        constraints = request.synthesis_constraints.constraints
        assert len(constraints) == 4  # Should have 4 constraints

        # Check each constraint
        constraint_dict = {
            (c.channel_name, c.constraint_name.value): c.constraint_value
            for c in constraints
        }

        assert constraint_dict[("temperature", "min")] == 18.5
        assert constraint_dict[("temperature", "max")] == 30.0
        assert constraint_dict[("humidity", "max")] == 60.0
        assert constraint_dict[("pressure", "max")] == 500.0

    def test_with_strict_projection_type_all_constraints(
        self, base_dynamic_data, base_config
    ):
        """Test conversion with 'strict' projection type and all constraint types."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {
                "temperature": {
                    "min": 18.5,
                    "max": 30.0,
                    "argmax": 50,
                    "argmin": 10,
                },
                "_projection_type_": "strict",
            }
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # Verify constraints were properly extracted
        assert result.synthesis_constraints is not None
        assert len(result.synthesis_constraints.constraints) == 4

        # Create a dictionary of constraints for easier verification
        constraints_dict = {
            c.constraint_name: (c.channel_name, c.constraint_value)
            for c in result.synthesis_constraints.constraints
        }

        # Verify each constraint type
        assert constraints_dict[ConstraintType.MIN] == ("temperature", 18.5)
        assert constraints_dict[ConstraintType.MAX] == ("temperature", 30.0)
        assert constraints_dict[ConstraintType.ARGMAX] == ("temperature", 50)
        assert constraints_dict[ConstraintType.ARGMIN] == ("temperature", 10)

        # Verify projection type
        assert (
            result.synthesis_constraints.projection_during_synthesis == "strict"
        )

    def test_with_strict_projection_type_multiple_channels(
        self, base_dynamic_data, base_config
    ):
        """Test conversion with 'strict' projection type and multiple channels."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {
                "temperature": {"min": 18.5, "max": 30.0},
                "humidity": {"min": 40.0, "max": 80.0},
                "_projection_type_": "strict",
            }
        )

        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        result = convert_dynamic_time_series_data_to_synthefy_request(
            data_with_constraints, **modified_config
        )

        # Verify constraints were properly extracted
        assert result.synthesis_constraints is not None
        assert len(result.synthesis_constraints.constraints) == 4

        # Group constraints by channel
        channel_constraints = {}
        for c in result.synthesis_constraints.constraints:
            if c.channel_name not in channel_constraints:
                channel_constraints[c.channel_name] = {}
            channel_constraints[c.channel_name][c.constraint_name] = (
                c.constraint_value
            )

        # Verify temperature constraints
        assert channel_constraints["temperature"][ConstraintType.MIN] == 18.5
        assert channel_constraints["temperature"][ConstraintType.MAX] == 30.0

        # Verify humidity constraints
        assert channel_constraints["humidity"][ConstraintType.MIN] == 40.0
        assert channel_constraints["humidity"][ConstraintType.MAX] == 80.0

    def test_with_invalid_constraint_combinations(
        self, base_dynamic_data, base_config
    ):
        """Test handling of invalid constraint combinations with projection types."""
        test_cases = [
            {
                "constraints": {
                    "temperature": {"argmax": 50},
                    "_projection_type_": "clipping",
                },
                "expected_error": "When projection_during_synthesis='clipping', only 'min' and 'max' constraints are supported",
            },
            {
                "constraints": {
                    "temperature": {"argmin": 10},
                    "_projection_type_": "clipping",
                },
                "expected_error": "When projection_during_synthesis='clipping', only 'min' and 'max' constraints are supported",
            },
        ]

        for test_case in test_cases:
            modified_data = copy.deepcopy(base_dynamic_data.root)
            modified_data["_constraints_"] = json.dumps(
                test_case["constraints"]
            )

            data_with_constraints = DynamicTimeSeriesData(root=modified_data)

            modified_config = base_config.copy()
            modified_config["selected_action"] = SelectedAction.SYNTHESIS

            with pytest.raises(ValueError, match=test_case["expected_error"]):
                _ = convert_dynamic_time_series_data_to_synthefy_request(
                    data_with_constraints, **modified_config
                )

    def test_with_invalid_constraint_values(
        self, base_dynamic_data, base_config
    ):
        """Test that invalid constraint values cause validation failure."""
        # Test with string that's not a number
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"min": "invalid"}, "_projection_type_": "strict"}
        )
        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value for temperature:min: invalid",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )

        # Test with None value
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"max": None}}
        )
        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value type for temperature:max: NoneType",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )

        # Test with dictionary object
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"argmax": {}}}
        )
        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value type for temperature:argmax: dict",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )

        # Test with list object
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"argmin": []}}
        )
        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value type for temperature:argmin: list",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )

    def test_constraints_parsing_invalid2(self, base_dynamic_data, base_config):
        """Test parsing invalid constraints."""
        modified_data = copy.deepcopy(base_dynamic_data.root)
        modified_data["_constraints_"] = json.dumps(
            {"temperature": {"max": "not_a_number"}}
        )
        data_with_constraints = DynamicTimeSeriesData(root=modified_data)

        modified_config = base_config.copy()
        modified_config["selected_action"] = SelectedAction.SYNTHESIS

        with pytest.raises(
            ValueError,
            match="Failed to create window: Invalid constraint value for temperature:max: not_a_number",
        ):
            convert_dynamic_time_series_data_to_synthefy_request(
                data_with_constraints, **modified_config
            )


class TestConvertSynthefyResponseToDynamicTimeSeriesData:
    """Test suite for convert_synthefy_response_to_dynamic_time_series_data function."""

    @pytest.fixture
    def basic_window(self):
        """Create a basic window with timeseries and metadata."""
        return SynthefyTimeSeriesWindow(
            id=0,
            name="test_window",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[20.0, 21.0, 22.0]),
                OneTimeSeries(name="humidity", values=[50.0, 51.0, 52.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[
                    OneDiscreteMetaData(
                        name="status", values=["good", "good", "bad"]
                    ),
                ],
                continuous_conditions=[
                    OneContinuousMetaData(
                        name="quality", values=[0.9, 0.8, 0.7]
                    ),
                ],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-01", "2023-01-02", "2023-01-03"],
            ),
        )

    @pytest.fixture
    def basic_response(self, basic_window):
        """Create a basic response with one window."""
        return SynthefyResponse(
            windows=[basic_window],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[],
        )

    @pytest.fixture
    def basic_response_with_anomaly(self, basic_window):
        """Create a basic response with one window and anomaly timestamps."""
        return SynthefyResponse(
            windows=[basic_window],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[
                TimeStamps(name="timestamp", values=["2023-01-02"])
            ],
        )

    def test_basic_conversion(self, basic_response):
        """Test basic conversion from SynthefyResponse to DynamicTimeSeriesData."""
        result = convert_synthefy_response_to_dynamic_time_series_data(
            basic_response
        )
        assert isinstance(result, DynamicTimeSeriesData)
        assert set(result.root.keys()) == {
            "temperature",
            "humidity",
            "status",
            "quality",
            "timestamp",  # window_idx not included for single window
        }
        assert list(result.root["temperature"].values()) == [20.0, 21.0, 22.0]  # pyright: ignore
        assert list(result.root["humidity"].values()) == [50.0, 51.0, 52.0]  # pyright: ignore
        assert list(result.root["status"].values()) == ["good", "good", "bad"]  # pyright: ignore
        assert list(result.root["quality"].values()) == [0.9, 0.8, 0.7]  # pyright: ignore
        assert list(result.root["timestamp"].values()) == [  # pyright: ignore
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ]
        assert (
            "window_idx" not in result.root
        )  # Verify window_idx is not present

    def test_multiple_windows(self):
        """Test conversion with multiple windows."""
        window1 = SynthefyTimeSeriesWindow(
            id=0,
            name="window1",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[20.0, 21.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[],
                continuous_conditions=[],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-01", "2023-01-02"],
            ),
        )

        window2 = SynthefyTimeSeriesWindow(
            id=1,
            name="window2",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[22.0, 23.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[],
                continuous_conditions=[],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-03", "2023-01-04"],
            ),
        )

        response = SynthefyResponse(
            windows=[window1, window2],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[],
        )

        result = convert_synthefy_response_to_dynamic_time_series_data(response)

        assert isinstance(result, DynamicTimeSeriesData)
        assert (
            "window_idx" in result.root
        )  # Verify window_idx is present for multiple windows
        assert list(result.root["temperature"].values()) == [  # pyright: ignore
            20.0,
            21.0,
            22.0,
            23.0,
        ]
        assert list(result.root["timestamp"].values()) == [  # pyright: ignore
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
        ]
        assert list(result.root["window_idx"].values()) == [0, 0, 1, 1]  # pyright: ignore

    def test_multiple_windows_with_metadata(self):
        """Test conversion with multiple windows including metadata."""
        window1 = SynthefyTimeSeriesWindow(
            id=0,
            name="window1",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[20.0, 21.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[
                    OneDiscreteMetaData(name="status", values=["good", "good"]),
                ],
                continuous_conditions=[
                    OneContinuousMetaData(name="quality", values=[0.9, 0.8]),
                ],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-01", "2023-01-02"],
            ),
        )

        window2 = SynthefyTimeSeriesWindow(
            id=1,
            name="window2",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[22.0, 23.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[
                    OneDiscreteMetaData(name="status", values=["bad", "bad"]),
                ],
                continuous_conditions=[
                    OneContinuousMetaData(name="quality", values=[0.7, 0.6]),
                ],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-03", "2023-01-04"],
            ),
        )

        response = SynthefyResponse(
            windows=[window1, window2],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[],
        )

        result = convert_synthefy_response_to_dynamic_time_series_data(response)

        assert isinstance(result, DynamicTimeSeriesData)
        assert list(result.root["temperature"].values()) == [  # pyright: ignore
            20.0,
            21.0,
            22.0,
            23.0,
        ]
        assert list(result.root["timestamp"].values()) == [  # pyright: ignore
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
        ]
        assert list(result.root["status"].values()) == [  # pyright: ignore
            "good",
            "good",
            "bad",
            "bad",
        ]
        assert list(result.root["quality"].values()) == [0.9, 0.8, 0.7, 0.6]  # pyright: ignore
        assert list(result.root["window_idx"].values()) == [0, 0, 1, 1]  # pyright: ignore

    def test_empty_windows(self):
        """Test conversion with empty windows list."""
        response = SynthefyResponse(
            windows=[],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[],
        )

        with pytest.raises(ValueError, match="Response contains no windows"):
            convert_synthefy_response_to_dynamic_time_series_data(response)

    def test_inconsistent_windows(self):
        """Test conversion with inconsistent window structures."""
        window1 = SynthefyTimeSeriesWindow(
            id=0,
            name="window1",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[20.0, 21.0]),
            ],
            metadata=MetaData(
                discrete_conditions=[],
                continuous_conditions=[],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-01", "2023-01-02"],
            ),
        )

        window2 = SynthefyTimeSeriesWindow(
            id=1,
            name="window2",
            timeseries_data=[
                OneTimeSeries(name="temperature", values=[22.0, 23.0]),
                OneTimeSeries(
                    name="humidity", values=[50.0, 51.0]
                ),  # Extra column
            ],
            metadata=MetaData(
                discrete_conditions=[],
                continuous_conditions=[],
            ),
            timestamps=TimeStamps(
                name="timestamp",
                values=["2023-01-03", "2023-01-04"],
            ),
        )

        response = SynthefyResponse(
            windows=[window1, window2],
            combined_text="test",
            forecast_timestamps=[],
            anomaly_timestamps=[],
        )

        with pytest.raises(
            ValueError, match="Inconsistent data structure across windows"
        ):
            convert_synthefy_response_to_dynamic_time_series_data(response)

    def test_anomaly_timestamps(self, basic_response_with_anomaly):
        result = convert_synthefy_response_to_dynamic_time_series_data(
            basic_response_with_anomaly
        )
        assert isinstance(result, DynamicTimeSeriesData)
        assert list(result.root["is_anomaly"].values()) == [0, 1, 0]  # pyright: ignore


class TestTrimResponseToForecastWindow:
    """Test suite for trim_response_to_forecast_window function."""

    @pytest.fixture
    def sample_response(self):
        """Create a sample response with multiple time series."""
        return DynamicTimeSeriesData(
            root={
                "temperature": {
                    0: 20.0,
                    1: 21.0,
                    2: 22.0,
                    3: 23.0,
                    4: 24.0,
                },
                "humidity": {
                    0: 45.0,
                    1: 46.0,
                    2: 47.0,
                    3: 48.0,
                    4: 49.0,
                },
            }
        )

    def test_basic_trimming(self, sample_response):
        """Test basic trimming functionality with standard inputs."""
        window_size = 5
        forecast_length = 2

        result = trim_response_to_forecast_window(
            sample_response, window_size, forecast_length
        )
        # Check correct length
        assert len(result.root["temperature"]) == forecast_length
        assert len(result.root["humidity"]) == forecast_length

        # Check correct values (should only contain last 2 values)
        assert result.root["temperature"] == {0: 23.0, 1: 24.0}
        assert result.root["humidity"] == {0: 48.0, 1: 49.0}

    def test_zero_forecast_length(self, sample_response):
        """Test behavior when forecast_length is 0."""
        window_size = 5
        forecast_length = 0
        result = trim_response_to_forecast_window(
            sample_response, window_size, forecast_length
        )

        # Should return a deep copy of original response without modifications
        assert result.model_dump() == sample_response.model_dump()

    def test_forecast_length_equals_window_size(self, sample_response):
        """Test when forecast_length equals window_size."""
        window_size = 5
        forecast_length = 5

        result = trim_response_to_forecast_window(
            sample_response, window_size, forecast_length
        )

        # Should return all values with reindexed keys
        assert len(result.root["temperature"]) == window_size
        assert result.root["temperature"] == {
            i: sample_response.root["temperature"][i]
            for i in range(window_size)
        }

    def test_empty_response(self):
        """Test behavior with empty response."""
        empty_response = DynamicTimeSeriesData(root={})
        window_size = 5
        forecast_length = 2

        result = trim_response_to_forecast_window(
            empty_response, window_size, forecast_length
        )

        assert result.root == {}

    def test_multiple_series_consistency(self):
        """Test that all time series are trimmed consistently."""
        response = DynamicTimeSeriesData(
            root={
                "series1": {i: float(i) for i in range(10)},
                "series2": {i: float(i * 2) for i in range(10)},
                "series3": {i: float(i * 3) for i in range(10)},
            }
        )

        window_size = 10
        forecast_length = 3

        result = trim_response_to_forecast_window(
            response, window_size, forecast_length
        )

        # Check all series have same length
        lengths = [len(v) for v in result.root.values()]
        assert all(length == forecast_length for length in lengths)

        # Check correct values and sequential keys
        for i in range(forecast_length):
            assert result.root["series1"][i] == float(i + 7)
            assert result.root["series2"][i] == float((i + 7) * 2)
            assert result.root["series3"][i] == float((i + 7) * 3)

    def test_string_keys(self):
        """Test behavior with string keys instead of integer keys."""
        response = DynamicTimeSeriesData(
            root={
                "temperature": {
                    "t0": 20.0,
                    "t1": 21.0,
                    "t2": 22.0,
                    "t3": 23.0,
                    "t4": 24.0,
                },
                "humidity": {
                    "t0": 45.0,
                    "t1": 46.0,
                    "t2": 47.0,
                    "t3": 48.0,
                    "t4": 49.0,
                },
            }
        )

        window_size = 5
        forecast_length = 2

        result = trim_response_to_forecast_window(
            response, window_size, forecast_length
        )

        # Check correct length
        assert len(result.root["temperature"]) == forecast_length
        assert len(result.root["humidity"]) == forecast_length

        # Check that keys are converted to integers
        assert result.root["temperature"] == {0: 23.0, 1: 24.0}
        assert result.root["humidity"] == {0: 48.0, 1: 49.0}

    def test_mixed_key_types(self):
        """Test behavior with mixed key types (strings and integers)."""
        response = DynamicTimeSeriesData(
            root={
                "series1": {
                    0: 1.0,
                    "1": 2.0,
                    2: 3.0,
                    "t3": 4.0,
                    4: 5.0,
                },
                "series2": {
                    "0": 10.0,
                    1: 20.0,
                    "2": 30.0,
                    3: 40.0,
                    "4": 50.0,
                },
            }
        )

        window_size = 5
        forecast_length = 3

        result = trim_response_to_forecast_window(
            response, window_size, forecast_length
        )

        # Check correct length
        assert len(result.root["series1"]) == forecast_length
        assert len(result.root["series2"]) == forecast_length

        # Check that all keys are converted to sequential integers
        assert result.root["series1"] == {0: 3.0, 1: 4.0, 2: 5.0}
        assert result.root["series2"] == {0: 30.0, 1: 40.0, 2: 50.0}

    def test_timestamp_string_keys(self):
        """Test behavior with timestamp-like string keys."""
        response = DynamicTimeSeriesData(
            root={
                "temperature": {
                    "2023-01-01": 20.0,
                    "2023-01-02": 21.0,
                    "2023-01-03": 22.0,
                    "2023-01-04": 23.0,
                    "2023-01-05": 24.0,
                },
                "humidity": {
                    "2023-01-01": 45.0,
                    "2023-01-02": 46.0,
                    "2023-01-03": 47.0,
                    "2023-01-04": 48.0,
                    "2023-01-05": 49.0,
                },
            }
        )

        window_size = 5
        forecast_length = 2

        result = trim_response_to_forecast_window(
            response, window_size, forecast_length
        )

        # Check correct length
        assert len(result.root["temperature"]) == forecast_length
        assert len(result.root["humidity"]) == forecast_length

        # Check that keys are converted to sequential integers
        assert result.root["temperature"] == {0: 23.0, 1: 24.0}
        assert result.root["humidity"] == {0: 48.0, 1: 49.0}

    def test_non_sequential_string_keys(self):
        """Test behavior with non-sequential string keys."""
        response = DynamicTimeSeriesData(
            root={
                "series1": {
                    "a": 1.0,
                    "c": 2.0,
                    "b": 3.0,
                    "e": 4.0,
                    "d": 5.0,
                },
                "series2": {
                    "v1": 10.0,
                    "v3": 20.0,
                    "v2": 30.0,
                    "v5": 40.0,
                    "v4": 50.0,
                },
            }
        )

        window_size = 5
        forecast_length = 2

        result = trim_response_to_forecast_window(
            response, window_size, forecast_length
        )

        # Check correct length
        assert len(result.root["series1"]) == forecast_length
        assert len(result.root["series2"]) == forecast_length

        # Check that keys are converted to sequential integers
        # Values should be taken from the last 2 entries in original order
        assert result.root["series1"] == {0: 4.0, 1: 5.0}
        assert result.root["series2"] == {0: 40.0, 1: 50.0}


def test_s3_prefix_exists_true():
    mock_s3_client = MagicMock()
    mock_s3_client.list_objects_v2.return_value = {
        "Contents": [{"Key": "test/key"}]
    }

    result = s3_prefix_exists(mock_s3_client, "test-bucket", "test/")

    assert result is True
    mock_s3_client.list_objects_v2.assert_called_once_with(
        Bucket="test-bucket", Prefix="test/", MaxKeys=1
    )


def test_s3_prefix_exists_false():
    mock_s3_client = MagicMock()
    mock_s3_client.list_objects_v2.return_value = {}

    result = s3_prefix_exists(mock_s3_client, "test-bucket", "test/")

    assert result is False


def test_delete_s3_objects_success():
    mock_s3_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "test/key1"}, {"Key": "test/key2"}]}
    ]
    mock_s3_client.get_paginator.return_value = mock_paginator

    delete_s3_objects(mock_s3_client, "test-bucket", "test/")

    mock_s3_client.delete_objects.assert_called_once_with(
        Bucket="test-bucket",
        Delete={"Objects": [{"Key": "test/key1"}, {"Key": "test/key2"}]},
    )


def test_delete_s3_objects_batch_deletion():
    mock_s3_client = MagicMock()
    mock_paginator = MagicMock()

    # Create more than 1000 objects
    objects = [{"Key": f"test/key{i}"} for i in range(1500)]
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": obj["Key"]} for obj in objects]}
    ]
    mock_s3_client.get_paginator.return_value = mock_paginator

    delete_s3_objects(mock_s3_client, "test-bucket", "test/")

    # Should have been called twice - once for first 1000, once for remaining 500
    assert mock_s3_client.delete_objects.call_count == 2

    # Verify first batch of 1000
    first_call_args = mock_s3_client.delete_objects.call_args_list[0][1]
    assert len(first_call_args["Delete"]["Objects"]) == 1000

    # Verify second batch of 500
    second_call_args = mock_s3_client.delete_objects.call_args_list[1][1]
    assert len(second_call_args["Delete"]["Objects"]) == 500


def test_delete_s3_objects_empty():
    mock_s3_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"Contents": []}]
    mock_s3_client.get_paginator.return_value = mock_paginator

    delete_s3_objects(mock_s3_client, "test-bucket", "test/")

    mock_s3_client.delete_objects.assert_not_called()


class TestGetSettings:
    """Test suite for get_settings function."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables."""
        monkeypatch.setenv("SYNTHEFY_CONFIG_PATH", "/mock/config.yaml")
        monkeypatch.setenv("SYNTHEFY_DATASETS_BASE", "/data/datasets")
        monkeypatch.setenv("SYNTHEFY_PACKAGE_BASE", "/app/package")
        monkeypatch.setenv("SYNTHEFY_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("JSON_SAVE_PATH", "/data/json")

    @pytest.fixture
    def mock_yaml_config(self):
        """Sample YAML configuration with environment variables."""
        return {
            "dataset_path": "/data/datasets/${dataset_name}",  # Already expanded
            "bucket_name": "test-bucket",
            "preprocess_config_path": "/app/package/config/preprocess.yaml",  # Already expanded
            "synthesis_config_path": "/app/package/config/synthesis.yaml",  # Already expanded
            "forecast_config_path": "/app/package/config/forecast.yaml",  # Already expanded
            "synthesis_model_path": "/data/datasets/synthesis/best_model.ckpt",  # Already expanded
            "forecast_model_path": "/data/datasets/forecast/best_model.ckpt",  # Already expanded
            "json_save_path": "/data/json",
            "show_gt_synthesis_timeseries": True,
            "show_gt_forecast_timeseries": True,
            "return_only_synthetic_in_streaming_response": True,
            "max_file_size": 1073741824,
        }

    @patch("synthefy_pkg.app.utils.api_utils.load_yaml_config")
    def test_get_settings_with_dataset_name(
        self, mock_load_yaml, mock_env_vars, mock_yaml_config
    ):
        """Test getting settings with dataset name substitution."""
        mock_load_yaml.return_value = mock_yaml_config

        settings = get_settings(SynthesisSettings, dataset_name="test_dataset")

        assert settings.dataset_path == "/data/datasets/test_dataset"
        assert (
            settings.synthesis_config_path
            == "/app/package/config/synthesis.yaml"
        )
        assert (
            settings.synthesis_model_path
            == "/data/datasets/synthesis/best_model.ckpt"
        )
        assert settings.show_gt_synthesis_timeseries is True
        assert settings.return_only_synthetic_in_streaming_response is True

    @patch("synthefy_pkg.app.utils.api_utils.load_yaml_config")
    def test_get_settings_missing_config_path(
        self, mock_load_yaml, monkeypatch
    ):
        """Test error when SYNTHEFY_CONFIG_PATH is not set."""
        monkeypatch.delenv("SYNTHEFY_CONFIG_PATH", raising=False)

        with pytest.raises(
            RuntimeError,
            match="SYNTHEFY_CONFIG_PATH environment variable not set",
        ):
            get_settings(SynthesisSettings)

    @patch("synthefy_pkg.app.utils.api_utils.load_yaml_config")
    def test_get_settings_yaml_load_error(self, mock_load_yaml, mock_env_vars):
        """Test error when YAML config cannot be loaded."""
        mock_load_yaml.side_effect = RuntimeError("Failed to load config")

        with pytest.raises(RuntimeError, match="Failed to load config"):
            get_settings(SynthesisSettings)

    @patch("synthefy_pkg.app.utils.api_utils.load_yaml_config")
    def test_get_settings_different_classes(
        self, mock_load_yaml, mock_env_vars, mock_yaml_config
    ):
        """Test getting settings for different settings classes."""
        mock_yaml_config["dataset_name"] = "mock_dataset_name"
        mock_load_yaml.return_value = mock_yaml_config

        # Test SynthesisSettings
        synthesis_settings = get_settings(SynthesisSettings)
        assert isinstance(synthesis_settings, SynthesisSettings)
        assert (
            synthesis_settings.synthesis_config_path
            == "/app/package/config/synthesis.yaml"
        )

        # Test PreprocessSettings
        preprocess_settings = get_settings(PreprocessSettings)
        assert isinstance(preprocess_settings, PreprocessSettings)
        assert (
            preprocess_settings.preprocess_config_path
            == "/app/package/config/preprocess.yaml"
        )

        # Test ForecastSettings
        forecast_settings = get_settings(ForecastSettings)
        assert isinstance(forecast_settings, ForecastSettings)
        assert (
            forecast_settings.forecast_config_path
            == "/app/package/config/forecast.yaml"
        )


class TestTempDirectoryOperations:
    def test_get_user_tmp_dir_without_dataset(self):
        # Test without dataset name
        expected_path = "/tmp/test_user"
        with patch("os.makedirs") as mock_makedirs:
            result = get_user_tmp_dir("test_user")
            assert result == expected_path
            mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

    def test_get_user_tmp_dir_with_dataset(self):
        # Test with dataset name
        expected_path = "/tmp/test_user/test_dataset"
        with patch("os.makedirs") as _:
            result = get_user_tmp_dir("test_user", "test_dataset")
            assert result == expected_path

    def test_handle_file_upload_success(self):
        # Create mock file
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.csv"
        mock_file.file = MagicMock()

        with (
            patch("builtins.open", create=True) as mock_open,
            patch("shutil.copyfileobj") as mock_copyfileobj,
        ):
            result = handle_file_upload(mock_file, "/tmp/test")

            assert result == "/tmp/test/test.csv"
            mock_open.assert_called_once_with("/tmp/test/test.csv", "wb")
            mock_copyfileobj.assert_called_once()

    def test_cleanup_tmp_dir_success(self):
        with patch("shutil.rmtree") as mock_rmtree:
            cleanup_tmp_dir("/tmp/test")
            mock_rmtree.assert_called_once_with("/tmp/test")

    def test_cleanup_tmp_dir_error(self):
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("loguru.logger.warning") as mock_logger,
        ):
            mock_rmtree.side_effect = Exception("Test error")

            cleanup_tmp_dir("/tmp/test")

            mock_rmtree.assert_called_once_with("/tmp/test")
            mock_logger.assert_called_once_with(
                "Error cleaning up temporary files: Test error"
            )

    def test_cleanup_local_directories(self, tmp_path):
        """Test cleanup_local_directories function"""
        # Create test files and directories
        test_file = tmp_path / "test.txt"
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        test_file.write_text("test")

        # Create nested structure
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "nested.txt").write_text("test")

        paths = [
            str(test_file),
            str(test_dir),
            str(tmp_path / "nonexistent"),
        ]

        # Test cleanup
        cleanup_local_directories(paths)

        # Verify cleanup
        assert not test_file.exists(), "File was not cleaned up"
        assert not test_dir.exists(), "Directory was not cleaned up"

        # Test with empty list
        cleanup_local_directories([])


class TestCreateWindowNameFromGroupLabels:
    def test_basic_functionality(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1"]),
            OneDiscreteMetaData(name="store_category", values=["clothes"]),
        ]
        group_label_cols = ["store_id", "store_category"]

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert result == "store_id=1,store_category=clothes"

    def test_empty_group_labels(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1"]),
            OneDiscreteMetaData(name="store_category", values=["clothes"]),
        ]
        group_label_cols = []

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert result == "Window"  # default name

    def test_no_matching_group_labels(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1"]),
            OneDiscreteMetaData(name="store_category", values=["clothes"]),
        ]
        group_label_cols = [
            "product_id",
            "product_category",
        ]  # non-matching labels

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert result == "Window"  # default name

    def test_multiple_group_labels(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=["1"]),
            OneDiscreteMetaData(name="store_category", values=["clothes"]),
            OneDiscreteMetaData(name="product_id", values=["123"]),
            OneDiscreteMetaData(name="product_category", values=["shirts"]),
        ]
        group_label_cols = ["store_id", "store_category", "product_id"]

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert result == "store_id=1,store_category=clothes,product_id=123"

    def test_empty_discrete_conditions(self):
        discrete_conditions = []
        group_label_cols = ["store_id", "store_category"]

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert result == "Window"  # default name

    def test_custom_default_name(self):
        discrete_conditions = []
        group_label_cols = ["store_id", "store_category"]
        default_name = "Custom Window"

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols, default_name
        )
        assert result == "Custom Window"

    def test_empty_values_in_discrete_conditions(self):
        discrete_conditions = [
            OneDiscreteMetaData(name="store_id", values=[]),
            OneDiscreteMetaData(name="store_category", values=["clothes"]),
        ]
        group_label_cols = ["store_id", "store_category"]

        result = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols
        )
        assert (
            result == "store_category=clothes"
        )  # only includes the non-empty condition


class TestFilterWindowDataframeByWindowFilters:
    @pytest.fixture
    def test_df(self):
        # Create sample data
        dates = pd.date_range(
            start="2024-01-01", end="2024-01-01 23:59:59", freq="H"
        )
        n_samples = len(dates)

        assert n_samples % 2 == 0
        window_size = n_samples // 2
        # First, create window indices
        window_idx = np.repeat([0, 1], window_size)

        # Create group labels that are constant within each window
        group_label_1 = ["EU", "NA"]  # one for each window
        group_label_2 = ["dev1", "dev2"]  # one for each window

        # Repeat the values for each timestamp within the window
        group_label_1 = np.repeat(group_label_1, window_size)
        group_label_2 = np.repeat(group_label_2, window_size)

        data = {
            # timeseries columns
            "timeseries_1": np.random.uniform(0, 10, n_samples),
            "timeseries_2": np.random.uniform(-5, 5, n_samples),
            # continuous columns
            "continuous_1": np.linspace(0.0, 10.0, n_samples),
            # Discrete columns
            "time_invariant_discrete": ["A"] * window_size
            + ["B"] * window_size,
            "time_varying_discrete": ["1", "2"] * (window_size // 2)
            + ["3", "4"] * (window_size // 2),
            # Group label columns (constant within each window)
            "group_label_1": group_label_1,
            "group_label_2": group_label_2,
            # Timestamp column
            "timestamp": dates,
            # Window index
            "window_idx": window_idx,
        }

        df = pd.DataFrame(data)

        # Verify that group labels are constant within each window
        for idx in df["window_idx"].unique():
            window_data = df[df["window_idx"] == idx]
            assert len(window_data["group_label_1"].unique()) == 1
            assert len(window_data["group_label_2"].unique()) == 1

        return df

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_group_labels(self, test_df):
        group_labels = OneDiscreteMetaDataRange(
            name="group_label_1-group_label_2",
            options=["EU-dev1", "NA-dev2"],
        )

        valid_window_indices = filter_window_dataframe_group_labels(
            test_df, group_labels
        )
        assert valid_window_indices == [0, 1]

        group_labels = OneDiscreteMetaDataRange(
            name="group_label_1-group_label_2",
            options=["EU-dev1"],
        )

        valid_window_indices = filter_window_dataframe_group_labels(
            test_df, group_labels
        )
        assert valid_window_indices == [0]

        group_labels = OneDiscreteMetaDataRange(
            name="group_label_1-group_label_2",
            options=["NA-NA"],
        )

        valid_window_indices = filter_window_dataframe_group_labels(
            test_df, group_labels
        )
        assert valid_window_indices == []

    @pytest.mark.asyncio
    async def test_filter_continuous_no_conditions(self, test_df):
        """Test that empty conditions list returns all window indices."""
        no_conditions = []
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, no_conditions
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_continuous_all_windows_satisfy(self, test_df):
        """Test condition that all windows should satisfy."""
        min_val = test_df["continuous_1"].min() - 1
        max_val = test_df["continuous_1"].max() + 1
        all_valid_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1", min_val=min_val, max_val=max_val
            )
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, all_valid_condition
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_continuous_no_windows_satisfy(self, test_df):
        """Test condition that no windows should satisfy."""
        impossible_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1", min_val=100, max_val=200
            )
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, impossible_condition
        )
        assert valid_indices == []

    @pytest.mark.asyncio
    async def test_filter_continuous_nonexistent_column(self, test_df):
        """Test handling of non-existent column."""
        invalid_column_condition = [
            OneContinuousMetaDataRange(
                name="non_existent_column", min_val=0, max_val=10
            )
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, invalid_column_condition
        )
        assert sorted(valid_indices) == []

    @pytest.mark.asyncio
    async def test_filter_continuous_exact_minmax(self, test_df):
        """Test edge case with exact min/max values."""
        exact_min = test_df["continuous_1"].min()
        exact_max = test_df["continuous_1"].max()
        edge_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1", min_val=exact_min, max_val=exact_max
            )
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, edge_condition
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_continuous_partial_window_satisfaction(self, test_df):
        """Test case where only one window satisfies the condition."""
        partial_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1",
                min_val=0,
                max_val=5,  # Only window 0 should fully satisfy this
            )
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, partial_condition
        )
        assert valid_indices == [0]  # Only window 0 should satisfy

    @pytest.mark.asyncio
    async def test_filter_continuous_multiple_conditions_all_satisfy(
        self, test_df
    ):
        """Test multiple conditions where all windows satisfy."""
        multi_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1",
                min_val=0,
                max_val=10,  # Only window 0 satisfies this
            ),
            OneContinuousMetaDataRange(
                name="timeseries_1",
                min_val=0,
                max_val=10,  # all windows should satisfy this
            ),
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, multi_condition
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_continuous_multiple_conditions_partial_satisfy(
        self, test_df
    ):
        """Test multiple conditions where only one window satisfies all conditions."""
        multi_condition = [
            OneContinuousMetaDataRange(
                name="continuous_1",
                min_val=0,
                max_val=5,  # Only window 0 satisfies this
            ),
            OneContinuousMetaDataRange(
                name="timeseries_1",
                min_val=0,
                max_val=10,  # all windows satisfy this
            ),
        ]
        valid_indices = filter_window_dataframe_continuous_conditions(
            test_df, multi_condition
        )
        assert sorted(valid_indices) == [0]

    @pytest.mark.asyncio
    async def test_filter_discrete_no_conditions(self, test_df):
        """Test that empty conditions list returns all window indices."""
        no_conditions = []
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, no_conditions
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_discrete_all_windows_satisfy(self, test_df):
        """Test condition that all windows should satisfy."""
        all_valid_condition = [
            OneDiscreteMetaDataRange(
                name="time_invariant_discrete", options=["A", "B"]
            )
        ]
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, all_valid_condition
        )
        assert sorted(valid_indices) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_discrete_no_windows_satisfy(self, test_df):
        """Test condition that no windows should satisfy."""
        impossible_condition = [
            OneDiscreteMetaDataRange(
                name="time_invariant_discrete",
                options=["C"],  # Neither window has value C
            )
        ]
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, impossible_condition
        )
        assert valid_indices == []

    @pytest.mark.asyncio
    async def test_filter_discrete_nonexistent_column(self, test_df):
        """Test handling of non-existent column."""
        invalid_column_condition = [
            OneDiscreteMetaDataRange(
                name="non_existent_column", options=["A", "B"]
            )
        ]
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, invalid_column_condition
        )
        assert valid_indices == []

    @pytest.mark.asyncio
    async def test_filter_discrete_partial_window_satisfaction(self, test_df):
        """Test case where only one window satisfies the condition."""
        partial_condition = [
            OneDiscreteMetaDataRange(
                name="time_invariant_discrete",
                options=["A"],  # Only window 0 should satisfy this
            )
        ]
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, partial_condition
        )
        assert valid_indices == [0]

    @pytest.mark.asyncio
    async def test_filter_discrete_multiple_conditions(self, test_df):
        """Test multiple conditions where only specific windows satisfy all conditions."""
        multi_condition = [
            OneDiscreteMetaDataRange(
                name="time_invariant_discrete", options=["A", "B"]
            ),
            OneDiscreteMetaDataRange(
                name="time_varying_discrete",
                options=["1", "2"],  # Note: values need to be strings
            ),
        ]
        valid_indices = filter_window_dataframe_discrete_conditions(
            test_df, multi_condition
        )
        assert sorted(valid_indices) == [0]

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_by_window_filters_empty(
        self, test_df
    ):
        """Test that empty filters return the full dataframe."""
        empty_filters = WindowFilters(
            group_label_cols=None,
            metadata_range=MetaDataRange(
                continuous_conditions=[], discrete_conditions=[]
            ),
            timestamps_range=None,
            timeseries_range=[],
        )
        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, empty_filters
        )
        assert len(filtered_df) == len(test_df)
        assert sorted(filtered_df["window_idx"].unique()) == [0, 1]

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_by_window_filters_group_only(
        self, test_df
    ):
        """Test filtering with only group labels."""
        group_only_filters = WindowFilters(
            group_label_cols=OneDiscreteMetaDataRange(
                name="group_label_1-group_label_2", options=["EU-dev1"]
            ),
            metadata_range=MetaDataRange(
                continuous_conditions=[], discrete_conditions=[]
            ),
            timestamps_range=None,
            timeseries_range=[],
        )
        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, group_only_filters
        )
        assert len(filtered_df["window_idx"].unique()) == 1
        assert list(filtered_df["window_idx"].unique()) == [0]
        assert all(filtered_df["group_label_1"] == "EU")
        assert all(filtered_df["group_label_2"] == "dev1")

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_by_window_filters_combined_metadata(
        self, test_df
    ):
        """Test filtering with combined discrete and continuous metadata filters."""
        combined_filters = WindowFilters(
            group_label_cols=None,
            metadata_range=MetaDataRange(
                continuous_conditions=[
                    OneContinuousMetaDataRange(
                        name="continuous_1",
                        min_val=0,
                        max_val=5,  # Only window 0 should satisfy
                    )
                ],
                discrete_conditions=[
                    OneDiscreteMetaDataRange(
                        name="time_invariant_discrete",
                        options=["A"],  # Only window 0 should satisfy
                    )
                ],
            ),
            timestamps_range=None,
            timeseries_range=[],
        )
        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, combined_filters
        )
        assert len(filtered_df["window_idx"].unique()) == 1
        assert list(filtered_df["window_idx"].unique()) == [0]
        assert all(filtered_df["time_invariant_discrete"] == "A")
        assert all(filtered_df["continuous_1"] <= 5)

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_by_window_filters_all_types(
        self, test_df
    ):
        """Test filtering with all filter types combined."""
        all_filters = WindowFilters(
            group_label_cols=OneDiscreteMetaDataRange(
                name="group_label_1-group_label_2",
                options=["EU-dev1", "NA-dev2"],
            ),
            metadata_range=MetaDataRange(
                continuous_conditions=[
                    OneContinuousMetaDataRange(
                        name="continuous_1", min_val=0, max_val=10
                    )
                ],
                discrete_conditions=[
                    OneDiscreteMetaDataRange(
                        name="time_invariant_discrete", options=["A", "B"]
                    )
                ],
            ),
            timestamps_range=None,
            timeseries_range=[
                OneContinuousMetaDataRange(
                    name="timeseries_1", min_val=0, max_val=10
                )
            ],
        )
        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, all_filters
        )
        assert len(filtered_df["window_idx"].unique()) == 2
        assert sorted(filtered_df["window_idx"].unique()) == [0, 1]
        assert set(filtered_df["time_invariant_discrete"].unique()) == {
            "A",
            "B",
        }
        assert all(filtered_df["continuous_1"] <= 10)
        assert all(filtered_df["timeseries_1"] <= 10)

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_by_window_filters_no_matches(
        self, test_df
    ):
        """Test filtering when no windows match the criteria."""
        no_match_filters = WindowFilters(
            group_label_cols=OneDiscreteMetaDataRange(
                name="group_label_1-group_label_2", options=["invalid-combo"]
            ),
            metadata_range=MetaDataRange(
                continuous_conditions=[], discrete_conditions=[]
            ),
            timestamps_range=None,
            timeseries_range=[],
        )
        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, no_match_filters
        )
        assert len(filtered_df) == 0
        assert len(filtered_df["window_idx"].unique()) == 0

    @pytest.mark.asyncio
    async def test_filter_window_dataframe_reset_window_indices(self, test_df):
        """Test that window indices are reset after filtering."""
        # Create filters that will only keep window 1
        filters = WindowFilters(
            group_label_cols=OneDiscreteMetaDataRange(
                name="group_label_1-group_label_2",
                options=["NA-dev2"],  # Only matches window 1
            ),
            metadata_range=MetaDataRange(
                continuous_conditions=[],
                discrete_conditions=[],
            ),
            timestamps_range=None,
            timeseries_range=[],
        )

        filtered_df = await filter_window_dataframe_by_window_filters(
            test_df, filters
        )

        # Verify that window indices are reset to start from 0
        assert len(filtered_df["window_idx"].unique()) == 1
        assert list(filtered_df["window_idx"].unique()) == [
            0
        ]  # Should be reset to 0
        assert all(filtered_df["group_label_1"] == "NA")
        assert all(filtered_df["group_label_2"] == "dev2")


class TestApplyMetadataVariations:
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        df = pd.DataFrame(
            {
                "window_idx": [5, 5, 5, 6, 6, 6],
                "temperature": [25.0, 25.0, 25.0, 30.0, 30.0, 30.0],
                "pressure": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                "status": ["ok", "ok", "ok", "fail", "fail", "fail"],
            }
        )
        return df

    @pytest.mark.asyncio
    async def test_apply_single_perturbation(self, sample_df):
        """Test applying a single perturbation to a numeric column."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )
        # All rows in window 0 should be modified
        assert (
            result[result["window_idx"] == 0]["temperature"] == 30.0
        ).all()  # 25 + 5
        # All rows in window 1 should be modified
        assert (
            result[result["window_idx"] == 1]["temperature"] == 35.0
        ).all()  # 30 + 5

    @pytest.mark.asyncio
    async def test_apply_exact_value(self, sample_df):
        """Test applying exact values to columns."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="pending",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # All rows in window 0 should be modified
        assert (result[result["window_idx"] == 0]["status"] == "pending").all()
        # All rows in window 1 should remain unchanged
        assert (result[result["window_idx"] == 1]["status"] == "fail").all()

    @pytest.mark.asyncio
    async def test_window_index_range(self, sample_df):
        """Test proper handling of window index range."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=1,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Window 0 should be removed - 1 replaces 0 and window_idx is updated
        assert result.shape[0] == 3
        assert (result[result["window_idx"] == 0]["temperature"] == 35.0).all()

    @pytest.mark.asyncio
    async def test_all_perturbation_types(self, sample_df):
        """Test all perturbation types on numeric data."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ],
            [
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.SUBTRACT,
                )
            ],
            [
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                )
            ],
            [
                MetaDataVariation(
                    order=3,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.DIVIDE,
                )
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Each window should have all rows with the same value
        for window_idx, expected_temp in enumerate(
            [30.0, 20.0, 50.0, 12.5]
        ):  # 25+5, 25-5, 25*2, 25/2
            assert (
                result[result["window_idx"] == window_idx]["temperature"]
                == expected_temp
            ).all()

    @pytest.mark.asyncio
    async def test_mixed_exact_and_perturbation(self, sample_df):
        """Test combining exact value for string and perturbation for numeric."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="maintenance",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # All rows in window 0 should be modified
        assert (
            result[result["window_idx"] == 0]["status"] == "maintenance"
        ).all()
        assert (
            result[result["window_idx"] == 0]["temperature"] == 50.0
        ).all()  # 25 * 2

        # All rows in window 1 should remain unchanged
        assert (result[result["window_idx"] == 1]["status"] == "fail").all()
        assert (result[result["window_idx"] == 1]["temperature"] == 30.0).all()

    @pytest.mark.asyncio
    async def test_multiple_columns_same_variation(self, sample_df):
        """Test applying the same type of variation to multiple columns simultaneously."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=1,
                    name="pressure",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        assert (
            result[result["window_idx"] == 0]["temperature"] == 50.0
        ).all()  # 25 * 2
        assert (
            result[result["window_idx"] == 0]["pressure"] == 2.0
        ).all()  # 1 * 2
        assert (
            result[result["window_idx"] == 1]["temperature"] == 60.0
        ).all()  # 30 * 2
        assert (
            result[result["window_idx"] == 1]["pressure"] == 4.0
        ).all()  # 2 * 2

    @pytest.mark.asyncio
    async def test_empty_metadata_variations(self, sample_df):
        """Test behavior with empty metadata variations list."""
        metadata_variations = []

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        assert result.equals(sample_df)

    @pytest.mark.asyncio
    async def test_invalid_column_name(self, sample_df):
        """Test error handling when column name doesn't exist."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="nonexistent_column",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ]
        ]

        with pytest.raises(KeyError):
            await apply_metadata_variations(
                df=sample_df.copy(),
                metadata_variations=metadata_variations,
                window_start_idx=0,
                window_inclusive_end_idx=1,
                window_size=3,
            )

    @pytest.mark.asyncio
    async def test_type_mismatch_error(self, sample_df):
        """Test error handling when trying to apply numeric perturbation to string column."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ]
        ]

        with pytest.raises(ValueError, match="contains non-numeric values"):
            await apply_metadata_variations(
                df=sample_df.copy(),
                metadata_variations=metadata_variations,
                window_start_idx=0,
                window_inclusive_end_idx=1,
                window_size=3,
            )

    @pytest.mark.asyncio
    async def test_exact_value_type_mismatch(self, sample_df):
        """Test error handling when trying to set wrong type for exact value."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value="not_a_number",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                )
            ]
        ]

        with pytest.raises(
            ValueError, match="Value not_a_number is not the same type"
        ):
            await apply_metadata_variations(
                df=sample_df.copy(),
                metadata_variations=metadata_variations,
                window_start_idx=0,
                window_inclusive_end_idx=1,
                window_size=3,
            )

    @pytest.mark.asyncio
    async def test_invalid_window_indices(self, sample_df):
        """Test behavior with invalid window indices."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=2,  # Beyond available windows
            window_inclusive_end_idx=3,
            window_size=3,
        )

        assert result.empty

    @pytest.mark.asyncio
    async def test_division_by_zero(self, sample_df):
        """Test error handling when attempting division by zero."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.DIVIDE,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Should contain inf values after division by zero
        assert (
            result[result["window_idx"] == 0]["temperature"]
            .isin([float("inf")])
            .all()
        )

    @pytest.mark.asyncio
    async def test_multiple_variations_sequence(self, sample_df):
        """Test applying multiple variations in sequence to ensure proper window indexing."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ],
            [
                MetaDataVariation(
                    order=1,
                    name="status",
                    value="maintenance",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                )
            ],
            [
                MetaDataVariation(
                    order=2,
                    name="pressure",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                )
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check window indexing is correct
        assert len(result["window_idx"].unique()) == 3
        assert all(result["window_idx"].isin([0, 1, 2]))

        # Check first variation
        assert (
            result[result["window_idx"] == 0]["temperature"] == 30.0
        ).all()  # 25 + 5

        # Check second variation
        assert (
            result[result["window_idx"] == 1]["status"] == "maintenance"
        ).all()

        # Check third variation
        assert (
            result[result["window_idx"] == 2]["pressure"] == 2.0
        ).all()  # 1 * 2

    @pytest.mark.asyncio
    async def test_null_values_handling(self):
        """Test handling of null values in the dataframe."""
        # Create a sample dataframe with null values
        df_with_nulls = pd.DataFrame(
            {
                "window_idx": [0, 0, 0, 1, 1, 1],
                "temperature": [25.0, None, 25.0, 30.0, None, 30.0],
                "pressure": [1.0, 1.0, None, 2.0, 2.0, None],
                "status": ["ok", None, "ok", "fail", None, "fail"],
            }
        )

        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=1,
                    name="status",
                    value="maintenance",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ]
        ]

        result = await apply_metadata_variations(
            df=df_with_nulls.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Check that null values remain null after perturbation
        assert (
            result[result["window_idx"] == 0]["temperature"].isna().sum() == 1
        )
        assert (
            result[result["window_idx"] == 1]["temperature"].isna().sum() == 1
        )

        # Check that non-null values are correctly transformed
        assert (
            result[result["window_idx"] == 0]["temperature"]
            .dropna()
            .eq(50.0)
            .all()
        )
        assert (
            result[result["window_idx"] == 1]["temperature"]
            .dropna()
            .eq(60.0)
            .all()
        )

        # Check that exact values replace null values
        assert (result["status"] == "maintenance").all()

    @pytest.mark.asyncio
    async def test_numeric_exact_value(self, sample_df):
        """Test setting a numeric column to an exact value."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=42.0,
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                )
            ]
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Check that all temperature values in both windows are set to exactly 42.0
        assert (result[result["window_idx"] == 0]["temperature"] == 42.0).all()
        assert (result[result["window_idx"] == 1]["temperature"] == 42.0).all()

        # Verify other columns remain unchanged
        assert (result[result["window_idx"] == 0]["pressure"] == 1.0).all()
        assert (result[result["window_idx"] == 1]["pressure"] == 2.0).all()
        assert (result[result["window_idx"] == 0]["status"] == "ok").all()
        assert (result[result["window_idx"] == 1]["status"] == "fail").all()

    @pytest.mark.asyncio
    async def test_apply_metadata_variations_reset_window_indices(
        self, sample_df
    ):
        """Test that window indices are reset when applying metadata variations."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                )
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=1,  # Only apply to window 1
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # check that the index has been properly reset
        assert result.index.is_unique
        assert result.index.is_monotonic_increasing

        # Verify window indices are reset
        assert len(result["window_idx"].unique()) == 2
        assert list(sorted(result["window_idx"].unique())) == [0, 1]

        # Verify first variation (window 0 in result)
        assert (
            result[result["window_idx"] == 0]["temperature"] == 35.0
        ).all()  # 30 + 5

        # Verify second variation (window 1 in result)
        assert (
            result[result["window_idx"] == 1]["temperature"] == 40.0
        ).all()  # 30 + 10


class TestGetTrainConfigFileName:
    """Test suite for the get_train_config_file_name utility function."""

    @pytest.mark.parametrize(
        "dataset_name, task, job_name, expected_filename",
        [
            (
                "mydataset",
                "forecast",
                None,
                "config_mydataset_forecasting.yaml",
            ),
            (
                "another_data",
                "synthesis",
                None,
                "config_another_data_synthesis.yaml",
            ),
            (
                "dataset123",
                "forecast",
                "job_abc",
                "config_dataset123_job_abc_forecasting.yaml",
            ),
            (
                "data_v2",
                "synthesis",
                "experiment_x",
                "config_data_v2_experiment_x_synthesis.yaml",
            ),
            (
                "special-chars_dataset",
                "forecast",
                "job-with-hyphens",
                "config_special-chars_dataset_job-with-hyphens_forecasting.yaml",
            ),
        ],
    )
    def test_get_train_config_file_name(
        self, dataset_name, task, job_name, expected_filename
    ):
        """Tests the get_train_config_file_name function with various inputs."""
        filename = get_train_config_file_name(dataset_name, task, job_name)
        assert filename == expected_filename


class TestDetectTimeFrequency:
    def test_detect_yearly_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=5, freq="Y")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "year"

    def test_detect_monthly_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=12, freq="M")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "month"

    def test_detect_weekly_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=10, freq="W")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "week"

    def test_detect_daily_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_detect_hourly_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=24, freq="H")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "hour"

    def test_detect_minute_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=60, freq="T")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "minute"

    def test_detect_second_frequency(self):
        dates = pd.date_range(start="2020-01-01", periods=60, freq="S")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "second"

    def test_detect_multi_period_frequencies(self):
        # Test 3-hour frequency
        dates = pd.date_range(start="2020-01-01", periods=8, freq="3H")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 3
        assert result.unit == "hour"

        # Test 15-minute frequency
        dates = pd.date_range(start="2020-01-01", periods=4, freq="15T")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 15
        assert result.unit == "minute"

        # Test 2-day frequency
        dates = pd.date_range(start="2020-01-01", periods=5, freq="2D")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 2
        assert result.unit == "day"

    def test_detect_anchor_point_frequencies(self):
        # Test weekly frequencies with different anchor points
        dates = pd.date_range(start="2020-01-01", periods=10, freq="W-SUN")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "week"

        dates = pd.date_range(start="2020-01-01", periods=10, freq="W-MON")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "week"

        # Test annual frequencies with different anchor points
        dates = pd.date_range(start="2020-01-01", periods=5, freq="A-JAN")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "year"

        dates = pd.date_range(start="2020-01-01", periods=5, freq="A-DEC")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "year"

    def test_detect_quarterly_frequencies(self):
        # Test basic quarterly frequency
        dates = pd.date_range(start="2020-01-01", periods=8, freq="Q")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "quarter"

        # Test quarterly frequencies with different anchor points
        dates = pd.date_range(start="2020-01-01", periods=8, freq="Q-JAN")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "quarter"

        dates = pd.date_range(start="2020-01-01", periods=8, freq="Q-DEC")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "quarter"

    def test_irregular_timestamps(self):
        # Create irregular timestamps that don't follow a pattern
        irregular_dates = [
            "2020-01-01",
            "2020-01-03",
            "2020-01-08",
            "2020-02-15",
            "2020-04-01",
        ]
        # Should now detect the most common frequency (2 days is most common)
        result = detect_time_frequency(irregular_dates)
        assert result is not None
        assert result.value == 2
        assert result.unit == "day"

    def test_single_timestamp(self):
        assert detect_time_frequency(["2020-01-01"]) is None

    def test_empty_dataframe(self):
        assert detect_time_frequency([]) is None

    def test_invalid_timestamp_format(self):
        timestamps = ["invalid_date", "2020-01-01", "2020-01-02"]
        assert detect_time_frequency(timestamps) is None

    def test_mixed_frequencies(self):
        # Create timestamps with mixed frequencies
        dates = [
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",  # hourly
            "2020-01-02 00:00:00",  # daily
            "2020-01-09 00:00:00",  # weekly
        ]
        # With fallback, should detect the most common interval (hour level)
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "hour"

    def test_string_timestamp_conversion(self):
        # Test with string timestamps that need to be converted
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_different_timestamp_formats(self):
        # Test with different valid timestamp formats
        dates = [
            "2020-01-01 00:00:00",
            "2020/01/02 00:00:00",
            "2020.01.03 00:00:00",
        ]
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_irregular_timestamps_minute_level(self):
        # Create irregular timestamps with minute-level intervals
        timestamps = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:15:00",  # 15 minutes later
                "2023-01-01 10:28:00",  # 13 minutes later
                "2023-01-01 10:43:00",  # 15 minutes later
                "2023-01-01 11:00:00",  # 17 minutes later
                "2023-01-01 11:15:00",  # 15 minutes later
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should detect most common interval (15 minutes = minute level)
        assert result is not None
        assert result.value == 15
        assert result.unit == "minute"

    def test_irregular_timestamps_hour_level(self):
        # Create irregular timestamps with hour-level intervals
        timestamps = pd.to_datetime(
            [
                "2023-01-01 08:00:00",
                "2023-01-01 10:00:00",  # 2 hours later
                "2023-01-01 13:00:00",  # 3 hours later
                "2023-01-01 15:00:00",  # 2 hours later
                "2023-01-01 17:00:00",  # 2 hours later
                "2023-01-01 19:00:00",  # 2 hours later
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should detect most common interval (2 hours = hour level)
        assert result is not None
        assert result.value == 2
        assert result.unit == "hour"

    def test_irregular_timestamps_month_level(self):
        # Create irregular timestamps with roughly monthly intervals
        timestamps = pd.to_datetime(
            [
                "2023-01-15",
                "2023-02-20",  # ~36 days later
                "2023-03-18",  # ~26 days later
                "2023-04-22",  # ~35 days later
                "2023-05-25",  # ~33 days later
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should detect most common interval (36 days = 4 weeks)
        assert result is not None
        assert result.value == 4
        assert result.unit == "week"

    def test_irregular_weekly_pattern(self):
        # Create irregular timestamps with weekly-ish intervals
        timestamps = pd.to_datetime(
            [
                "2023-01-01",  # Sunday
                "2023-01-09",  # Monday (8 days later)
                "2023-01-15",  # Sunday (6 days later)
                "2023-01-22",  # Sunday (7 days later)
                "2023-01-30",  # Monday (8 days later)
                "2023-02-05",  # Sunday (6 days later)
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should detect most common interval (6 days is most common)
        assert result is not None
        assert result.value == 6
        assert result.unit == "day"

    def test_very_irregular_timestamps(self):
        # Create very irregular timestamps that should still get some frequency
        timestamps = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:05:00",  # 5 minutes
                "2023-01-01 10:35:00",  # 30 minutes
                "2023-01-01 11:40:00",  # 65 minutes
                "2023-01-01 11:45:00",  # 5 minutes
                "2023-01-01 12:15:00",  # 30 minutes
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should still detect minute-level frequency
        assert result is not None
        assert result.value == 5
        assert result.unit == "minute"

    def test_irregular_timestamps_with_large_gaps(self):
        # Create timestamps with some large gaps
        timestamps = pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-02",  # 1 day
                "2023-01-03",  # 1 day
                "2023-02-01",  # 29 days (big gap)
                "2023-02-02",  # 1 day
                "2023-02-03",  # 1 day
            ]
        )
        result = detect_time_frequency(timestamps)
        # Should detect daily frequency (most common is 1 day)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_insufficient_data_for_mode_detection(self):
        # Single timestamp should still return None
        timestamps = pd.to_datetime(["2023-01-01"])
        result = detect_time_frequency(timestamps)
        assert result is None

    def test_pandas_inference_success_vs_fallback(self):
        # Test that regular data still uses pandas inference
        regular_timestamps = pd.date_range(
            start="2020-01-01", periods=10, freq="D"
        )
        result = detect_time_frequency(regular_timestamps)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

        # Test that irregular data uses fallback
        irregular_timestamps = pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-03",  # 2 days
                "2020-01-04",  # 1 day
                "2020-01-06",  # 2 days
                "2020-01-07",  # 1 day
                "2020-01-09",  # 2 days
            ]
        )
        result = detect_time_frequency(irregular_timestamps)
        # Should detect most common interval (2 days) using fallback
        assert result is not None
        assert result.value == 2
        assert result.unit == "day"


class TestDetectFrequencyFromMode:
    """Test the _detect_frequency_from_mode helper function directly."""

    def test_detect_frequency_from_mode_minutes(self):
        # Import the private function for testing
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:15:00",  # 15 minutes
                "2023-01-01 10:30:00",  # 15 minutes
                "2023-01-01 10:47:00",  # 17 minutes
                "2023-01-01 11:02:00",  # 15 minutes
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 15
        assert result.unit == "minute"

    def test_detect_frequency_from_mode_hours(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-01 08:00:00",
                "2023-01-01 10:00:00",  # 2 hours
                "2023-01-01 12:00:00",  # 2 hours
                "2023-01-01 15:00:00",  # 3 hours
                "2023-01-01 17:00:00",  # 2 hours
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 2
        assert result.unit == "hour"

    def test_detect_frequency_from_mode_days(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-03",  # 2 days
                "2023-01-04",  # 1 day
                "2023-01-06",  # 2 days
                "2023-01-07",  # 1 day
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_detect_frequency_from_mode_weeks(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-01",
                "2023-01-08",  # 7 days
                "2023-01-15",  # 7 days
                "2023-01-23",  # 8 days
                "2023-01-29",  # 6 days
                "2023-02-05",  # 7 days
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 1
        assert result.unit == "week"

    def test_detect_frequency_from_mode_months(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-15",
                "2023-02-17",  # ~33 days
                "2023-03-20",  # ~31 days
                "2023-04-18",  # ~29 days
                "2023-05-20",  # ~32 days
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 4
        assert result.unit == "week"

    def test_detect_frequency_from_mode_years(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2020-03-15",
                "2021-03-20",  # ~370 days
                "2022-03-18",  # ~363 days
                "2023-03-22",  # ~369 days
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 12
        assert result.unit == "month"

    def test_detect_frequency_from_mode_seconds(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        timestamps = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:00:30",  # 30 seconds
                "2023-01-01 10:01:00",  # 30 seconds
                "2023-01-01 10:01:32",  # 32 seconds
                "2023-01-01 10:02:02",  # 30 seconds
            ]
        )
        result = _detect_frequency_from_mode(timestamps)
        assert result is not None
        assert result.value == 30
        assert result.unit == "second"

    def test_detect_frequency_from_mode_edge_cases(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        # Empty timestamps
        result = _detect_frequency_from_mode(
            pd.Series([], dtype="datetime64[ns]")
        )
        assert result is None

        # Single timestamp
        result = _detect_frequency_from_mode(pd.to_datetime(["2023-01-01"]))
        assert result is None

    def test_detect_frequency_from_mode_boundary_values(self):
        from synthefy_pkg.app.utils.api_utils import _detect_frequency_from_mode

        # Test boundary between minute and hour (59 minutes = minute, 61 minutes = hour)
        timestamps_59min = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-01 10:59:00",  # 59 minutes
                "2023-01-01 11:58:00",  # 59 minutes
            ]
        )
        result = _detect_frequency_from_mode(timestamps_59min)
        assert result is not None
        assert result.value == 59
        assert result.unit == "minute"

        # Test boundary between hour and day (23 hours = hour, 25 hours = day)
        timestamps_23hr = pd.to_datetime(
            [
                "2023-01-01 10:00:00",
                "2023-01-02 09:00:00",  # 23 hours
                "2023-01-03 08:00:00",  # 23 hours
            ]
        )
        result = _detect_frequency_from_mode(timestamps_23hr)
        assert result is not None
        assert result.value == 23
        assert result.unit == "hour"


class TestTimeFrequencyComprehensive:
    """Additional comprehensive tests for time frequency detection."""

    def test_detect_sub_second_frequencies(self):
        """Test detection of sub-second frequencies."""
        # Test millisecond frequency
        dates = pd.date_range(
            start="2020-01-01 10:00:00", periods=5, freq="100ms"
        )
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 100
        assert result.unit == "millisecond"
        assert str(result) == "100 milliseconds"

        # Test microsecond frequency - should detect milliseconds via fallback
        dates = pd.date_range(
            start="2020-01-01 10:00:00", periods=5, freq="500ms"
        )
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 500
        assert result.unit == "millisecond"

    def test_detect_business_day_frequency(self):
        """Test detection of business day frequency."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="B")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

    def test_timezone_aware_timestamps(self):
        """Test handling of timezone-aware timestamps."""
        # UTC timezone
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D", tz="UTC")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"

        # EST timezone
        dates = pd.date_range(
            start="2020-01-01", periods=5, freq="H", tz="US/Eastern"
        )
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 1
        assert result.unit == "hour"

    def test_leap_year_handling(self):
        """Test data spanning leap years."""
        # February 29th should be handled correctly
        dates = pd.to_datetime(
            [
                "2020-02-29",  # Leap year
                "2021-02-28",  # Non-leap year
                "2022-02-28",
                "2023-02-28",
            ]
        )
        result = detect_time_frequency(dates)
        assert result is not None
        # Should detect roughly yearly frequency (measured in days/months)
        assert result.unit in ["day", "month", "year"]

    def test_very_large_numeric_parts(self):
        """Test frequencies with very large numeric parts."""
        # Create timestamps 500 days apart
        timestamps = pd.to_datetime(
            [
                "2020-01-01",
                "2021-05-16",  # ~500 days later
                "2022-09-28",  # ~500 days later
                "2024-02-09",  # ~500 days later
            ]
        )
        result = detect_time_frequency(timestamps)
        assert result is not None
        # Should detect year-level frequency (since gaps are ~1.4 years each)
        assert result.unit == "year"
        assert result.value >= 1  # Should be at least 1 year

    def test_time_frequency_model_properties(self):
        """Test the TimeFrequency model's properties and methods."""
        from synthefy_pkg.app.data_models import TimeFrequency

        # Test basic properties
        freq = TimeFrequency(value=1, unit="day")
        assert str(freq) == "1 day"

        freq = TimeFrequency(value=15, unit="minute")
        assert str(freq) == "15 minutes"

        # Test string representation for different values
        freq = TimeFrequency(value=1, unit="hour")
        assert str(freq) == "1 hour"

        freq = TimeFrequency(value=30, unit="second")
        assert str(freq) == "30 seconds"

        freq = TimeFrequency(value=500, unit="millisecond")
        assert str(freq) == "500 milliseconds"

    def test_time_frequency_model_validation(self):
        """Test TimeFrequency model validation."""
        from synthefy_pkg.app.data_models import TimeFrequency

        # Valid creation
        freq = TimeFrequency(value=15, unit="minute")
        assert freq.value == 15
        assert freq.unit == "minute"

        # Test validation errors
        with pytest.raises(ValueError):
            TimeFrequency(value=0, unit="minute")  # zero value

        with pytest.raises(ValueError):
            TimeFrequency(value=-5, unit="hour")  # negative value

        with pytest.raises(ValueError):
            TimeFrequency(value=1, unit="invalid_unit")  # invalid unit

    def test_complex_frequency_patterns(self):
        """Test complex pandas frequency patterns."""
        # Test week with anchor
        dates = pd.date_range(start="2020-01-01", periods=8, freq="2W-TUE")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 2
        assert result.unit == "week"

        # Test month with anchor
        dates = pd.date_range(start="2020-01-31", periods=6, freq="2M")
        result = detect_time_frequency(dates)
        assert result is not None
        assert result.value == 2
        assert result.unit == "month"

    def test_mixed_precision_timestamps(self):
        """Test timestamps with different precision levels."""
        mixed_timestamps = [
            "2020-01-01",
            "2020-01-02 00:00:00.000000",
            "2020-01-03T00:00:00",
            "2020-01-04 00:00:00.000",
        ]
        result = detect_time_frequency(mixed_timestamps)
        # Mixed precision should still be detected as daily
        assert result is not None
        assert result.value == 1
        assert result.unit == "day"


class TestFormatTimestampWithOptionalFractionalSeconds:
    """Test the format_timestamp_with_optional_fractional_seconds function."""

    def test_format_timestamp_no_fractional_seconds(self):
        """Test timestamp formatting without fractional seconds."""
        timestamp = pd.Timestamp("2010-02-05T00:00:00")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2010-02-05T00:00:00"

    def test_format_timestamp_with_milliseconds(self):
        """Test timestamp formatting with millisecond precision."""
        timestamp = pd.Timestamp("2010-02-05T00:00:00.142")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2010-02-05T00:00:00.142"

    def test_format_timestamp_with_microseconds(self):
        """Test timestamp formatting with full microsecond precision."""
        timestamp = pd.Timestamp("2010-02-05T00:00:00.142567")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2010-02-05T00:00:00.142567"

    def test_format_timestamp_with_trailing_zeros(self):
        """Test timestamp formatting removes trailing zeros."""
        timestamp = pd.Timestamp("2010-02-05T00:00:00.142000")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2010-02-05T00:00:00.142"

    def test_format_timestamp_with_single_microsecond(self):
        """Test timestamp formatting with single microsecond."""
        timestamp = pd.Timestamp("2010-02-05T00:00:00.000001")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2010-02-05T00:00:00.000001"

    def test_format_timestamp_with_datetime_object(self):
        """Test timestamp formatting with datetime object."""
        timestamp = datetime(2010, 2, 5, 0, 0, 0, 142000)
        result = format_timestamp_with_optional_fractional_seconds(
            pd.Timestamp(timestamp)
        )
        assert result == "2010-02-05T00:00:00.142"

    def test_format_timestamp_various_precisions(self):
        """Test timestamp formatting with various precision levels."""
        test_cases = [
            ("2020-01-01T12:30:45", "2020-01-01T12:30:45"),
            ("2020-01-01T12:30:45.1", "2020-01-01T12:30:45.1"),
            ("2020-01-01T12:30:45.12", "2020-01-01T12:30:45.12"),
            ("2020-01-01T12:30:45.123", "2020-01-01T12:30:45.123"),
            ("2020-01-01T12:30:45.1234", "2020-01-01T12:30:45.1234"),
            ("2020-01-01T12:30:45.12345", "2020-01-01T12:30:45.12345"),
            ("2020-01-01T12:30:45.123456", "2020-01-01T12:30:45.123456"),
            ("2020-01-01T12:30:45.100000", "2020-01-01T12:30:45.1"),
            ("2020-01-01T12:30:45.123000", "2020-01-01T12:30:45.123"),
        ]

        for input_str, expected in test_cases:
            timestamp = pd.Timestamp(input_str)
            result = format_timestamp_with_optional_fractional_seconds(
                timestamp
            )
            assert result == expected, (
                f"Failed for {input_str}: expected {expected}, got {result}"
            )

    def test_format_timestamp_edge_cases(self):
        """Test timestamp formatting with edge cases."""
        # Test minimum date
        timestamp = pd.Timestamp("1900-01-01T00:00:00")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "1900-01-01T00:00:00"

        # Test maximum microseconds
        timestamp = pd.Timestamp("2020-12-31T23:59:59.999999")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2020-12-31T23:59:59.999999"

        # Test leap year
        timestamp = pd.Timestamp("2020-02-29T12:30:45.123456")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2020-02-29T12:30:45.123456"

    def test_format_timestamp_different_constructors(self):
        """Test timestamp formatting with different pandas Timestamp constructors."""
        # From string
        timestamp = pd.Timestamp("2020-01-01T12:00:00.123")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2020-01-01T12:00:00.123"

        # From year, month, day, etc.
        timestamp = pd.Timestamp(
            year=2020, month=1, day=1, hour=12, microsecond=123456
        )
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2020-01-01T12:00:00.123456"

        # From datetime
        dt = datetime(2020, 1, 1, 12, 0, 0, 123456)
        timestamp = pd.Timestamp(dt)
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        assert result == "2020-01-01T12:00:00.123456"

    def test_format_timestamp_nanosecond_precision(self):
        """Test timestamp formatting with nanosecond precision (should be truncated to microseconds)."""
        # Pandas timestamps have nanosecond precision, but datetime objects only have microsecond
        # The function should handle this correctly
        timestamp = pd.Timestamp("2020-01-01T12:00:00.123456789")
        result = format_timestamp_with_optional_fractional_seconds(timestamp)
        # Should be truncated to microseconds
        assert result == "2020-01-01T12:00:00.123456"


class TestOrderPreservationInModifications:
    """Test suite for ensuring order of modifications is preserved correctly."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing order preservation."""
        df = pd.DataFrame(
            {
                "window_idx": [0, 0, 0, 1, 1, 1],
                "temperature": [25.0, 25.0, 25.0, 30.0, 30.0, 30.0],
                "pressure": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                "status": ["ok", "ok", "ok", "fail", "fail", "fail"],
            }
        )
        return df

    @pytest.mark.asyncio
    async def test_metadata_variations_order_preservation(self, sample_df):
        """Test that metadata variations are applied in the correct order."""
        # Create variations with specific order to test
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="maintenance",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Check that we have 4 windows (2 variation lists * 2 original windows)
        assert len(result["window_idx"].unique()) == 4

        # Check that the first variation (window 0) has temperature = (25 + 5) * 2 = 60
        assert (result[result["window_idx"] == 0]["temperature"] == 60.0).all()

        # Check that the second variation (window 2) has status = "maintenance"
        assert (
            result[result["window_idx"] == 2]["status"] == "maintenance"
        ).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_overlapping_columns(
        self, sample_df
    ):
        """Test order preservation when multiple variations modify the same column."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.SUBTRACT,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check that the order is preserved: (25 + 10) * 2 - 5 = 65
        assert (result[result["window_idx"] == 0]["temperature"] == 65.0).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_exact_value_overwrite(
        self, sample_df
    ):
        """Test that exact value modifications overwrite previous modifications in order."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=42.0,
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check that the exact value overwrites and then the addition is applied: 42 + 5 = 47
        assert (result[result["window_idx"] == 0]["temperature"] == 47.0).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_multiple_columns(
        self, sample_df
    ):
        """Test order preservation across multiple columns."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="pressure",
                    value=3.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=3,
                    name="status",
                    value="critical",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check temperature: (25 + 5) * 2 = 60
        assert (result[result["window_idx"] == 0]["temperature"] == 60.0).all()

        # Check pressure: 1 + 3 = 4
        assert (result[result["window_idx"] == 0]["pressure"] == 4.0).all()

        # Check status: "critical"
        assert (result[result["window_idx"] == 0]["status"] == "critical").all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_division_by_zero(
        self, sample_df
    ):
        """Test order preservation when division by zero occurs."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=0.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.DIVIDE,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check that division by zero results in inf, and then addition still works
        assert (
            result[result["window_idx"] == 0]["temperature"] == float("inf")
        ).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_string_operations(
        self, sample_df
    ):
        """Test order preservation with string column operations."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="warning",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=1,
                    name="status",
                    value="critical",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
                MetaDataVariation(
                    order=2,
                    name="status",
                    value="resolved",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check that the last exact value overwrites all previous ones
        assert (result[result["window_idx"] == 0]["status"] == "resolved").all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_empty_variations(
        self, sample_df
    ):
        """Test order preservation when some variations are empty."""
        metadata_variations = [
            [],  # Empty variation
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
            [],  # Another empty variation
            [
                MetaDataVariation(
                    order=0,
                    name="status",
                    value="active",
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Check that we have 8 windows (4 variation lists * 2 original windows)
        assert len(result["window_idx"].unique()) == 8

        # Check first variation (temperature + 10) - window 2
        assert (result[result["window_idx"] == 2]["temperature"] == 35.0).all()

        # Check second variation (status = "active") - window 6
        assert (result[result["window_idx"] == 6]["status"] == "active").all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_complex_operations(
        self, sample_df
    ):
        """Test order preservation with complex mathematical operations."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=10.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.SUBTRACT,
                ),
                MetaDataVariation(
                    order=3,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.DIVIDE,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check the complex calculation: ((25 + 5) * 2 - 10) / 2 = 25
        assert (result[result["window_idx"] == 0]["temperature"] == 25.0).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_window_index_preservation(
        self, sample_df
    ):
        """Test that window indices are preserved in the correct order."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=1.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=3.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=1,
            window_size=3,
        )

        # Check that we have 6 windows (3 variation lists * 2 original windows)
        assert len(result["window_idx"].unique()) == 6
        assert list(result["window_idx"].unique()) == [0, 1, 2, 3, 4, 5]

        # Check that each window has the correct temperature value
        # Each variation list gets a copy of the original data, then modified
        # Windows 0, 2, 4 correspond to the first, second, third variations applied to window 0
        # Windows 1, 3, 5 correspond to the first, second, third variations applied to window 1
        assert (
            result[result["window_idx"] == 0]["temperature"] == 26.0
        ).all()  # 25 + 1
        assert (
            result[result["window_idx"] == 1]["temperature"] == 31.0
        ).all()  # 30 + 1
        assert (
            result[result["window_idx"] == 2]["temperature"] == 27.0
        ).all()  # 25 + 2
        assert (
            result[result["window_idx"] == 3]["temperature"] == 32.0
        ).all()  # 30 + 2
        assert (
            result[result["window_idx"] == 4]["temperature"] == 28.0
        ).all()  # 25 + 3
        assert (
            result[result["window_idx"] == 5]["temperature"] == 33.0
        ).all()  # 30 + 3

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_negative_values(
        self, sample_df
    ):
        """Test order preservation with negative values."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=-5.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=-2.0,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check the calculation: (25 + (-5)) * (-2) = 20 * (-2) = -40
        assert (result[result["window_idx"] == 0]["temperature"] == -40.0).all()

    @pytest.mark.asyncio
    async def test_metadata_variations_order_with_floating_point_precision(
        self, sample_df
    ):
        """Test order preservation with floating point precision."""
        metadata_variations = [
            [
                MetaDataVariation(
                    order=0,
                    name="temperature",
                    value=0.1,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
                MetaDataVariation(
                    order=1,
                    name="temperature",
                    value=0.3,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.MULTIPLY,
                ),
                MetaDataVariation(
                    order=2,
                    name="temperature",
                    value=0.5,
                    perturbation_or_exact_value="perturbation",
                    perturbation_type=PerturbationType.ADD,
                ),
            ],
        ]

        result = await apply_metadata_variations(
            df=sample_df.copy(),
            metadata_variations=metadata_variations,
            window_start_idx=0,
            window_inclusive_end_idx=0,
            window_size=3,
        )

        # Check the calculation: ((25 + 0.1) * 0.3) + 0.5 = 25.1 * 0.3 + 0.5 = 7.53 + 0.5 = 8.03
        expected_value = ((25.0 + 0.1) * 0.3) + 0.5
        actual_values = result[result["window_idx"] == 0]["temperature"].values
        assert all(
            abs(actual - expected_value) < 1e-10 for actual in actual_values
        )


if __name__ == "__main__":
    pytest.main()
