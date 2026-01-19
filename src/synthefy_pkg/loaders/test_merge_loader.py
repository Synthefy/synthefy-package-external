import os

# add home directory to path
import sys
from collections import Counter
from datetime import timezone
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from synthefy_pkg.loaders.merge_loader import MergeLoader


@pytest.fixture
def merge_loader():
    return MergeLoader("")  # You'll need to provide a valid config path


@pytest.fixture
def sample_data():
    # Create sample data for each frequency
    data = {
        "Yearly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="Y"),
                "value": [1, 2, 3],
            }
        ),
        "Quarterly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=4, freq="Q"),
                "value": [1, 2, 3, 4],
            }
        ),
        "Monthly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=12, freq="M"),
                "value": range(1, 13),
            }
        ),
        "Weekly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=52, freq="W"),
                "value": range(1, 53),
            }
        ),
        "Daily": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=365, freq="D"),
                "value": range(1, 366),
            }
        ),
        "Hourly": pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=24 * 7, freq="H"
                ),
                "value": range(1, 24 * 7 + 1),
            }
        ),
    }
    return data


def test_downsample_adjacent_frequencies(merge_loader, sample_data):
    # Test downsampling between adjacent frequency levels
    frequency_pairs = [
        ("Hourly", "Daily"),
        ("Daily", "Weekly"),
        ("Weekly", "Monthly"),
        ("Monthly", "Quarterly"),
        ("Quarterly", "Yearly"),
    ]

    for from_freq, to_freq in frequency_pairs:
        df = sample_data[from_freq]
        result, new_freq = merge_loader._downsample_one_level(df, from_freq)

        assert new_freq == to_freq
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "timestamp" in result.columns
        assert len(result) < len(df)  # Downsampled data should have fewer rows


def test_upsample_adjacent_frequencies(merge_loader, sample_data):
    # Test upsampling between adjacent frequency levels
    frequency_pairs = [
        ("Yearly", "Quarterly"),
        ("Quarterly", "Monthly"),
        ("Monthly", "Weekly"),
        ("Weekly", "Daily"),
        ("Daily", "Hourly"),
    ]

    for from_freq, to_freq in frequency_pairs:
        df = sample_data[from_freq]
        result, new_freq = merge_loader._upsample_one_level(df, from_freq)

        assert new_freq == to_freq
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "timestamp" in result.columns
        assert len(result) > len(df)  # Upsampled data should have more rows


def test_resample_metadata_all_frequencies(merge_loader, sample_data):
    all_frequencies = [
        "Yearly",
        "Quarterly",
        "Monthly",
        "Weekly",
        "Daily",
        "Hourly",
    ]

    # Test conversion between all possible frequency pairs
    for from_freq in all_frequencies:
        df = sample_data[from_freq]
        for to_freq in all_frequencies:
            if from_freq != to_freq:
                result = merge_loader._resample_metadata(df, from_freq, to_freq)
                assert isinstance(result, pd.DataFrame)
                assert not result.empty
                assert "timestamp" in result.columns


def test_invalid_frequency_conversion(merge_loader, sample_data):
    # Test invalid frequency conversion
    with pytest.raises(ValueError):
        merge_loader._downsample_one_level(sample_data["Yearly"], "Yearly")

    with pytest.raises(ValueError):
        merge_loader._upsample_one_level(sample_data["Hourly"], "Hourly")


@pytest.fixture
def sample_data_with_nans():
    # Create sample data with NaN values
    return {
        "Monthly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=12, freq="M"),
                "value": [
                    1,
                    np.nan,
                    3,
                    np.nan,
                    5,
                    6,
                    np.nan,
                    8,
                    9,
                    np.nan,
                    11,
                    12,
                ],
                "other_value": [
                    np.nan,
                    2,
                    np.nan,
                    4,
                    np.nan,
                    6,
                    7,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        ),
        "Weekly": pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=8, freq="W"),
                "value": [1, np.nan, 3, np.nan, 5, np.nan, 7, 8],
                "other_value": [np.nan, 2, np.nan, 4, np.nan, 6, np.nan, 8],
            }
        ),
    }


def test_downsample_with_nans(merge_loader, sample_data_with_nans):
    """Test downsampling behavior with NaN values"""
    # Test Monthly to Quarterly downsampling
    monthly_df = sample_data_with_nans["Monthly"]
    result, new_freq = merge_loader._downsample_one_level(monthly_df, "Monthly")

    assert new_freq == "Quarterly"
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Check that NaN handling in mean aggregation works as expected
    assert not bool(result["value"].isna().all())  # Shouldn't be all NaN
    assert bool(
        result["other_value"].isna().any()
    )  # Should preserve some NaN where all values in period were NaN


def test_upsample_with_nans(merge_loader, sample_data_with_nans):
    """Test upsampling behavior with NaN values"""
    # Test Weekly to Daily upsampling
    weekly_df = sample_data_with_nans["Weekly"]
    result, new_freq = merge_loader._upsample_one_level(weekly_df, "Weekly")

    assert new_freq == "Daily"
    assert isinstance(result, pd.DataFrame)
    # Check that NaN values are properly forwarded
    assert (
        result["value"].isna().sum() == weekly_df["value"].isna().sum() * 7
    )  # Each weekly NaN becomes 7 daily NaNs
    assert not bool(result["value"].isna().all())  # Shouldn't be all NaN


def test_resample_metadata_nan_preservation(
    merge_loader, sample_data_with_nans
):
    """Test that resampling preserves NaN patterns appropriately"""
    monthly_df = sample_data_with_nans["Monthly"]

    # Test both up and down sampling
    quarterly_result = merge_loader._resample_metadata(
        monthly_df, "Monthly", "Quarterly"
    )
    weekly_result = merge_loader._resample_metadata(
        monthly_df, "Monthly", "Weekly"
    )
    # Check NaN handling in both directions
    assert not quarterly_result["value"].isna().all()
    assert not weekly_result["value"].isna().all()
    assert quarterly_result["other_value"].isna().any()
    assert weekly_result["value"].isna().any()


def test_consecutive_nan_handling(merge_loader):
    """Test handling of consecutive NaN values"""
    # Create data with consecutive NaN values
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=12, freq="M"),
            "value": [
                1,
                np.nan,
                np.nan,
                np.nan,
                5,
                6,
                np.nan,
                np.nan,
                9,
                10,
                np.nan,
                12,
            ],
        }
    )

    # Test downsampling
    result, _ = merge_loader._downsample_one_level(df, "Monthly")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Check that consecutive NaNs are handled appropriately in aggregation
    assert not bool(result["value"].isna().any())
    assert not bool(result["value"].isna().all())


def test_edge_case_all_nans(merge_loader):
    """Test handling of a column with all NaN values"""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=12, freq="M"),
            "value": [np.nan] * 12,
            "valid_value": range(12),
        }
    )

    # Test downsampling
    result, _ = merge_loader._downsample_one_level(df, "Monthly")

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result["value"], pd.Series)
    assert bool(
        result["value"].isna().all()
    )  # All-NaN column should remain all-NaN
    assert isinstance(result["valid_value"], pd.Series)
    assert not bool(
        result["valid_value"].isna().any()
    )  # Valid column should remain valid


@pytest.fixture
def sample_metadata_generator():
    def create_generator() -> Generator[
        tuple[str, pd.DataFrame, dict[str, str | list[str]]], None, None
    ]:
        # Create sample metadata entries with different frequencies
        metadata1 = (
            "series1",
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=24, freq="H"
                    ),
                    "value": range(24),
                }
            ),
            {"frequency": "Hourly", "length": 24},
        )

        metadata2 = (
            "series2",
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=7, freq="D"
                    ),
                    "value": range(7),
                }
            ),
            {"frequency": "Daily", "length": 7},
        )

        metadata3 = (
            "series3",
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=12, freq="M"
                    ),
                    "value": range(12),
                }
            ),
            {"frequency": ["Monthly"], "length": 12},  # Test list frequency
        )

        for item in [metadata1, metadata2, metadata3]:
            yield item

    return create_generator


def test_generate_metadata_all_frequency_dict_basic(
    merge_loader, sample_metadata_generator
):
    """Test basic functionality of metadata dictionary generation"""
    merge_loader.config["metadata_n_longest"] = 0
    result = merge_loader._generate_metadata_all_frequency_dict(
        sample_metadata_generator()
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(item[0], dict) for item in result)
    assert all(isinstance(item[1], dict) for item in result)


def test_generate_metadata_all_frequency_dict_frequencies(
    merge_loader, sample_metadata_generator
):
    """Test that all frequencies are generated for each metadata series"""
    merge_loader.config["metadata_n_longest"] = 0
    result = merge_loader._generate_metadata_all_frequency_dict(
        sample_metadata_generator()
    )

    for metadata_dict, _ in result:
        # Check all frequencies are present
        assert set(metadata_dict.keys()) == set(
            ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        )

        # Check all values are DataFrames
        assert all(
            isinstance(df, pd.DataFrame) for df in metadata_dict.values()
        )


def test_generate_metadata_all_frequency_dict_n_longest(
    merge_loader, sample_metadata_generator
):
    """Test n_longest parameter functionality"""
    # Set n_longest to 2
    merge_loader.config["metadata_n_longest"] = 2
    result = merge_loader._generate_metadata_all_frequency_dict(
        sample_metadata_generator()
    )

    assert len(result) == 2
    # Should keep the two longest series (24 and 12 length)
    lengths = [meta["length"] for _, meta in result]
    assert sorted(lengths, reverse=True) == [24, 12]


def test_generate_metadata_all_frequency_dict_timestamp_handling(
    merge_loader, sample_metadata_generator
):
    """Test timestamp column handling"""
    merge_loader.config["metadata_n_longest"] = 0
    result = merge_loader._generate_metadata_all_frequency_dict(
        sample_metadata_generator()
    )
    count = 0
    for metadata_dict, _ in result:
        for key in metadata_dict.keys():
            df = metadata_dict[key]
            count += 1
            assert (
                list(df.columns).count("timestamp") == 0
            )  # No timestamp in regular columns
            # assert isinstance(df.index, pd.DatetimeIndex) # note: we may want this check, but future code handles it
            assert df.index.name == "timestamp"


def test_generate_metadata_all_frequency_dict_empty(merge_loader):
    """Test handling of empty generator"""
    merge_loader.config["metadata_n_longest"] = 0
    empty_generator = (x for x in [])  # Create empty generator
    result = merge_loader._generate_metadata_all_frequency_dict(empty_generator)

    assert isinstance(result, list)
    assert len(result) == 0


def test_generate_metadata_all_frequency_dict_list_frequency(merge_loader):
    """Test handling of frequency as list"""
    merge_loader.config["metadata_n_longest"] = 0

    def generator():
        yield (
            "series1",
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2020-01-01", periods=12, freq="M"
                    ),
                    "value": range(12),
                }
            ),
            {"frequency": ["Monthly", "Daily"], "length": 12},
        )

    result = merge_loader._generate_metadata_all_frequency_dict(generator())
    assert len(result) == 1
    assert isinstance(result[0][0], dict)
    assert len(result[0][0]) == 6  # Should have all frequency levels


@pytest.fixture
def sample_merge_data():
    # Create sample data with timezone
    datas = pd.DataFrame(
        {"value": range(5)},
        index=pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC"),
    )

    # Create sample metadata
    metadata_dict = (
        {
            "Daily": pd.DataFrame(
                {"metadata_val": [10, 20, 30]},
                index=pd.date_range("2020-01-01", periods=3, freq="D"),
            )
        },
        {"columns": [{"column_id": "metadata_val"}], "frequency": "Daily"},
    )

    dmd = {"frequency": "Daily", "columns": []}

    return datas, metadata_dict, dmd


def test_merge_slice_metadata_basic(merge_loader, sample_merge_data):
    """Test basic merging functionality"""
    datas, metadata_dict, dmd = sample_merge_data
    timestamp_col = datas.index
    metadata_seen_count = Counter()

    result_df, result_dmd = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert isinstance(result_df, pd.DataFrame)
    assert "metadata_val_1" in result_df.columns
    assert len(result_dmd["columns"]) == 1
    assert result_df.index.equals(datas.index)


def test_merge_slice_metadata_timezone_handling(
    merge_loader, sample_merge_data
):
    """Test timezone handling"""
    datas, metadata_dict, dmd = sample_merge_data
    # Create metadata with no timezone
    metadata_dict[0]["Daily"].index = pd.date_range(
        "2020-01-01", periods=3, freq="D", tz=None
    )

    timestamp_col = datas.index
    metadata_seen_count = Counter()

    result_df, _ = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert result_df.index.tz == datas.index.tz


def test_merge_slice_metadata_duplicate_handling(
    merge_loader, sample_merge_data
):
    """Test handling of duplicate timestamps in metadata"""
    datas, metadata_dict, dmd = sample_merge_data
    # Create metadata with duplicate timestamps
    duplicate_index = pd.date_range("2020-01-01", periods=2, freq="D").repeat(2)
    metadata_dict[0]["Daily"] = pd.DataFrame(
        {"metadata_val": [1, 2, 3, 4]}, index=duplicate_index
    )

    timestamp_col = datas.index
    metadata_seen_count = Counter()

    result_df, _ = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert not result_df.index.duplicated().any().item()


def test_merge_slice_metadata_counter(merge_loader, sample_merge_data):
    """Test metadata column counter functionality"""
    datas, metadata_dict, dmd = sample_merge_data
    timestamp_col = datas.index
    metadata_seen_count = Counter()

    # Call merge twice to test counter
    merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )
    result_df, _ = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert "metadata_val_2" in result_df.columns


def test_merge_slice_metadata_time_range_clipping(merge_loader):
    """Test that metadata is properly clipped to data time range"""
    # Create data with specific time range
    datas = pd.DataFrame(
        {"value": range(3)},
        index=pd.date_range("2020-01-02", periods=3, freq="D", tz="UTC"),
    )

    # Create metadata with wider time range
    metadata_dict = (
        {
            "Daily": pd.DataFrame(
                {"metadata_val": range(5)},
                index=pd.date_range(
                    "2020-01-01", periods=5, freq="D", tz="UTC"
                ),
            )
        },
        {"columns": [{"column_id": "metadata_val"}], "frequency": "Daily"},
    )

    dmd = {"frequency": "Daily", "columns": []}
    timestamp_col = datas.index
    metadata_seen_count = Counter()

    result_df, _ = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert len(result_df) == len(datas)
    assert result_df.index.min() == datas.index.min()
    assert result_df.index.max() == datas.index.max()


def test_merge_slice_metadata_nearest_matching(merge_loader):
    """Test nearest timestamp matching"""
    # Create data with specific timestamps
    datas = pd.DataFrame(
        {"value": range(3)},
        index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"),
    )

    # Create metadata with offset timestamps
    metadata_dict = (
        {
            "Daily": pd.DataFrame(
                {"metadata_val": [10, 20, 30]},
                index=pd.date_range(
                    "2020-01-01 12:00:00", periods=3, freq="D", tz="UTC"
                ),
            )
        },
        {"columns": [{"column_id": "metadata_val"}], "frequency": "Daily"},
    )

    dmd = {"frequency": "Daily", "columns": []}
    timestamp_col = datas.index
    metadata_seen_count = Counter()

    result_df, _ = merge_loader._merge_slice_metadata(
        metadata_dict, datas, dmd, timestamp_col, metadata_seen_count
    )

    assert len(result_df) == len(datas)
    assert not result_df["metadata_val_1"].isna().any().item()


# pytest tests/test_merge_loader.py -v
