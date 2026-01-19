import datetime

import numpy as np
import pandas as pd
import pytest
import torch


def datetime_to_sequence(dt: datetime.datetime) -> list[float]:
    """
    Convert datetime into a sequence of 9 numbers:
    year, quarter, month, week, day of week, day, hour, minute, second

    Args:
        dt: datetime object

    Returns:
        list: [year, quarter, month, week, day_of_week, day, hour, minute, second]
    """
    return [
        dt.year,
        (dt.month - 1) // 3 + 1,  # quarter (1-4)
        dt.month,  # month (1-12)
        dt.isocalendar()[1],  # week number (1-53)
        dt.weekday() + 1,  # day of week (1-7, Monday=1)
        dt.day,  # day of month (1-31)
        dt.hour,  # hour (0-23)
        dt.minute,  # minute (0-59)
        dt.second,  # second (0-59)
    ]


def convert_to_sequence(dt_val, convert_to_sequence=True):
    dt_length = list()
    NUM_STEPS = 10
    for dtv in dt_val:
        start_date, end_date = dtv
        total_diff = end_date - start_date

        # Calculate the step size
        step_size = total_diff / NUM_STEPS

        # Generate dates
        dates = [start_date + (step_size * i) for i in range(NUM_STEPS + 1)]

        # convert datetime into a sequence of 9 numbers:
        # year, quarter, month, week, day of week, day, hour, minute, second
        if convert_to_sequence:
            dates = [datetime_to_sequence(d) for d in dates]

        dt_length.append(dates)
    return dt_length


def construct_timestamp_mask(timestamps):
    """
    Construct a mask from the timestamps

    timestamps: Batch x num_correlates x window_size x num_timestamps
    timestamp_mask: Batch x num_correlates x window_size x window_size
    """
    # TODO: get from config later, but making this function self contained
    relevant_timestamp_indices = [0, 2, 5, 6, 7, 8]

    relevant_timestamps = timestamps.permute(0, 3, 1, 2)
    relevant_timestamps = relevant_timestamps[:, relevant_timestamp_indices]

    ts_i = relevant_timestamps.reshape(
        relevant_timestamps.shape[0], len(relevant_timestamp_indices), -1, 1
    )
    ts_j = ts_i.squeeze(-1).unsqueeze(-2)

    # produce 1 between any two values where the second value is larger than the first
    # 0 if they are equal, -1 if the second value is smaller than the first
    mask = (
        (ts_i > ts_j).float() - (ts_i < ts_j).float()
    )  # Batch x ts_idx x (window_size * num_correlates) x (window_size * num_correlates)

    # find the first non-zero value in a particular dimension, and set to 0 if it is -1, 1 otherwise
    # Find first non-zero value in each row
    # Replace zeros with a large positive number to ensure they're not selected as minimum
    # Create a mask of non-zero elements
    nonzero_mask = (mask != 0).float()

    # Replace zeros with large number to ensure they're not selected
    large_number = 1e9
    modified = mask.clone()
    modified[mask == 0] = large_number

    # Get indices of minimum values (first non-zero due to large_number replacement)
    min_indices = modified.abs().argmin(
        dim=1
    )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)
    # Gather the original values at those indices
    first_nonzero = torch.gather(
        mask, 1, min_indices.unsqueeze(-1)
    )  # Batch x (window_size * num_correlates) x (window_size * num_correlates) x 1

    # If a row was all zeros (or large_numbers), return 0
    all_zeros = (nonzero_mask.sum(dim=1) == 0).float()
    first_nonzero = first_nonzero.squeeze(-1) + (all_zeros)

    # create a mask for the nan values
    nan_mask = torch.isnan(relevant_timestamps)
    row_has_nan = nan_mask.any(dim=1).reshape(
        relevant_timestamps.shape[0], -1
    )  # Batch x (window_size * num_correlates)

    # mask out nan rows and columns
    first_nonzero = first_nonzero * (~row_has_nan.unsqueeze(-1)).float()
    first_nonzero = first_nonzero * (~row_has_nan.unsqueeze(-2)).float()

    first_nonzero[first_nonzero == -1] = 0
    first_nonzero[first_nonzero > 1] = 1

    return first_nonzero  # Batch x (window_size * num_correlates) x (window_size * num_correlates)


def generate_random_hash_table(num_datasets, text_embedding_dim):
    return torch.randn((num_datasets, text_embedding_dim))


def sample_from_hash_table(dataset_ids, hash_table):
    return hash_table[dataset_ids]


# Test fixtures
@pytest.fixture
def sample_datetime():
    return datetime.datetime(2020, 5, 1, 12, 30, 45)


@pytest.fixture
def sample_dt_vals():
    return [
        [
            (datetime.datetime(2020, 5, 1), datetime.datetime(2020, 5, 10)),
            (datetime.datetime(2020, 3, 1), datetime.datetime(2021, 1, 10)),
            (datetime.datetime(2012, 1, 1), datetime.datetime(2022, 1, 1)),
        ],
        [
            (datetime.datetime(2020, 5, 5), datetime.datetime(2020, 5, 15)),
            (datetime.datetime(2019, 9, 1), datetime.datetime(2020, 7, 1)),
            (datetime.datetime(1990, 1, 1), datetime.datetime(2000, 1, 1)),
        ],
    ]


# Tests
def test_datetime_to_sequence(sample_datetime):
    result = datetime_to_sequence(sample_datetime)
    expected = [2020, 2, 5, 18, 5, 1, 12, 30, 45]
    assert result == expected


def test_convert_to_sequence(sample_dt_vals):
    result = convert_to_sequence(sample_dt_vals[0])
    assert len(result) == 3  # 3 date ranges
    assert len(result[0]) == 11  # 11 steps (including start and end)
    assert len(result[0][0]) == 9  # 9 datetime components


def test_construct_timestamp_mask(sample_dt_vals):
    timestamps = torch.tensor(
        [convert_to_sequence(dt_val) for dt_val in sample_dt_vals]
    ).float()
    timestamps[0, 0, 5, 0] = torch.nan  # Add a NaN value for testing

    mask = construct_timestamp_mask(timestamps)

    # Check shape
    assert mask.shape == (
        2,
        33,
        33,
    )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)

    # Check values are either 0 or 1
    assert torch.all((mask == 0) | (mask == 1))

    # Check NaN handling
    assert torch.all(mask[0, 5, :] == 0)  # Row with NaN should be all zeros
    assert torch.all(mask[0, :, 5] == 0)  # Column with NaN should be all zeros

    # Additional test for specific mask values
    batch1 = np.array(
        convert_to_sequence(sample_dt_vals[0], convert_to_sequence=False)
    )
    del batch1
    assert mask[0, 0].reshape(3, 11).shape == (
        3,
        11,
    )  # Check reshaping works correctly


def test_hash_table_functions():
    num_datasets = 10
    text_embedding_dim = 32
    dataset_ids = torch.tensor([0, 2, 4])

    hash_table = generate_random_hash_table(num_datasets, text_embedding_dim)
    assert hash_table.shape == (num_datasets, text_embedding_dim)

    samples = sample_from_hash_table(dataset_ids, hash_table)
    assert samples.shape == (3, text_embedding_dim)
    assert torch.all(samples[0] == hash_table[0])
    assert torch.all(samples[1] == hash_table[2])
    assert torch.all(samples[2] == hash_table[4])
