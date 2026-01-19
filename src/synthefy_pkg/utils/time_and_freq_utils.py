"""
Universalized frequency notation:
Basic Frequencies:
'T' or 'min' - Minutely
'H' or 'h' - Hourly
'D' - Daily
'W' - Weekly
'M' - Monthly (end of month)
'MS' - Monthly start
'Q' - Quarterly (end of quarter)
'QS' - Quarterly start
'Y' - Yearly (end of year)
'YS' - Yearly start
Business Frequencies:
'B' - Business day
'BM' - Business month end
'BMS' - Business month start
'BQ' - Business quarter end
'BQS' - Business quarter start
Sub-daily:
's' - Secondly
'ms' - Milliseconds
'us' - Microseconds
'ns' - Nanoseconds
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame
from numpy.typing import NDArray
from tabpfn_time_series import DefaultFeatures

COMPILE = True

# Frequency hierarchies
SYNTHEFY_FREQUENCY_HIERARCHY: list[str] = [
    "Hourly",
    "Daily",
    "Weekly",
    "Monthly",
    "Quarterly",
    "Yearly",
]

SYNTHEFY_FREQUENCY_HIERARCHY_BASE: list[str] = [
    "Hourly",
    "Daily",
    "Monthly",
    "Yearly",
]

# Legacy frequencies to seconds mapping
FREQUENCIES_TO_SECONDS: dict[str, int] = {
    "secondly": 1,
    "minutely": 60,
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
    "monthly": 2628000,
    "quarterly": 7884000,
    "yearly": 31536000,
}

# Frequency name to pandas frequency code mapping
FREQ_MAP: dict[str, str] = {
    "minutely": "T",
    "hourly": "H",
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def create_synthetic_date_range(
    frequency: str, periods: int, max_years_back: int = 10
) -> pd.DatetimeIndex:
    """
    Create a synthetic date range with random start time.

    Parameters
    ----------
    frequency : str
        The frequency string (e.g., 'hourly', 'daily', 'weekly')
    periods : int
        Number of periods in the date range
    max_years_back : int, default=10
        Maximum number of years in the past to sample end time from

    Returns
    -------
    pd.DatetimeIndex
        Generated date range
    """

    # Sample a random start time within the specified years back
    end_time = pd.Timestamp.now() - pd.Timedelta(
        seconds=np.random.randint(0, int(max_years_back * 365.25 * 24 * 3600))
    )
    if frequency in FREQ_MAP.values():
        pandas_freq = frequency
    else:
        pandas_freq = FREQ_MAP.get(
            frequency, "H"
        )  # Default to hourly if not found

    date_range = pd.date_range(end=end_time, periods=periods, freq=pandas_freq)

    return date_range


def convert_timestamp_to_features(
    date_range: Union[pd.DatetimeIndex, NDArray[np.datetime64]],
    device: Optional[Union[torch.device, str]] = None,
) -> tuple[torch.Tensor, list[str]]:
    """
    Convert a pandas DatetimeIndex or numpy datetime64 array to time feature tensor using TabPFN-TS features.

    Parameters
    ----------
    date_range : Union[pd.DatetimeIndex, np.ndarray]
        The date range to convert to features (can be DatetimeIndex or numpy datetime64 array)
    device : torch.device, optional
        Device to place the tensor on

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        A tuple containing:
        - Time features tensor of shape (seq_len, num_time_features)
        - List of column names for the time features
    """

    # Create DataFrame with timestamp and dummy columns
    df = pd.DataFrame({"timestamp": date_range})
    df["item_id"] = "0"  # dummy item_id
    df["target"] = 0  # dummy target values

    # Create TimeSeriesDataFrame and apply feature transformations
    timefeatures = TimeSeriesDataFrame(df)
    for func in [
        DefaultFeatures.add_calendar_features,
        DefaultFeatures.add_running_index,
    ]:
        timefeatures = timefeatures.groupby(
            level="item_id", group_keys=False
        ).apply(func)

    # Get column names before converting to tensor (excluding 'target')
    timefeatures_df = timefeatures.drop(columns=["target"])
    column_names = timefeatures_df.columns.tolist()

    # Convert to tensor
    timefeatures_tensor = torch.tensor(
        timefeatures_df.values, dtype=torch.float32, device=device
    )

    return timefeatures_tensor, column_names


def add_synthetic_time_features(
    X: torch.Tensor,
    d: list[torch.Tensor],
    available_frequencies: list[str],
    max_features: int,
    series_flags: torch.Tensor,
    frequencies: Optional[Union[list[str], np.ndarray]] = None,
    start_times: Any = None,  # TODO: NOT SUPPORTED YET
) -> tuple[
    torch.Tensor, list[torch.Tensor], int, list[str], list[str], list[int]
]:
    """
    Add synthetic time features to the input tensor a la TabPFN-TS.

    Creates calendar features and running index using timestamp utilities,
    then replaces the first k features of X with time features.

    Parameters
    ----------
    X : torch.Tensor
        Input features tensor of shape (batch_size, seq_len, max_features)
    d : List[torch.Tensor]
        List of tensors indicating number of actual features per dataset
    available_frequencies : List[str]
        List of available frequency strings to choose from
    max_features : int
        Maximum number of features allowed

    Returns
    -------
    tuple
        (X_with_time_features, updated_d_list)
    """

    # Choose random frequencies for each dataset in the batch
    if frequencies is None:
        frequencies = np.random.choice(
            available_frequencies, replace=True, size=X.shape[0]
        ).tolist()

    timefeatures_columns = []

    start_times = list()  # TODO: start times not supported yet
    assert isinstance(frequencies, list)
    for freq in frequencies:
        date_range = create_synthetic_date_range(freq, X.shape[1])
        timefeatures_tensor, timefeatures_column_names = (
            convert_timestamp_to_features(date_range, X.device)
        )
        timefeatures_columns.append(timefeatures_tensor)
        start_times.append(date_range[0])

    timefeatures_batch = torch.stack(timefeatures_columns)

    # ONLY if adding the time features will exceed max_features,
    # remove the last features to stay within limit
    # Roll X by the number of time features to make space at the beginning
    # Only roll non-tabular data
    num_time_features = timefeatures_batch.shape[-1]

    # Vectorized rolling and assignment for non-tabular data only
    X = X.clone()
    if series_flags.any():
        X[series_flags] = torch.roll(
            X[series_flags], num_time_features, dims=-1
        )
        X[series_flags, :, :num_time_features] = timefeatures_batch[
            series_flags
        ]

    # Update d to account for the additional time features
    updated_d = [
        torch.clamp(dv + num_time_features, 0, max_features)
        for sflag, dv in zip(series_flags, d)
        if sflag
    ]

    return (
        X,
        updated_d,
        num_time_features,
        timefeatures_column_names,
        frequencies,
        start_times,
    )
