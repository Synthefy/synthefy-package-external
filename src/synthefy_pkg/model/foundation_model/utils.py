import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange, reduce, repeat
from loguru import logger
from omegaconf import ListConfig

NULL_TOKEN = -999999

# Default lags to generate for covariate features (t-1, t-2, t-3)
DEFAULT_COVARIATE_LAGS: List[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    36,
    40,
]

SKIP_COVARIATES = [
    "running_index",
    # "target",
    "year",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month_sin",
    "day_of_month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "month_of_year_sin",
    "month_of_year_cos",
]


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def continuous_to_token(
    continuous_input: torch.Tensor, continuous_mask: torch.Tensor
) -> torch.Tensor:
    """
    Convert continuous values to tokenized representation.
    Each continuous value is represented by 9 elements:
    - 1 element for sign (0 = negative, 1 = positive)
    - 4 elements for whole number digits
    - 4 elements for decimal digits

    If a value is equal to the NULL_TOKEN, all 9 elements will be set to -1.

    Args:
        continuous_input: Input tensor of continuous values
        continuous_mask: Mask tensor of shape [..., 9] with 1s for non-null tokens and 0s for null tokens

    Returns:
        Tokenized representation with shape [..., 9] for each input value
    """

    # Create output tensor with extra dimension
    orig_shape = continuous_input.shape
    output_shape = orig_shape + (9,)
    tokens = torch.zeros(output_shape, device=continuous_input.device)

    # Create mask for null tokens
    null_mask = continuous_mask

    # Process non-null values
    valid_values = continuous_input.clone()

    # Get sign (1 for positive, 0 for negative)
    sign = (valid_values >= 0).float()

    # Get absolute values
    abs_values = torch.abs(valid_values)

    # Extract whole part and decimal part
    whole_part = torch.floor(abs_values)
    decimal_part = abs_values - whole_part

    # Convert whole part to 4 digits
    thousands = torch.fmod(torch.floor(whole_part / 1000), 10)
    hundreds = torch.fmod(torch.floor(whole_part / 100), 10)
    tens = torch.fmod(torch.floor(whole_part / 10), 10)
    ones = torch.fmod(whole_part, 10)

    # Convert decimal part to 4 digits
    tenths = torch.fmod(torch.floor(decimal_part * 10), 10)
    hundredths = torch.fmod(torch.floor(decimal_part * 100), 10)
    thousandths = torch.fmod(torch.floor(decimal_part * 1000), 10)
    ten_thousandths = torch.fmod(torch.floor(decimal_part * 10000), 10)

    # Fill in the tokens tensor
    tokens[..., 0] = sign
    tokens[..., 1] = thousands
    tokens[..., 2] = hundreds
    tokens[..., 3] = tens
    tokens[..., 4] = ones
    tokens[..., 5] = tenths
    tokens[..., 6] = hundredths
    tokens[..., 7] = thousandths
    tokens[..., 8] = ten_thousandths

    # remove the extra dimension that comes from the continuous values
    tokens = tokens.squeeze(-2)

    # Set all token elements to -1 for null tokens
    if null_mask.any():
        # Expand null_mask to match token dimensions
        expanded_null_mask = null_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(
            expanded_null_mask,
            torch.tensor(-1.0, device=tokens.device),
            tokens,
        )

    return tokens


def obtain_timestamps_in_seconds(timestamps):
    # timestamps = timestamps.int()
    relevant_timestamp_indices = [
        0,
        2,
        5,
        6,
        7,
        8,
    ]  # year, month, day, hour, minute, second
    offsets = [12, 31, 24, 60, 60, 1]
    prod = []
    for i in range(0, len(offsets)):
        prod.append(np.prod(offsets[i:]))
    multiplies = torch.tensor(prod).to(timestamps.device).float()
    multiplies = multiplies.unsqueeze(0).unsqueeze(
        0
    )  # 1 X 1 X NUM_TIMESTAMP_FEATURES

    relevant_timestamps = timestamps[..., relevant_timestamp_indices]
    timestamps_in_seconds = torch.sum(relevant_timestamps * multiplies, dim=-1)
    return timestamps_in_seconds


def convert_timestamp_in_seconds_to_readable(timestamps):
    """
    converts the timestamps from seconds to a readable format
    Reverses the encoding done in obtain_timestamps_in_seconds

    Args:
        timestamps: scalar (int/float) or tensor of shape [...] containing encoded timestamp values

    Returns:
        If input is scalar: single datetime.datetime object
        If input is tensor: list of datetime.datetime objects preserving the original shape
    """
    # These are the same offsets and multipliers used in obtain_timestamps_in_seconds
    offsets = [12, 31, 24, 60, 60, 1]
    prod = []
    for i in range(0, len(offsets)):
        prod.append(np.prod(offsets[i:]))

    def _convert_single_timestamp(ts_val):
        """Helper function to convert a single timestamp value"""
        # Extract components by integer division and modulo
        year = ts_val // prod[0]  # 32140800
        remaining = ts_val % prod[0]

        month = remaining // prod[1]  # 2678400
        remaining = remaining % prod[1]

        day = remaining // prod[2]  # 86400
        remaining = remaining % prod[2]

        hour = remaining // prod[3]  # 3600
        remaining = remaining % prod[3]

        minute = remaining // prod[4]  # 60
        second = remaining % prod[4]

        # Convert to integers and add bounds checking
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        second = int(second)

        # Add bounds checking to ensure valid datetime values
        year = max(1, min(9999, year))  # datetime.datetime year range
        month = max(1, min(12, month))  # Valid month range

        # Handle day bounds based on month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            max_day = 31
        elif month in [4, 6, 9, 11]:
            max_day = 30
        else:  # February
            # Simple leap year check
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                max_day = 29
            else:
                max_day = 28

        day = max(1, min(max_day, day))
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        second = max(0, min(59, second))

        try:
            return datetime.datetime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
            )
        except ValueError:
            # If still invalid, return a default datetime with debug info
            logger.warning(
                f"Warning: Invalid datetime components - year:{year}, month:{month}, day:{day}, hour:{hour}, minute:{minute}, second:{second}"
            )
            logger.warning(f"Original timestamp value: {ts_val}")
            # Return a safe default
            return datetime.datetime(2000, 1, 1, 0, 0, 0)

    # Check if input is scalar (Python number or 0-dimensional tensor)
    if isinstance(timestamps, (int, float)):
        return _convert_single_timestamp(timestamps)
    elif isinstance(timestamps, torch.Tensor) and timestamps.dim() == 0:
        return _convert_single_timestamp(timestamps.item())
    else:
        # Handle tensor input - preserve shape
        if isinstance(timestamps, torch.Tensor):
            original_shape = timestamps.shape
            flat_timestamps = timestamps.flatten()

            results = []
            for ts in flat_timestamps:
                results.append(_convert_single_timestamp(ts.item()))

            # Reshape results to match original shape
            def reshape_list(flat_list, shape):
                if len(shape) == 0:
                    return flat_list[0]
                elif len(shape) == 1:
                    return flat_list
                else:
                    size = shape[0]
                    remainder = shape[1:]
                    chunk_size = len(flat_list) // size
                    return [
                        reshape_list(
                            flat_list[i * chunk_size : (i + 1) * chunk_size],
                            remainder,
                        )
                        for i in range(size)
                    ]

            return reshape_list(results, original_shape)
        else:
            # Handle other array-like inputs
            try:
                # Try to convert to tensor first
                ts_tensor = torch.tensor(timestamps)
                return _convert_single_timestamp(ts_tensor.item())
            except Exception as e:
                raise ValueError(
                    f"Unsupported input type:{e} {type(timestamps)}"
                )


def apply_invalid_target_masks(attn_mask, invalid_mask, target_mask):
    """
    combines an attention mask with invalid and target masks, then ensures self attention
    is still allowed

    attn_mask: Batch x (num_correlates x window_size) x (num_correlates x window_size)
    invalid_mask: Batch x (num_correlates x window_size)
    target_mask: Batch x (num_correlates x window_size)

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
    """
    attn_mask = torch.logical_or(attn_mask, invalid_mask.unsqueeze(-1))
    attn_mask = torch.logical_or(attn_mask, invalid_mask.unsqueeze(-2))

    # applying the target mask (B, 1, NT)
    attn_mask = torch.logical_or(attn_mask, target_mask.unsqueeze(-2))

    # applying self attention mask
    attn_mask = torch.logical_and(
        attn_mask,
        (1 - torch.eye(attn_mask.shape[-1]))
        .unsqueeze(0)
        .bool()
        .repeat(attn_mask.shape[0], 1, 1)
        .to(attn_mask.device),
    )

    return attn_mask


def obtain_causal_attn_mask(
    timestamps, invalid_mask, target_mask, present_leak=False
):
    """
    Construct an attention mask from the timestamps

    timestamps: Batch x (num_correlates x window_size) x num_timestamps
    timestamp_mask: Batch x (num_correlates x window_size)

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
        with 0s for valid tokens and -inf for invalid tokens
    """
    # TODO: get from config later, but making this function self contained
    timestamps_in_seconds = obtain_timestamps_in_seconds(timestamps)
    ts_i = timestamps_in_seconds.unsqueeze(
        -1
    )  # B X NUM_CORRELATES * TIME_SERIES_LENGTH X 1
    ts_j = timestamps_in_seconds.unsqueeze(
        -2
    )  # B X 1 X NUM_CORRELATES * TIME_SERIES_LENGTH

    # produce 1 between any two values where the second value is larger than the first
    # 0 if they are equal, -1 if the second value is smaller than the first
    if present_leak:
        # if the timestamp is the same, allows attention
        causal_attn_mask = (
            ts_i < ts_j
        )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)
    else:
        causal_attn_mask = (
            ts_i <= ts_j
        )  # Batch x (window_size * num_correlates) x (window_size * num_correlates)

    # applying the invalid mask
    causal_attn_mask = apply_invalid_target_masks(
        causal_attn_mask, invalid_mask, target_mask
    )

    causal_attn_mask = causal_attn_mask.float()

    causal_attn_mask = causal_attn_mask.unsqueeze(
        1
    )  # Batch x 1 x (window_size * num_correlates) x (window_size * num_correlates)

    large_negative_number = -1e15
    causal_attn_mask = causal_attn_mask * large_negative_number

    return causal_attn_mask


def obtain_future_triangular_attn_mask(
    batch_size,
    num_correlates,
    time_series_length,
    invalid_mask,
    target_mask,
    device,
    dtype,
):
    """
    everything is unmasked except for future values from the same correlate

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
        with 0s for valid tokens and -inf for invalid tokens
    """
    # Create a mask of ones for the upper triangular part of each correlate's time series
    indices = torch.arange(time_series_length, device=device)
    upper_tri = indices.unsqueeze(0) > indices.unsqueeze(
        1
    )  # Creates a time_series_length x time_series_length boolean mask
    upper_tri = upper_tri.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and correlate dimensions
    upper_tri = upper_tri.expand(
        batch_size, num_correlates, time_series_length, time_series_length
    )
    upper_tri = repeat(
        upper_tri.unsqueeze(-2),
        "b nc t1 1 t2 -> b nc t1 c t2",
        nc=num_correlates,
        c=num_correlates,
        t1=time_series_length,
        t2=time_series_length,
    )
    complete_mask = upper_tri.float()
    complete_mask = rearrange(complete_mask, "b nc t1 c t2 -> b (nc t1) (c t2)")
    complete_mask = apply_invalid_target_masks(
        complete_mask, invalid_mask, target_mask
    )
    complete_mask = complete_mask.float().unsqueeze(1)
    large_negative_number = get_large_negative_number(dtype)
    complete_mask = complete_mask * large_negative_number

    return complete_mask.to(device)


def obtain_interpolation_attn_mask(
    batch_size,
    num_correlates,
    time_series_length,
    invalid_mask,
    target_mask,
    device,
    dtype,
):
    """
    everything is unmasked except for invalid and target tokens

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
        with 0s for valid tokens and -inf for invalid tokens
    """
    complete_mask = (
        torch.zeros(
            batch_size,
            num_correlates * time_series_length,
            num_correlates * time_series_length,
        )
        .to(device)
        .bool()
    )
    complete_mask = apply_invalid_target_masks(
        complete_mask, invalid_mask, target_mask
    )
    complete_mask = complete_mask.float().unsqueeze(1)
    large_negative_number = get_large_negative_number(dtype)
    complete_mask = complete_mask * large_negative_number
    return complete_mask.to(device)


def obtain_future_correlate_attn_mask(
    batch_size,
    num_correlates,
    time_series_length,
    invalid_mask,
    target_mask,
    device,
    dtype,
):
    """
    everything is unmasked except for invalid and target tokens, and future tokens from the same correlate

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
        with 0s for valid tokens and -inf for invalid tokens
    """
    complete_mask = (
        torch.zeros(
            batch_size,
            num_correlates,
            time_series_length,
            num_correlates,
            time_series_length,
        )
        .to(device)
        .bool()
    )

    # Vectorized replacement for the nested loops
    # Create indices for time steps
    time_indices = torch.arange(time_series_length, device=device)

    # Create upper triangular mask for future times: j < k means position (j,k) is future
    future_mask = time_indices.unsqueeze(0) > time_indices.unsqueeze(
        1
    )  # [time_series_length, time_series_length]

    # Create diagonal mask for same correlates
    correlate_indices = torch.arange(num_correlates, device=device)
    same_correlate_mask = correlate_indices.unsqueeze(
        0
    ) == correlate_indices.unsqueeze(1)  # [num_correlates, num_correlates]

    # Combine masks: mask future times within same correlate
    # Expand dimensions to match complete_mask shape
    combined_mask = same_correlate_mask.unsqueeze(-1).unsqueeze(
        -1
    ) & future_mask.unsqueeze(0).unsqueeze(0)
    # Shape: [num_correlates, num_correlates, time_series_length, time_series_length]
    complete_mask = rearrange(
        combined_mask.unsqueeze(0), "1 nc c t1 t2 -> 1 (nc t1) (c t2)"
    )
    complete_mask = repeat(
        complete_mask, "1 nct1 ct2 -> b nct1 ct2", b=batch_size
    )
    complete_mask = apply_invalid_target_masks(
        complete_mask, invalid_mask, target_mask
    )
    complete_mask = complete_mask.float().unsqueeze(1)
    large_negative_number = get_large_negative_number(dtype)
    complete_mask = complete_mask * large_negative_number
    return complete_mask.to(device)


def obtain_row_attn_mask(
    batch_size,
    num_correlates,
    time_series_length,
    invalid_mask,
    target_mask,
    device,
    dtype,
):
    """
    attention only on the rows of the target mask (across correlates, but same time)

    returns: Batch x (num_correlates x window_size) x (num_correlates x window_size)
        with 0s for valid tokens and -inf for invalid tokens
    """
    # Create a mask where each position can only attend to positions at the same time step
    # across different correlates
    time_indices = torch.arange(time_series_length, device=device)
    time_mask = time_indices.unsqueeze(0) == time_indices.unsqueeze(
        1
    )  # time_series_length x time_series_length
    time_mask = (
        time_mask.unsqueeze(1).unsqueeze(0).unsqueeze(0)
    )  # Add batch and correlate dimensions
    time_mask = time_mask.expand(
        batch_size,
        num_correlates,
        time_series_length,
        num_correlates,
        time_series_length,
    )
    empty_mask = (
        ~time_mask
    )  # Invert the mask to get 0s where we want attention and 1s where we don't

    empty_mask = rearrange(empty_mask, "b nc t1 c t2 -> b (nc t1) (c t2)")
    empty_mask = apply_invalid_target_masks(
        empty_mask, invalid_mask, target_mask
    )
    empty_mask = empty_mask.float().unsqueeze(1)
    large_negative_number = get_large_negative_number(dtype)
    empty_mask = empty_mask * large_negative_number
    return empty_mask.to(device)


def obtain_column_attn_mask(
    batch_size,
    num_correlates,
    time_series_length,
    invalid_mask,
    target_mask,
    device,
    dtype,
):
    """
    attention only on the columns of the target mask (across time, but same correlate)
    """
    time_indices = torch.arange(num_correlates, device=device)
    time_mask = time_indices.unsqueeze(0) == time_indices.unsqueeze(
        1
    )  # num_correlates x num_correlates
    time_mask = (
        time_mask.unsqueeze(-1).unsqueeze(1).unsqueeze(0)
    )  # Add batch and time dimensions
    time_mask = time_mask.expand(
        batch_size,
        num_correlates,
        time_series_length,
        num_correlates,
        time_series_length,
    )
    empty_mask = (
        ~time_mask
    )  # Invert the mask to get 0s where we want attention and 1s where we don't

    empty_mask = rearrange(empty_mask, "b nc t1 c t2 -> b (nc t1) (c t2)")
    empty_mask = apply_invalid_target_masks(
        empty_mask, invalid_mask, target_mask
    )
    empty_mask = empty_mask.float().unsqueeze(1)
    large_negative_number = get_large_negative_number(dtype)
    empty_mask = empty_mask * large_negative_number
    return empty_mask.to(device)


def obtain_train_test_block_attn_mask(
    train_sizes,
    mask,
    time_series_length,
    num_correlates,
    batch_size,
    device,
    dtype,
):
    """
    train_sizes_expanded: B
    mask: B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
    time_series_length: int
    num_correlates: int
    batch_size: int
    device: torch.device
    dtype: torch.dtype

    return B X (NUM_CORRELATES X TIME_SERIES_LENGTH) X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        with -inf only for test-test tokens in different rows, 0 elsewhere
    """

    # these set of operations ensure that every train token can attend to every other train token
    # they also ensure that every test token (be it target or feature) can attend to every other train token
    # now, we will ensure that the test feature and target token can attend only to the corresponding feature and target train tokens
    # TODO: replace the repeats with view
    train_sizes_expanded = train_sizes.unsqueeze(-1).unsqueeze(-1)
    time_indices = torch.arange(time_series_length).to(device)
    train_mask = (
        time_indices.unsqueeze(0).unsqueeze(0) >= train_sizes_expanded
    ).expand(batch_size, num_correlates, time_series_length)
    train_mask = rearrange(train_mask, "b nc t -> b (nc t)")
    train_mask = torch.logical_and(train_mask, mask)
    train_mask = train_mask.unsqueeze(-1).expand(
        batch_size,
        num_correlates * time_series_length,
        num_correlates * time_series_length,
    )
    train_mask = rearrange(
        train_mask,
        "b (nc t1) (c t2) -> b nc t1 c t2",
        b=batch_size,
        nc=num_correlates,
        t1=time_series_length,
        c=num_correlates,
        t2=time_series_length,
    )

    # Vectorized replacement for the nested loops
    # Create boolean mask: True where t < train_sizes[b] (reuse time_indices and train_sizes)
    mask_condition = time_indices.unsqueeze(0) < train_sizes.unsqueeze(-1)

    # Expand the mask to match train_mask's last dimension
    # We want to zero out train_mask[b, :, :, :, t] where mask_condition[b, t] is True
    expanded_mask = (
        mask_condition.unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .expand(
            batch_size,
            num_correlates,
            time_series_length,
            num_correlates,
            time_series_length,
        )
    )

    # Apply the mask: set to 0 where condition is True
    train_mask = torch.where(
        expanded_mask,
        torch.tensor(0, device=device, dtype=train_mask.dtype),
        train_mask,
    )

    train_mask = rearrange(train_mask, "b nc t1 c t2 -> b (nc t1) (c t2)")

    self_attn_mask = torch.eye(time_series_length).to(device).bool()
    self_attn_mask = repeat(
        self_attn_mask,
        "t1 t2 -> (nc t1) t2",
        nc=num_correlates,
        t1=time_series_length,
        t2=time_series_length,
    )
    self_attn_mask = repeat(
        self_attn_mask,
        "(nc1 t1) t2 -> (nc1 t1) (nc2 t2)",
        nc1=num_correlates,
        t1=time_series_length,
        t2=time_series_length,
        nc2=num_correlates,
    )
    self_attn_mask = repeat(
        self_attn_mask,
        "(nc1 t1) (nc2 t2) -> b (nc1 t1) (nc2 t2)",
        b=batch_size,
        nc1=num_correlates,
        t1=time_series_length,
        nc2=num_correlates,
        t2=time_series_length,
    )
    self_attn_mask = torch.logical_not(self_attn_mask)

    # we now ensure that the test feature and target token can attend only to the corresponding feature and target train tokens
    attn_mask = torch.logical_and(train_mask, self_attn_mask)
    # torch.set_printoptions(threshold=10000)
    # print(attn_mask.int(), rearrange(attn_mask.int(), "b (nc t1) (c t2) -> b nc t1 c t2", b=batch_size, nc=num_correlates, t1=time_series_length, c=num_correlates, t2=time_series_length))

    attn_mask = attn_mask.unsqueeze(
        1
    ).float()  # Batch x 1 x (window_size * num_correlates) x (window_size * num_correlates)

    large_negative_number = get_large_negative_number(dtype)
    attn_mask = attn_mask * large_negative_number
    return attn_mask.to(device)


def obtain_mask(elem, category=None):
    """
    gets the mask for invalid tokens based on the elem tensor equal to the NULL_TOKEN, and nan values
    """
    nan_mask = torch.isnan(elem)
    null_mask = elem == NULL_TOKEN
    mask = torch.logical_or(nan_mask, null_mask)
    mask_einops = reduce(mask, "b nc t f -> b nc t", "sum").bool()
    mask = mask.sum(dim=-1).bool()
    assert torch.equal(mask, mask_einops), "The mask is not equal"
    if category == "textual":
        invalid_mask = (elem == 0).all(dim=-1)
        mask = torch.logical_or(mask, invalid_mask)
    return mask


def generate_target_mask(
    mask_mixing_rates: list[float],
    target_masking_schemes: list[str],
    target_filtering_schemes: list[str],
    block_target_mask_mean: int,
    block_target_mask_range: int,
    block_mask_num: int,
    block_mask_every: bool,
    row_mask_ratio: float,
    row_mask_min: int,
    row_use_train_test: bool,
    target_mask_ratio: float,
    time_series_length: int,
    batch_size: int,
    num_correlates: int,
    device: Union[torch.device, str],
    train_sizes: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    time_columns: int = -1,
    num_actual_correlates: Optional[torch.Tensor] = None,
):
    """generates a target mask, that is, which values to predict.
    The availiable target masks are:
    - block: picks several random column and masks out a random length sequence of values
        special case: if block_target_mask_mean == time_series_length / 2 and block_target_mask_range == 0 and block_mask_num == 1, then we mask out the future half of the time series
    - row: masks out a random number of rows, excluding the first row_mask_min rows
    - train_test: masks out the test tokens (bottom test last column, returned as a grid)
    - train_test_block: masks out the test tokens in a block (bottom num_test rows)
    - train_test_last: masks out the test tokens in the last column (bottom test last column, but returned as a sequence)
    - train_test_block_filter: prevents masking of the first n train rows (top num_train rows)
    - time_indices_filter: prevents masking of the time indicies (time_columns)
    - random: masks out a random number of values (random_mask_ratio)
    - random_last: masks out a random number of values in the last column (random_mask_ratio)
    Parameters:
    num_correlates: int, the total number of correlates in the batch (max capacity of model)
    num_actual_correlates: int, the number of correlates in the batch (actual number of useful columns in each table in the batch)
    Returns:
    - A target mask tensor of shape (B X (NUM_CORRELATES X TIME_SERIES_LENGTH)) or (B X TIME_SERIES_LENGTH) for last-only masks.
          Masking strategy: 1 implies position is masked (value should be predicted)
    - a dictionary of applied masks
    """

    target_masks = list()
    filter_masks = list()

    # Convert ListConfig to list if needed
    if isinstance(mask_mixing_rates, ListConfig):
        mask_mixing_rates = list(mask_mixing_rates)

    if len(mask_mixing_rates) == 1 and len(target_masking_schemes) > 1:
        mask_mixing_rates = list(mask_mixing_rates) * len(
            target_masking_schemes
        )
    assert len(target_masking_schemes) == len(mask_mixing_rates), (
        "target_masking_schemes and mask_mixing_rates must have the same length"
    )

    # make sure at least one masking scheme is used
    masks_to_apply_dict = {
        scheme: np.random.rand() < rate
        for scheme, rate in zip(target_masking_schemes, mask_mixing_rates)
    }
    give_up_counter = 0
    while (
        np.sum(list(masks_to_apply_dict.values())) == 0 and give_up_counter < 10
    ):
        masks_to_apply_dict = {
            scheme: np.random.rand() < rate
            for scheme, rate in zip(target_masking_schemes, mask_mixing_rates)
        }
        give_up_counter += 1

    if give_up_counter == 10:
        most_likely_mask = np.argmax(mask_mixing_rates)
        # logger.trace(
        #     f"giving up, choosing {target_masking_schemes[most_likely_mask]} with rate {mask_mixing_rates[most_likely_mask]}"
        # )
        masks_to_apply_dict[target_masking_schemes[most_likely_mask]] = True

    assert sum(masks_to_apply_dict.values()) > 0, (
        "No mask mixing rate is set to True"
    )

    if "block" in target_masking_schemes and masks_to_apply_dict["block"]:
        # if self.block_target_mask_mean > 0, then we mask blocks of a certain length
        #     If the range is 0 and the number of mask is length / 2 and the number of masks is num_correlates, mask out the future
        #     otherwise, select block_mask_num random blocks of length self.block_target_mask_mean +- self.block_target_mask_range
        # if self.block_target_mask_mean <= 0, then we mask a random number of values between 0 and self.target_mask_ratio
        assert block_target_mask_mean > 0, (
            "block_target_mask_mean must be provided for block_masking_scheme"
        )
        if (
            block_target_mask_mean == time_series_length / 2
            and block_target_mask_range == 0
            and block_mask_num == num_correlates
        ):
            # this is the case where we mask the future half of the time series
            target_mask = torch.zeros(
                batch_size,
                num_correlates,
                time_series_length,
                device=device,
            )
            target_mask[..., time_series_length // 2 :] = 1
            target_mask = target_mask.reshape(
                batch_size,
                num_correlates * time_series_length,
            )
            target_masks.append(target_mask)
        else:
            # randomly decide on lengths for the block masks
            if block_target_mask_range == 0:
                target_mask_lengths = torch.full(
                    (batch_size, block_mask_num),
                    block_target_mask_mean,
                    device=device,
                )
            else:
                target_mask_lengths = torch.randint(
                    max(0, block_target_mask_mean - block_target_mask_range),
                    min(
                        time_series_length,
                        block_target_mask_mean + block_target_mask_range,
                    ),
                    (batch_size, block_mask_num),
                    device=device,
                )  # b, bmn

            # Ensure start position + mask length doesn't exceed time_series_length
            max_starts = torch.clamp(
                time_series_length - target_mask_lengths,
                min=0,
            )

            # Generate random starting positions for each mask (in time)
            random_starts = torch.floor(
                torch.rand(batch_size, block_mask_num, device=device)
                * (max_starts + 1)
            ).long()  # b, bmn

            if num_actual_correlates is None:
                num_actual_correlates = (
                    torch.ones(batch_size, device=device) * num_correlates
                )

            if block_mask_every:
                assert block_mask_num == num_correlates, (
                    "block_mask_every requires block_mask_num == num_correlates"
                )
                # Put a mask in every column. Padded correlates and time columns are handled later.
                chosen_column_indices = repeat(
                    torch.arange(num_correlates, device=device),
                    "bmn -> b bmn",
                    b=batch_size,
                )  # view, not clone
            else:
                # Pick the columns which will contain the masks: repeats allowed
                chosen_column_indices = (
                    torch.rand(batch_size, block_mask_num, device=device)
                    * (num_actual_correlates.unsqueeze(-1) - time_columns)
                    + time_columns
                ).floor()  # b, bmn

            # Fill in
            target_mask = torch.zeros(
                batch_size, num_correlates, time_series_length, device=device
            )

            # Create time offsets for masking contiguous regions
            max_length = target_mask_lengths.max().item()
            time_offsets = (
                torch.arange(max_length, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, max_length)

            # Compute all time indices that should be masked
            time_indices = (
                random_starts.unsqueeze(-1) + time_offsets
            )  # (batch_size, block_mask_num, max_length)

            # Create mask for valid time indices (within the specified length for each block)
            valid_mask = time_offsets < target_mask_lengths.unsqueeze(
                -1
            )  # (batch_size, block_mask_num, max_length)

            # Also ensure time indices don't exceed time_series_length
            time_valid_mask = time_indices < time_series_length
            valid_mask = valid_mask & time_valid_mask

            # Get the valid indices (unpacked into batch, mask, time)
            valid_positions = torch.where(valid_mask)
            batch_idx = valid_positions[0]  # batch dimension
            mask_idx = valid_positions[1]  # block_mask_num dimension
            time_offset_idx = valid_positions[2]  # time offset dimension

            # Get the corresponding batch, correlate, and time indices
            final_batch_indices = batch_idx
            final_correlate_indices = chosen_column_indices[
                batch_idx, mask_idx
            ].long()
            final_time_indices = time_indices[
                batch_idx, mask_idx, time_offset_idx
            ]

            # Set the mask values to 1
            target_mask[
                final_batch_indices, final_correlate_indices, final_time_indices
            ] = 1
            # Reshape to match expected output shape
            target_mask = target_mask.reshape(
                batch_size,
                num_correlates * time_series_length,
            )
            target_mask = target_mask.to(device)
            target_masks.append(target_mask)

    if "row" in target_masking_schemes and masks_to_apply_dict["row"]:
        assert row_mask_ratio > 0, (
            "row_mask_ratio must be provided for row_masking_scheme"
        )
        target_mask = torch.zeros(
            batch_size, num_correlates, time_series_length
        ).to(device)

        # select row_mask_ratio random rows from each batch
        # Generate random row indices for each batch
        if row_use_train_test:
            assert train_sizes is not None, (
                "train_sizes must be provided for row_use_train_test"
            )
            max_rows = min(int(train_sizes.min().item()), time_series_length)
        else:
            max_rows = time_series_length
        num_rows_to_mask = int(row_mask_ratio * time_series_length)
        # Ensure rows_to_mask doesn't exceed max_rows
        num_rows_to_mask = min(num_rows_to_mask, max_rows)

        # Generate unique random row indices for each batch
        # Use torch.randperm to ensure unique indices
        random_rows = torch.stack(
            [
                torch.randperm(max_rows, device=device)[:num_rows_to_mask]
                for _ in range(batch_size)
            ]
        )

        # Vectorized approach: use advanced indexing to set selected rows to 1
        # Create batch indices for advanced indexing
        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, num_rows_to_mask)
        )

        # Flatten indices for advanced indexing
        batch_indices_flat = batch_indices.flatten()
        random_rows_flat = random_rows.flatten()

        unmask_rows_num = max(row_mask_min, time_columns)

        # Set all correlates for the selected rows to 1
        target_mask[batch_indices_flat, unmask_rows_num:, random_rows_flat] = 1

        target_mask = target_mask.to(device)
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")
        target_masks.append(target_mask)
    if "time_indices_filter" in target_filtering_schemes:
        unmask_rows_num = max(row_mask_min, time_columns)
        assert unmask_rows_num >= 0, (
            "unmask_rows_num must be provided for time_indices_filter"
        )
        target_mask = torch.ones(
            batch_size, num_correlates, time_series_length
        ).to(device)
        target_mask[:, :unmask_rows_num, :] = 0
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")
        filter_masks.append(target_mask.clone())

    if (
        "train_test" in target_masking_schemes
        and masks_to_apply_dict["train_test"]
    ):
        assert train_sizes is not None and seq_lens is not None, (
            "train_sizes and seq_lens must be provided for train_test_masking_scheme"
        )
        time_indices = torch.arange(time_series_length).to(device)
        use_seq_lens = seq_lens.squeeze(-1)
        use_train_sizes = train_sizes.squeeze(-1)
        target_mask = torch.logical_and(
            (time_indices.unsqueeze(0) >= use_train_sizes.unsqueeze(-1)),
            (time_indices.unsqueeze(0) < use_seq_lens.unsqueeze(-1)),
        )
        target_mask = torch.concatenate(
            [
                torch.zeros(
                    (batch_size, num_correlates - 1, time_series_length)
                )
                .bool()
                .to(device),
                target_mask.unsqueeze(1),  # target mask at the last dim
            ],
            dim=1,
        )  # concatenate to the correlates
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")
        target_masks.append(target_mask)
    if (
        "train_test_block" in target_masking_schemes
        and masks_to_apply_dict["train_test_block"]
    ) or ("train_test_block_filter" in target_filtering_schemes):
        assert train_sizes is not None and seq_lens is not None, (
            "train_sizes and seq_lens must be provided for train_test_masking_scheme"
        )
        time_indices = torch.arange(time_series_length).to(device)
        use_seq_lens = seq_lens.squeeze(-1)
        use_train_sizes = train_sizes.squeeze(-1)
        target_mask = torch.logical_and(
            (time_indices.unsqueeze(0) >= use_train_sizes.unsqueeze(-1)),
            (time_indices.unsqueeze(0) < use_seq_lens.unsqueeze(-1)),
        )
        unmask_rows_num = max(row_mask_min, time_columns)
        target_mask = repeat(
            target_mask, "b t -> b nc t", nc=num_correlates - unmask_rows_num
        )
        target_mask = torch.concatenate(
            [
                torch.zeros((batch_size, unmask_rows_num, time_series_length))
                .bool()
                .to(device),
                target_mask,  # target mask at the last dim
            ],
            dim=1,
        )  # concatenate to the correlates
        target_mask = rearrange(target_mask, "b nc t -> b (nc t)")
        if "train_test_block_filter" in target_filtering_schemes:
            filter_masks.append(target_mask.clone())
        if "train_test_block" in target_masking_schemes:
            target_masks.append(target_mask)
    if (
        "train_test_last" in target_masking_schemes
        and masks_to_apply_dict["train_test_last"]
    ):
        assert train_sizes is not None and seq_lens is not None, (
            "train_sizes and seq_lens must be provided for train_test_last_mask"
        )
        time_indices = torch.arange(time_series_length).to(device)
        use_train_sizes = train_sizes.squeeze(-1)
        use_seq_lens = seq_lens.squeeze(-1)
        target_mask = torch.logical_and(
            (time_indices.unsqueeze(0) >= use_train_sizes.unsqueeze(-1)),
            (time_indices.unsqueeze(0) < use_seq_lens.unsqueeze(-1)),
        )  # the same as above, but returning b t as the target mask for the last column
        target_masks.append(target_mask)
    if "random" in target_masking_schemes and masks_to_apply_dict["random"]:
        # target masking scheme is random
        # randomly mask
        target_mask = (
            torch.rand(
                batch_size,
                num_correlates * time_series_length,
            )
            < target_mask_ratio
        ).to(device)  # B X (NUM_CORRELATES X TIME_SERIES_LENGTH)
        target_masks.append(target_mask)
    if (
        "random_last" in target_masking_schemes
        and masks_to_apply_dict["random_last"]
    ):
        target_mask = (
            torch.rand(
                batch_size,
                time_series_length,
            )
            < target_mask_ratio
        ).to(device)  # B X TIME_SERIES_LENGTH
        target_masks.append(target_mask)
    assert not (
        "train_test_last" in target_masking_schemes and len(target_masks) > 1
    ), "train_test_last and other masking schemes are mutually exclusive"
    assert not (
        "random_last" in target_masking_schemes and len(target_masks) > 1
    ), "random_last and other masking schemes are mutually exclusive"

    # take the or of all the masks
    if len(target_masks) == 0:
        # No masks created, return all zeros
        target_mask = torch.zeros(
            batch_size, num_correlates * time_series_length
        ).to(device)
    elif len(target_masks) == 1:
        # Only one mask, return it directly
        target_mask = target_masks[0]
    else:
        # Multiple masks, combine them with logical OR
        target_mask = target_masks[0]
        for i, next_target_mask in enumerate(target_masks[1:]):
            target_mask = torch.logical_or(target_mask, next_target_mask)
    for i, filter_mask in enumerate(filter_masks):
        target_mask = torch.logical_and(filter_mask, target_mask)

    return target_mask, masks_to_apply_dict


def make_covariate_lag_feature(
    covariate_columns: Optional[List[str]] = None,
    lags: List[int] = DEFAULT_COVARIATE_LAGS,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a feature generator that adds lagged versions of the specified covariates.

    Notes
    -----
    - Must not reorder rows, because upstream feature pipeline splits train/test by position.
    - Only lags covariate columns (never touches the 'target' column).
    - Safe if some covariates are missing in the current dataframe; they will be skipped.
    """

    # Keep only strictly positive lags
    positive_lags = [lag for lag in lags if isinstance(lag, int) and lag > 0]

    def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
        # Do not change row order; just add shifted columns
        result = df.copy()
        if not positive_lags:
            return result

        cols_to_lag = (
            df.columns if covariate_columns is None else covariate_columns
        )
        for col in cols_to_lag:
            if col in SKIP_COVARIATES:
                continue
            series = result[col]
            for lag in positive_lags:
                lagged_series = series.shift(lag)
                # Fill NaN values with forward fill, then backward fill, then 0
                # Collect lagged columns in a dict, then concat once at the end for performance
                if "_lagged_cols" not in locals():
                    _lagged_cols = {}
                _lagged_cols[f"{col}_lag{lag}"] = (
                    lagged_series.ffill().bfill().fillna(0)
                )

        return result

    return _add_lags
