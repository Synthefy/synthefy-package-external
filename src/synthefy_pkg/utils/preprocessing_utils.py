import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def safe_division(a, b):
    """
    Safely divides two numbers.
    Covers cases like:
        ipdb> 0.6/0.2
            2.9999999999999996
    """
    # Calculate result and round to nearest integer if very close to it
    result = a / b
    # If result is very close to an integer (within a small epsilon), return that integer
    nearest_int = round(result)
    if abs(result - nearest_int) < 1e-10:
        return nearest_int
    # Otherwise, floor the result as requested
    return int(result)


def _calculate_min_length_needed(
    window_size: int, stride: int, train_val_split: Dict[str, float]
) -> Tuple[int, Dict[str, float]]:
    """
    Calculates the minimum length needed for time series data to create windows across all splits.

    This function determines the minimum required length of a time series segment that allows
    creating at least one window for each non-zero split (train, validation, test) while
    maintaining appropriate proportions.
    The function calculates the minimum length needed for windows across all splits (non overlapping) and updates
    the split ratios based on the actual window counts.
    The function also ensures that the split ratios are maintained while creating windows.
    The function returns the minimum length needed for windows across all splits and the updated split ratios.
    Zero windows are allowed for val and test splits if the split ratios are 0.
    For train split min number of windows allowed is 1.
    If train split is 1.0, the function returns 1 for min_length_needed and 0 for all split ratios.

    Parameters:
    -----------
    window_size : int
        Size of the sliding window.
    stride : int
        Step size between consecutive windows.
    train_val_split : Dict[str, float]
        Dictionary with "train" and "val" proportions (must sum to <= 1.0).
        The test proportion is calculated as 1 - train - val.

    Returns:
    --------
    Tuple[int, Dict[str, float]]
        - min_length_needed: Minimum length of data required to create windows.
        - adjusted_train_val_split: Adjusted split ratios based on actual window counts.
    """
    # Calculate test split proportion
    test_split_ratio = 1 - train_val_split["train"] - train_val_split["val"]

    # Find the minimum non-zero ratio among the splits
    non_zero_ratios = [
        r
        for r in [
            train_val_split["train"],
            train_val_split["val"],
            test_split_ratio,
        ]
        if r > 0
    ]
    min_split_ratio = min(non_zero_ratios)

    # Calculate how many windows each split should have to maintain ratios
    # Using a tolerance approach to handle floating-point precision issues
    # If the smallest ratio gets 1 window, how many should others have?
    num_train_windows = max(
        1, safe_division(train_val_split["train"], min_split_ratio)
    )
    num_val_windows = max(
        0, safe_division(train_val_split["val"], min_split_ratio)
    )
    num_test_windows = max(0, safe_division(test_split_ratio, min_split_ratio))

    # Update split ratios based on actual window counts
    # Calculate the actual lengths used by each split
    train_length = (
        window_size + (num_train_windows - 1) * stride
        if num_train_windows > 0
        else 0
    )
    val_length = (
        window_size + (num_val_windows - 1) * stride
        if num_val_windows > 0
        else 0
    )
    test_length = (
        window_size + (num_test_windows - 1) * stride
        if num_test_windows > 0
        else 0
    )
    total_length = train_length + val_length + test_length

    # Update split ratios based on actual lengths
    # The values of train val test split ratios can be updated only if
    # the division by min_split_ratio is not an integer.
    # This is done to ensure that no unused length is left in any split.
    # Therefore prevents train/val/test sets from being empty.
    adjusted_train_val_split = {
        "train": train_length / total_length if total_length > 0 else 0,
        "val": val_length / total_length if total_length > 0 else 0,
        "test": test_length / total_length if total_length > 0 else 0,
    }

    return total_length, adjusted_train_val_split


def _partition_group_length(
    group_len: int,
    min_length_needed: int,
    user_defined_shuffle_partitions_len: Optional[int] = None,
) -> List[int]:
    """
    Partitions a time series group into optimal segments for windowing.

    This function divides a group of time series data into partitions that maximize
    data utilization while ensuring each partition is at least the minimum required length.
    If the group length is less than the minimum length needed, the function returns a list with the group length,
    which means that some of train/val/test splits will have no windows for this partition segment.


    Parameters:
    -----------
    group_len : int
        The total length of the group (number of time steps).
    min_length_needed : int
        Minimum length required for creating windows in each partition.
    user_defined_shuffle_partitions_len : Optional[int], default=None
        If provided, the length of partitions to use for shuffling.
        If not provided, the number of partitions will be determined by the group length and the window size.

    Returns:
    --------
    List[int]
        A list of integers representing the size of each partition.
        The sum of all partition sizes equals the original group length.
    """
    if user_defined_shuffle_partitions_len is not None:
        min_partition_size = user_defined_shuffle_partitions_len
    else:
        min_partition_size = min(group_len, min_length_needed)

    # If group_len is less than min_length_needed, return just one partition
    if group_len < min_length_needed:
        return [group_len]

    # Calculate number of partitions
    num_partitions = max(1, group_len // min_partition_size)

    # Create partitions of the optimal size
    partition_lens = [min_partition_size] * num_partitions

    # Handle any remaining length
    remainder = group_len - (min_partition_size * num_partitions)
    if remainder > 0:
        partition_lens.append(remainder)

    # Ensure the sum of partitions equals the original group length
    try:
        if sum(partition_lens) != group_len:
            raise ValueError(
                "Partition sum mismatch! The sum of partitions does not equal the original group length."
            )
    except Exception as e:
        logger.error(f"Error in partition calculation: {str(e)}")
        raise

    return partition_lens


def _calculate_windows_and_length(
    length: int, window_size: int, stride: int
) -> Tuple[int, int]:
    """
    Calculates the number of possible windows and the actual data length used.

    Parameters:
    -----------
    length : int
        The total length of the data segment to window.
    window_size : int
        The size of each window.
    stride : int
        The step size between consecutive windows.

    Returns:
    --------
    Tuple[int, int]
        - num_windows: The number of windows that can be generated.
        - length_used: The actual length of the data segment covered by these windows.
          Will be 0 if no windows can be created.
    """
    if length < window_size:
        return (
            0,
            0,
        )  # Cannot create any windows if length is smaller than window size

    num_windows = max(0, (length - window_size) // stride + 1)
    length_used = (
        (num_windows - 1) * stride + window_size if num_windows > 0 else 0
    )
    return num_windows, length_used


def calculate_split_windows(
    group_len: int,
    train_val_split: Dict[str, float],
    window_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: int,
    allow_overlap: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Calculates the number of windows for each data split while preserving split proportions.

    This function determines how many windows can be created for training, validation,
    and testing splits based on the specified proportions and stride values. When
    allow_overlap=True, windows can overlap between splits and are allocated based
    on total available windows in the group.

    Parameters:
    -----------
    group_len : int
        Length of the time series group.
    train_val_split : Dict[str, float]
        Dictionary with "train" and "val" proportions for splitting the data.
    window_size : int
        Size of the sliding window.
    train_stride : int
        Stride for training set windows.
    val_stride : int
        Stride for validation set windows.
    test_stride : int
        Stride for test set windows.
    allow_overlap : bool, default=False
        If True, allows overlapping windows between splits and allocates based
        on total windows available. If False, uses sequential non-overlapping splits.

    Returns:
    --------
    Tuple[int, int, int, int]
        - num_train_windows: Number of windows for training.
        - num_val_windows: Number of windows for validation.
        - num_test_windows: Number of windows for testing.
        - unused_length: Length of data that couldn't be used for any windows.
    """
    # Handle special case where window size equals group length
    if window_size == group_len:
        return 1, 0, 0, 0  # Assign one window to train split

    if allow_overlap:
        # For overlapping mode: use train_stride for all splits and allocate total windows
        # Calculate total possible windows using train_stride
        total_windows, total_length_used = _calculate_windows_and_length(
            group_len, window_size, train_stride
        )

        if total_windows == 0:
            return 0, 0, 0, group_len

        # Allocate windows based on proportions
        num_train_windows = max(
            1, int(total_windows * train_val_split["train"])
        )
        num_val_windows = max(0, int(total_windows * train_val_split["val"]))

        # Give remaining windows to test
        num_test_windows = max(
            0, total_windows - num_train_windows - num_val_windows
        )

        # Calculate unused length
        unused_length = max(0, group_len - total_length_used)

    else:
        # Original non-overlapping logic
        # Calculate number of windows for training
        num_train_rows = int(group_len * train_val_split["train"])
        num_train_windows, train_length_used = _calculate_windows_and_length(
            num_train_rows, window_size, train_stride
        )
        remaining_length = group_len - train_length_used

        # Calculate validation proportion of remaining length
        test_split = 1 - train_val_split["train"] - train_val_split["val"]
        denominator = train_val_split["val"] + test_split
        val_ratio = (
            train_val_split["val"] / denominator if denominator > 0 else 0
        )

        num_val_rows = int(remaining_length * val_ratio)
        num_val_windows, val_length_used = _calculate_windows_and_length(
            num_val_rows, window_size, val_stride
        )
        final_remaining_length = remaining_length - val_length_used

        # Calculate test windows from remaining length
        num_test_windows, test_length_used = _calculate_windows_and_length(
            final_remaining_length, window_size, test_stride
        )

        # Calculate total used and unused length
        total_used_length = (
            train_length_used + val_length_used + test_length_used
        )
        unused_length = max(0, group_len - total_used_length)

        # Redistribute windows if train has none but other splits have some
        if num_train_windows == 0:
            return 0, 0, 0, group_len

    return (
        num_train_windows,
        num_val_windows,
        num_test_windows,
        unused_length,
    )


def generate_window_start_indices(
    group_start_row: int,
    num_train_windows: int,
    num_val_windows: int,
    num_test_windows: int,
    window_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: int,
    window_from_beginning: bool,
    unused_length: int,
    allow_overlap: bool = False,
) -> List[int]:
    """
    Generates the starting indices for all windows across train, validation, and test splits.

    Calculates the precise row indices where each window should start in the original
    time series data, respecting the specified strides for each split. When allow_overlap=True,
    windows can overlap between splits by using stride-based positioning instead of gap-based.

    Parameters:
    -----------
    group_start_row : int
        Starting row index for this group in the original dataset.
    num_train_windows : int
        Number of windows for training.
    num_val_windows : int
        Number of windows for validation.
    num_test_windows : int
        Number of windows for testing.
    window_size : int
        Size of the sliding window.
    train_stride : int
        Stride for training set windows.
    val_stride : int
        Stride for validation set windows.
    test_stride : int
        Stride for test set windows.
    window_from_beginning : bool
        If True, windows start from beginning of group data.
        If False, windows are positioned to include the latest data points.
    unused_length : int
        Length of data that couldn't be used for any windows.
    allow_overlap : bool, default=False
        If True, allows overlapping windows between splits by using stride-based positioning.
        If False, uses gap-based positioning to prevent overlap (original behavior).

    Returns:
    --------
    List[int]
        A list of all window starting indices (row positions) in the original dataset,
        ordered by train, validation, then test splits.
    """
    start_offset = 0 if window_from_beginning else unused_length
    current_pos = group_start_row + start_offset

    offset_between_splits = train_stride if allow_overlap else window_size

    train_start_indices = [
        current_pos + (i * train_stride) for i in range(num_train_windows)
    ]
    if num_train_windows > 0:
        current_pos = train_start_indices[-1] + offset_between_splits

    val_start_indices = [
        current_pos + (i * val_stride) for i in range(num_val_windows)
    ]
    if num_val_windows > 0:
        current_pos = val_start_indices[-1] + offset_between_splits

    test_start_indices = [
        current_pos + (i * test_stride) for i in range(num_test_windows)
    ]

    return train_start_indices + val_start_indices + test_start_indices


def assign_indices_to_splits(
    window_idx: int,
    num_train_windows: int,
    num_val_windows: int,
    num_test_windows: int,
    shuffle: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Assigns consecutive window indices to train, validation, and test splits.

    Parameters:
    -----------
    window_idx : int
        Starting window index for the current batch of windows.
    num_train_windows : int
        Number of windows to assign to training.
    num_val_windows : int
        Number of windows to assign to validation.
    num_test_windows : int
        Number of windows to assign to testing.
    shuffle : bool, default=False
        If True, shuffles the window indices before assigning to splits.

    Returns:
    --------
    Tuple[List[int], List[int], List[int]]
        - train_indices: List of window indices for training.
        - val_indices: List of window indices for validation.
        - test_indices: List of window indices for testing.
    """
    total_windows = num_train_windows + num_val_windows + num_test_windows
    windows_inds_list = list(range(window_idx, window_idx + total_windows))

    # Shuffle the indices if requested
    if shuffle:
        random.shuffle(windows_inds_list)

    # Assign to splits
    train_indices = windows_inds_list[:num_train_windows]
    val_indices = windows_inds_list[
        num_train_windows : num_train_windows + num_val_windows
    ]
    test_indices = windows_inds_list[num_train_windows + num_val_windows :]

    return train_indices, val_indices, test_indices


def find_group_train_val_test_split_inds_stratified(
    group_len_dict: Dict[str, int],
    train_val_split: Dict[str, float],
    shuffle: bool,
    window_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: int,
    window_from_beginning: bool = False,
    user_defined_shuffle_partitions_len: Optional[int] = None,
    allow_overlap: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Creates stratified train/val/test splits for windowed time series data across multiple groups.

    This function handles the complex task of creating window indices for different
    splits while maintaining proportions and ensuring stratification by group. It supports both
    shuffled and sequential splitting strategies, with optional overlap between splits.

    Parameters:
    -----------
    group_len_dict : Dict[str, int]
        Dictionary mapping group keys to their lengths (number of time steps).
    train_val_split : Dict[str, float]
        Dictionary with "train" and "val" proportions. Test = 1 - train - val.
    shuffle : bool
        If True, partitions each group for better randomization.
        If True, uses train_stride for all splits to ensure consistent shuffling.
    window_size : int
        Size of the sliding window.
    train_stride : int
        Stride for training windows.
    val_stride : int
        Stride for validation windows.
    test_stride : int
        Stride for test windows.
    window_from_beginning : bool, default=False
        If True, windows start from beginning of group data.
        If False, windows are positioned to include the latest data points.
    user_defined_shuffle_partitions_len : Optional[int], default=None
        If provided, the number of partitions to use for shuffling.
        If not provided, the number of partitions will be determined by the group length and the window size.
    allow_overlap : bool, default=False
        If True, allows overlapping windows between train/val/test splits.
        If False, creates non-overlapping splits (original behavior).

    Returns:
    --------
    Tuple[Dict[str, List[int]], Dict[str, List[int]]]
        - split_inds_dict: Dictionary mapping "train", "val", "test" to lists of window indices.
        - window_start_indices: Dictionary mapping group keys to lists of window start positions
          (row indices in the original dataset).

    Raises:
    -------
    ValueError
        If the total number of indices doesn't match the total number of windows.
    """
    split_inds_dict = {"train": [], "val": [], "test": []}
    window_start_indices = {}
    window_idx = 0
    group_start_row = 0

    # If shuffle is enabled, use consistent stride for proper randomization
    if shuffle or allow_overlap:
        logger.info("Shuffle is enabled, using train_stride for all splits")
        val_stride = test_stride = train_stride
        min_length_needed, adjusted_train_val_split = (
            _calculate_min_length_needed(
                window_size, train_stride, train_val_split
            )
        )
    actual_train_val_split = train_val_split
    for group_key, group_len in group_len_dict.items():
        window_start_indices[group_key] = []
        # Partition the group length using the helper function
        if allow_overlap:
            # Calculate the number of windows for this partition
            (
                num_train_windows,
                num_val_windows,
                num_test_windows,
                unused_length,
            ) = calculate_split_windows(
                group_len,
                actual_train_val_split,
                window_size,
                train_stride,
                val_stride,
                test_stride,
                allow_overlap,
            )
            window_start_indices[group_key] = generate_window_start_indices(
                group_start_row,
                num_train_windows,
                num_val_windows,
                num_test_windows,
                window_size,
                train_stride,
                val_stride,
                test_stride,
                window_from_beginning,
                unused_length,
                allow_overlap,
            )
            # Assign indices to splits
            train_indices, val_indices, test_indices = assign_indices_to_splits(
                window_idx,
                num_train_windows,
                num_val_windows,
                num_test_windows,
                shuffle,
            )

            # Extend the split dictionaries
            split_inds_dict["train"].extend(train_indices)
            split_inds_dict["val"].extend(val_indices)
            split_inds_dict["test"].extend(test_indices)

            # Update counters for next group
            window_idx += num_train_windows + num_val_windows + num_test_windows

        else:
            if shuffle:
                partition_lens = _partition_group_length(
                    group_len,
                    min_length_needed,
                    user_defined_shuffle_partitions_len,
                )
            else:
                partition_lens = [group_len]

            segment_start_row = group_start_row
            for segment_len in partition_lens:
                # If there is only one partition, use the original split ratios
                if shuffle:
                    actual_train_val_split = (
                        train_val_split
                        if (
                            segment_len < min_length_needed
                            or user_defined_shuffle_partitions_len is not None
                        )
                        else adjusted_train_val_split
                    )
                # Calculate the number of windows for this partition
                (
                    num_train_windows,
                    num_val_windows,
                    num_test_windows,
                    unused_length,
                ) = calculate_split_windows(
                    segment_len,
                    actual_train_val_split,
                    window_size,
                    train_stride,
                    val_stride,
                    test_stride,
                )

                # Generate start indices for each split
                window_start_indices[group_key].extend(
                    generate_window_start_indices(
                        segment_start_row,
                        num_train_windows,
                        num_val_windows,
                        num_test_windows,
                        window_size,
                        train_stride,
                        val_stride,
                        test_stride,
                        window_from_beginning,
                        unused_length,
                    )
                )

                # Assign indices to splits
                train_indices, val_indices, test_indices = (
                    assign_indices_to_splits(
                        window_idx,
                        num_train_windows,
                        num_val_windows,
                        num_test_windows,
                    )
                )

                # Extend the split dictionaries
                split_inds_dict["train"].extend(train_indices)
                split_inds_dict["val"].extend(val_indices)
                split_inds_dict["test"].extend(test_indices)

                # Update counters for next group
                window_idx += (
                    num_train_windows + num_val_windows + num_test_windows
                )
                segment_start_row += segment_len

        group_start_row += group_len

        # Verify the window alignment if not starting from beginning
        if (
            not window_from_beginning
            and len(window_start_indices[group_key]) > 0
            and not shuffle
        ):
            try:
                expected_pos = (
                    group_start_row - group_len + group_len - window_size
                )
                actual_pos = window_start_indices[group_key][-1]
                if actual_pos != expected_pos:
                    raise ValueError(
                        f"Window alignment error: expected position {expected_pos} but got {actual_pos} "
                        f"for group {group_key}"
                    )
            except Exception as e:
                logger.error(f"Window alignment error: {str(e)}")
                raise

    # Verify no samples were lost
    total_indices = sum(len(v) for v in split_inds_dict.values())
    total_windows = sum(len(v) for v in window_start_indices.values())

    if not shuffle and total_indices != total_windows:
        try:
            error_msg = f"Total indices ({total_indices}) != total windows ({total_windows})"
            logger.error(error_msg)
            logger.error(
                f"Split distribution: {[len(v) for v in split_inds_dict.values()]}"
            )
            logger.error(
                f"Windows distribution: {[len(v) for v in window_start_indices.values()]}"
            )
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Validation error in splits: {str(e)}")
            raise

    return split_inds_dict, window_start_indices


def validate_split_indices(split_inds_dict: Dict[str, List[int]]) -> None:
    """
    Validates that all data splits contain at least one sample.

    Parameters:
    -----------
    split_inds_dict : Dict[str, List[int]]
        Dictionary with split names as keys and lists of window indices as values.

    Raises:
    -------
    ValueError
        If any of the splits (train, val, test) have empty indices lists.
        Suggests adjusting window_size, stride, or train_val_split values.
    """
    for key, indices in split_inds_dict.items():
        if not indices:
            raise ValueError(
                f"Split '{key}' has no samples. This shouldn't be empty - "
                f"please adjust your window_size and/or stride and/or"
                f"train_val_split values to ensure all splits have data."
            )


def get_overlap_matrix(pos_array1, pos_array2, window_size):
    """
    Creates a boolean vector indicating which positions in the first array overlap with any in the second.

    This function efficiently computes overlaps between two arrays of window positions using
    NumPy broadcasting, without requiring explicit loops. A window at position i overlaps with
    another window at position j if |i - j| < window_size.

    Parameters:
    -----------
    pos_array1 : numpy.ndarray
        Array of window starting positions for the first set.
    pos_array2 : numpy.ndarray
        Array of window starting positions for the second set.
    window_size : int
        Size of each window, used to determine if windows overlap.

    Returns:
    --------
    numpy.ndarray
        Boolean vector of shape (len(pos_array1),) where True indicates
        that the window overlaps with at least one window in pos_array2.

    Examples:
    ---------
    >>> pos_array1 = np.array([0, 10, 20, 30])  # Window starting positions for set 1
    >>> pos_array2 = np.array([5, 15, 25])      # Window starting positions for set 2
    >>> window_size = 10                         # Each window is 10 units long
    >>> overlap_vector = get_overlap_matrix(pos_array1, pos_array2, window_size)
    >>> overlap_vector
    array([True, True, True, True])
    # This shows all windows in pos_array1 overlap with at least one window in pos_array2
    """
    if len(pos_array1) == 0 or len(pos_array2) == 0:
        # Return a correctly shaped empty boolean array instead of a 2D one
        return np.zeros(len(pos_array1), dtype=bool)

    # Create position matrices for broadcasting
    pos1 = pos_array1.reshape(-1, 1)
    pos2 = pos_array2.reshape(1, -1)

    # Calculate absolute differences
    abs_diff = np.abs(pos1 - pos2)

    # Return overlap matrix where True indicates overlap
    overlap = abs_diff < window_size
    return np.any(overlap, axis=1)


def remove_overlapping_windows(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    window_start_positions: List[int],
    window_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Removes overlapping windows between train, validation, and test sets with prioritization.

    This function ensures data leakage is prevented by removing windows that overlap across splits.
    It uses a priority system where test windows are fully preserved, validation windows are kept
    unless they overlap with test, and train windows are kept unless they overlap with either test
    or validation windows.

    Parameters:
    -----------
    train_indices : List[int]
        List of window indices for the training set.
    val_indices : List[int]
        List of window indices for the validation set.
    test_indices : List[int]
        List of window indices for the test set.
    window_start_positions : List[int]
        List of all window start positions (row indices in the original dataset).
    window_size : int
        Size of each window.

    Returns:
    --------
    Tuple[List[int], List[int], List[int]]
        - clean_train_indices: Training window indices with no overlaps.
        - clean_val_indices: Validation window indices with no overlaps with test.
        - test_indices: Unchanged test window indices.

    Notes:
    ------
    The function implements a fast, matrix-based approach with no explicit loops
    for maximum performance on large datasets.
    """
    # First, build a mapping from indices to their positions in window_start_positions
    all_indices = sorted(train_indices + val_indices + test_indices)
    index_to_pos = {idx: i for i, idx in enumerate(all_indices)}

    # Get the actual window positions for each set
    train_positions = np.array(
        [window_start_positions[index_to_pos[idx]] for idx in train_indices]
    )
    val_positions = np.array(
        [window_start_positions[index_to_pos[idx]] for idx in val_indices]
    )
    test_positions = np.array(
        [window_start_positions[index_to_pos[idx]] for idx in test_indices]
    )

    # Check for overlaps between test and train
    train_test_overlap_mask = get_overlap_matrix(
        train_positions, test_positions, window_size
    )

    # Keep train indices that don't overlap with test
    remaining_train_indices = np.array(train_indices)[~train_test_overlap_mask]
    remaining_train_positions = train_positions[~train_test_overlap_mask]

    # Check for overlaps between test and val
    val_test_overlap_mask = get_overlap_matrix(
        val_positions, test_positions, window_size
    )

    # Keep val indices that don't overlap with test
    remaining_val_indices = np.array(val_indices)[~val_test_overlap_mask]
    remaining_val_positions = val_positions[~val_test_overlap_mask]

    # Check for overlaps between remaining train and remaining val
    train_val_overlap_mask = get_overlap_matrix(
        remaining_train_positions, remaining_val_positions, window_size
    )

    # Apply the train-val overlap mask to the remaining train indices
    final_train_indices = remaining_train_indices[~train_val_overlap_mask]

    return (
        final_train_indices.tolist(),
        remaining_val_indices.tolist(),
        test_indices,
    )


def create_group_label_to_split_idx_dict(
    group_label_to_window_start_row_indices: Dict[Any, List[int]],
    df_row_inds_dataset_types: Dict[str, List[int]],
    allow_overlap: bool = False,
) -> Dict[str, Dict[int, Any]]:
    """
    Creates a mapping between split indices and group labels.

    Parameters:
    -----------
    group_label_to_window_start_row_indices : Dict[Any, List[int]]
        Dictionary mapping group label tuples to their window start row indices
        example - (window_size:4, stride=2)
            {
                '("group1", "A")': [0, 2, 6],
                '("group1", "B")': [10, 14, 18]
            }
            Note that the group label tuple is a string! so use eval() to convert it to a tuple.

    df_row_inds_dataset_types : Dict[str, List[int]]
        Dictionary mapping split names ('train', 'val', 'test') to row indices
        of the dataset.
        example - (window_size:4, stride=2)

            {
                "train": [0, 1, 2, 3, 4, 5, 14, 15, 16, 17],
                "val": [6, 7, 8, 9, 18, 19, 20, 21],
                "test": [10, 11, 12, 13]
            }

    Returns:
    --------
    Dict[str, Dict[int, Any]]
        Dictionary with structure:
        {
            "train_idx": {0: group_label_tuple, 1: group_label_tuple, ...},
            "val_idx": {0: group_label_tuple, 1: group_label_tuple, ...},
            "test_idx": {0: group_label_tuple, 1: group_label_tuple, ...}
        }
    """
    tmp_sets = {k: set(v) for k, v in df_row_inds_dataset_types.items()}
    split_inds_counter = {"train": 0, "val": 0, "test": 0}
    group_label_to_split_idx_dict = {
        "train": {},
        "val": {},
        "test": {},
    }

    already_seen_window_start_row_idxs = set()
    for (
        gl_tup,
        window_start_row_idxs,
    ) in group_label_to_window_start_row_indices.items():
        for window_start_row_idx in window_start_row_idxs:
            # Check to make sure that the split_window_idx is not already in the dictionary.
            # If it is, then there is leakage in the splits.
            if (
                window_start_row_idx in already_seen_window_start_row_idxs
                and not allow_overlap
            ):
                raise ValueError(
                    f"Window start row index {window_start_row_idx} is already associated with a group label. "
                    "This indicates an entire window of data is leaking into another split. This should not happen."
                )

            if window_start_row_idx in tmp_sets["train"]:
                group_label_to_split_idx_dict["train"][
                    split_inds_counter["train"]
                ] = eval(gl_tup)
                split_inds_counter["train"] += 1
            elif window_start_row_idx in tmp_sets["val"]:
                group_label_to_split_idx_dict["val"][
                    split_inds_counter["val"]
                ] = eval(gl_tup)
                split_inds_counter["val"] += 1
            elif window_start_row_idx in tmp_sets["test"]:
                group_label_to_split_idx_dict["test"][
                    split_inds_counter["test"]
                ] = eval(gl_tup)
                split_inds_counter["test"] += 1
            already_seen_window_start_row_idxs.add(window_start_row_idx)

    return group_label_to_split_idx_dict
