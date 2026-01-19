import torch

from synthefy_pkg.preprocessing.fmv2_sharded_mix_and_split import (
    retrieve_timestamp_from_window,
)


def extract_times_from_window_batch(batch, window_size):
    """
    Extract the times from the window batch
    """
    start_time = retrieve_timestamp_from_window(
        batch["timeseries"].cpu().numpy(), window_size, idx=0, as_seconds=True
    )
    end_time = retrieve_timestamp_from_window(
        batch["timeseries"].cpu().numpy(), window_size, idx=-1, as_seconds=True
    )
    return start_time, end_time


def extract_id_from_window_batch(batch):
    """
    Extract the id from the window batch
    """
    return batch["timeseries"][:, -1].to(torch.int32)


def identify_overlapping_windows(
    start_times_1, end_times_1, start_times_2, end_times_2, device="cuda:0"
):
    """
    Identify if any windows in the batch overlap with each other using PyTorch tensors.

    Args:
        start_times: Array of start times for each window
        end_times: Array of end times for each window
        device: Device to run computations on

    Returns:
        torch.Tensor: Boolean tensor indicating overlapping windows
    """
    # Convert to tensors and move to device
    start_times_1 = torch.tensor(start_times_1, device=device)
    end_times_1 = torch.tensor(end_times_1, device=device)
    start_times_2 = torch.tensor(start_times_2, device=device)
    end_times_2 = torch.tensor(end_times_2, device=device)

    # Create broadcasted comparisons
    # start_i <= end_j for all i,j
    start_less_end = start_times_1.unsqueeze(1) <= end_times_2.unsqueeze(0)

    # end_i >= start_j for all i,j
    end_greater_start = end_times_1.unsqueeze(1) >= start_times_2.unsqueeze(0)

    # Combine conditions
    overlap_matrix = start_less_end & end_greater_start

    return overlap_matrix.float()  # Convert boolean to float


def identify_window_distance(
    start_times_1, end_times_1, start_times_2, end_times_2, device="cuda:0"
):
    """
    Calculate the minimum distance between all pairs of time windows using PyTorch tensors.
    The distance is defined as the closest point between two time segments.

    Args:
        start_times: Array of start times for each window
        end_times: Array of end times for each window
        device: Device to run computations on

    Returns:
        torch.Tensor: Matrix where matrix[i,j] is the minimum distance between window i and j
    """
    # Convert to tensors and move to device
    starts_1 = torch.tensor(start_times_1, device=device)
    ends_1 = torch.tensor(end_times_1, device=device)
    starts_2 = torch.tensor(start_times_2, device=device)
    ends_2 = torch.tensor(end_times_2, device=device)

    # Create broadcasted tensors
    start_i = starts_1.unsqueeze(1)  # Shape: (n_windows, 1)
    end_i = ends_1.unsqueeze(1)  # Shape: (n_windows, 1)
    start_j = starts_2.unsqueeze(0)  # Shape: (1, n_windows)
    end_j = ends_2.unsqueeze(0)  # Shape: (1, n_windows)

    # Calculate distances
    dist_before = start_j - end_i
    dist_after = start_i - end_j

    # Identify overlapping windows
    overlap = (start_i <= end_j) & (end_i >= start_j)

    # Combine distances
    distances = torch.minimum(dist_before, dist_after)
    distances[overlap] = 0  # Set overlapping windows to distance 0

    # Set diagonal to 0
    distances.fill_diagonal_(0)

    return distances


def identify_window_overlap_percentage(
    start_times_1, end_times_1, start_times_2, end_times_2, device="cuda:0"
):
    """
    Calculate the percentage of overlap between two time series windows.

    Args:
        start_times_1: Array of start times for first set of windows
        end_times_1: Array of end times for first set of windows
        start_times_2: Array of start times for second set of windows
        end_times_2: Array of end times for second set of windows
        device: Device to run computations on

    Returns:
        torch.Tensor: Matrix where matrix[i,j] is the percentage of overlap between window i and j
    """
    # Convert to tensors and move to device
    starts_1 = torch.tensor(start_times_1, device=device)
    ends_1 = torch.tensor(end_times_1, device=device)
    starts_2 = torch.tensor(start_times_2, device=device)
    ends_2 = torch.tensor(end_times_2, device=device)

    # Create broadcasted tensors
    start_i = starts_1.unsqueeze(1)  # Shape: (n_windows, 1)
    end_i = ends_1.unsqueeze(1)  # Shape: (n_windows, 1)
    start_j = starts_2.unsqueeze(0)  # Shape: (1, n_windows)
    end_j = ends_2.unsqueeze(0)  # Shape: (1, n_windows)

    # Calculate overlap start and end times
    overlap_start = torch.maximum(start_i, start_j)
    overlap_end = torch.minimum(end_i, end_j)

    # Calculate overlap duration
    overlap_duration = torch.clamp(overlap_end - overlap_start, min=0)

    # Calculate total duration of each window
    window_duration_1 = end_i - start_i
    window_duration_2 = end_j - start_j

    # Calculate percentage of overlap
    # For each window pair, we take the minimum of the two window durations
    # to get the maximum possible overlap
    max_possible_overlap = torch.maximum(window_duration_1, window_duration_2)

    # Calculate percentage (avoid division by zero)
    overlap_percentage = torch.where(
        max_possible_overlap > 0,
        overlap_duration / max_possible_overlap,
        torch.zeros_like(overlap_duration),
    )

    overlap_percentage[
        overlap_percentage < 1e-8
    ] = -0.3  # zero overlap especially penalized

    return overlap_percentage
