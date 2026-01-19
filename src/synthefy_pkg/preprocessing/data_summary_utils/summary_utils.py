import numpy as np
from loguru import logger
from scipy.interpolate import interp1d


def _interpolate_nans(
    data: np.ndarray, method: str = "linear", drop_edge_nans: bool = False
) -> np.ndarray:
    """
    Interpolate NaN values in a time series.

    Args:
        data: Input time series with potential NaN values
        method: Interpolation method ('linear', 'forward', 'backward')
        drop_edge_nans: If True, drop trailing and preceding NaNs before interpolation

    Returns:
        Interpolated time series with NaN values filled
    """
    if not np.isnan(data).any():
        return data

    # Handle dropping edge NaNs if requested
    if drop_edge_nans:
        # Find first and last non-NaN indices
        valid_mask = ~np.isnan(data)
        if not valid_mask.any():
            # All values are NaN
            return np.array([])

        first_valid = np.argmax(valid_mask)
        last_valid = len(data) - 1 - np.argmax(valid_mask[::-1])

        # Trim the data to remove edge NaNs
        data = data[first_valid : last_valid + 1]

    # Create index for non-NaN values
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    valid_values = data[valid_mask]

    # Handle case where all values are NaN - return all zeros regardless of method
    if len(valid_values) == 0:
        logger.debug("All values are NaN, filling with 0.0")
        return np.zeros_like(data)

    if len(valid_values) < 2:
        # If less than 2 valid values, use forward/backward fill
        if method == "forward":
            return np.full_like(
                data, valid_values[0] if len(valid_values) > 0 else 0.0
            )
        elif method == "backward":
            return np.full_like(
                data, valid_values[-1] if len(valid_values) > 0 else 0.0
            )
        else:
            return np.full_like(
                data, np.nanmean(data) if not np.isnan(data).all() else 0.0
            )

    # Create full index
    full_indices = np.arange(len(data))

    if method == "linear":
        # Linear interpolation
        interp_func = interp1d(
            valid_indices, valid_values, kind="linear", bounds_error=False
        )
        interpolated = interp_func(full_indices)
        # Handle extrapolation manually
        interpolated[full_indices < valid_indices[0]] = valid_values[0]
        interpolated[full_indices > valid_indices[-1]] = valid_values[-1]

        # Post-process: fill any remaining NaNs with forward fill
        nan_mask = np.isnan(interpolated)
        if nan_mask.any():
            logger.debug(
                f"Warning: Linear interpolation left {nan_mask.sum()} NaNs, filling with forward fill"
            )
            # Use forward fill for any remaining NaNs
            last_valid = valid_values[0]
            for i in range(len(interpolated)):
                if not nan_mask[i]:
                    last_valid = interpolated[i]
                else:
                    interpolated[i] = last_valid
    elif method == "forward":
        # Forward fill
        interpolated = np.full_like(data, np.nan)
        interpolated[valid_indices] = valid_values
        last_valid = valid_values[0]
        for i in range(len(data)):
            if not np.isnan(interpolated[i]):
                last_valid = interpolated[i]
            else:
                interpolated[i] = last_valid
    elif method == "backward":
        # Backward fill
        interpolated = np.full_like(data, np.nan)
        interpolated[valid_indices] = valid_values
        next_valid = valid_values[-1]
        for i in range(len(data) - 1, -1, -1):
            if not np.isnan(interpolated[i]):
                next_valid = interpolated[i]
            else:
                interpolated[i] = next_valid
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return interpolated
