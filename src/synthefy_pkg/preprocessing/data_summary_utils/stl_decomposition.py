"""
STL (Seasonal and Trend decomposition using Loess) decomposition for time series.

This module provides comprehensive STL decomposition functionality using statsmodels,
including parameter tuning, robust estimation, and visualization capabilities.
"""

from collections import Counter
from typing import Any, List, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import STL


def detect_period(data, min_period, max_period):
    """Detect period using FFT power spectrum with improved robustness."""
    # Compute FFT
    fft = np.fft.fft(data)
    power = np.abs(fft) ** 2

    # Focus on positive frequencies (excluding DC component)
    freqs = np.fft.fftfreq(len(data))
    positive_freqs = freqs > 0

    if not np.any(positive_freqs):
        return min_period

    # Find peaks in power spectrum
    power_positive = power[positive_freqs]
    freqs_positive = freqs[positive_freqs]

    # Apply threshold to avoid noise
    threshold = np.mean(power_positive) + 2 * np.std(power_positive)
    significant_power = power_positive > threshold

    if not np.any(significant_power):
        return min_period

    # Find dominant frequency among significant peaks
    dominant_idx = np.argmax(power_positive * significant_power)
    dominant_freq = freqs_positive[dominant_idx]

    if dominant_freq != 0:
        period = int(round(float(1 / abs(dominant_freq))))

        # Additional validation: check if period makes sense
        if min_period <= period <= max_period:
            # Verify period by checking autocorrelation at this lag
            if len(data) >= 2 * period:
                autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
                if autocorr > 0.1:  # Only accept if there's some correlation
                    return period

    # Fallback: try common periods
    common_periods = [7, 12, 24, 30, 52, 365]
    for p in common_periods:
        if min_period <= p <= max_period and p <= len(data) // 2:
            return p

    return min_period


def perform_stl_decomposition(
    time_series: Union[pd.Series, np.ndarray, torch.Tensor],
    period: Optional[Union[int, List[int]]] = None,
    seasonal: Optional[Union[int, List[int]]] = None,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    robust: bool = False,
    return_components: bool = True,
    return_numpy: bool = True,
    num_periods: Optional[int] = None,
    multi_period: bool = False,
) -> Union[dict, Any]:
    """
    Perform STL (Seasonal and Trend decomposition using Loess) decomposition on a time series.

    This function supports both single-period and multi-period seasonal decomposition.
    For multi-period decomposition, it efficiently extracts seasonal components for multiple
    periods using optimized algorithms.

    Parameters
    ----------
    time_series : Union[pd.Series, np.ndarray, torch.Tensor]
        Input time series data. If numpy array, should be 1-dimensional.
        If torch.Tensor, will be converted to numpy array.
    period : Optional[Union[int, list[int]]], default=None
        Period(s) of the seasonal component(s). If None, will be inferred from the data.
        For multi-period: list of periods. For single-period: single integer.
    seasonal : Optional[Union[int, list[int]]], default=None
        Seasonal smoothing parameter(s). Must be odd. If None, will be set automatically.
        For multi-period: list of seasonals. For single-period: single integer.
    trend : Optional[int], default=None
        Trend smoothing parameter. Must be odd. If None, will be set automatically.
    low_pass : Optional[int], default=None
        Low-pass smoothing parameter. Must be odd. If None, will be set automatically.
    seasonal_deg : int, default=1
        Degree of seasonal LOESS (0, 1, or 2).
    trend_deg : int, default=1
        Degree of trend LOESS (0, 1, or 2).
    low_pass_deg : int, default=1
        Degree of low-pass LOESS (0, 1, or 2).
    seasonal_jump : int, default=1
        Jump size for seasonal LOESS.
    trend_jump : int, default=1
        Jump size for trend LOESS.
    low_pass_jump : int, default=1
        Jump size for low-pass LOESS.
    robust : bool, default=False
        Whether to use robust estimation.
    return_components : bool, default=True
        If True, returns a dictionary with all components.
        If False, returns only the STL result object.
    return_numpy : bool, default=True
        If True, converts pandas Series components to numpy arrays.
    num_periods : Optional[int], default=None
        Maximum number of periods to detect (for multi-period auto-detection).
    multi_period : bool, default=False
        If True, performs multi-period decomposition. If False, single-period.

    Returns
    -------
    Union[dict, statsmodels.tsa.seasonal.STLResult]
        If return_components=True: Dictionary containing:
            - 'trend': Trend component
            - 'seasonal': Seasonal component (combined for multi-period)
            - 'resid': Residual component
            - 'weights': Weights used in robust estimation (if robust=True)
            - 'stl_result': The full STL result object
            - 'seasonal_components': List of individual seasonal components (multi-period only)
            - 'periods': List of periods used (multi-period only)
            - 'stl_results': List of STL results (multi-period only)
        If return_components=False: The STL result object directly
    """

    # ===== UNIFIED INPUT VALIDATION =====
    # Convert torch.Tensor to numpy
    if isinstance(time_series, torch.Tensor):
        time_series = time_series.detach().cpu().numpy()

    # Convert numpy array to pandas Series
    if isinstance(time_series, np.ndarray):
        if time_series.ndim != 1:
            raise ValueError(
                "If numpy array, time_series must be 1-dimensional"
            )
        time_series = pd.Series(time_series)

    # Validate input type
    if not isinstance(time_series, pd.Series):
        raise ValueError(
            "time_series must be a pandas Series or 1D numpy array"
        )

    # Validate length
    if len(time_series) < 15:
        raise ValueError("Time series must have at least 15 observations")

    n = len(time_series)

    # Handle NaN values
    if time_series.isna().any():
        time_series = time_series.fillna(time_series.mean())

    # ===== PERIOD AND PARAMETER PREPARATION =====
    if multi_period:
        # Multi-period mode
        if period is None or (
            isinstance(period, (list, tuple)) and len(period) == 0
        ):
            periods = _detect_multiple_periods(
                time_series, max_periods=num_periods or 3
            )
        elif isinstance(period, int):
            periods = [period]
        else:
            periods = period.copy()  # Avoid modifying original

        # Sort periods (shortest first for optimal decomposition)
        periods = sorted(periods)

        # Validate periods
        for p in periods:
            if p <= 1 or p >= n:
                raise ValueError(f"Period {p} must be between 2 and {n - 1}")

        # Prepare seasonal parameters
        if seasonal is None:
            seasonals = periods.copy()
        elif isinstance(seasonal, int):
            seasonals = [seasonal] * len(periods)
        else:
            seasonals = seasonal.copy()

        # Ensure all seasonals are odd
        seasonals = [s + (s % 2 == 0) for s in seasonals]

    else:
        # Single-period mode
        if period is None:
            periods = detect_period(time_series, min_period=2, max_period=n - 1)
        else:
            periods = period

        # Validate single period
        if isinstance(periods, (list, tuple)):
            # This should not happen in single-period mode, but handle gracefully
            periods = periods[0] if periods else 2

        if periods <= 1 or periods >= n:
            raise ValueError(
                f"Period must be between 2 and {n - 1}, got {periods}"
            )

        # Prepare seasonal parameter
        if seasonal is None:
            seasonals = periods
        else:
            seasonals = seasonal

        # Ensure odd
        if isinstance(seasonals, (list, tuple)):
            seasonals = seasonals[0] if seasonals else 3

        if seasonals % 2 == 0:
            seasonals += 1

    # ===== STL DECOMPOSITION EXECUTION =====
    if multi_period:
        # Multi-period decomposition
        if isinstance(periods, (list, tuple)) and len(periods) == 1:
            # Single period - use optimized single-period path
            return _single_period_stl(
                time_series,
                periods[0] if isinstance(periods, (list, tuple)) else periods,
                seasonals[0]
                if isinstance(seasonals, (list, tuple))
                else seasonals,
                trend,
                low_pass,
                seasonal_deg,
                trend_deg,
                low_pass_deg,
                seasonal_jump,
                trend_jump,
                low_pass_jump,
                robust,
                return_components,
                return_numpy,
            )
        else:
            # Multiple periods - use optimized multi-period path
            # Ensure periods and seasonals are lists for multi-period function
            periods_list = (
                periods if isinstance(periods, (list, tuple)) else [periods]
            )
            seasonals_list = (
                seasonals
                if isinstance(seasonals, (list, tuple))
                else [seasonals]
            )

            return _multi_period_stl(
                time_series,
                periods_list,
                seasonals_list,
                trend,
                low_pass,
                seasonal_deg,
                trend_deg,
                low_pass_deg,
                seasonal_jump,
                trend_jump,
                low_pass_jump,
                robust,
                return_components,
                return_numpy,
            )
    else:
        # Single-period decomposition (original logic)
        period = periods  # periods is now a single value
        seasonal = seasonals  # seasonals is now a single value

        # Ensure time_series is a pandas Series for STL
        if isinstance(time_series, np.ndarray):
            time_series = pd.Series(time_series)

        # Perform STL decomposition
        stl = STL(
            time_series,
            period=period,
            seasonal=seasonal,
            trend=trend,
            low_pass=low_pass,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            seasonal_jump=seasonal_jump,
            trend_jump=trend_jump,
            low_pass_jump=low_pass_jump,
            robust=robust,
        )

        stl_result = stl.fit()

        if not return_components:
            return stl_result

        # Extract components
        components = {
            "trend": stl_result.trend,
            "seasonal": stl_result.seasonal,
            "resid": stl_result.resid,
            "stl_result": stl_result,
        }

        # Add weights if robust estimation was used
        if robust:
            components["weights"] = stl_result.weights

        if return_numpy:
            components["trend"] = components["trend"].to_numpy()
            components["seasonal"] = components["seasonal"].to_numpy()
            components["resid"] = components["resid"].to_numpy()

        return components


def plot_stl_decomposition(
    time_series: Union[pd.Series, np.ndarray],
    period: Optional[Union[int, List[int]]] = None,
    figsize: tuple = (12, 10),
    title: Optional[str] = None,
    multi_period: bool = False,
    **stl_kwargs,
) -> tuple[Any, List[Any]]:
    """
    Perform STL decomposition and create a comprehensive visualization.

    This function supports both single-period and multi-period decomposition plotting.
    For multi-period decomposition, it creates additional subplots showing individual
    seasonal components.

    Parameters
    ----------
    time_series : Union[pd.Series, np.ndarray]
        Input time series data.
    period : Optional[Union[int, list[int]]], default=None
        Period(s) for seasonal decomposition. For multi-period: list of periods.
    figsize : tuple, default=(12, 10)
        Figure size for the plot. Will be adjusted automatically for multi-period.
    title : Optional[str], default=None
        Title for the entire plot.
    multi_period : bool, default=False
        If True, performs multi-period decomposition and creates additional plots.
    **stl_kwargs
        Additional keyword arguments passed to perform_stl_decomposition.

    Returns
    -------
    tuple
        (fig, axes) - matplotlib figure and axes objects.

    Raises
    ------
    ImportError
        If matplotlib is not available.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> data = np.sin(2 * np.pi * np.arange(100) / 7) + 0.1 * np.random.randn(100)
    >>> ts = pd.Series(data, index=dates)
    >>>
    >>> # Single-period plotting
    >>> fig, axes = plot_stl_decomposition(ts, period=7)
    >>> plt.show()
    >>>
    >>> # Multi-period plotting
    >>> fig, axes = plot_stl_decomposition(ts, period=[7, 30], multi_period=True)
    >>> plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Please install it with: pip install matplotlib"
        )

    if isinstance(time_series, pd.Series):
        # Remove duplicate indices, keeping the first occurrence
        time_series = time_series[~time_series.index.duplicated(keep="first")]
        logger.info(
            f"Removed duplicate indices. Data length: {len(time_series)}"
        )
    elif isinstance(time_series, np.ndarray):
        # For numpy arrays, we don't need to handle duplicates
        pass

    # Perform STL decomposition
    components = perform_stl_decomposition(
        time_series, period=period, multi_period=multi_period, **stl_kwargs
    )

    # Determine number of subplots based on multi-period mode
    if multi_period and "seasonal_components" in components:
        total_plots = 4 + len(
            components["seasonal_components"]
        )  # Original + individual seasonals
        # Adjust figsize for more plots
        figsize = (figsize[0], figsize[1] * (total_plots / 4))
    else:
        total_plots = 4

    # Create subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=figsize, sharex=True)

    # Ensure axes is always a list for type safety
    if total_plots == 1:
        axes_list = [axes]  # Single subplot case
    else:
        # When total_plots > 1, axes is a numpy array
        if hasattr(axes, "flatten"):
            # Convert numpy array to list and ensure each element is an Axes object
            axes_list = []
            for ax in axes.flatten():
                axes_list.append(ax)
        else:
            axes_list = list(axes)  # Fallback for other array-like objects

    # Type assertion to help static type checkers understand these are matplotlib axes
    axes_list = cast(List[Axes], axes_list)

    # Ensure all components have the same length
    expected_length = len(time_series)
    for component_name in ["trend", "seasonal", "resid"]:
        if len(components[component_name]) != expected_length:
            logger.warning(
                f"Warning: {component_name} component length ({len(components[component_name])}) "
                f"doesn't match time series length ({expected_length})"
            )

    # Plot original time series
    if isinstance(time_series, pd.Series):
        # drop values with the same index
        x_axis = time_series.index
        # Since we know it's a pandas Series, we can safely access .values
        plot_data = time_series.values
        axes_list[0].plot(
            x_axis,
            plot_data,
            "b-",
            linewidth=1,
            label="Original",
        )
    else:
        x_axis = np.arange(len(time_series))
        axes_list[0].plot(
            x_axis, time_series, "b-", linewidth=1, label="Original"
        )

    # Type assertion for matplotlib axes
    assert hasattr(axes_list[0], "set_title"), "Expected matplotlib axes object"
    axes_list[0].set_title("Original Time Series")
    axes_list[0].legend()
    axes_list[0].grid(True, alpha=0.3)

    # Plot trend component
    axes_list[1].plot(
        x_axis, components["trend"], "r-", linewidth=1.5, label="Trend"
    )
    assert hasattr(axes_list[1], "set_title"), "Expected matplotlib axes object"
    axes_list[1].set_title("Trend Component")
    axes_list[1].legend()
    axes_list[1].grid(True, alpha=0.3)

    # Plot seasonal component
    axes_list[2].plot(
        x_axis, components["seasonal"], "g-", linewidth=1, label="Seasonal"
    )
    assert hasattr(axes_list[2], "set_title"), "Expected matplotlib axes object"
    axes_list[2].set_title("Seasonal Component")
    axes_list[2].legend()
    axes_list[2].grid(True, alpha=0.3)

    # Plot residual component
    axes_list[3].plot(
        x_axis, components["resid"], "k-", linewidth=0.8, label="Residual"
    )
    assert hasattr(axes_list[3], "set_title"), "Expected matplotlib axes object"
    axes_list[3].set_title("Residual Component")
    axes_list[3].legend()
    axes_list[3].grid(True, alpha=0.3)

    # Add horizontal line at y=0 for residual plot
    axes_list[3].axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # ===== MULTI-PERIOD ADDITIONAL PLOTS =====
    if multi_period and "seasonal_components" in components:
        # Plot combined seasonal component
        axes_list[4].plot(
            x_axis,
            components["seasonal"],
            "m-",
            linewidth=1.5,
            label="Combined Seasonal",
        )
        assert hasattr(axes_list[4], "set_title"), (
            "Expected matplotlib axes object"
        )
        axes_list[4].set_title("Combined Seasonal Component")
        axes_list[4].legend()
        axes_list[4].grid(True, alpha=0.3)

        # Plot individual seasonal components
        for i, (seasonal_comp, period_val) in enumerate(
            zip(components["seasonal_components"], components["periods"])
        ):
            plot_idx = 5 + i
            if plot_idx < len(axes_list):
                axes_list[plot_idx].plot(
                    x_axis,
                    seasonal_comp,
                    "c-",
                    linewidth=1,
                    label=f"Period {period_val}",
                )
                assert hasattr(axes_list[plot_idx], "set_title"), (
                    "Expected matplotlib axes object"
                )
                axes_list[plot_idx].set_title(
                    f"Seasonal Component (Period {period_val})"
                )
                axes_list[plot_idx].legend()
                axes_list[plot_idx].grid(True, alpha=0.3)

    # Set overall title
    if title:
        if multi_period and "periods" in components:
            fig.suptitle(
                f"{title} - Multi-Period Decomposition (Periods: {components['periods']})",
                fontsize=16,
            )
        else:
            fig.suptitle(title, fontsize=16)
    else:
        if multi_period and "periods" in components:
            fig.suptitle(
                f"Multi-Period STL Decomposition (Periods: {components['periods']})",
                fontsize=16,
            )
        else:
            fig.suptitle("STL Decomposition", fontsize=16)

    # Adjust layout
    plt.tight_layout()

    return fig, axes_list


def _single_period_stl(
    time_series: Union[pd.Series, np.ndarray],
    period: int,
    seasonal: int,
    trend: Optional[int],
    low_pass: Optional[int],
    seasonal_deg: int,
    trend_deg: int,
    low_pass_deg: int,
    seasonal_jump: int,
    trend_jump: int,
    low_pass_jump: int,
    robust: bool,
    return_components: bool,
    return_numpy: bool,
) -> Union[dict, Any]:
    """Optimized single-period STL decomposition."""

    # Ensure time_series is a pandas Series for STL
    if isinstance(time_series, np.ndarray):
        time_series = pd.Series(time_series)

    stl = STL(
        time_series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
        robust=robust,
    )

    stl_result = stl.fit()

    if not return_components:
        return stl_result

    components = {
        "trend": stl_result.trend,
        "seasonal": stl_result.seasonal,
        "resid": stl_result.resid,
        "stl_result": stl_result,
        "seasonal_components": [stl_result.seasonal],
        "periods": [period],
        "stl_results": [stl_result],
        "denormalized_reconstructed": stl_result.trend
        + stl_result.seasonal
        + stl_result.resid,
    }

    if robust:
        components["weights"] = stl_result.weights

    if return_numpy:
        components["trend"] = components["trend"].to_numpy()
        components["seasonal"] = components["seasonal"].to_numpy()
        components["resid"] = components["resid"].to_numpy()
        components["seasonal_components"] = [
            comp.to_numpy() if hasattr(comp, "to_numpy") else np.array(comp)
            for comp in components["seasonal_components"]
        ]

    return components


def _multi_period_stl(
    time_series: Union[pd.Series, np.ndarray],
    periods: Union[List[int], tuple],
    seasonals: Union[List[int], tuple],
    trend: Optional[int],
    low_pass: Optional[int],
    seasonal_deg: int,
    trend_deg: int,
    low_pass_deg: int,
    seasonal_jump: int,
    trend_jump: int,
    low_pass_jump: int,
    robust: bool,
    return_components: bool,
    return_numpy: bool,
) -> Union[dict, Any]:
    """Optimized multi-period STL decomposition with reduced memory overhead."""

    n = len(time_series)
    seasonal_components = []
    stl_results = []

    # Pre-allocate arrays for better memory efficiency
    if isinstance(time_series, pd.Series):
        # Since we know it's a pandas Series, we can safely access .values
        current_data = (
            time_series.values.copy()
        )  # Work with numpy array directly
        time_index = time_series.index
    else:
        current_data = np.array(time_series).copy()
        time_index = np.arange(len(time_series))

    seasonal_array = np.zeros((len(periods), n))

    # Perform sequential decomposition with optimized memory usage
    for i, (period, seasonal_param) in enumerate(zip(periods, seasonals)):
        # Convert back to pandas Series for STL (required by statsmodels)
        current_series = pd.Series(current_data, index=time_index)

        # Perform STL decomposition
        stl = STL(
            current_series,
            period=period,
            seasonal=seasonal_param,
            trend=trend,
            low_pass=low_pass,
            seasonal_deg=seasonal_deg,
            trend_deg=trend_deg,
            low_pass_deg=low_pass_deg,
            seasonal_jump=seasonal_jump,
            trend_jump=trend_jump,
            low_pass_jump=low_pass_jump,
            robust=robust,
        )

        stl_result = stl.fit()
        stl_results.append(stl_result)

        # Extract seasonal component and store efficiently
        if hasattr(stl_result.seasonal, "values"):
            seasonal_component = stl_result.seasonal.values
        else:
            seasonal_component = np.array(stl_result.seasonal)
        seasonal_array[i] = seasonal_component
        seasonal_components.append(stl_result.seasonal)

        # Update current_data for next iteration (vectorized operation)
        if i < len(periods) - 1:
            current_data = current_data.astype(
                float
            ) - seasonal_component.astype(float)

    # Combine seasonal components efficiently
    combined_seasonal = np.sum(seasonal_array, axis=0)
    combined_seasonal = pd.Series(combined_seasonal, index=time_index)

    # Get final trend and residual from the last decomposition
    final_trend = stl_results[-1].trend
    final_resid = stl_results[-1].resid

    if not return_components:
        return stl_results[-1]

    # Prepare output components
    components = {
        "trend": final_trend,
        "seasonal": combined_seasonal,
        "resid": final_resid,
        "stl_result": stl_results[-1],
        "seasonal_components": seasonal_components,
        "periods": periods,
        "stl_results": stl_results,
        "denormalized_reconstructed": final_trend
        + combined_seasonal
        + final_resid,
    }

    if robust:
        components["weights"] = stl_results[-1].weights

    if return_numpy:
        components["trend"] = components["trend"].to_numpy()
        components["seasonal"] = components["seasonal"].to_numpy()
        components["resid"] = components["resid"].to_numpy()
        components["seasonal_components"] = [
            comp.to_numpy() if hasattr(comp, "to_numpy") else np.array(comp)
            for comp in components["seasonal_components"]
        ]

    return components


def _detect_multiple_periods(
    time_series: Union[pd.Series, np.ndarray], max_periods: int = 3
) -> List[int]:
    """
    Optimized multiple seasonal period detection using improved FFT analysis.

    This version reduces computational overhead and improves accuracy.
    """
    # Handle both pandas Series and numpy arrays
    if isinstance(time_series, pd.Series):
        # Since we know it's a pandas Series, we can safely access .values
        data = np.asarray(time_series.values)
    else:
        data = np.asarray(time_series)
    n = len(data)

    # Early exit for short series
    if n < 50:
        return [7]  # Default for very short series

    # Use power-of-2 length for FFT efficiency
    fft_length = 2 ** int(np.log2(n))
    if fft_length < n:
        fft_length *= 2

    # Pad data to power-of-2 length for efficient FFT
    padded_data = np.pad(data, (0, fft_length - n), mode="edge")

    # Compute FFT efficiently
    fft = np.fft.rfft(padded_data)  # Use rfft for real data (faster)
    power = np.abs(fft) ** 2

    # Focus on relevant frequency range (avoid very short/long periods)
    min_period = 3
    max_period = min(n // 4, 365)  # Reasonable upper bound

    # Convert to periods and filter
    freqs = np.fft.rfftfreq(fft_length)
    periods = 1 / (freqs + 1e-10)  # Avoid division by zero

    # Filter periods in reasonable range
    valid_mask = (periods >= min_period) & (periods <= max_period)
    valid_periods = periods[valid_mask]
    valid_power = power[valid_mask]

    if len(valid_periods) == 0:
        return [7]  # Default fallback

    # Find peaks more efficiently
    from scipy.signal import find_peaks

    # Normalize power for better peak detection
    normalized_power = valid_power / np.max(valid_power)

    # Find peaks with adaptive threshold
    threshold = 0.1 + 0.3 * np.std(normalized_power)
    peaks, properties = find_peaks(
        normalized_power, height=threshold, distance=5, prominence=0.1
    )

    if len(peaks) == 0:
        return [7]  # Default fallback

    # Score peaks by power and prominence
    peak_scores = []
    for peak_idx in peaks:
        score = (
            normalized_power[peak_idx] * 0.7
            + properties["prominences"][list(peaks).index(peak_idx)] * 0.3
        )
        peak_scores.append((valid_periods[peak_idx], score))

    # Sort by score and take top periods
    peak_scores.sort(key=lambda x: x[1], reverse=True)
    detected_periods = [int(round(p[0])) for p in peak_scores[:max_periods]]

    # Remove duplicates and validate
    detected_periods = list(
        dict.fromkeys(detected_periods)
    )  # Preserve order, remove duplicates

    # Fallback to common periods if not enough detected
    if len(detected_periods) < max_periods:
        common_periods = [7, 12, 24, 30, 52, 365]
        for p in common_periods:
            if p not in detected_periods and min_period <= p <= max_period:
                detected_periods.append(p)
                if len(detected_periods) >= max_periods:
                    break

    return detected_periods[:max_periods]
