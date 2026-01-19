import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from synthefy_pkg.fm_evals.formats.metrics import ForecastMetrics


def mae(history: np.ndarray, target: np.ndarray, forecast: np.ndarray) -> float:
    """Compute Mean Absolute Error, robust to NaNs and empty inputs."""
    # Ensure inputs are numpy arrays
    target = np.asarray(target)
    forecast = np.asarray(forecast)

    if target.size == 0 or forecast.size == 0:
        return float("nan")

    # Check if there are any valid (non-NaN) values in both arrays
    valid_mask = ~np.isnan(target) & ~np.isnan(forecast)
    if not np.any(valid_mask):
        return float("nan")

    return float(np.nanmean(np.abs(target - forecast)))


def mape(
    history: np.ndarray, target: np.ndarray, forecast: np.ndarray
) -> float:
    """Compute Mean Absolute Percentage Error, robust to NaNs, zeros, and empty inputs."""
    # Ensure inputs are numpy arrays
    target = np.asarray(target)
    forecast = np.asarray(forecast)

    if target.size == 0 or forecast.size == 0:
        return np.nan

    # Create mask to exclude zeros and NaNs from target
    valid_mask = (target != 0) & ~np.isnan(target) & ~np.isnan(forecast)

    if not np.any(valid_mask):
        return np.nan

    # Compute MAPE only on valid values
    valid_target = target[valid_mask]
    valid_forecast = forecast[valid_mask]

    return float(
        np.nanmean(np.abs((valid_forecast - valid_target) / valid_target))
    )


def nmae(
    history: np.ndarray, target: np.ndarray, forecast: np.ndarray
) -> float:
    """Compute Normalized Mean Absolute Error, robust to NaNs and empty inputs."""
    # Ensure inputs are numpy arrays
    history = np.asarray(history)
    target = np.asarray(target)
    forecast = np.asarray(forecast)

    if target.size == 0 or forecast.size == 0 or history.size == 0:
        return np.nan

    # Check if there are any valid (non-NaN) values in both arrays
    valid_mask = ~np.isnan(target) & ~np.isnan(forecast)
    if not np.any(valid_mask):
        return np.nan

    # Reshape inputs to 2D arrays as required by sklearn
    history_2d = history.reshape(-1, 1)
    target_2d = target.reshape(-1, 1)
    forecast_2d = forecast.reshape(-1, 1)

    # Fit scaler on history data and transform target and forecast
    scaler = StandardScaler()
    scaler.fit(history_2d)

    target_scaled = np.asarray(scaler.transform(target_2d)).ravel()
    forecast_scaled = np.asarray(scaler.transform(forecast_2d)).ravel()

    return mae(history, target_scaled, forecast_scaled)


def mse(history: np.ndarray, target: np.ndarray, forecast: np.ndarray) -> float:
    """Compute Mean Squared Error, robust to NaNs and empty inputs."""
    # Ensure inputs are numpy arrays
    target = np.asarray(target)
    forecast = np.asarray(forecast)

    if target.size == 0 or forecast.size == 0:
        return np.nan

    valid_mask = ~np.isnan(target) & ~np.isnan(forecast)
    if not np.any(valid_mask):
        return np.nan

    return float(np.nanmean(np.square(target - forecast)))


def compute_on_df(df: pd.DataFrame, col: str) -> dict:
    """
    Compute MAE and MAPE for the target region using the given column and its forecast.
    Args:
        df: DataFrame as produced by join_as_dfs
        col: column name (ground truth)
    Returns:
        dict with keys 'mae' and 'mape'
    """
    target_mask = df["split"] == "target"
    target = df.loc[target_mask, col].to_numpy()
    forecast_col = f"{col}_forecast"
    if forecast_col not in df.columns:
        raise ValueError(
            f"Forecast column {forecast_col} not found in DataFrame."
        )
    forecast = df.loc[target_mask, forecast_col].to_numpy()
    # history is not used in these metrics, but for API compatibility:
    history = df.loc[df["split"] == "history", col].to_numpy()

    # Ensure all inputs are numpy arrays
    target = np.asarray(target)
    forecast = np.asarray(forecast)
    history = np.asarray(history)

    return {
        "mae": mae(history, target, forecast),
        "mape": mape(history, target, forecast),
        "nmae": nmae(history, target, forecast),
        "mse": mse(history, target, forecast),
    }


def compute_sample_metrics(eval_sample, forecast_sample) -> ForecastMetrics:
    """
    Compute metrics for a single sample.
    """
    # Ensure all inputs are numpy arrays
    true = np.asarray(eval_sample.target_values)
    pred = np.asarray(forecast_sample.values)
    history = np.asarray(eval_sample.history_values)

    # Use the existing mae and mape functions
    mae_val = mae(history, true, pred)
    mape_val = mape(history, true, pred)
    nmae_val = nmae(history, true, pred)
    mse_val = mse(history, true, pred)

    # Ensure sample_id is a string for ForecastMetrics
    sample_id = str(eval_sample.sample_id)

    return ForecastMetrics(
        sample_id=sample_id,
        mae=mae_val,
        mape=mape_val,
        nmae=nmae_val,
        mse=mse_val,
        median_mae=mae_val,  # For single sample, median equals mean
        median_mape=mape_val,  # For single sample, median equals mean
        median_nmae=nmae_val,
        median_mse=mse_val,
    )
