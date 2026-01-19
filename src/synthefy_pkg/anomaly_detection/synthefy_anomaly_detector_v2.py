import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from prophet import Prophet

from synthefy_pkg.anomaly_detection.utils.basic_utils import save_anomaly_results
from synthefy_pkg.anomaly_detection.utils.plotting_utils import (
    plot_anomaly_windows,
    plot_concurrent_anomalies,
)
from synthefy_pkg.utils.basic_utils import check_missing_cols

COMPILE = True

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))
DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")

NUM_WEEKLY_LAGS = 5  # Number of weeks to look back for pattern detection

VALID_FREQUENCY_UNITS = ["T", "min", "H", "D", "W", "M"]


@dataclass
class AnomalyMetadata:
    """
    Metadata container for each detected anomaly.

    Attributes:
        timestamp: The timestamp when the anomaly occurred
        score: Normalized anomaly score (higher means more anomalous)
        original_value: The actual value at the anomaly point
        predicted_value: The expected/predicted value at the anomaly point
        group_metadata: Additional metadata about the group this anomaly belongs to
    """

    timestamp: Any
    score: float
    original_value: float
    predicted_value: float
    group_metadata: Dict[str, Any]


def _parse_and_validate_frequency(freq_str: str) -> Dict[str, Any]:
    """
    Parse frequency string and generate all frequency-related metadata and configurations.

    Args:
        freq_str: Frequency string (e.g., '5T', '22H', '2D')

    Returns:
        Dictionary containing:
        - periods_per_day: Number of periods in a day
        - periods_per_week: Number of periods in a week
        - periods_per_month: Number of periods in a month
        - min_points_required: Minimum required points for analysis
        - windows: Default window configurations for anomaly detection
        - freq_unit: Frequency unit (T, H, D, W, M)

    Examples:
        '5T' -> 288 periods per day (24 * 60 / 5)
        '22H' -> ~1.09 periods per day (24 / 22)
        'D' -> 1 period per day
    """
    # Extract numeric part if it exists
    numeric_part = "".join(filter(str.isdigit, freq_str))
    unit_part = "".join(filter(str.isalpha, freq_str))

    if not unit_part:
        raise ValueError(
            f"Invalid frequency string: {freq_str}. Must contain a valid unit (T, H, D, W, M)"
        )

    if unit_part not in VALID_FREQUENCY_UNITS:
        raise ValueError(
            f"Invalid frequency unit: {unit_part}, choose from {VALID_FREQUENCY_UNITS}"
        )

    num = int(numeric_part) if numeric_part else 1

    # Base periods in a day for each unit
    base_periods = {
        "T": {"day": 24 * 60, "week": 24 * 60 * 7, "month": 24 * 60 * 30},
        "min": {"day": 24 * 60, "week": 24 * 60 * 7, "month": 24 * 60 * 30},
        "H": {"day": 24, "week": 24 * 7, "month": 24 * 30},
        "D": {"day": 1, "week": 7, "month": 30},
        "W": {"day": 1 / 7, "week": 1, "month": 4},
        "M": {"day": 1 / 30, "week": 1 / 4, "month": 1},
    }

    # Calculate actual periods
    periods = {
        "periods_per_day": base_periods[unit_part]["day"] / num,
        "periods_per_week": base_periods[unit_part]["week"] / num,
        "periods_per_month": base_periods[unit_part]["month"] / num,
    }

    # Generate window configurations based on frequency type
    if unit_part in ["T", "min"]:  # Minutely data
        windows = {
            "volatility": max(1, int(periods["periods_per_day"] / 24)),  # 1 hour worth
            "pattern": max(1, int(periods["periods_per_day"])),  # 1 day worth
            "min_points": max(1, int(periods["periods_per_day"] * 2)),  # 2 days worth
            "baseline_multiplier": 7,
        }
    elif unit_part == "H":  # Hourly data
        windows = {
            "volatility": max(1, int(24 / num)),  # 1 day worth
            "pattern": max(1, int((24 * 7) / num)),  # 1 week worth
            "min_points": max(1, int((24 * 2) / num)),  # 2 days worth
            "baseline_multiplier": 7,
        }
    elif unit_part == "D":  # Daily data
        windows = {
            "volatility": max(1, int(7 / num)),  # 1 week worth
            "pattern": max(1, int(28 / num)),  # 4 weeks worth
            "min_points": max(1, int(14 / num)),  # 2 weeks worth
            "baseline_multiplier": 4,
        }
    elif unit_part == "W":  # Weekly data
        windows = {
            "volatility": max(1, int(4 / num)),  # 1 month worth
            "pattern": max(1, int(12 / num)),  # 3 months worth
            "min_points": max(1, int(8 / num)),  # 2 months worth
            "baseline_multiplier": 4,
        }
    else:  # Monthly data
        windows = {
            "volatility": max(1, int(3 / num)),  # 1 quarter worth
            "pattern": max(1, int(12 / num)),  # 1 year worth
            "min_points": max(1, int(12 / num)),  # 1 year worth
            "baseline_multiplier": 4,
        }

    return {
        **periods,
        "min_points_required": windows["min_points"],
        "windows": windows,
        "freq_unit": unit_part,
    }


def normalize_score(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series using z-score normalization.

    Args:
        series: The pandas Series to normalize.

    Returns:
        A pandas Series with normalized values.
    """
    if series.empty or series.std() == 0:
        return series
    return (series - series.mean()) / (series.std() + 1e-6)


def _calculate_long_term_patterns(df: pd.DataFrame, freq_unit: str) -> pd.Series:
    """
    Calculate patterns for weekly/monthly data.
    """
    # For weekly data, compare with previous 12 weeks
    # For monthly data, compare with previous 12 months and same month last year
    n_periods = 12

    historical_values = pd.DataFrame()
    for i in range(1, n_periods + 1):
        historical_values[f"lag_{i}"] = df["y"].shift(i)

    if freq_unit == "M":
        # Add same month last year
        historical_values["last_year"] = df["y"].shift(12)

    # Calculate mean and std of historical values
    historical_mean = historical_values.mean(axis=1)
    historical_std = historical_values.std(axis=1)

    # Calculate normalized deviation from pattern
    pattern_score = np.abs(df["y"] - historical_mean) / (historical_std + 1e-6)

    return pattern_score


def calculate_volatility_metrics(
    df: pd.DataFrame, volatility_window: int, baseline_window: int
) -> pd.DataFrame:
    """
    Calculate volatility metrics and scores for a given DataFrame.

    Args:
        df: DataFrame with 'y' column for time series data
        volatility_window: Window size for rolling calculations
        baseline_window: Window size for baseline calculations

    Returns:
        DataFrame with added volatility metrics and scores
    """
    # Calculate multiple volatility metrics
    df["rolling_std"] = df["y"].rolling(window=volatility_window).std()
    df["rolling_mad"] = (
        df["y"]
        .rolling(window=volatility_window)
        .apply(lambda x: np.median(np.abs(x - np.median(x))))
    )
    df["ewm_std"] = df["y"].ewm(span=volatility_window).std()

    # Calculate baseline statistics using configurable window
    df["rolling_mean_std"] = df["rolling_std"].rolling(window=baseline_window).mean()
    df["rolling_mean_mad"] = df["rolling_mad"].rolling(window=baseline_window).mean()
    df["rolling_mean_ewm"] = df["ewm_std"].rolling(window=baseline_window).mean()

    # Calculate volatility scores
    df["std_score"] = (df["rolling_std"] - df["rolling_mean_std"]) / df[
        "rolling_mean_std"
    ]
    df["mad_score"] = (df["rolling_mad"] - df["rolling_mean_mad"]) / df[
        "rolling_mean_mad"
    ]
    df["ewm_score"] = (df["ewm_std"] - df["rolling_mean_ewm"]) / df["rolling_mean_ewm"]

    # Combine scores
    df["volatility_score"] = df[["std_score", "mad_score", "ewm_score"]].mean(axis=1)

    return df


def _calculate_pattern_scores(
    df: pd.DataFrame, freq_unit: str, periods_per_week: float
) -> pd.Series:
    """
    Calculate pattern-based anomaly scores using time-based features.

    Args:
        df: DataFrame with 'ds' and 'y' columns
        freq_unit: Frequency type ('H', 'T', 'min', 'D')
        periods_per_week: Number of periods in a week

    Returns:
        Series containing the final pattern scores
    """
    pattern_df = df.copy()

    # Convert periods_per_week to integer for shifting
    periods_per_week_int = int(round(periods_per_week))

    # Add time features adjusted for frequency
    if "".join(filter(str.isalpha, freq_unit)) in ["H", "T", "min"]:
        pattern_df["hour"] = pattern_df["ds"].dt.hour

    pattern_df["day_of_week"] = pattern_df["ds"].dt.dayofweek

    # Calculate weekly patterns
    weekly_lags = pd.DataFrame()
    for i in range(1, NUM_WEEKLY_LAGS):
        weekly_lags[f"lag_{i}w"] = pattern_df["y"].shift(i * periods_per_week_int)

    weekly_score = np.abs(pattern_df["y"] - weekly_lags.mean(axis=1)) / (
        weekly_lags.std(axis=1) + 1e-6
    )

    # Calculate time-of-day patterns
    tod_pattern = pattern_df.groupby("hour")["y"].transform("mean")
    tod_std = pattern_df.groupby("hour")["y"].transform("std")
    tod_score = np.abs(pattern_df["y"] - tod_pattern) / (tod_std + 1e-6)

    # Calculate day-of-week patterns
    dow_pattern = pattern_df.groupby("day_of_week")["y"].transform("mean")
    dow_std = pattern_df.groupby("day_of_week")["y"].transform("std")
    dow_score = np.abs(pattern_df["y"] - dow_pattern) / (dow_std + 1e-6)

    # Combine pattern scores
    pattern_score = pd.DataFrame(
        {"weekly_score": weekly_score, "tod_score": tod_score, "dow_score": dow_score}
    ).mean(axis=1)
    assert isinstance(pattern_score, pd.Series)
    return pattern_score


class AnomalyDetector:
    """
    Anomaly detection system that supports multiple detection methods.

    Supports the following anomaly types:
    - Peak: Single-point outliers using Prophet forecasting
    - Scattered: Periods of high volatility using rolling statistics
    - Multiple Week: Deviations from historical weekly patterns

    The detector can be configured via the preprocessing config file
    in `examples/configs/preprocessing_configs/` with the following structure:
    ```
    anomaly_detection:
        model:
            type: "synthefy_anomaly_detection_v2"
            params:
                daily_seasonality: true  # Enable daily seasonality patterns
                weekly_seasonality: true # Enable weekly seasonality patterns
                # Additional Prophet parameters can be configured here
        frequency:
            type: "H"        # Frequency type: H (hourly), D (daily), T/min (minutely)
            windows:
                volatility: 168  # Window size for scattered anomalies (e.g. 168 hours = 1 week)
                pattern: 168     # Window size for pattern analysis (e.g. 168 hours = 1 week)
                min_points: 50   # Minimum required data points for analysis
                baseline_multiplier: 7  # Multiplier for baseline window (e.g. 7x volatility window)
        sd_thresholds:
            peak: 2.5         # Z-score threshold for peak anomalies
            scattered: 2.5    # Threshold for volatility-based anomalies
            out_of_pattern: 2.5  # Threshold for pattern deviation anomalies
        weights:              # Weights for combining different anomaly types
            peak: 1.0        # Weight for peak anomalies
            scattered: 0.8   # Weight for scattered anomalies
            out_of_pattern: 0.6  # Weight for pattern anomalies
    ```
    """

    timestamps_col: List[str]
    timeseries_cols: List[str]
    anomaly_config: Dict[str, Any]
    group_labels_cols: List[str]
    freq_metadata: Dict[str, Any]

    def __init__(self, config_source: Union[str, Dict[str, Any]]):
        """
        Initialize the anomaly detector with configuration.

        Args:
            config_source: Path to the configuration file
        """
        (
            self.timestamps_col,
            self.timeseries_cols,
            self.anomaly_config,
            self.group_labels_cols,
            self.config,
        ) = self._get_config(config_source)
        self.anomaly_config, self.freq_metadata = self._validate_config()

    def _get_config(
        self, config_source: Union[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str], Dict[str, Any], List[str], Dict[str, Any]]:
        """
        Load the anomaly detection configuration from file or dictionary.

        Args:
            config_source: Path to the configuration file or dictionary

        Returns:
            Dictionary containing the anomaly detection configuration
        """
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config = json.load(f)
        elif isinstance(config_source, dict):
            config = config_source

        group_labels_cols = config.get("group_labels", {}).get("cols", [])
        if not group_labels_cols:
            logger.warning("No group labels provided in config")

        try:
            timeseries_cols = config["timeseries"]["cols"]
            timestamps_col = config["timestamps_col"]
            anomaly_config = config["anomaly_detection"]
        except KeyError as e:
            raise ValueError(f"Missing required param in config: {str(e)}")

        return (
            timestamps_col,
            timeseries_cols,
            anomaly_config,
            group_labels_cols,
            config,
        )

    def _validate_config(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate the anomaly detection configuration.
        """

        default_config = {
            "model": {
                "type": "synthefy_anomaly_detector_v2",
                "params": {
                    "daily_seasonality": True,
                    "weekly_seasonality": True,
                    "yearly_seasonality": False,
                    "interval_width": 0.95,
                },
            },
            "sd_thresholds": {
                "peak": 2.5,
                "scattered": 2.5,
                "out_of_pattern": 2.5,
            },
            "concurrent_anomalies": {
                "enabled": False,
                "time_window": "1H",
                # Minimum number of KPIs that must show anomalies within the window
                "min_kpis_for_concurrent_anomaly": 2,
                # Whether to consider different anomaly types together
                "combine_types": True,
            },
        }

        anomaly_config = self.anomaly_config

        # Validate and merge with defaults for each section
        # Model configuration
        if "model" not in anomaly_config:
            logger.warning("No model configuration found, using defaults")
            anomaly_config["model"] = default_config["model"]
        else:
            if "type" not in anomaly_config["model"]:
                logger.warning("No model type specified, using default")
                anomaly_config["model"]["type"] = default_config["model"]["type"]
            elif anomaly_config["model"]["type"] != "synthefy_anomaly_detector_v2":
                raise ValueError(
                    f"Invalid model type '{anomaly_config['model']['type']}'. "
                    "Only 'synthefy_anomaly_detector_v2' is supported."
                )

            # Validate model params
            if "params" not in anomaly_config["model"]:
                logger.warning("No model parameters specified, using defaults")
                anomaly_config["model"]["params"] = default_config["model"]["params"]

        if "model" in anomaly_config:
            if anomaly_config["model"]["type"] == "synthefy_anomaly_detector_v2":
                valid_prophet_params = {
                    "daily_seasonality",
                    "weekly_seasonality",
                    "yearly_seasonality",
                    "interval_width",
                    "seasonality_mode",
                    "changepoint_prior_scale",
                    "seasonality_prior_scale",
                    "holidays_prior_scale",
                    "changepoint_range",
                }

                invalid_params = (
                    set(anomaly_config["model"]["params"].keys()) - valid_prophet_params
                )
                if invalid_params:
                    raise ValueError(
                        f"Invalid `synthefy_anomaly_detector_v2` parameters in config: {invalid_params}. "
                        f"Valid parameters are: {valid_prophet_params}"
                    )

                # Validate parameter types
                params = anomaly_config["model"]["params"]
                if "daily_seasonality" in params and not isinstance(
                    params["daily_seasonality"], bool
                ):
                    raise ValueError("daily_seasonality must be a boolean")
                if "weekly_seasonality" in params and not isinstance(
                    params["weekly_seasonality"], bool
                ):
                    raise ValueError("weekly_seasonality must be a boolean")
                if "yearly_seasonality" in params and not isinstance(
                    params["yearly_seasonality"], bool
                ):
                    raise ValueError("yearly_seasonality must be a boolean")
                if "interval_width" in params and not (
                    isinstance(params["interval_width"], (int, float))
                    and 0 < params["interval_width"] < 1
                ):
                    raise ValueError("interval_width must be a float between 0 and 1")

        if "frequency" not in anomaly_config:
            raise ValueError(
                "Missing required 'frequency' configuration in anomaly_detection"
            )

        freq_config = anomaly_config["frequency"]
        if "type" not in freq_config:
            raise ValueError("Missing required 'type' in frequency configuration")

        freq_type = freq_config["type"]

        freq_metadata = _parse_and_validate_frequency(freq_type)
        self.freq_unit = freq_metadata.pop("freq_unit")

        # Use the metadata for windows configuration
        if "windows" not in freq_config:
            logger.warning(
                f"No windows specified for {freq_type} frequency, using defaults"
            )
            freq_config["windows"] = freq_metadata.pop("windows")
        else:
            # Validate that all required windows are present
            required_windows = {
                "volatility",
                "pattern",
                "min_points",
                "baseline_multiplier",
            }
            missing_windows = required_windows - set(freq_config["windows"].keys())
            if missing_windows:
                logger.warning(
                    f"Missing window parameters: {missing_windows}. Using frequency-specific defaults for missing values"
                )
                for window in missing_windows:
                    freq_config["windows"][window] = freq_metadata.pop("windows")[
                        window
                    ]

        # Thresholds configuration
        if "sd_thresholds" not in anomaly_config:
            logger.warning("No thresholds configuration found, using defaults")
            anomaly_config["sd_thresholds"] = default_config["sd_thresholds"]

        # Add default concurrent anomalies configuration
        if "concurrent_anomalies" not in anomaly_config:
            logger.warning(
                "No concurrent anomalies configuration found, using defaults"
            )
            anomaly_config["concurrent_anomalies"] = default_config[
                "concurrent_anomalies"
            ]
        else:
            # If concurrent_anomalies exists but some fields are missing, use defaults
            for key, default_value in default_config["concurrent_anomalies"].items():
                if key not in anomaly_config["concurrent_anomalies"]:
                    logger.warning(
                        f"Missing {key} in concurrent_anomalies config, using default: {default_value}"
                    )
                    anomaly_config["concurrent_anomalies"][key] = default_value

        return anomaly_config, freq_metadata

    def load_data(self) -> pd.DataFrame:
        """
        Load data.
        """
        data_path = os.path.join(str(DATASETS_BASE), self.config["filename"])
        try:
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            elif data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            else:
                raise ValueError(
                    f"Unsupported file format. File must be .parquet or .csv: {data_path}"
                )
        except Exception as e:
            raise FileNotFoundError(f"Error loading data: {str(e)}")

        # Validate required columns
        required_cols = (
            self.timestamps_col + self.group_labels_cols + self.timeseries_cols
        )
        check_missing_cols(required_cols, df)

        df = df[required_cols]
        df[self.timestamps_col[0]] = pd.to_datetime(df[self.timestamps_col[0]])
        assert isinstance(df, pd.DataFrame)
        return df

    def _find_anomalies_using_prophet(
        self, df: pd.DataFrame, uncertainty_interval: float = 0.95
    ) -> pd.DataFrame:
        """
        Use Prophet to detect anomalies in a single time series.

        Args:
            df: DataFrame with 'ds' (timestamp) and 'y' (value) columns
            uncertainty_interval: Prophet prediction interval width (0 to 1)

        Returns:
            DataFrame containing detected anomalies with predictions and scores

        Raises:
            ValueError: If data format is invalid
        """
        if not all(col in df.columns for col in ["ds", "y"]):
            raise ValueError("Data must contain 'ds' and 'y' columns")

        model = Prophet(**self.anomaly_config["model"]["params"])
        model.fit(df)
        forecast = model.predict(df)

        forecast["anomaly"] = 0
        forecast["y"] = df["y"].values

        # Mark points outside prediction intervals as anomalies
        forecast.loc[forecast["yhat_upper"] < forecast["y"], "anomaly"] = 1
        forecast.loc[forecast["yhat_lower"] > forecast["y"], "anomaly"] = 1
        anomalies = forecast[forecast["anomaly"] == 1].copy()
        anomalies["residuals"] = np.abs(anomalies["y"] - anomalies["yhat"])

        anomalies["local_std"] = (forecast["yhat_upper"] - forecast["yhat_lower"]) / 4
        anomalies["residuals"] = anomalies["residuals"] / anomalies["local_std"]

        return anomalies

    def _detect_scattered_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect periods of high volatility using rolling statistics.

        Args:
            df: DataFrame with 'ds' and 'y' columns

        Returns:
            DataFrame containing periods of high volatility with scores

        Notes:
            Uses multiple statistical methods:
            - Rolling standard deviation
            - Rolling median absolute deviation (MAD)
            - Exponentially weighted moving standard deviation
        """
        volatility_window = self.anomaly_config["frequency"]["windows"]["volatility"]

        # Get baseline window multiplier from config
        baseline_multiplier = self.anomaly_config["frequency"]["windows"][
            "baseline_multiplier"
        ]
        baseline_window = volatility_window * baseline_multiplier

        if len(df) < baseline_window:
            logger.warning(
                f"Insufficient data points ({len(df)}) for scattered anomaly detection. "
                f"Required: {baseline_window} points for baseline window"
            )
            return pd.DataFrame()

        df = calculate_volatility_metrics(df, volatility_window, baseline_window)

        # Mark anomalies
        sd_threshold = self.anomaly_config["sd_thresholds"]["scattered"]
        df["anomaly"] = 0
        df.loc[df["volatility_score"] > sd_threshold, "anomaly"] = 1

        anomalies = df[df["anomaly"] == 1].copy()
        anomalies["residuals"] = anomalies["volatility_score"]

        assert isinstance(anomalies, pd.DataFrame)
        return anomalies

    def _detect_out_of_pattern_anomalies(
        self, df: pd.DataFrame, pattern_window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Detect deviations from historical patterns.

        Args:
            df: DataFrame with 'ds' and 'y' columns
            pattern_window: Number of hours for pattern detection
                          (defaults to config value)

        Returns:
            DataFrame containing pattern anomalies with scores

        Notes:
            Uses multiple pattern detection methods:
            - Weekly seasonality comparison
            - Time-of-day pattern comparison
            - Day-of-week pattern comparison
        """
        if pattern_window is None:
            pattern_window = self.anomaly_config["frequency"]["windows"]["pattern"]

        # Use stored frequency metadata instead of recalculating
        if self.freq_metadata["periods_per_day"] >= 1:  # Sub-daily or daily data
            df["pattern_score"] = _calculate_pattern_scores(
                df, self.freq_unit, self.freq_metadata["periods_per_week"]
            )
        else:  # Weekly or monthly data
            df["pattern_score"] = _calculate_long_term_patterns(df, self.freq_unit)

        # Mark anomalies
        sd_threshold = self.anomaly_config["sd_thresholds"]["out_of_pattern"]
        df["anomaly"] = 0
        df.loc[df["pattern_score"] > sd_threshold, "anomaly"] = 1

        anomalies = df[df["anomaly"] == 1].copy()
        anomalies["residuals"] = anomalies["pattern_score"]

        return anomalies

    def _process_group(
        self, group_info: Tuple[Union[str, Tuple], pd.DataFrame, str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process a single group of time series data for all anomaly types.

        Args:
            group_info: Tuple containing (group_key, group_df, kpi_column)

        Returns:
            Dictionary containing anomaly DataFrames for each detection method
        """
        group_key, group, kpi_column = group_info

        # Data validation and preprocessing
        if group[kpi_column].nunique() == 1:
            logger.warning(f"Skipping group {group_key} with only one unique value")
            return {}
        if len(group) < self.anomaly_config["frequency"]["windows"]["min_points"]:
            logger.warning(f"Skipping group {group_key} with insufficient data points")
            return {}

        if len(group) != group[self.timestamps_col[0]].nunique():
            # logger.error(f"Skipping group {group_key} with duplicate timestamps")
            logger.warning(f"Group {group_key} has duplicate timestamps")
            group = group.drop_duplicates()

        group = group.rename(
            columns={self.timestamps_col[0]: "ds", kpi_column: "y"}
        ).copy()

        group["ds"] = pd.to_datetime(group["ds"]).dt.tz_localize("UTC")
        group["ds"] = group["ds"].dt.tz_localize(None)
        group = group.sort_values("ds")
        group = group.dropna(subset=["y"])

        if group.empty:
            logger.error(f"Skipping group {group_key} with no valid data points")
            return {}

        peak_anomalies = self._find_anomalies_using_prophet(group)
        scattered_anomalies = self._detect_scattered_anomalies(group)
        weekly_anomalies = self._detect_out_of_pattern_anomalies(group)

        # Add group information only if group labels exist
        for df in [peak_anomalies, scattered_anomalies, weekly_anomalies]:
            if not df.empty and self.group_labels_cols:
                for i, col in enumerate(self.group_labels_cols):
                    # Handle both string and tuple group keys
                    if isinstance(group_key, tuple):
                        df[col] = group_key[i]
                    elif isinstance(group_key, str):
                        df[col] = group_key

        return {
            "peak": peak_anomalies,
            "scattered": scattered_anomalies,
            "out_of_pattern": weekly_anomalies,
        }

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        num_anomalies_limit: int = 50,
        min_anomaly_score: Optional[float] = None,
        n_jobs: int = -1,
    ) -> Tuple[
        Dict[str, Dict[str, Dict[str, List[AnomalyMetadata]]]],
        Dict[str, List[Dict[str, Any]]],
    ]:
        """
        Main method to detect anomalies in the data (parallel version).

        Args:
            df: data
            num_anomalies_limit: Maximum number of anomalies to return per type
            min_anomaly_score: Minimum anomaly score to include (optional)
            n_jobs: Number of parallel jobs (-1 for all cores)

        Returns:
            Tuple containing:
            - Dictionary of individual anomalies by KPI/type/group
            - Dictionary of concurrent anomalies across KPIs
        """
        results = {}

        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        for kpi in self.timeseries_cols:
            logger.info(f"Processing KPI: {kpi}")

            results[kpi] = {
                "peak": {},
                "scattered": {},
                "out_of_pattern": {},
            }

            kpi_data = df[
                [self.timestamps_col[0]] + self.group_labels_cols + [kpi]
            ].copy()

            # Handle no group labels case
            if not self.group_labels_cols:
                # Create a single group with a default key
                group_data = [("single_group", kpi_data, kpi)]
            else:
                grouped_data = list(kpi_data.groupby(self.group_labels_cols))
                group_data = [
                    (
                        name,
                        group,
                        kpi,
                    )
                    for name, group in grouped_data
                ]

            # Process groups in parallel
            all_anomalies_dict = {
                method: pd.DataFrame()
                for method in ["peak", "scattered", "out_of_pattern"]
            }

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_group = {
                    executor.submit(self._process_group, data): data[0]
                    for data in group_data
                }

                for future in as_completed(future_to_group):
                    group_result = future.result()
                    for method in all_anomalies_dict:
                        if method in group_result and not group_result[method].empty:
                            all_anomalies_dict[method] = pd.concat(
                                [all_anomalies_dict[method], group_result[method]]
                            )

            # Convert to final format and apply filters
            for method, anomalies_df in all_anomalies_dict.items():
                if not anomalies_df.empty:
                    if min_anomaly_score is not None:
                        anomalies_df = anomalies_df[
                            anomalies_df["residuals"] >= min_anomaly_score
                        ]

                    method_results = self._extract_anomalies(
                        anomalies_df,
                        self.anomaly_config["sd_thresholds"][method],
                        num_anomalies_limit,
                    )
                    results[kpi][method].update(method_results)

        # Detect concurrent anomalies
        concurrent_results = self._detect_concurrent_anomalies(results)

        return results, concurrent_results

    def _extract_anomalies(
        self,
        anomalies_df: pd.DataFrame,
        sd_threshold: float,
        limit: Optional[int] = None,
    ) -> Dict[str, List[AnomalyMetadata]]:
        """
        Convert anomaly DataFrame to structured format with metadata.

        Args:
            anomalies_df: DataFrame containing anomalies
            sd_threshold: Score threshold for filtering
            limit: Maximum number of anomalies to return per group

        Returns:
            Dictionary mapping group keys to lists of AnomalyMetadata
        """
        results = {}

        # Filter anomalies by threshold before processing
        anomalies_df = anomalies_df[anomalies_df["residuals"] >= sd_threshold]

        # Handle no group labels case
        if not self.group_labels_cols:
            # Process all anomalies as a single group
            group = anomalies_df.sort_values("residuals", ascending=False)
            if limit:
                group = group.head(limit)

            results["single_group"] = [
                AnomalyMetadata(
                    timestamp=row["ds"],
                    score=row["residuals"],
                    original_value=row["y"],
                    predicted_value=row.get("yhat", row["y"]),
                    group_metadata={},  # Empty dict when no group labels
                )
                for _, row in group.iterrows()
            ]
        else:
            # Original group-based processing
            for group_key, group in anomalies_df.groupby(self.group_labels_cols):
                group_key_str = (
                    "-".join(str(x) for x in group_key)
                    if isinstance(group_key, tuple)
                    else str(group_key)
                )
                group = group.sort_values("residuals", ascending=False)
                if limit:
                    group = group.head(limit)

                results[group_key_str] = [
                    AnomalyMetadata(
                        timestamp=row["ds"],
                        score=row["residuals"],
                        original_value=row["y"],
                        predicted_value=row.get("yhat", row["y"]),
                        group_metadata={
                            col: row[col] for col in self.group_labels_cols
                        },
                    )
                    for _, row in group.iterrows()
                ]

        return results

    def _detect_concurrent_anomalies(
        self,
        all_kpi_results: Dict[str, Dict[str, Dict[str, List[AnomalyMetadata]]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect anomalies that occur across multiple KPIs within a specified time window.

        Args:
            all_kpi_results: Results dictionary from detect_anomalies

        Returns:
            Dictionary mapping timestamps to lists of concurrent anomalies
        """
        concurrent_config = self.anomaly_config["concurrent_anomalies"]
        if not concurrent_config["enabled"]:
            return {}

        time_window = pd.Timedelta(concurrent_config["time_window"])
        min_kpis_for_concurrent_anomaly = concurrent_config[
            "min_kpis_for_concurrent_anomaly"
        ]
        combine_types = concurrent_config["combine_types"]

        # Flatten all anomalies into a single list with their metadata
        all_anomalies = []
        for kpi, kpi_results in all_kpi_results.items():
            for anomaly_type, type_results in kpi_results.items():
                for group_key, anomalies in type_results.items():
                    for anomaly in anomalies:
                        all_anomalies.append(
                            {
                                "kpi": kpi,
                                "type": anomaly_type,
                                "group_key": group_key,
                                "timestamp": anomaly.timestamp.isoformat(),
                                "score": anomaly.score,
                                "original_value": anomaly.original_value,
                                "predicted_value": anomaly.predicted_value,
                                "group_metadata": anomaly.group_metadata,
                            }
                        )

        # Sort anomalies by timestamp
        all_anomalies.sort(key=lambda x: pd.Timestamp(x["timestamp"]))

        # Find clusters of concurrent anomalies
        concurrent_clusters = {}

        for i, current_anomaly in enumerate(all_anomalies):
            current_time = pd.Timestamp(current_anomaly["timestamp"])
            cluster_key = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # Find all anomalies within the time window
            concurrent = [current_anomaly]
            for other_anomaly in all_anomalies[i + 1 :]:
                if (
                    pd.Timestamp(other_anomaly["timestamp"]) - current_time
                    <= time_window
                ):
                    # Check if we should combine different anomaly types
                    if (
                        combine_types
                        or other_anomaly["type"] == current_anomaly["type"]
                    ):
                        # For same KPI, only include if it's a different type
                        if other_anomaly["kpi"] != current_anomaly["kpi"] or (
                            not combine_types
                            and other_anomaly["type"] != current_anomaly["type"]
                        ):
                            concurrent.append(other_anomaly)
                else:
                    break

            # Only keep clusters with enough distinct KPIs
            distinct_kpis = len(set(a["kpi"] for a in concurrent))
            if distinct_kpis >= min_kpis_for_concurrent_anomaly:
                concurrent_clusters[cluster_key] = {
                    "timestamp": current_time.isoformat(),
                    "anomalies": concurrent,
                    "distinct_kpis": distinct_kpis,
                    "total_score": sum(a["score"] for a in concurrent),
                    "kpis_involved": list(set(a["kpi"] for a in concurrent)),
                }

        return concurrent_clusters


def main(
    config_path: str, results_path: Optional[str] = None, visualize_plots: bool = False
):
    """
    Main function to run anomaly detection.

    Args:
        config_path: Path to the configuration file
        results_path: Optional path to previously saved anomaly results
    """
    output_path = os.path.join(
        str(DATASETS_BASE), config_path.split("_preprocessing")[0].split("config_")[1]
    )
    # Initialize detector
    detector = AnomalyDetector(config_source=config_path)
    df = detector.load_data()

    # Either load existing results or run detection
    if results_path:
        logger.info(f"Loading existing anomaly results from {results_path}")
        try:
            with open(results_path, "r") as f:
                json_results = json.load(f)

            # Convert JSON back to AnomalyMetadata objects
            results = {}
            for kpi, kpi_results in json_results.items():
                results[kpi] = {}
                for anomaly_type, type_results in kpi_results.items():
                    results[kpi][anomaly_type] = {}
                    for group_key, anomalies in type_results.items():
                        results[kpi][anomaly_type][group_key] = [
                            AnomalyMetadata(
                                timestamp=pd.Timestamp(a["timestamp"]),
                                score=float(a["score"]),
                                original_value=float(a["original_value"]),
                                predicted_value=float(a["predicted_value"]),
                                group_metadata=a["group_metadata"],
                            )
                            for a in anomalies
                        ]
        except Exception as e:
            raise ValueError(f"Error loading results file: {str(e)}")

        concurrent_results_path = results_path.replace(".json", "_concurrent.json")
        logger.info(f"Loading existing anomaly results from {concurrent_results_path}")
        try:
            with open(concurrent_results_path, "r") as f:
                concurrent_results = json.load(f)

        except Exception as e:
            raise ValueError(f"Error loading results file: {str(e)}")
    else:
        logger.info(f"Starting anomaly detection using config: {config_path}")
        results, concurrent_results = detector.detect_anomalies(df)

        # Save results with timestamp
        results_filename = "anomaly_detection_results_baseline"
        save_anomaly_results(results, concurrent_results, output_path, results_filename)

    if visualize_plots:
        # Create plots directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = os.path.join(
            str(DATASETS_BASE), "anomaly_plots", f"detection_run_{timestamp}"
        )
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots for all KPIs and anomaly types
        logger.info("Starting to generate plots for all KPIs and anomaly types")
        plot_anomaly_windows(
            df=df,
            results=results,
            timestamps_col=detector.timestamps_col,
            max_plots_per_type=20,  # Top 20 anomalies per type
            save_path=plots_dir,
            plot_top_anomalies=True,
            freq_type="".join(
                filter(str.isalpha, detector.anomaly_config["frequency"]["type"])
            ),
        )

        # Generate plots for concurrent anomalies
        logger.info("Starting to generate plots for concurrent anomalies")
        plot_concurrent_anomalies(
            df=df,
            concurrent_results=concurrent_results,
            timestamps_col=detector.timestamps_col,
            max_plots=20,
            save_path=plots_dir,
            freq_type="".join(
                filter(str.isalpha, detector.anomaly_config["frequency"]["type"])
            ),
        )

        logger.info(
            f"Anomaly detection and plotting completed. Plots saved to: {plots_dir}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to previously saved anomaly results (optional)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--visualize_plots",
        help="Whether to generate debug plots",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.config, args.results, args.visualize_plots)
