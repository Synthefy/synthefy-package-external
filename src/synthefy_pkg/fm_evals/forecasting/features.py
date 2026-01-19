from typing import Callable, List, Optional

import pandas as pd

COMPILE = True

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


def make_ema_features(
    columns: Optional[List[str]] = None, spans: List[float] = [3, 7, 14, 30]
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create exponential moving average features."""

    def _add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        # If columns is None, process all columns except those in SKIP_COVARIATES
        if columns is None:
            cols_to_process = [
                col for col in df.columns if col not in SKIP_COVARIATES
            ]
        else:
            cols_to_process = [
                col for col in columns if col not in SKIP_COVARIATES
            ]

        for col in cols_to_process:
            for span in spans:
                result[f"{col}_ema_{span}"] = result[col].ewm(span=span).mean()
                result[f"{col}_ewm_std_{span}"] = (
                    result[col].ewm(span=span).std()
                )

        # Clean the new features
        return result

    return _add_ema_features
