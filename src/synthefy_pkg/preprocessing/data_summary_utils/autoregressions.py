"""Autoregression methods for time series analysis."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl


def autocorrelation(
    df: pl.DataFrame,
    column: str,
    max_lag: Optional[int] = None,
    method: str = "pearson",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate autocorrelation function for a time series column.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        max_lag: Maximum lag to compute (default: len(df) // 4)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Tuple of (lags, autocorrelations)
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    series = df.get_column(column).drop_nulls().to_numpy()
    n = len(series)

    if max_lag is None:
        max_lag = n // 4

    lags = np.arange(0, max_lag + 1)
    autocorr = np.zeros_like(lags, dtype=float)

    # Calculate autocorrelation for each lag
    for i, lag in enumerate(lags):
        if lag == 0:
            autocorr[i] = 1.0
        else:
            # Split series into two parts
            x1 = series[:-lag]
            x2 = series[lag:]

            if method == "pearson":
                # Pearson correlation
                corr = np.corrcoef(x1, x2)[0, 1]
                autocorr[i] = corr if not np.isnan(corr) else 0.0
            else:
                # For other methods, use numpy correlation
                autocorr[i] = np.corrcoef(x1, x2)[0, 1]
                if np.isnan(autocorr[i]):
                    autocorr[i] = 0.0

    return lags, autocorr


def arima_analysis(
    df: pl.DataFrame,
    column: str,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    max_iter: int = 50,
) -> Dict[str, Any]:
    """
    Perform ARIMA analysis on a time series column.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        order: (p, d, q) order for ARIMA model
        seasonal_order: (P, D, Q, s) seasonal order (optional)
        max_iter: Maximum iterations for model fitting

    Returns:
        Dictionary with model results and diagnostics
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    series = df.get_column(column).drop_nulls().to_numpy()

    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        # Test for stationarity
        adf_result = adfuller(series)

        # Fit ARIMA model
        if seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(series, order=order)

        fitted_model = model.fit(maxiter=max_iter)  # type: ignore

        # Get residuals and predictions
        residuals = fitted_model.resid  # type: ignore
        fitted_values = fitted_model.fittedvalues  # type: ignore

        # Calculate AIC and BIC
        aic = fitted_model.aic  # type: ignore
        bic = fitted_model.bic  # type: ignore

        # Ljung-Box test for residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)

        return {
            "model": fitted_model,
            "residuals": residuals,
            "fitted_values": fitted_values,
            "aic": aic,
            "bic": bic,
            "adf_statistic": adf_result[0],
            "adf_pvalue": adf_result[1],
            "ljung_box_statistic": lb_result["lb_stat"].iloc[-1],
            "ljung_box_pvalue": lb_result["lb_pvalue"].iloc[-1],
            "order": order,
            "seasonal_order": seasonal_order,
        }

    except ImportError:
        raise ImportError("statsmodels required for ARIMA analysis")


def garch_analysis(
    df: pl.DataFrame,
    column: str,
    order: Tuple[int, int] = (1, 1),
    mean_model: str = "AR",
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """
    Perform GARCH analysis on a time series column.

    Args:
        df: Polars DataFrame
        column: Column name for analysis
        order: (p, q) order for GARCH model
        mean_model: Mean model type ('AR', 'MA', 'ARMA', 'Constant')
        max_iter: Maximum iterations for model fitting

    Returns:
        Dictionary with GARCH model results and volatility estimates
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    series = df.get_column(column).drop_nulls().to_numpy()

    try:
        from arch import arch_model

        # Calculate returns if series is not already returns
        if np.any(
            series > 100
        ):  # Simple heuristic: if values > 100, assume prices
            returns = np.diff(np.log(series))
        else:
            returns = series

        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]

        # Fit GARCH model
        if mean_model == "AR":
            model = arch_model(
                returns, vol="GARCH", p=order[0], q=order[1], mean="AR", lags=1
            )
        else:
            model = arch_model(
                returns, vol="GARCH", p=order[0], q=order[1], mean="Constant"
            )

        fitted_model = model.fit(disp="off", maxiter=max_iter)  # type: ignore

        # Get volatility estimates
        volatility = fitted_model.conditional_volatility

        # Get model parameters
        params = fitted_model.params

        # Calculate AIC and BIC
        aic = fitted_model.aic
        bic = fitted_model.bic

        # Ljung-Box test for standardized residuals
        standardized_resids = fitted_model.resid / volatility
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(standardized_resids, lags=10, return_df=True)

        return {
            "model": fitted_model,
            "volatility": volatility,
            "returns": returns,
            "standardized_residuals": standardized_resids,
            "parameters": params,
            "aic": aic,
            "bic": bic,
            "ljung_box_statistic": lb_result["lb_stat"].iloc[-1],
            "ljung_box_pvalue": lb_result["lb_pvalue"].iloc[-1],
            "order": order,
            "mean_model": mean_model,
        }

    except ImportError:
        raise ImportError("arch package required for GARCH analysis")
