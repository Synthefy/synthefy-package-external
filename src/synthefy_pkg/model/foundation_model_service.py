import os
import random
import time
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from autogluon.timeseries import TimeSeriesDataFrame
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
from tabpfn_time_series import (
    DefaultFeatures,
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
)
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG

# This line will suppress all warnings
from synthefy_pkg.app.config import SynthefyFoundationModelSettings
from synthefy_pkg.app.data_models import (
    ConfidenceInterval,
    FMDataPointModification,
    ForecastDataset,
    ForecastGroup,
    MetadataDataFrame,
    MetaDataVariation,
    PerturbationsOrExactValues,
    PerturbationType,
    SynthefyFoundationModelMetadata,
    SynthefyFoundationModelTypeMetadata,
    SynthefyTimeSeriesModelType,
)
from synthefy_pkg.app.utils.api_utils import (
    format_timestamp_with_optional_fractional_seconds,
)
from synthefy_pkg.app.utils.fm_model_utils import get_foundation_model_info
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v3e import (
    SynthefyFoundationForecastingModelV3E,
)
from synthefy_pkg.model.foundation_model.utils import (
    DEFAULT_COVARIATE_LAGS,
    make_covariate_lag_feature,
)
from synthefy_pkg.utils.synthesis_utils import load_forecast_model

warnings.filterwarnings("ignore")

COMPILE = True

FM_SETTINGS = SynthefyFoundationModelSettings()  # type: ignore

COMPILE = True

GEMINI_MODELS = {
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
}


@lru_cache(maxsize=1)
def get_gemini_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODELS["gemini-2.0-flash-lite"],
        temperature=0.1,
        max_retries=1,
        google_api_key=api_key,
    )


def _validate_and_sanitize_timezone(timezone_value: Any) -> str:
    """
    Validate and sanitize timezone values from metadata.

    Args:
        timezone_value: The timezone value from metadata (could be any type)

    Returns:
        str: A valid timezone string, defaults to "UTC" if invalid
    """
    # Handle None or empty values
    if not timezone_value:
        logger.warning("Timezone value is None or empty, using UTC")
        return "UTC"

    # Convert to string if not already
    timezone_str = str(timezone_value).strip()

    # Handle empty string after stripping
    if not timezone_str:
        logger.warning("Timezone value is None or empty, using UTC")
        return "UTC"

    # Handle float timezone offsets (e.g., "1.0", "-5.0", "5.5")
    try:
        timezone_offset = float(timezone_str)
        # Convert float offset to pandas-compatible format
        if timezone_offset == 0:
            gmt_timezone = "UTC"
        else:
            # Calculate hours and minutes from decimal offset
            hours = int(timezone_offset)
            minutes = int(abs(timezone_offset - hours) * 60)

            if timezone_offset > 0:
                # Use +HH:MM format for positive offsets
                gmt_timezone = f"+{hours:02d}:{minutes:02d}"
            else:
                # Use -HH:MM format for negative offsets
                gmt_timezone = f"{hours:03d}:{minutes:02d}"

        logger.info(
            f"Converted numeric timezone offset '{timezone_str}' to '{gmt_timezone}'"
        )
        timezone_str = gmt_timezone
    except ValueError:
        # Not a numeric value, continue with string processing
        pass

    # Handle special invalid cases
    if timezone_str in ["nan", "None"]:
        logger.warning(f"Invalid timezone value '{timezone_str}', using UTC")
        return "UTC"

    # Handle UTC+HH:MM or UTC-HH:MM format
    if timezone_str.startswith("UTC"):
        if timezone_str == "UTC":
            return "UTC"

        # Extract the offset part (e.g., "+1:00" from "UTC+1:00")
        offset_part = timezone_str[3:]  # Remove "UTC" prefix

        # Validate and normalize the offset format
        if offset_part and (
            offset_part.startswith("+") or offset_part.startswith("-")
        ):
            try:
                # Parse and normalize the offset format (e.g., "-4:00" -> "-04:00")
                import re

                match = re.match(r"^([+-])(\d{1,2}):(\d{2})$", offset_part)
                if match:
                    sign, hours, minutes = match.groups()
                    normalized_offset = f"{sign}{int(hours):02d}:{minutes}"

                    # Test if the normalized offset is valid by trying to parse it
                    test_ts = pd.Timestamp("2023-01-01")
                    test_ts.tz_localize(normalized_offset)
                    logger.info(
                        f"Converted timezone '{timezone_str}' to '{normalized_offset}'"
                    )
                    return normalized_offset
                else:
                    raise ValueError(f"Invalid offset format: {offset_part}")
            except (
                ValueError,
                TypeError,
                KeyError,
                AmbiguousTimeError,
                NonExistentTimeError,
                Exception,
            ) as e:
                logger.warning(
                    f"Invalid UTC offset format '{timezone_str}', using UTC: {e}"
                )
                return "UTC"

    # Try to parse as a valid timezone
    try:
        # Test if it's a valid timezone by trying to localize to it
        test_ts = pd.Timestamp("2023-01-01")
        test_ts.tz_localize(timezone_str)
        return timezone_str
    except (
        ValueError,
        TypeError,
        KeyError,
        AmbiguousTimeError,
        NonExistentTimeError,
        Exception,
    ):
        # Try common timezone formats
        common_mappings = {
            "EST": "America/New_York",
            "PST": "America/Los_Angeles",
            "CST": "America/Chicago",
            "MST": "America/Denver",
            "UTC": "UTC",
            "GMT": "UTC",
        }

        if timezone_str in common_mappings:
            logger.info(
                f"Mapped timezone '{timezone_str}' to '{common_mappings[timezone_str]}'"
            )
            return common_mappings[timezone_str]

        # If all else fails, default to UTC
        logger.warning(f"Could not parse timezone '{timezone_str}', using UTC")
        return "UTC"


def handle_missing_values_and_get_time_varying_non_numeric_columns(
    df: pd.DataFrame, non_target_columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Handle missing values in the dataframe and identify time-varying non-numeric columns.

    Args:
        df: pd.DataFrame
            The dataframe to handle missing values in
        non_target_columns: List[str]
            The columns to handle missing values in

    Returns:
        Tuple[pd.DataFrame, List[str]]
            A tuple containing:
            - Processed dataframe ready for training
            - List of time-varying non-numeric columns
    """
    time_varying_non_numeric_columns = []
    df_copy = df.copy()

    for col in non_target_columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            # For numeric non_target_columns, use forward fill, then backward fill, then fill with 0
            df_copy[col] = df_copy[col].ffill().bfill().fillna(0)
        else:
            # For non-numeric non_target_columns, fill with "unknown"
            df_copy[col] = df_copy[col].ffill().bfill().fillna("unknown")
            if df_copy[col].nunique() > 1:
                time_varying_non_numeric_columns.append(col)

    return df_copy, time_varying_non_numeric_columns


def identify_time_invariant_columns(
    df: pd.DataFrame, non_target_columns: List[str]
) -> List[str]:
    """Identify time-invariant columns (columns with only one unique value)"""
    if not non_target_columns:
        return []
    return (
        df[non_target_columns]
        .columns[df[non_target_columns].nunique() == 1]
        .tolist()
    )


def setup_train_test_dfs(
    tabpfn_dataframe: pd.DataFrame,
    target_columns: List[str],
    datetime_column: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Set up training and test dataframes for TabPFN time series forecasting.

    This function processes the input dataframe to:
    1. Identify and separate target columns from other columns
    2. Handle missing values appropriately based on column type
    3. Identify time-invariant columns and remove them from the dataframe
    4. Prepare the dataframe for time series forecasting

    Parameters
    ----------
    tabpfn_dataframe : pd.DataFrame
        The input dataframe containing the time series data
    target_columns : List[str]
        List of column names that are the target variables to be forecasted
    datetime_column : str
        Name of the column containing datetime information

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        A tuple containing:
        - Processed dataframe ready for training
        - List of all non-target columns
        - List of time-invariant columns
    """

    # Check for duplicate columns and drop them if found
    # This can happen when the group filter is used as a covariate
    if tabpfn_dataframe.columns.duplicated().any():
        duplicate_cols = tabpfn_dataframe.columns[
            tabpfn_dataframe.columns.duplicated()
        ].tolist()
        tabpfn_dataframe = tabpfn_dataframe.loc[
            :, ~tabpfn_dataframe.columns.duplicated()
        ]
        logger.debug(f"Dropped duplicate columns: {duplicate_cols}")

    all_columns_except_target_columns = []
    time_invariant_columns = []
    # Get all non-target and non-datetime columns
    all_columns_except_target_columns = [
        col
        for col in tabpfn_dataframe.columns
        if col not in target_columns and col != datetime_column
    ]

    tabpfn_dataframe, time_varying_non_numeric_columns = (
        handle_missing_values_and_get_time_varying_non_numeric_columns(
            tabpfn_dataframe, all_columns_except_target_columns
        )
    )

    # Remove time varying non-numeric non_target_columns
    tabpfn_dataframe = tabpfn_dataframe.drop(
        columns=time_varying_non_numeric_columns
    )
    all_columns_except_target_columns = [
        col
        for col in all_columns_except_target_columns
        if col not in time_varying_non_numeric_columns
    ]

    time_invariant_columns = identify_time_invariant_columns(
        tabpfn_dataframe, all_columns_except_target_columns
    )

    logger.debug(
        f"Found {len(all_columns_except_target_columns)} non-target columns, "
        f"of which {len(time_invariant_columns)} are time invariant"
    )
    tabpfn_dataframe = tabpfn_dataframe.rename(
        columns={datetime_column: "timestamp"}
    )
    tabpfn_dataframe = tabpfn_dataframe[
        ["timestamp"] + target_columns + all_columns_except_target_columns
    ]

    return (
        tabpfn_dataframe,
        all_columns_except_target_columns,
        time_invariant_columns,
    )


def apply_point_modifications(
    df: pd.DataFrame,
    timestamp_column: str,
    point_modifications: List[FMDataPointModification] | None,
) -> pd.DataFrame:
    """
    Efficiently apply a batch of data point modifications to a DataFrame based on timestamp.
    This function updates specific values in the DataFrame according to a list of modifications,
    where each modification specifies a timestamp and a dictionary of column-value pairs to update
    at that timestamp.

    Parameters
    ----------
    df : pd.DataFrame - must include a timestamp_column
    timestamp_column : str - the name of the column in `df` that contains the timestamps.
    point_modifications : List[FMDataPointModification] | None - a list of FMDataPointModification objects, each specifying:
        - `timestamp` (str): The timestamp of the row to modify.
        - `modification_dict` (dict): A mapping of column names to new values for that row.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified modifications applied.

    Notes
    -----
    - If a modification refers to a timestamp or column not present in the DataFrame, it will be ignored.
    """
    if not point_modifications:
        return df

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Get the set of available timestamps in the dataframe for efficient lookup
    df_timestamps = set(df[timestamp_column])

    # Merge modifications for the same timestamp and filter out non-existent timestamps
    merged_mods = {}
    skipped_timestamps = []

    for mod in point_modifications:
        ts = pd.to_datetime(mod.timestamp)
        if ts not in df_timestamps:
            skipped_timestamps.append(str(ts))
            continue
        if ts not in merged_mods:
            merged_mods[ts] = {}
        merged_mods[ts].update(mod.modification_dict)

    # Log skipped timestamps if any
    if skipped_timestamps:
        logger.warning(
            f"Skipped {len(skipped_timestamps)} point modifications with timestamps not found in dataframe: "
            f"{skipped_timestamps[:5]}{'...' if len(skipped_timestamps) > 5 else ''}"
        )

    # Log duplicate timestamps if any
    if len(merged_mods) < len(point_modifications) - len(skipped_timestamps):
        logger.warning(
            f"Duplicate timestamps found in point_modifications: "
            f"{[str(ts) for ts in merged_mods.keys()]}. "
            "Modifications for the same timestamp have been merged; later values overwrite earlier ones."
        )

    # If no valid modifications remain, return the original dataframe
    if not merged_mods:
        logger.debug("No valid point modifications to apply")
        return df

    mod_df = pd.DataFrame(
        [
            {**mod_dict, timestamp_column: ts}
            for ts, mod_dict in merged_mods.items()
        ]
    )

    # Set the timestamp as index for both DataFrames
    df.set_index(timestamp_column, inplace=True)
    mod_df.set_index(timestamp_column, inplace=True)

    # Only update columns that exist in both DataFrames
    update_cols = [col for col in mod_df.columns if col in df.columns]

    logger.debug(
        f"Will modify {len(mod_df)} rows of {df.shape[0]} rows for columns: {update_cols}"
    )

    # Use DataFrame.update for efficient in-place update
    df.update(mod_df[update_cols])

    # Reset index to restore the timestamp column
    df.reset_index(inplace=True)

    return df


def apply_full_and_range_modifications(
    df: pd.DataFrame,
    full_and_range_modifications: List[MetaDataVariation] | None,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Apply unified modifications (full and range) to the dataframe.

    Args:
        df: The dataframe to modify
        full_and_range_modifications: List of modifications to apply
        timestamp_column: Name of the timestamp column for range filtering

    Returns:
        Modified dataframe

    The modifications are applied in order (sorted by the 'order' field).
    - 'full' modifications: applied to the entire dataframe
    - 'range' modifications: applied only to rows within the specified timestamp range
    """
    if not full_and_range_modifications or df is None or len(df) == 0:
        return df

    # Sort modifications by order
    sorted_modifications = sorted(
        full_and_range_modifications, key=lambda x: x.order
    )

    for modification in sorted_modifications:
        logger.debug(
            f"Applying {modification.modification_type} modification (order {modification.order}): {modification} to {modification.name}"
        )

        if modification.name not in df.columns:
            logger.warning(
                f"Column {modification.name} not found in dataframe. Skipping modification."
            )
            continue

        # Determine which rows to modify
        if modification.modification_type == "full":
            # Apply to all rows
            mask = pd.Series([True] * len(df), index=df.index)
        else:  # range modification
            df_timestamps = pd.to_datetime(df[timestamp_column])
            mask = pd.Series([True] * len(df), index=df.index)

            if modification.min_timestamp is not None:
                mask &= df_timestamps >= modification.min_timestamp
                logger.debug(
                    f"Applied min_timestamp filter: {modification.min_timestamp}"
                )

            if modification.max_timestamp is not None:
                mask &= df_timestamps <= modification.max_timestamp
                logger.debug(
                    f"Applied max_timestamp filter: {modification.max_timestamp}"
                )

            logger.debug(
                f"Range modification will apply to {mask.sum()} out of {len(df)} rows"
            )

        # Apply the modification to the selected rows
        if modification.perturbation_or_exact_value == "perturbation":
            if modification.perturbation_type == PerturbationType.ADD:
                df.loc[mask, modification.name] += modification.value
            elif modification.perturbation_type == PerturbationType.SUBTRACT:
                df.loc[mask, modification.name] -= modification.value
            elif modification.perturbation_type == PerturbationType.MULTIPLY:
                df.loc[mask, modification.name] *= modification.value
            elif modification.perturbation_type == PerturbationType.DIVIDE:
                df.loc[mask, modification.name] /= modification.value
        else:
            df.loc[mask, modification.name] = modification.value

    return df


def _generate_llm_explanation(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    covariate_columns_to_leak: List[str] | None = None,
    target_column: str | None = None,
) -> str:
    """Generate an explanation for the forecast using the foundation model."""
    if covariate_columns_to_leak is None:
        covariate_columns_to_leak = []
    forecast_to_explain = forecast_df.reset_index()[
        ["timestamp", "target"] + covariate_columns_to_leak
    ]

    forecast_to_explain_str = []
    for idx, row in forecast_to_explain.iterrows():
        covariate_str = ""
        if covariate_columns_to_leak:
            covariate_str = ", ".join(
                f"{col}: {round(row[col], 3) if pd.notna(row[col]) else 'NA'}"
                for col in covariate_columns_to_leak
                if col in row
            )
        if covariate_str:
            forecast_to_explain_str.append(
                f"Timestamp: {row['timestamp']}. {target_column}: {round(row['target'], 3) if pd.notna(row['target']) else 'NA'}, Covariates: {covariate_str}"
            )
        else:
            forecast_to_explain_str.append(
                f"Timestamp: {row['timestamp']}. {target_column}: {round(row['target'], 3) if pd.notna(row['target']) else 'NA'}"
            )
    forecast_to_explain_str = "\n".join(forecast_to_explain_str)

    history_str = []
    for idx, row in df.iterrows():
        covariate_str = ""
        if covariate_columns_to_leak:
            covariate_str = ", ".join(
                f"{col}: {round(row[col], 3) if pd.notna(row[col]) else 'NA'}"
                for col in covariate_columns_to_leak
                if col in row
            )
        if covariate_str:
            history_str.append(
                f"Timestamp: {row['timestamp']}. {target_column}: {round(row[target_column], 3) if target_column and pd.notna(row[target_column]) else 'NA'}, Covariates: {covariate_str}"
            )
        else:
            history_str.append(
                f"Timestamp: {row['timestamp']}. {target_column}: {round(row[target_column], 3) if target_column and pd.notna(row[target_column]) else 'NA'}"
            )
    history_str = "\n".join(history_str)

    prompt = f"""
        You are a time series forecasting expert forecasting the {target_column}. Analyze this forecasted data and provide ONLY actionable insights:

        ## Historical Data (most recent last):
        {history_str}

        ## Forecasted Data:
        {forecast_to_explain_str}

        Provide exactly 2 bullet points formatted as markdown:
        1. How are exogenous factors (covariates) affecting the target in the forecast? If they are not affecting the target, DON'T SAY THEY ARE. (1 sentence)
        2. Give actionable insights for the business.
        The bullets should be explicit, to the point, and not vague.

        Keep each point under 25-40 words. Be direct and practical, and make the response human-readable and easy to interpret.
        DO NOT HALLUCINATE.

        Examples of good responses:
        ```
        • Temperature increases drive 20% higher sales; optimize inventory for warm weather.
        • Stock more summer items and reduce inventory for winter/fall.

        • Marketing spend correlates with 15% sales lift; maintain current ad budget.
        • Increase marketing budget by 10% during peak seasons for maximum ROI.

        • Website traffic doubles during sales; prepare server capacity.
        • Scale up CDN and database resources before promotions.

        • System Score is strongly correlated with latency and and memory bandwidth.
        • Turn off unused services to reduce latency.
        ```

        Here's the analysis of the data:

    """
    logger.debug(f"Prompt: {prompt}")
    try:
        llm = get_gemini_llm()
        start_time = time.time()
        response = llm.invoke(prompt)
        end_time = time.time()
        logger.debug(
            f"Time taken to generate explanation: {end_time - start_time} seconds"
        )
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)
    except Exception:
        return "Could not generate explanation of the forecast."


def _generate_shap_analysis(
    train_df: TimeSeriesDataFrame,
    test_df: TimeSeriesDataFrame,
    predictor: TabPFNTimeSeriesPredictor,
    target_column: str,
    feature_names: List[str],
    n_background_samples: int = 200,
    n_explanation_samples: int = 20,
    shap_method: str = "kernel",
) -> Dict[str, Any] | None:
    """
    Generate SHAP analysis for TabPFN time series predictions.

    Args:
        train_df: Training dataframe with features
        test_df: Test dataframe with features
        predictor: TabPFN predictor instance
        target_column: Name of the target column
        feature_names: List of feature names to analyze
        n_background_samples: Number of background samples for SHAP
        n_explanation_samples: Number of samples to explain
        shap_method: SHAP computation method. Available options:
            - "kernel": KernelExplainer for black-box models (default)

    Returns:
        Dictionary containing SHAP analysis results with feature importance scores

    Note:
        - KernelExplainer: Very flexible but slow, works with any black-box model
    """
    try:
        logger.info(
            f"🔄 SHAP analysis in progress for {target_column} "
            f"(background samples: {n_background_samples}, explanation samples: {n_explanation_samples})"
        )

        # Create a prediction function wrapper for SHAP
        def prediction_function(X: np.ndarray | pd.DataFrame):
            """Wrapper function for TabPFN predictions that SHAP can use."""
            try:
                # Convert input to the format expected by TabPFN
                if isinstance(X, np.ndarray):
                    # If X is a numpy array, convert to DataFrame
                    X_df = pd.DataFrame(
                        X, columns=["item_id", "timestamp"] + feature_names
                    )
                else:
                    X_df = X.copy()

                # Ensure we have the required columns for TabPFN
                if "target" not in X_df.columns:
                    X_df["target"] = np.nan

                if "timestamp" not in X_df.columns:
                    raise ValueError(
                        "timestamp column not found in X_df - This should not be possible"
                    )
                if "item_id" not in X_df.columns:
                    raise ValueError(
                        "item_id column not found in X_df - This should not be possible"
                    )

                # Set index for TimeSeriesDataFrame
                X_df = X_df.set_index(["item_id", "timestamp"])

                # Create TimeSeriesDataFrame
                ts_df = TimeSeriesDataFrame(X_df)

                # Make prediction using the predictor
                predictions = predictor.predict(
                    train_df,  # Use original train_df as background
                    ts_df,
                    quantile_config=sorted(
                        list(
                            set(TABPFN_TS_DEFAULT_QUANTILE_CONFIG + [0.1, 0.9])
                        )
                    ),
                )

                # Return the target predictions
                return predictions["target"].values

            except Exception as e:
                import traceback

                logger.error(
                    f"Error in SHAP prediction function: {e}\n{traceback.format_exc()}"
                )
                raise e

        # background_data: Used by SHAP as the reference dataset to estimate the expected value of the model output.
        # It should represent the typical (historical) distribution of the features. Here, we sample from the training data.
        train_df_regular = train_df.reset_index()
        background_data = train_df_regular.sample(
            n=min(n_background_samples, len(train_df_regular)),
            random_state=42,
        ).values

        # explanation_data: The specific samples for which we want to compute SHAP values (i.e., explain the model's predictions).
        # Here, we use the first n_explanation_samples from the test set.
        test_df_regular = test_df.reset_index()
        explanation_data = test_df_regular.head(n_explanation_samples).values

        # Create SHAP explainer based on specified method
        logger.debug(f"Creating SHAP {shap_method.title()}Explainer")

        try:
            if shap_method.lower() == "kernel":
                explainer = shap.KernelExplainer(
                    prediction_function, background_data, link="identity"
                )
                # Calculate SHAP values with more samples for stability
                logger.debug("Calculating SHAP values with KernelExplainer")
                shap_values = explainer.shap_values(
                    explanation_data,
                    nsamples=min(
                        100, len(background_data)
                    ),  # More samples = more stable
                    random_state=42,  # Fixed seed for reproducibility
                )
            else:
                logger.warning(
                    f"Unknown SHAP method '{shap_method}', Only supports 'kernel', falling back to it."
                )
                explainer = shap.KernelExplainer(
                    prediction_function, background_data, link="identity"
                )
                shap_values = explainer.shap_values(
                    explanation_data,
                    nsamples=min(100, len(background_data)),
                    random_state=42,
                )

            # Get raw shap values for each feature (without taking mean)
            if shap_values is not None:
                if not hasattr(shap_values, "shape"):
                    # Handle case where shap_values might not be a numpy array
                    shap_values = np.array(shap_values)

                # Take absolute values but don't average - keep all values
                abs_shap_values = np.abs(shap_values)

                # Create dict with feature names as keys and lists of shap values as values
                feature_importance_dict = {}
                for i, feature_name in enumerate(feature_names):
                    feature_importance_dict[feature_name] = abs_shap_values[
                        :, i
                    ].tolist()
            else:
                logger.error(
                    "SHAP values are None, cannot calculate feature importance"
                )
                return None

            # remove shap features
            shap_features_to_remove = [
                "target",
                "hour_of_day_cos",
                "hour_of_day_sin",
                "day_of_week_cos",
                "day_of_week_sin",
                "day_of_month_sin",
                "day_of_month_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "week_of_year_sin",
                "week_of_year_cos",
                "month_of_year_sin",
                "month_of_year_cos",
                "year",
            ]

            feature_importance_dict = {
                k: v
                for k, v in feature_importance_dict.items()
                if k not in shap_features_to_remove
            }

            if "running_index" in feature_importance_dict:
                feature_importance_dict[f"{target_column}_features"] = (
                    feature_importance_dict.pop("running_index")
                )

            # Sort by mean of absolute SHAP values
            sorted_features = dict(
                sorted(
                    feature_importance_dict.items(),
                    key=lambda x: np.mean(x[1]),
                    reverse=True,
                )
            )
            return sorted_features
        except Exception as e:
            logger.error(f"Error in SHAP analysis for {target_column}: {e}")
            return None

    except Exception as e:
        logger.error(f"Error in SHAP analysis for {target_column}: {e}")
        return None


class TabPFNPredictor:
    def __init__(self):
        self.predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)

        self.selected_features = [
            DefaultFeatures.add_running_index,
            DefaultFeatures.add_calendar_features,
        ]
        logger.debug(f"Selected features: {self.selected_features}")

    def drop_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop all the datetime column from the dataframe.
        """
        logger.debug(
            f"Dropping datetime columns from DataFrame with shape {df.shape}"
        )
        columns_before = df.columns.tolist()

        # Get all datetime columns first, then drop them all at once
        datetime_cols = [
            col
            for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        if datetime_cols:
            logger.debug(f"Dropping datetime columns: {datetime_cols}")
            df = df.drop(columns=datetime_cols)

        columns_after = df.columns.tolist()
        logger.debug(
            f"Dropped {len(columns_before) - len(columns_after)} datetime columns"
        )
        return df

    def _process_ground_truth(
        self,
        ground_truth_df: pd.DataFrame | None,
        current_target_column: str,
        future_time_stamps: List[datetime],
        datetime_column: str,
    ) -> List[float | None]:
        """
        Process ground truth data for a target column and future timestamps.
        """
        # Default to None values
        ground_truth_values: List[float | None] = [None] * len(
            future_time_stamps
        )
        if ground_truth_df is None:
            logger.debug("No ground truth dataframe provided")
            return ground_truth_values

        if current_target_column not in ground_truth_df.columns:
            logger.debug(
                f"Column '{current_target_column}' not found in ground truth dataframe"
            )
            return ground_truth_values

        # Only proceed if we have ground truth data
        logger.debug(
            f"Found ground truth data for column '{current_target_column}'"
        )

        # Convert future timestamps to a DataFrame
        future_df = pd.DataFrame(
            {datetime_column: pd.to_datetime(future_time_stamps)}
        )

        # Ensure both dataframes have datetime type for the timestamp column
        ground_truth_df[datetime_column] = pd.to_datetime(
            ground_truth_df[datetime_column]
        )

        merged_df = pd.merge(
            future_df,
            ground_truth_df[[datetime_column, current_target_column]],
            on=datetime_column,
            how="left",
        )

        # Convert matched values to list, preserving None for missing values
        matched_values = merged_df[current_target_column].tolist()

        # Convert NaN to None (NaN isn't JSON serializable)
        ground_truth_values = [
            float(val) if pd.notna(val) else None for val in matched_values
        ]

        match_count = sum(1 for val in ground_truth_values if val is not None)
        logger.debug(
            f"Found {match_count} matching timestamps out of {len(future_time_stamps)}"
        )
        return ground_truth_values

    def _find_timestamp_column(self, df: pd.DataFrame) -> str:
        """
        Find the timestamp column in a dataframe.
        Returns the column name if found, raises ValueError if not found.
        """
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        raise ValueError("No datetime column found in dataframe")

    def _process_metadata_dataframes(
        self,
        metadata_dataframes: List[MetadataDataFrame],
        target_df: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        Process metadata dataframes by matching timestamps with target dataframe.
        Only keeps metadata rows where timestamp is before or equal to target timestamp.

        Parameters:
        ----------
        metadata_dataframes: List[MetadataDataFrame]
            A list of MetadataDataFrame objects
        target_df: pd.DataFrame
            The target dataframe to match timestamps with

        Returns:
        -------
        List[pd.DataFrame]
            A list of processed metadata dataframes. Each df must be of len(target_df)
        """
        logger.debug("Processing metadata dataframes")
        no_datetime_metadata_dfs: List[pd.DataFrame] = [None] * len(
            metadata_dataframes
        )  # type: ignore

        # Get target timestamp column and convert to UTC? TODO - find tz of target_df
        target_timestamp_col = self._find_timestamp_column(target_df)
        target_timestamps = pd.to_datetime(target_df[target_timestamp_col])

        try:
            if target_timestamps.dt.tz is None:
                target_timestamps = target_timestamps.dt.tz_localize("UTC")
            else:
                target_timestamps = target_timestamps.dt.tz_convert("UTC")
        except Exception as e:
            logger.error(
                f"Could not convert/interpret the timezone of target dataframe: {e}."
                "Please make sure the target dataframe uses the same timezone."
            )
            raise e

        for i, meta_df in enumerate(metadata_dataframes):
            logger.debug(
                f"Processing metadata dataframe {i + 1}/{len(metadata_dataframes)}"
            )

            meta_df_copy = meta_df.df.copy()
            # Get timezone from metadata_json if available, default to UTC
            timestamp_column = meta_df.timestamp_key or "timestamp"
            timezone = "UTC"
            if meta_df.metadata_json is not None:
                raw_timezone = meta_df.metadata_json.get("timezone", "UTC")
                # Validate and sanitize timezone
                timezone = _validate_and_sanitize_timezone(raw_timezone)

            # Convert timestamp column to datetime with timezone
            try:
                meta_df_copy[timestamp_column] = pd.to_datetime(
                    meta_df_copy[timestamp_column], utc=False
                )

                # If timezone-naive, localize to specified timezone
                if meta_df_copy[timestamp_column].dt.tz is None:
                    meta_df_copy[timestamp_column] = meta_df_copy[
                        timestamp_column
                    ].dt.tz_localize(timezone)

                # Convert to UTC if not already in UTC
                if meta_df_copy[timestamp_column].dt.tz != "UTC":
                    meta_df_copy[timestamp_column] = meta_df_copy[
                        timestamp_column
                    ].dt.tz_convert("UTC")

                # Sort the dataframe by timestamp
                meta_df_copy = meta_df_copy.sort_values(timestamp_column)
            except Exception as e:
                logger.error(
                    f"Could not convert/interpret the timezone of metadata dataframe {i + 1}: {e}"
                )
                raise e

            # Find matching indices
            matching_indices = []
            for target_ts in target_timestamps:
                # Find the most recent "metadata timestamp that's <= target timestamp
                mask = meta_df_copy[timestamp_column] <= target_ts
                if mask.any():
                    # Get the index of the most recent matching row
                    matching_idx = meta_df_copy[mask].index[-1]
                    matching_indices.append(matching_idx)

            if not matching_indices:
                logger.warning(
                    f"No matching timestamps found for metadata dataframe {i + 1}"
                )
                continue

            # Get matching rows and drop timestamp column
            matched_df = meta_df_copy.loc[matching_indices].drop(
                columns=[timestamp_column]
            )

            # Reindex to match target dataframe
            matched_df.index = target_df.index[: len(matched_df)]

            # Add the column name to the matched_df
            if (
                meta_df.metadata_json is not None
                and meta_df.metadata_json.get("columns") is not None
            ):
                logger.debug(
                    f"Adding column names to matched_df for metadata dataframe {i + 1}"
                )
                matched_df.columns = [
                    meta_df.metadata_json["columns"][i]["title"]
                    for i in range(len(meta_df.metadata_json["columns"]))
                ]
            else:
                logger.debug(
                    f"No column names found for metadata dataframe {i + 1}"
                )

            no_datetime_metadata_dfs[i] = matched_df
            logger.debug(f"Processed metadata dataframe {i + 1}")

        # Remove None entries
        no_datetime_metadata_dfs = [
            df for df in no_datetime_metadata_dfs if df is not None
        ]

        return no_datetime_metadata_dfs

    def _validate_no_duplicate_timestamps(
        self, target_df: pd.DataFrame, datetime_column: str
    ) -> None:
        """
        Validate that the target dataframe does not contain duplicate timestamps
        and is sorted by the datetime column.

        Parameters:
        ----------
        target_df: pd.DataFrame
            The dataframe to validate
        datetime_column: str
            The name of the datetime column to check for duplicates

        Raises:
        ------
        ValueError: If duplicate timestamps are found or if timestamps are not sorted
        """
        logger.debug(
            f"Checking for duplicate timestamps in column: {datetime_column}"
        )

        # Count occurrences of each timestamp
        timestamp_counts = target_df[datetime_column].value_counts()

        # Find duplicates (timestamps with count > 1)
        duplicates = timestamp_counts[timestamp_counts > 1]

        if not duplicates.empty:
            # Get the first few duplicate timestamps for the error message
            duplicate_examples = duplicates.head(3).index.tolist()
            formatted_examples = [str(ts) for ts in duplicate_examples]

            logger.error(
                f"Found {len(duplicates)} duplicate timestamps in {datetime_column}"
            )
            logger.error(
                f"Examples of duplicate timestamps: {', '.join(formatted_examples)}"
            )

            raise ValueError(
                f"Duplicate timestamps found in column '{datetime_column}'. "
                f"Found {len(duplicates)} timestamps with multiple entries. "
                f"Examples: {', '.join(formatted_examples)}"
            )

        logger.debug(f"No duplicate timestamps found in {datetime_column}")

        # Check if timestamps are sorted
        if not target_df[datetime_column].is_monotonic_increasing:
            logger.error(f"Timestamps in {datetime_column} are not sorted")
            raise ValueError(
                f"Timestamps in column '{datetime_column}' must be sorted in ascending order"
            )

    def predict(
        self,
        target_df: pd.DataFrame,
        covariate_columns: List[str],
        metadata_dataframes: List[MetadataDataFrame],
        target_columns: List[str],
        forecasting_timestamp: datetime,
        future_time_stamps: List[datetime],
        timestamp_column: str,
        ground_truth_df: pd.DataFrame | None = None,
        remove_all_metadata: bool = False,
        covariate_columns_to_leak: List[str] | None = None,
        metadata_dataframes_leak_idxs: List[int] | None = None,
        quantiles: List[float] = [0.1, 0.9],
        do_llm_explanation: bool = False,
        do_shap_analysis: bool = False,
        shap_method: str = "kernel",
    ) -> ForecastDataset:
        """
        Generate predictions for a list of datasets using the foundation model.

        Parameters:
        ----------
        target_df: the customer uploaded dataset without any preprocessing.
                   Since this can include NaNs, will ffill.bfill(0)

        covariate_columns: List[str]
            A list of column names in the target_df that
            should be used as covariates (these do not include the target columns)

        metadata_dataframes : List[MetadataDataFrame]
            A list of pandas DataFrames, each representing a time series
            dataset. Each DataFrame is expected to contain a datetime
            index or a column identifying the time axis, as well as the
            target columns to be predicted.

        target_columns : List[str]
            A list of column names in the target_df that
            should be forecasted.
            These columns represent the variables the model will learn to
            predict (e.g. "ride_count").

        forecasting_timestamp : datetime
            The point in time immediately preceding the prediction
            window. This timestamp is used as the cutoff — all data up
            to and including this point is used to inform future
            predictions.

        future_time_stamps : List[datetime]
            A list of datetime values for which forecasts are required.
            These represent the prediction horizon — i.e., the specific
            timestamps for which model output should be generated.

        timestamp_column : str
            The name of the column containing timestamp information.

        ground_truth_df: pd.DataFrame | None
            A dataframe with ground truth values for the target columns.
            The dataframe must have a datetime column and the target columns.

        covariate_columns_to_leak: List[str] | None
            A list of column names in the target_df that should be leaked to the future.

        metadata_dataframes_leak_idxs: List[int] | None
            A list of indices of the metadata_dataframes that should be leaked to the future.

        quantiles: List[float]
            A list of quantiles to return -> Must be of length 2 if provided.

        full_and_range_modifications: List[MetaDataVariation] | None
            A list of unified modifications that can be applied to full data or specific time ranges.
            These are applied in order (sorted by the 'order' field) before point modifications.

        point_modifications: List[FMDataPointModification] | None
            A list of data point modifications to apply to the target_df.
            These are applied after full_and_range_modifications.

        do_llm_explanation: bool
            Whether to use LLM to explain the forecast(s).

        do_shap_analysis: bool
            Whether to use SHAP to explain the forecast(s).

        shap_method: str
            SHAP computation method. Options: "kernel".
            Default is "kernel".

        Returns:
        -------
        ForecastDataset
            The ForecastDataset contains predictions for the specified
            target columns across all provided future_time_stamps. The
            resulting ForecastDataset will include a 'date_hour'
            (or equivalent) column
            along with the forecasted target columns.

        Notes:
        ------
        - There can be no duplicate feature names/column names in the target_df/metadata_df
        """
        logger.debug(
            f"Starting prediction with target_df shape: {target_df.shape}"
        )
        logger.debug(f"Target columns to forecast: {target_columns}")
        logger.debug(f"Forecasting timestamp: {forecasting_timestamp}")
        logger.debug(
            f"Future timestamps range: {future_time_stamps[0]} to {future_time_stamps[-1]} ({len(future_time_stamps)} points)"
        )
        logger.debug(
            f"Number of metadata dataframes: {len(metadata_dataframes)}"
        )
        logger.debug(f"{quantiles=}")
        logger.debug(f"Using timestamp column: {timestamp_column}")

        logger.debug("Filling (if any) NaNs with 0")
        datetime_column = timestamp_column

        # Configure feature generators: base time features + covariate lags
        # Use the user-provided covariates and default lag set [1,2,3]
        try:
            selected_features_multivariate = [
                DefaultFeatures.add_running_index,
                DefaultFeatures.add_calendar_features,
                make_covariate_lag_feature(covariate_columns),
            ]
            logger.debug(
                f"Configured feature generators with covariate lags for columns: {covariate_columns}"
            )
        except Exception as e:
            logger.error(
                f"Failed to configure covariate lag features, proceeding with base features only: {e}"
            )

        if covariate_columns_to_leak:
            if any(
                col not in covariate_columns
                for col in covariate_columns_to_leak
            ):
                raise ValueError(
                    f"All columns in {covariate_columns_to_leak=} must be in {covariate_columns=}"
                )

        if metadata_dataframes_leak_idxs:
            if any(
                idx >= len(metadata_dataframes)
                for idx in metadata_dataframes_leak_idxs
            ):
                raise ValueError(
                    f"All {metadata_dataframes_leak_idxs=} must be less than {len(metadata_dataframes)=}"
                )

        # keep only the target and covariate columns in the target_df
        target_df = target_df[
            [datetime_column] + target_columns + covariate_columns
        ]

        # Validate no duplicate timestamps
        self._validate_no_duplicate_timestamps(target_df, datetime_column)
        if ground_truth_df is not None:
            self._validate_no_duplicate_timestamps(
                ground_truth_df, datetime_column
            )

        # TabPFN dataframe.
        tabpfn_dataframe = target_df.copy()

        # Process metadata dataframes
        no_datetime_metadata_dfs = self._process_metadata_dataframes(
            metadata_dataframes, target_df
        )

        logger.debug("Concatenating target dataframe with metadata dataframes")
        tabpfn_dataframe = pd.concat(
            [tabpfn_dataframe, *no_datetime_metadata_dfs], axis=1
        )
        logger.debug(
            f"Combined dataframe shape after concatenation: {tabpfn_dataframe.shape}"
        )

        # Note: Modifications are now applied once after downloading data in do_forecast_core_code
        # so target_df and ground_truth_df already have modifications applied

        # save this for leaking metadata/covariates
        original_tabpfn_dataframe = tabpfn_dataframe.copy()
        # Remove all data after the forecasting_timestamp
        logger.debug(
            f"Filtering data to remove points after forecasting timestamp: {forecasting_timestamp}"
        )
        rows_before = len(tabpfn_dataframe)
        tabpfn_dataframe = tabpfn_dataframe[
            tabpfn_dataframe[datetime_column] <= forecasting_timestamp
        ]
        rows_after = len(tabpfn_dataframe)
        logger.debug(
            f"Removed {rows_before - rows_after} rows after filtering by timestamp"
        )
        # Convert the datetime column to pd.Timestamp
        logger.debug(
            f"Converting datetime column {datetime_column} to pd.Timestamp"
        )
        tabpfn_dataframe[datetime_column] = pd.to_datetime(
            tabpfn_dataframe[datetime_column]
        )

        (
            tabpfn_dataframe,
            all_columns_except_target_columns,
            time_invariant_columns,
        ) = setup_train_test_dfs(
            tabpfn_dataframe,
            target_columns,
            datetime_column,
        )

        if remove_all_metadata:
            all_columns_except_target_columns = []
            time_invariant_columns = []

        # Make a prediction df, with the future_time_stamps as the index and the target_columns as the columns
        final_predictions: ForecastDataset = ForecastDataset(
            timestamps=[
                format_timestamp_with_optional_fractional_seconds(
                    pd.Timestamp(timestamp)
                )
                for timestamp in future_time_stamps
            ],
            values=[],
        )

        for current_target_column in target_columns:
            logger.debug(f"Processing target column: {current_target_column}")
            # Make a copy of the tabpfn_dataframe with datetime_column, current_target_column and all_columns_except_target_columns
            current_target_df = tabpfn_dataframe[
                ["timestamp", current_target_column]
                + all_columns_except_target_columns
            ]

            # Add a `item_id` column to the dataframe with all with value 0
            logger.debug("Adding item_id column")
            current_target_df["item_id"] = (
                "-".join(time_invariant_columns)
                if len(time_invariant_columns) > 0
                else "0"
            )

            # Change datetime column name to `timestamp`
            current_target_df = current_target_df.set_index(
                ["item_id", "timestamp"]
            )

            # Rename the current_target_column to `target`
            current_target_df = current_target_df.rename(
                columns={current_target_column: "target"}
            )

            train_df = TimeSeriesDataFrame(current_target_df)
            logger.debug(f"Training dataframe shape: {train_df.shape}")

            # Generate the test dataframe with the future_time_stamps and same columns as the train dataframe
            test_dataframe = pd.DataFrame(
                {
                    "target": [np.nan]
                    * len(future_time_stamps),  # Use NaN as placeholder
                    "item_id": [
                        "-".join(time_invariant_columns)
                        if len(time_invariant_columns) > 0
                        else "0"
                    ]
                    * len(future_time_stamps),
                    "timestamp": future_time_stamps,
                }
            )

            test_df = pd.DataFrame(test_dataframe)
            original_tabpfn_dataframe = original_tabpfn_dataframe[
                original_tabpfn_dataframe[datetime_column]
                > forecasting_timestamp  # must be > since forecast_timestamp is last historical point
            ]
            try:
                if covariate_columns_to_leak:
                    # no need to leak time invariant columns (it will be done below)
                    for col in covariate_columns_to_leak:
                        if col in time_invariant_columns:
                            continue
                        # Get the index of the forecasting_timestamp in the original_tabpfn_dataframe
                        future_metadata_values = []
                        for idx, timestamp in enumerate(
                            original_tabpfn_dataframe[datetime_column]
                        ):
                            if timestamp >= forecasting_timestamp:
                                future_metadata_values.append(
                                    original_tabpfn_dataframe[col].iloc[idx]
                                )
                            if len(future_metadata_values) == len(test_df):
                                break
                        logger.debug(
                            f"Found {len(future_metadata_values)=} for leaking covariate {col=}"
                        )
                        if len(future_metadata_values) == 0:
                            continue
                        if len(future_metadata_values) < len(test_df):
                            future_metadata_values.extend(
                                [None]
                                * (len(test_df) - len(future_metadata_values))
                            )
                        test_df[col] = future_metadata_values
            except Exception as e:
                logger.error(
                    f"Error leaking covariates: {e}, continuing without leaking covariates"
                )

            try:
                if metadata_dataframes_leak_idxs:
                    for idx in metadata_dataframes_leak_idxs:
                        # col_name in original_tabpfn_dataframe is: metadata_dataframes[0].metadata_json['columns'][0]['title']
                        try:
                            col = metadata_dataframes[idx].metadata_json[  # type: ignore
                                "columns"
                            ][  # type: ignore
                                0
                            ]["title"]  # type: ignore
                            if col not in original_tabpfn_dataframe.columns:
                                raise ValueError(
                                    f"Column {col} not found in original_dataframe - this should never happen"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error getting column name from metadata_dataframes: {e}"
                            )
                            raise e
                        # no need to leak time invariant columns (it will be done below)
                        if col in time_invariant_columns:
                            continue
                        # Get the index of the forecasting_timestamp in the original_tabpfn_dataframe
                        future_metadata_values = []
                        for idx, timestamp in enumerate(
                            original_tabpfn_dataframe[datetime_column]
                        ):
                            if timestamp >= forecasting_timestamp:
                                future_metadata_values.append(
                                    original_tabpfn_dataframe[col].iloc[idx]
                                )
                            if len(future_metadata_values) == len(test_df):
                                break
                        logger.debug(
                            f"Found {len(future_metadata_values)=} for leaking metadata {col=}"
                        )
                        if len(future_metadata_values) == 0:
                            continue
                        if len(future_metadata_values) < len(test_df):
                            future_metadata_values.extend(
                                [None]
                                * (len(test_df) - len(future_metadata_values))
                            )

                        test_df[col] = future_metadata_values
            except Exception as e:
                logger.error(
                    f"Error leaking metadata: {e}, continuing without leaking metadata"
                )

            for col in time_invariant_columns:
                # already added as a leak column
                if col in test_df.columns:
                    continue
                test_df[col] = [train_df[col].iloc[0]] * len(future_time_stamps)

            # Set the index to ['item_id', 'timestamp']
            test_df = test_df.set_index(["item_id", "timestamp"])

            test_df = TimeSeriesDataFrame(test_df)
            logger.debug(f"Final test dataframe shape: {test_df.shape}")

            univariate_train_df = TimeSeriesDataFrame(
                train_df[["target"]].copy()
            )
            univariate_test_df = TimeSeriesDataFrame(test_df[["target"]].copy())

            # Add features for TabPFN
            train_with_features, test_with_features = (
                FeatureTransformer.add_features(
                    train_df, test_df, selected_features_multivariate
                )
            )

            univariate_train_with_features, univariate_test_with_features = (
                FeatureTransformer.add_features(
                    univariate_train_df,
                    univariate_test_df,
                    self.selected_features,
                )
            )

            # Make predictions
            logger.info(f"Making predictions for {current_target_column}")
            start_time = time.time()
            try:
                pred = self.predictor.predict(
                    train_with_features,
                    test_with_features,
                    # note tabpfn doesnt support any outside DEFAULT_QUANTILE_CONFIG right now
                    quantile_config=sorted(
                        list(set(TABPFN_TS_DEFAULT_QUANTILE_CONFIG + quantiles))
                    ),
                )

                univariate_pred = self.predictor.predict(
                    univariate_train_with_features,
                    univariate_test_with_features,
                    # note tabpfn doesnt support any outside DEFAULT_QUANTILE_CONFIG right now
                    quantile_config=sorted(
                        list(set(TABPFN_TS_DEFAULT_QUANTILE_CONFIG + quantiles))
                    ),
                )
            except ValueError as ve:
                if "Number of samples" in str(
                    ve
                ) and "ignore_pretraining_limits" in str(ve):
                    raise ValueError(
                        "Dataset size exceeds model limit of 10000 samples. Try reducing the dataset size or contact Synthefy for support."
                    )
                raise ve
            except Exception as e:
                logger.error(
                    f"Error making predictions for {current_target_column}: {e}"
                )
                raise e

            prediction_time = time.time() - start_time
            logger.debug(
                f"Prediction completed in {prediction_time:.2f} seconds"
            )
            logger.debug(
                f"Prediction result shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}"
            )
            explanation = None
            if do_llm_explanation:
                explanation = _generate_llm_explanation(
                    df=tabpfn_dataframe,
                    forecast_df=pd.concat(
                        [
                            pred,
                            test_with_features[covariate_columns_to_leak],
                        ],
                        axis=1,
                    ),
                    covariate_columns_to_leak=covariate_columns_to_leak,
                    target_column=current_target_column,
                )

            # Generate SHAP analysis if requested
            shap_analysis = None
            if do_shap_analysis:
                try:
                    # Get feature names from the training data (include all features including target for time series)
                    feature_names = train_with_features.columns.tolist()

                    shap_analysis = _generate_shap_analysis(
                        train_df=train_with_features,
                        test_df=test_with_features,
                        predictor=self.predictor,
                        target_column=current_target_column,
                        feature_names=feature_names,
                        n_background_samples=min(
                            200, len(train_with_features)
                        ),  # More background samples
                        n_explanation_samples=min(10, len(test_with_features)),
                        shap_method=shap_method,
                    )
                    logger.debug(
                        f"SHAP analysis completed for {current_target_column}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating SHAP analysis for {current_target_column}: {e}"
                    )
                    shap_analysis = None

            target_forecast = ForecastGroup(
                target_column=current_target_column,
                forecasts=pred["target"].values.tolist(),
                confidence_intervals=[
                    ConfidenceInterval(
                        lower=pred[quantiles[0]].values[idx],
                        upper=pred[quantiles[1]].values[idx],
                    )
                    for idx in range(len(pred))
                ],
                univariate_forecasts=univariate_pred["target"].values.tolist(),
                univariate_confidence_intervals=[
                    ConfidenceInterval(
                        lower=univariate_pred[quantiles[0]].values[idx],
                        upper=univariate_pred[quantiles[1]].values[idx],
                    )
                    for idx in range(len(univariate_pred))
                ],
                explanation=explanation,
                shap_analysis=shap_analysis,
            )

            # Process ground truth data
            ground_truth_values = self._process_ground_truth(
                ground_truth_df,
                current_target_column,
                future_time_stamps,
                datetime_column,
            )

            target_forecast.ground_truth = ground_truth_values  # type: ignore

            logger.debug(
                f"Adding forecast for {current_target_column} to final predictions"
            )
            final_predictions.values.append(target_forecast)

        # Return the predictions as a pandas DataFrame
        logger.debug(
            f"Returning final predictions with {len(final_predictions.values)} target columns"
        )
        return final_predictions


class SynthefyFoundationModelPredictor:
    def __init__(self, model_type: SynthefyTimeSeriesModelType | str):
        logger.debug("Initializing SynthefyFoundationModelPredictor")
        self.foundation_model_metadata: SynthefyFoundationModelMetadata = (
            get_foundation_model_info(FM_SETTINGS.local_model_path)
        )

        if (
            model_type
            == SynthefyTimeSeriesModelType.FOUNDATION_MODEL_FORECASTING
        ):
            model_info = getattr(self.foundation_model_metadata, "forecasting")

            config = Configuration(
                config_filepath=str(model_info.local_config_path)
            )
            checkpoint = str(model_info.local_checkpoint_path)
            model, _, _ = load_forecast_model(config, checkpoint)
            self.predictor: SynthefyFoundationForecastingModelV3E = (
                model.decoder_model
            )  # type: ignore
        elif (
            model_type == SynthefyTimeSeriesModelType.FOUNDATION_MODEL_SYNTHESIS
        ):
            model_info = getattr(self.foundation_model_metadata, "synthesis")
            # TODO: Implement synthesis model
            raise NotImplementedError(
                "Synthefy synthesis model is not supported yet."
            )

    def predict(
        self,
        target_df: pd.DataFrame,
        covariate_columns: List[str],
        metadata_dataframes: List[MetadataDataFrame],
        target_columns: List[str],
        forecasting_timestamp: datetime,
        future_time_stamps: List[datetime],
        timestamp_column: str,
        ground_truth_df: pd.DataFrame | None = None,
        remove_all_metadata: bool = False,
        covariate_columns_to_leak: List[str] | None = None,
        metadata_dataframes_leak_idxs: List[int] | None = None,
        quantiles: List[float] = [0.1, 0.9],
        do_llm_explanation: bool = False,
        do_shap_analysis: bool = False,
        shap_method: str = "kernel",
    ) -> ForecastDataset:
        # Note: Modifications are now applied once after downloading data in do_forecast_core_code
        # so target_df and ground_truth_df already have modifications applied

        # Convert metadata_dataframes to the format expected by the foundation model
        # The foundation model expects List[pd.DataFrame] but we have List[MetadataDataFrame]
        metadata_dfs = []
        for meta_df in metadata_dataframes:
            if meta_df.df is not None:
                metadata_dfs.append(meta_df.df)

        return self.predictor.predict(
            target_df=target_df,
            covariate_columns=covariate_columns,
            metadata_dataframes=metadata_dfs,
            target_columns=target_columns,
            forecasting_timestamp=forecasting_timestamp,
            future_time_stamps=future_time_stamps,
            ground_truth_df=ground_truth_df,
            remove_all_metadata=remove_all_metadata,
            covariate_columns_to_leak=covariate_columns_to_leak,
            metadata_dataframes_leak_idxs=metadata_dataframes_leak_idxs,
            quantiles=quantiles,
            timestamp_column=timestamp_column,
        )


class UnsupportedModelError(Exception):
    pass


class FoundationModelService:
    @staticmethod
    @lru_cache(maxsize=4)
    def get_model(
        model_type: SynthefyTimeSeriesModelType | str,
        user_id: str | None = None,
        dataset_name: str | None = None,
    ):
        logger.debug("Getting foundation model")
        if (
            model_type == "tabpfn"
            or model_type == SynthefyTimeSeriesModelType.DEFAULT
        ):
            return TabPFNPredictor()
        elif (
            model_type
            == SynthefyTimeSeriesModelType.FOUNDATION_MODEL_FORECASTING
        ):
            return SynthefyFoundationModelPredictor(model_type=model_type)
        logger.error(f"Unsupported model type: {model_type}")
        raise UnsupportedModelError(f"Model '{model_type}' is not supported.")
