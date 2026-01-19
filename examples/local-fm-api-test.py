import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------- Setup API ---------------------------------------------------
X_API_KEY = "XXXX-XXXX"  # Replace with your API key
BASE_URL = "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=30.0)
ENDPOINT = "api/foundation_models/forecast/stream"


class FoundationModelForecastStreamRequest(BaseModel):
    # --------------- User uploaded data ---------------
    historical_timestamps: List[str]

    # from df.to_dict(orient='list')
    historical_timeseries_data: Dict[str, List[Any]]
    targets: List[str]  # must be present as keys in historical_timeseries_data

    # must be present in historical_timeseries_data if provided
    covariates: List[str] = Field(default_factory=list)
    # --------------- End user uploaded data ---------------

    # --------------- Synthefy Database context ---------------
    synthefy_metadata_info_combined: None  # Unused
    synthefy_metadata_leak_idxs: Optional[List[int]] = None  # Unused
    # --------------- End Synthefy Database context ---------------

    # --------------- Data for Forecasting ---------------
    # the timestamps for which we want to predict the targets' values
    forecast_timestamps: List[str]
    # from df.to_dict(orient='list'); future metadata that will be used
    future_timeseries_data: Dict[str, List[Any]] | None
    # --------------- End Data for Forecasting ---------------

    # Dict used to add constant context (will be same for each timestamp/repeated for the dfs)
    static_context: Dict[str, float | int | str] | None
    prompt: str | None  # Prompt/description of the task/data/etc

    quantiles: List[float] | None  # which quantiles to return


def make_api_call(request: FoundationModelForecastStreamRequest):
    # Make the API call
    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-API-Key": X_API_KEY},
    )
    return response


# ## Helper Functions
#
# We provide a comprehensive helper function to convert your data into the format our API expects:


def convert_df_to_synthefy_request(
    df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    target_cols: List[str],
    forecast_timestamps: List[str],
    timestamp_col: Optional[str] = None,  # auto-detect if not provided
    covariate_cols: List[str] = [],
) -> FoundationModelForecastStreamRequest:
    """Convert pandas DataFrames into a Synthefy API request.

    This function handles all the data preparation needed to make a forecast request.
    It supports:
    - Automatic timestamp detection
    - Multiple target variables
    - Optional covariates
    - Future known data
    - External data integration

    Args:
        df: Historical data DataFrame
        future_df: Future known data DataFrame (can be empty)
        target_cols: Columns to forecast
        forecast_timestamps: Timestamps to forecast for
        timestamp_col: Column containing timestamps (auto-detected if None)
        covariate_cols: Additional columns to use as features
        synthefy_metadata_info_combined: External data sources to use
        synthefy_metadata_leak_idxs: Which external data sources to use

    Returns:
        A properly formatted request object for the Synthefy API
    """
    df_copy = df.copy()
    # auto-detect timestamp column if not provided
    if timestamp_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                timestamp_col = col
                break
    else:
        df.loc[:, timestamp_col] = pd.to_datetime(df[timestamp_col])
        if future_df is not None:
            future_df.loc[:, timestamp_col] = pd.to_datetime(
                future_df[timestamp_col]
            )
    if not timestamp_col:
        raise ValueError("No timestamp column found")

    historical_timestamps = [
        ts.isoformat() for ts in df[timestamp_col].tolist()
    ]

    # drop timestamp column from df
    df_copy = df_copy.drop(columns=[timestamp_col])

    df_copy = df_copy[target_cols + covariate_cols]
    historical_timeseries_data = df_copy.to_dict(orient="list")
    historical_timeseries_data = {
        str(k): v for k, v in historical_timeseries_data.items()
    }

    # Get the future_timeseries_data (don't give the target columns)
    future_timeseries_data = None
    if future_df is not None:
        future_timeseries_data = future_df[covariate_cols].to_dict(
            orient="list"
        )
        future_timeseries_data = {
            str(k): v for k, v in future_timeseries_data.items()
        }

    # create request object
    request = FoundationModelForecastStreamRequest(
        historical_timestamps=historical_timestamps,
        historical_timeseries_data=historical_timeseries_data,
        targets=target_cols,
        covariates=covariate_cols,
        synthefy_metadata_info_combined=None,
        synthefy_metadata_leak_idxs=None,
        forecast_timestamps=forecast_timestamps,
        future_timeseries_data=future_timeseries_data,
        static_context=None,  # Not yet supported
        prompt=None,  # Not yet supported
        quantiles=None,
    )

    return request


def convert_response_to_df(
    response: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert API response to pandas DataFrames for easy analysis.

    Args:
        response: The API response dictionary

    Returns:
        Tuple of (forecast_df, quantiles_df) where:
        - forecast_df contains the point forecasts
        - quantiles_df contains the forecast quantiles
    """
    forecast_dict = {k: v for k, v in response["forecast"].items()}
    quantiles = {k: v for k, v in response["forecast_quantiles"].items()}
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df["timestamp"] = pd.to_datetime(response["forecast_timestamps"])
    quantiles_df = pd.DataFrame(quantiles)
    quantiles_df["timestamp"] = pd.to_datetime(response["forecast_timestamps"])
    return forecast_df, quantiles_df


if __name__ == "__main__":
    synthefy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # examples/local-fm-api-test.py
    df = pd.read_csv(f"{synthefy_root}/examples/walmart_sales.csv")

    # The last date is 2012-10-26. -> Let's forecast for the next 4 weeks.
    df = df[df["Store"] == "store_1"]

    # Create a basic forecast request
    request = convert_df_to_synthefy_request(
        df=df,
        future_df=None,  # No future data in this example
        target_cols=["Weekly_Sales"],
        forecast_timestamps=[
            "2012-11-02",
            "2012-11-09",
            "2012-11-16",
            "2012-11-23",
        ],
        timestamp_col="Date",
        covariate_cols=[],
    )

    # Let's examine our request
    print(json.dumps(request.model_dump(), indent=4))

    # Make the API call
    response = make_api_call(request)
    print(f"response {response.status_code}")

    # Convert response to DataFrames for analysis
    forecast_df, quantiles_df = convert_response_to_df(response.json())

    print(forecast_df.head())
    print(quantiles_df.head())
