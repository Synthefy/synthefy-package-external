import asyncio
import io
import json
import os
import threading
import traceback
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import boto3
import ijson
import pandas as pd
import pytz
import requests
from fastapi import HTTPException, status
from haver import Haver
from loguru import logger
from pydantic import BaseModel
from smart_open import open as smart_open

from synthefy_pkg.app.data_models import (
    HaverMetadataAccessInfo,
    MetadataDataFrame,
    PredictHQAggregationConfig,
    PredictHQMetadataAccessInfo,
    WeatherMetadataAccessInfo,
    WeatherStackLocation,
)
from synthefy_pkg.app.utils.api_utils import delete_s3_objects

COMPILE = True

# Rate limiting for WeatherStack API calls
WEATHERSTACK_RATE_LIMIT = asyncio.Semaphore(5)  # Max 5 concurrent requests
WEATHERSTACK_REQUEST_COUNT = 0

# Maximum number of chunks to prevent excessive API usage
MAX_WEATHERSTACK_CHUNKS = 50
MAX_WEATHERSTACK_CHUNKS_PER_DAY = 60

# Timezone offset validation limits based on real-world timezone boundaries
# UTC-12: Westernmost timezone (Baker Island, Howland Island - uninhabited US territories)
# UTC+14: Easternmost timezone (Line Islands, Kiribati - crosses International Date Line)
# These represent the maximum possible timezone offsets that exist on Earth
MIN_TIMEZONE_OFFSET_HOURS = -12
MAX_TIMEZONE_OFFSET_HOURS = 14

HAVER_API_KEY = os.getenv("HAVER_API_KEY")
if not HAVER_API_KEY:
    logger.error("HAVER_API_KEY environment variable not set")
haver = Haver(api_key=HAVER_API_KEY)

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
if not WEATHERSTACK_API_KEY:
    logger.error("WEATHERSTACK_API_KEY environment variable not set")

PREDICTHQ_API_KEY = os.getenv("PREDICTHQ_API_KEY")
if not PREDICTHQ_API_KEY:
    logger.error("PREDICTHQ_API_KEY environment variable not set")


# Mapping for PredictHQ category names (model field names to API category names)
PREDICTHQ_CATEGORY_MAPPING = {
    "academic": "academic",
    "concerts": "concerts",
    "conferences": "conferences",
    "expos": "expos",
    "festivals": "festivals",
    "performing_arts": "performing-arts",
    "sports": "sports",
    "community": "community",
    "daylight_savings": "daylight-savings",
    "observances": "observances",
    "politics": "politics",
    "public_holidays": "public-holidays",
    "school_holidays": "school-holidays",
    "severe_weather": "severe-weather",
}

# PredictHQ API limitations constants
PREDICTHQ_MAX_HISTORICAL_DAYS = 90  # Maximum days back we can query with current API plan (configurable for plan changes)

# Supported features in PredictHQ Features API (based on subscription level)
# These are the categories that have confirmed attendance-based features
PREDICTHQ_SUPPORTED_ATTENDANCE_FEATURES = {
    "concerts",
    "conferences",
    "expos",
    "festivals",
    "performing-arts",
    "sports",
    "community",
}

# These are the categories that have confirmed rank-based features
PREDICTHQ_SUPPORTED_RANK_FEATURES = {
    "daylight-savings",
    "observances",
    "politics",
    "public-holidays",
    "school-holidays",
    # "severe-weather",  # Removed: Not available with current subscription plan
}

# NOTE: The above constants should be updated based on your actual PredictHQ subscription plan.
# Some features may not be available depending on your subscription level.
# Always test against the actual API to confirm feature availability.

# Academic events have specific subtypes rather than a generic feature
PREDICTHQ_ACADEMIC_RANK_FEATURES = {
    "phq_rank_academic_exam",
    "phq_rank_academic_holiday",
}


def get_weatherstack_api_params(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    time_frequency: str,
    units: str,
) -> Dict[str, Any]:
    """Generate parameters for WeatherStack historical API call.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date for historical data
        end_date: End date for historical data
        time_frequency: Time frequency (e.g. "week", "day", "hour")
        units: Units for the data, 'm' for meters, 's' for scientific notation, 'f' for Fahrenheit

    Returns:
        Dictionary containing API parameters for WeatherStack
    """
    frequency_mapping = {
        "minute": 1,
        "hour": 1,
        "day": 24,
        "week": 24,
        "month": 24,
        "year": 24,
    }

    interval = frequency_mapping.get(time_frequency.lower(), 24)

    return {
        "access_key": WEATHERSTACK_API_KEY,
        "query": f"{latitude},{longitude}",
        "historical_date_start": start_date.strftime("%Y-%m-%d"),
        "historical_date_end": end_date.strftime("%Y-%m-%d"),
        "interval": interval,
        "hourly": 1,
        "units": units,
    }


def get_weatherstack_location_timezone_offset(
    latitude: float,
    longitude: float,
) -> float:
    """Get timezone offset for a location using WeatherStack API.

    Args:
        latitude: Location latitude coordinate
        longitude: Location longitude coordinate

    Returns:
        Timezone offset as a float (e.g., -5.0 for UTC-5)

    Raises:
        ValueError: If API request fails, returns invalid data, or offset is out of valid range
    """
    global WEATHERSTACK_REQUEST_COUNT
    WEATHERSTACK_REQUEST_COUNT += 1

    current_url = "https://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": f"{latitude},{longitude}",
    }

    logger.info(
        f"WeatherStack timezone API call #{WEATHERSTACK_REQUEST_COUNT} with params: {params}"
    )
    response = requests.get(current_url, params=params)

    try:
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")

        json_response = response.json()
        if "error" in json_response:
            raise ValueError(f"API error: {json_response['error']}")

        response_data = json_response["location"]
        offset = float(response_data["utc_offset"])

        # Validate timezone offset is within real-world limits
        # This prevents API errors, corrupted data, or invalid timezone calculations
        # from propagating through the system
        if not MIN_TIMEZONE_OFFSET_HOURS <= offset <= MAX_TIMEZONE_OFFSET_HOURS:
            raise ValueError(
                f"Invalid timezone offset: {offset}. Must be between {MIN_TIMEZONE_OFFSET_HOURS} and {MAX_TIMEZONE_OFFSET_HOURS} hours."
            )

        return offset
    except Exception as e:
        raise ValueError(
            f"Error getting weatherstack location timezone offset: {e}"
        )


def call_weatherstack_historical_api(
    params: Dict[str, Any], forecast: bool = False
) -> Dict[str, Any]:
    """Call WeatherStack historical or forecast API with given parameters.

    Args:
        params: API parameters dictionary
        forecast: Whether to call the forecast API instead of historical

    Returns:
        JSON response from WeatherStack API
    """
    global WEATHERSTACK_REQUEST_COUNT
    WEATHERSTACK_REQUEST_COUNT += 1

    history_url = (
        "https://api.weatherstack.com/forecast"
        if forecast
        else "https://api.weatherstack.com/historical"
    )

    logger.info(
        f"WeatherStack API call #{WEATHERSTACK_REQUEST_COUNT} to {history_url} with params: {params}"
    )
    response = requests.get(history_url, params=params)

    # Check for API errors
    if response.status_code != 200:
        logger.error(f"WeatherStack API error: HTTP {response.status_code}")
        return {}

    try:
        json_response = response.json()
        if "error" in json_response:
            logger.error(f"WeatherStack API error: {json_response['error']}")
            return {}
        return json_response
    except Exception as e:
        logger.error(f"Error parsing WeatherStack response: {e}")
        return {}


def weatherstack_to_df(api_response: Dict[str, Any]) -> pd.DataFrame:
    """Transform Weatherstack historical API response into a clean DataFrame.

    Args:
        api_response: Parsed JSON from Weatherstack API

    Returns:
        DataFrame with columns: ['timestamp', ...weather parameters...]
    """
    records = []
    historical = api_response.get("historical", {})
    for date, daydata in historical.items():
        for hour in daydata.get("hourly", []):
            time_str = str(hour.get("time", "0")).zfill(4)
            timestamp = f"{date} {time_str[:2]}:{time_str[2:]}"
            record = {"timestamp": timestamp}
            for k, v in hour.items():
                if k != "time":
                    record[k] = v
            records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d %H:%M"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _chunk_date_range(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_days: int = MAX_WEATHERSTACK_CHUNKS_PER_DAY,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into chunks of specified days.

    Args:
        start_date: Start date
        end_date: End date
        chunk_days: Number of days per chunk (default 60 for WeatherStack API limit)

    Returns:
        List of (chunk_start, chunk_end) tuples
    """
    current_time = pd.Timestamp.now()

    # Handle timezone-aware vs timezone-naive comparison
    if start_date.tz is not None and current_time.tz is None:
        current_time = current_time.tz_localize("UTC")
    elif start_date.tz is None and current_time.tz is not None:
        current_time = current_time.tz_localize(None)

    # Ensure end_date has the same timezone handling as start_date
    if start_date.tz is not None and end_date.tz is None:
        end_date = end_date.tz_localize(start_date.tz)
    elif start_date.tz is None and end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    if end_date > current_time:
        logger.warning(
            f"End date {end_date} is in the future. Clipping to current time {current_time}"
        )
        end_date = current_time

    if start_date > current_time:
        logger.warning(
            f"Start date {start_date} is in the future. No valid date range available."
        )
        return []

    # Calculate total days and warn if too many chunks would be created
    total_days = (end_date - start_date).days
    estimated_chunks = max(1, total_days // chunk_days)

    if (
        estimated_chunks > MAX_WEATHERSTACK_CHUNKS
    ):  # Limit to prevent excessive API usage
        logger.warning(
            f"Date range would create {estimated_chunks} chunks ({total_days} days). "
            f"This could use {estimated_chunks} API calls. Consider reducing the date range."
        )
        # Optionally limit the date range
        max_days = MAX_WEATHERSTACK_CHUNKS * chunk_days  # 50 chunks max
        if total_days > max_days:
            new_start_date = end_date - pd.Timedelta(days=max_days)
            logger.warning(
                f"Limiting date range from {start_date} to {new_start_date}"
            )
            start_date = new_start_date

    chunks = []
    current_start = start_date

    while current_start < end_date:
        chunk_end = min(
            current_start + pd.Timedelta(days=chunk_days - 1), end_date
        )
        chunks.append((current_start, chunk_end))
        current_start = chunk_end + pd.Timedelta(days=1)

    logger.info(f"Created {len(chunks)} date chunks for WeatherStack API calls")
    return chunks


async def _process_weatherstack_chunk(
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    latitude: float,
    longitude: float,
    time_frequency: str,
    units: str,
) -> pd.DataFrame:
    """Process a single weatherstack chunk by making API call and converting to DataFrame.

    Args:
        chunk_start: Start date for the chunk
        chunk_end: End date for the chunk
        latitude: Location latitude
        longitude: Location longitude
        time_frequency: Time frequency for data
        units: Units for the data, 'm' for meters, 's' for scientific notation, 'f' for Fahrenheit

    Returns:
        DataFrame containing processed weather data for the chunk
    """
    async with WEATHERSTACK_RATE_LIMIT:  # Rate limiting
        # Add a small delay between requests
        await asyncio.sleep(0.2)  # 200ms delay to avoid hitting rate limits

        params = get_weatherstack_api_params(
            latitude=latitude,
            longitude=longitude,
            start_date=chunk_start,
            end_date=chunk_end,
            time_frequency=time_frequency,
            units=units,
        )
        try:
            weatherstack_data = await asyncio.to_thread(
                call_weatherstack_historical_api, params
            )

            if not weatherstack_data:  # Handle empty response
                logger.warning(
                    f"Empty response for chunk {chunk_start} to {chunk_end}"
                )
                return pd.DataFrame()

            chunk_df = weatherstack_to_df(weatherstack_data)
            return chunk_df
        except Exception as e:
            logger.error(
                f"Error processing chunk {chunk_start} to {chunk_end}: {str(e)}"
            )
            return pd.DataFrame()


# TODO: optimize this function to be more efficient
def _process_weather_interval(
    i: int,
    target_timestamps_parsed: List[pd.Timestamp],
    combined_df: pd.DataFrame,
    aggregation_methods: Dict[str, str],
) -> Optional[pd.Series]:
    """Process a single weather data interval and return aggregated data.

    Args:
        i: Index of the target timestamp to process
        target_timestamps_parsed: List of parsed target timestamps
        combined_df: DataFrame with weather data
        aggregation_methods: Dictionary mapping weather parameters to aggregation methods

    Returns:
        Series with aggregated weather data for the interval, or None if no data
    """
    target_ts = target_timestamps_parsed[i]

    # Determine the interval start
    if i == 0:
        # For the first timestamp, use data from the beginning up to and including this timestamp
        interval_start = combined_df["timestamp"].min()
        interval_data = combined_df[combined_df["timestamp"] <= target_ts]
    else:
        # For subsequent timestamps, use data from previous timestamp (exclusive) to current (inclusive)
        interval_start = target_timestamps_parsed[i - 1]
        interval_data = combined_df[
            (combined_df["timestamp"] > interval_start)
            & (combined_df["timestamp"] <= target_ts)
        ]

    if interval_data.empty:
        logger.warning(
            f"No data found for interval {interval_start} to {target_ts}"
        )
        return None

    # Start with timestamp
    aggregated: Dict[str, Any] = {"timestamp": target_ts}

    # Aggregate each weather parameter
    for col in interval_data.columns:
        if col == "timestamp":
            continue

        col_data = interval_data[col]

        # Check if column contains numeric data
        if not pd.api.types.is_numeric_dtype(col_data):
            # For non-numeric data (e.g., wind direction strings), take the most recent value
            aggregated[col] = col_data.iloc[-1]
            logger.debug(
                f"Using most recent value for non-numeric column {col}: {aggregated[col]}"
            )
            continue

        method = aggregation_methods.get(col, "mean")  # Default to mean

        try:
            if method == "mean":
                result = col_data.mean()
            elif method == "max":
                result = col_data.max()
            elif method == "sum":
                result = col_data.sum()
            elif method == "min":
                result = col_data.min()
            else:
                result = col_data.mean()  # Default fallback

            # Handle NaN results
            if pd.isna(result):
                aggregated[col] = col_data.iloc[-1]
                logger.debug(f"NaN result for {col}, using most recent value")
            else:
                aggregated[col] = (
                    float(result)
                    if isinstance(result, (int, float))
                    else result
                )

        except (ValueError, TypeError) as e:
            # Fallback to most recent value if aggregation fails
            aggregated[col] = col_data.iloc[-1]
            logger.warning(
                f"Failed to aggregate column {col}, using most recent value: {e}"
            )

    logger.debug(f"Aggregated {len(interval_data)} data points for {target_ts}")
    return pd.Series(aggregated)


def _aggregate_weather_data_for_intervals(
    combined_df: pd.DataFrame, target_timestamps_parsed: List[pd.Timestamp]
) -> pd.DataFrame:
    """Aggregate weather data over intervals between target timestamps.

    Args:
        combined_df: DataFrame with weather data (columns: timestamp, weather_params...)
        target_timestamps_parsed: List of parsed target timestamps as pd.Timestamp objects, sorted

    Returns:
        DataFrame with aggregated weather data for each target timestamp
    """
    if combined_df.empty or not target_timestamps_parsed:
        return pd.DataFrame()

    # Parse target timestamps
    target_timestamps_parsed = sorted(target_timestamps_parsed)

    # Define aggregation methods for different weather parameters
    aggregation_methods = {
        "temperature": "mean",
        "humidity": "mean",
        "pressure": "mean",
        "visibility": "mean",
        "cloudcover": "mean",
        "heatindex": "mean",
        "dewpoint": "mean",
        "feelslike": "mean",
        "windchill": "mean",
        "uv_index": "mean",
        # Precipitation and wind - use sum/max for intensity measures
        "precip": "sum",  # Total precipitation
        "wind_speed": "mean",
        "windgust": "max",
        "wind_degree": "mean",  # Circular mean would be better, but simple mean for now
        # Chance/probability parameters - use mean
        "chanceofrain": "mean",
        "chanceofremdry": "mean",
        "chanceofwindy": "mean",
        "chanceofovercast": "mean",
        "chanceofsunshine": "mean",
        "chanceoffrost": "mean",
        "chanceofhightemp": "mean",
        "chanceoffog": "mean",
        "chanceofsnow": "mean",
        "chanceofthunder": "mean",
    }

    # Ensure timestamp column is datetime
    combined_df = combined_df.copy()
    combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # Process intervals in parallel using asyncio.to_thread would be ideal,
    # but since this is a sync function, we'll use sequential processing for now
    # Can be parallelized later with concurrent.futures if needed
    aggregated_rows = []
    for i in range(len(target_timestamps_parsed)):
        result = _process_weather_interval(
            i, target_timestamps_parsed, combined_df, aggregation_methods
        )
        if result is not None:
            aggregated_rows.append(result)

    if not aggregated_rows:
        logger.warning("No aggregated data could be generated")
        return pd.DataFrame()

    # Convert to DataFrame
    result_df = pd.DataFrame(aggregated_rows).reset_index(drop=True)
    logger.info(
        f"Aggregated weather data for {len(result_df)} target timestamps"
    )
    return result_df


def _detect_timestamp_format(
    target_timestamps: List[pd.Timestamp],
    original_timestamp_strings: Optional[List[str]] = None,
) -> str:
    """Detect timestamp format from a list of parsed timestamps and optionally original strings.

    Args:
        target_timestamps: List of parsed target timestamps
        original_timestamp_strings: Optional list of original timestamp strings before parsing

    Returns:
        Format string for timestamp parsing
    """
    try:
        if not target_timestamps:
            return "%Y-%m-%dT%H:%M:%S"  # Default to ISO format

        # If we have original strings, analyze them directly for better accuracy
        if original_timestamp_strings and len(original_timestamp_strings) > 0:
            return _detect_format_from_strings(original_timestamp_strings)

        # Fallback: analyze parsed timestamps (less accurate but still useful)
        return _detect_format_from_parsed_timestamps(target_timestamps)

    except Exception as e:
        logger.warning(f"Could not detect timestamp format: {e}")
        logger.error(f"Trace: {traceback.format_exc()}")
        return "%Y-%m-%dT%H:%M:%S"  # Default to ISO format


def _detect_format_from_strings(timestamp_strings: List[str]) -> str:
    """Detect timestamp format by analyzing original string patterns.

    Args:
        timestamp_strings: List of original timestamp strings

    Returns:
        Detected format string
    """
    if not timestamp_strings:
        return "%Y-%m-%dT%H:%M:%S"

    # Convert any Timestamp objects to strings
    string_samples = []
    for item in timestamp_strings[:3]:
        try:
            if isinstance(item, pd.Timestamp):  # It's a Timestamp object
                string_samples.append(item.strftime("%Y-%m-%d"))
            else:
                string_samples.append(str(item))
        except Exception as e:
            logger.warning(f"Error converting timestamp item to string: {e}")
            string_samples.append(str(item))

    # Take up to 3 samples for analysis
    samples = string_samples

    # Common timestamp patterns to check
    patterns = [
        # Year-month patterns
        (r"^\d{4}-\d{2}$", "%Y-%m"),  # 2023-01
        (r"^\d{4}/\d{2}$", "%Y/%m"),  # 2023/01
        # Date-only patterns
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),  # 2023-01-15
        (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),  # 2023/01/15
        (r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y"),  # 01/15/2023
        (r"^\d{2}-\d{2}-\d{4}$", "%m-%d-%Y"),  # 01-15-2023
        # ISO datetime patterns
        (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
            "%Y-%m-%dT%H:%M:%SZ",
        ),  # 2023-01-15T10:30:00Z
        (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:?\d{2}$",
            "%Y-%m-%dT%H:%M:%S%z",
        ),  # With timezone offset
        (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
            "%Y-%m-%dT%H:%M:%S",
        ),  # 2023-01-15T10:30:00
        (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$",
            "%Y-%m-%dT%H:%M",
        ),  # 2023-01-15T10:30
        # Space-separated datetime patterns
        (
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
            "%Y-%m-%d %H:%M:%S",
        ),  # 2023-01-15 10:30:00
        (
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$",
            "%Y-%m-%d %H:%M",
        ),  # 2023-01-15 10:30
        # Alternative datetime patterns
        (
            r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$",
            "%m/%d/%Y %H:%M:%S",
        ),  # 01/15/2023 10:30:00
        (
            r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$",
            "%m/%d/%Y %H:%M",
        ),  # 01/15/2023 10:30
    ]

    import re

    # Check each sample against patterns
    for sample in samples:
        sample = sample.strip()
        for pattern, format_str in patterns:
            if re.match(pattern, sample):
                logger.info(
                    f"Detected timestamp format: {format_str} from sample: {sample}"
                )
                return format_str

    # If no specific pattern matches, try to infer based on string structure
    sample = samples[0].strip()

    # Basic inference logic
    if "T" in sample:
        if sample.endswith("Z"):
            return "%Y-%m-%dT%H:%M:%SZ"
        elif "+" in sample[-6:] or "-" in sample[-6:]:
            return "%Y-%m-%dT%H:%M:%S%z"
        else:
            return "%Y-%m-%dT%H:%M:%S"
    elif " " in sample and ":" in sample:
        return "%Y-%m-%d %H:%M:%S"
    elif "-" in sample and sample.count("-") == 1:
        return "%Y-%m"  # Year-month format
    elif "-" in sample and sample.count("-") == 2:
        return "%Y-%m-%d"  # Date-only format
    elif "/" in sample:
        parts = sample.split("/")
        if len(parts) == 3 and len(parts[2]) == 4:  # MM/DD/YYYY
            return "%m/%d/%Y"
        elif len(parts) == 3 and len(parts[0]) == 4:  # YYYY/MM/DD
            return "%Y/%m/%d"
        elif len(parts) == 2:  # YYYY/MM
            return "%Y/%m"

    logger.warning(f"Could not match timestamp pattern for: {sample}")
    return "%Y-%m-%dT%H:%M:%S"  # Default fallback


def _detect_format_from_parsed_timestamps(
    target_timestamps: List[pd.Timestamp],
) -> str:
    """Detect timestamp format from parsed timestamps (fallback method).

    Args:
        target_timestamps: List of parsed timestamps

    Returns:
        Detected format string
    """
    # Get the first timestamp
    first_ts = target_timestamps[0]

    # Check if it has time components (hour, minute, second)
    has_time = not (
        first_ts.hour == 0 and first_ts.minute == 0 and first_ts.second == 0
    )

    # Check if it has timezone info
    if first_ts.tz is not None:
        # Check for common timezone formats
        if str(first_ts.tz) == "UTC":
            return "%Y-%m-%dT%H:%M:%SZ"  # UTC with Z
        else:
            return "%Y-%m-%dT%H:%M:%S%z"  # With offset

    # For timezone-naive timestamps
    if has_time:
        return "%Y-%m-%dT%H:%M:%S"
    else:
        # Check if all timestamps are at the start of their respective months
        # This might indicate original monthly format
        all_month_starts = all(
            ts.day == 1 and ts.hour == 0 and ts.minute == 0 and ts.second == 0
            for ts in target_timestamps[
                : min(5, len(target_timestamps))
            ]  # Check first 5
        )

        if all_month_starts and len(target_timestamps) > 1:
            # Check if timestamps are spaced monthly
            sorted_ts = sorted(target_timestamps)
            monthly_spacing = all(
                (sorted_ts[i + 1] - sorted_ts[i]).days
                >= 28  # At least 28 days (shortest month)
                for i in range(
                    min(3, len(sorted_ts) - 1)
                )  # Check first few intervals
            )

            if monthly_spacing:
                logger.info(
                    "Detected likely monthly timestamp format based on spacing and day=1 pattern"
                )
                return "%Y-%m"

        # Default to date format
        return "%Y-%m-%d"


def _create_weatherstack_metadata_dataframes(
    combined_df: pd.DataFrame,
    metadata_info: WeatherMetadataAccessInfo,
    timezone_offset: float,
    target_timestamps: List[pd.Timestamp],
    target_timezone: Optional[str] = None,
    original_timestamp_strings: Optional[List[str]] = None,
) -> List[MetadataDataFrame]:
    """Create MetadataDataFrame objects from processed weather data.

    Args:
        combined_df: Processed weather DataFrame
        metadata_info: WeatherStack metadata access information
        timezone_offset: Timezone offset for the location
        target_timestamps: List of target timestamps for description
        target_timezone: Optional timezone for the target timestamps
        original_timestamp_strings: Optional list of original timestamp strings before parsing

    Returns:
        List of MetadataDataFrame objects
    """
    weather_parameters = metadata_info.weather_parameters
    location_data = metadata_info.location_data

    # Filter to only requested columns
    requested_columns = ["timestamp"] + [
        param
        for param, enabled in weather_parameters.model_dump().items()
        if enabled and param in combined_df.columns
    ]
    result_df = combined_df[requested_columns]

    # Handle timezone conversion if target timezone is provided
    if target_timezone is not None:
        try:
            result_df = _convert_dataframe_timestamps_to_target_timezone(
                result_df, timezone_offset, target_timezone, "timestamp"
            )
            # Store timezone info consistently as string
            final_timezone = target_timezone
        except ValueError as e:
            logger.warning(
                f"Timezone conversion failed: {e}. Using location timezone offset."
            )
            final_timezone = f"UTC{int(timezone_offset):+d}:00"
    else:
        # Use location timezone offset in consistent format
        final_timezone = f"UTC{int(timezone_offset):+d}:00"

    # Format timestamps using the detected format string
    result_df["timestamp"] = result_df["timestamp"].dt.strftime(
        _detect_timestamp_format(target_timestamps, original_timestamp_strings)
    )

    metadata_dataframes = []
    for param in requested_columns:
        if param == "timestamp":
            continue

        param_df = result_df[["timestamp", param]].copy()
        param_df = param_df.rename(
            columns={"timestamp": "date", param: "value"}
        )

        metadata_json = {
            "data_source": "weather",
            "timestamp_columns": ["date"],
            "timezone": final_timezone,
            "columns": [
                {
                    "id": param,
                    "title": f"{param.title()} - {metadata_info.name}",
                    "column_id": "value",
                    "description": f"{param.title()} data for location {metadata_info.name} "
                    f"({location_data.latitude}, {location_data.longitude})",
                }
            ],
        }

        metadata_dataframes.append(
            MetadataDataFrame(
                df=param_df,
                metadata_json=metadata_json,
                description=f"WeatherStack {param.title()} data for {metadata_info.name}",
                timestamp_key="date",
            )
        )

    return metadata_dataframes


async def _fetch_weatherstack_data_for_date_range_async(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    latitude: float,
    longitude: float,
    time_frequency: str,
    units: str,
) -> pd.DataFrame:
    """Fetch weather data for a specific date range (async version).

    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching
        latitude: Location latitude
        longitude: Location longitude
        time_frequency: Time frequency for data
        units: Units for the data, 'm' for metric, 's' for scientific, 'f' for Fahrenheit

    Returns:
        DataFrame with weather data for the date range (empty if no data)
    """
    try:
        date_chunks = _chunk_date_range(start_date, end_date)
        if not date_chunks:
            return pd.DataFrame()

        tasks = [
            _process_weatherstack_chunk(
                chunk_start,
                chunk_end,
                latitude,
                longitude,
                time_frequency,
                units,
            )
            for chunk_start, chunk_end in date_chunks
        ]

        chunk_results = await asyncio.gather(*tasks)
        combined_df = pd.concat(chunk_results, ignore_index=True)

        if combined_df.empty:
            return pd.DataFrame()

        return (
            combined_df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return pd.DataFrame()


async def _process_weatherstack_without_specific_timestamps(
    metadata_info: WeatherMetadataAccessInfo,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    latitude: float,
    longitude: float,
    units: str,
) -> Optional[pd.DataFrame]:
    """Process WeatherStack data without specific timestamps.

    This is the original behavior where data is fetched for the full date range
    and then aggregated over intervals to match target_timestamps.

    Args:
        metadata_info: WeatherStack metadata access information
        start_time: Start time for the data
        end_time: End time for the data
        latitude: Location latitude
        longitude: Location longitude
        units: Units for the data, 'm' for metric, 's' for scientific, 'f' for Fahrenheit

    Returns:
        DataFrame with aggregated weather data or None if processing fails
    """
    try:
        time_frequency = metadata_info.frequency

        logger.info(
            f"Fetching data from {start_time} to {end_time} for {metadata_info.name}"
        )

        combined_df = await _fetch_weatherstack_data_for_date_range_async(
            start_time, end_time, latitude, longitude, time_frequency, units
        )

        if combined_df.empty:
            logger.warning("No weather data available for the time period")
            return None

        return combined_df if not combined_df.empty else None

    except Exception as e:
        logger.error(f"Error in fetching weather data: {str(e)}")
        return None


async def _fetch_weatherstack_single_date(
    date: pd.Timestamp,
    latitude: float,
    longitude: float,
    time_frequency: str,
    units: str,
) -> pd.DataFrame:
    """Fetch weather data for a single specific date from WeatherStack API.

    This function is designed for frequencies larger than daily (week, month, year).
    It fetches daily aggregate data for the specified date, not hourly data.

    Args:
        date: Specific date to fetch data for
        latitude: Location latitude
        longitude: Location longitude
        time_frequency: Time frequency for data (should be week, month, or year)
        units: Units for the data, 'm' for metric, 's' for scientific, 'f' for Fahrenheit

    Returns:
        DataFrame with daily aggregate weather data for the specific date (empty if no data)
    """
    async with WEATHERSTACK_RATE_LIMIT:  # Rate limiting
        # Add a small delay between requests
        await asyncio.sleep(0.1)  # 100ms delay to avoid hitting rate limits

        try:
            # Use the single date format: historical_date = YYYY-MM-DD
            # For frequencies larger than daily (week, month, year), we want daily aggregates, not hourly data
            params = {
                "access_key": WEATHERSTACK_API_KEY,
                "query": f"{latitude},{longitude}",
                "historical_date_start": date.strftime("%Y-%m-%d"),
                "historical_date_end": date.strftime("%Y-%m-%d"),
                "hourly": 1,
                "interval": 24,
                "units": units,
            }

            # No interval needed since we're getting daily aggregates

            logger.info(
                f"Fetching WeatherStack data for single date {date.strftime('%Y-%m-%d')} with params: {params}"
            )

            weatherstack_data = await asyncio.to_thread(
                call_weatherstack_historical_api, params
            )

            if not weatherstack_data:  # Handle empty response
                logger.warning(
                    f"Empty response for date {date.strftime('%Y-%m-%d')}"
                )
                return pd.DataFrame()

            weatherstack_df = weatherstack_to_df(weatherstack_data)
            # Convert to DataFrame - for daily aggregates, use a simpler conversion
            return weatherstack_df

        except Exception as e:
            logger.error(
                f"Error fetching data for date {date.strftime('%Y-%m-%d')}: {str(e)}"
            )
            return pd.DataFrame()


async def _process_weatherstack_for_specific_timestamps(
    metadata_info: WeatherMetadataAccessInfo,
    target_timestamps_parsed: List[pd.Timestamp],
    latitude: float,
    longitude: float,
    units: str,
) -> Optional[pd.DataFrame]:
    """Process WeatherStack data for specific timestamps only.

    This is the new efficient behavior where only the exact dates in target_timestamps
    are fetched from the API, without fetching all days in between.

    Args:
        metadata_info: WeatherStack metadata access information
        target_timestamps_parsed: List of specific timestamps to fetch data for
        latitude: Location latitude
        longitude: Location longitude
        units: Units for the data, 'm' for metric, 's' for scientific, 'f' for Fahrenheit

    Returns:
        DataFrame with weather data for specific timestamps or None if processing fails
    """
    try:
        # Extract unique dates from target timestamps
        unique_dates = list(set(ts.date() for ts in target_timestamps_parsed))
        unique_dates.sort()

        logger.info(
            f"Using specific timestamps mode: fetching data for {len(unique_dates)} unique dates"
        )

        # Convert dates back to pandas Timestamps for API calls
        unique_date_timestamps = [pd.Timestamp(date) for date in unique_dates]

        # Fetch data for each unique date in parallel
        logger.info(
            f"Making {len(unique_date_timestamps)} parallel WeatherStack API calls"
        )
        tasks = [
            _fetch_weatherstack_single_date(
                date_ts, latitude, longitude, metadata_info.frequency, units
            )
            for date_ts in unique_date_timestamps
        ]

        # Execute all API calls in parallel
        date_results = await asyncio.gather(*tasks)
        combined_df = pd.concat(date_results, ignore_index=True)

        if combined_df.empty:
            logger.warning(
                "No weather data retrieved for any of the target dates"
            )
            return None

        # Remove duplicates and sort by timestamp
        combined_df = (
            combined_df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        logger.info(
            f"Retrieved {len(combined_df)} weather data points from {len(unique_dates)} dates"
        )

        # Create a DataFrame with target timestamps and their corresponding dates
        target_df = pd.DataFrame(
            {
                "timestamp": target_timestamps_parsed,
                "date": [ts.date() for ts in target_timestamps_parsed],
            }
        )

        # Add date column to combined_df for merging
        combined_df = combined_df.copy()
        combined_df["date"] = combined_df["timestamp"].dt.date

        # Merge target timestamps with daily weather data based on date
        result_df = target_df.merge(
            combined_df.drop(
                "timestamp", axis=1
            ),  # Drop original timestamp, keep weather data
            on="date",
            how="left",
        ).drop("date", axis=1)  # Clean up the temporary date column

        # Check for any missing data
        missing_count = result_df.isnull().any(axis=1).sum()
        if missing_count > 0:
            logger.warning(
                f"Missing weather data for {missing_count} target timestamps"
            )

        if result_df.empty:
            logger.warning(
                "No target timestamps could be matched with weather data"
            )
            return None

        logger.info(
            f"Successfully matched {len(result_df)} target timestamps with weather data"
        )
        return result_df

    except Exception as e:
        logger.error(f"Error in specific timestamps processing: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


async def process_weatherstack_metadata_access_info(
    metadata_info: WeatherMetadataAccessInfo,
    target_timestamps: List[str],
) -> Optional[List[MetadataDataFrame]]:
    """Process WeatherStack metadata access info and return list of MetadataDataFrames.

    Args:
        metadata_info: WeatherStack metadata access information containing location and time details
        target_timestamps: List of exact timestamps in ISO format to fetch data for.

    Returns:
        List of MetadataDataFrame objects or None if processing fails

    Note:
        - When aggregate_intervals=True (default): Fetches data for full date range and aggregates over intervals
        - When aggregate_intervals=False: Only fetches data for specific dates in target_timestamps (more efficient)
    """
    try:
        latitude = metadata_info.location_data.latitude
        longitude = metadata_info.location_data.longitude

        timezone_offset = get_weatherstack_location_timezone_offset(
            latitude, longitude
        )

        time_frequency = metadata_info.frequency

        units = metadata_info.units

        start_time, end_time, target_timestamps_parsed, target_timezone = (
            _determine_start_end_times(target_timestamps, timezone_offset)
        )

        if (
            time_frequency.lower()
            in ["day", "week", "month", "quarter", "year"]
            and metadata_info.aggregate_intervals is False
        ):
            logger.info("Using specific timestamps mode for target timestamps")
            combined_df = await _process_weatherstack_for_specific_timestamps(
                metadata_info,
                target_timestamps_parsed,
                latitude,
                longitude,
                units,
            )
        else:
            logger.info("Using without specific timestamps mode")
            combined_df = (
                await _process_weatherstack_without_specific_timestamps(
                    metadata_info,
                    start_time,
                    end_time,
                    latitude,
                    longitude,
                    units,
                )
            )
            if (
                combined_df is not None
                and time_frequency.lower()
                in ["week", "month", "quarter", "year"]
                and metadata_info.aggregate_intervals is True
            ):
                logger.info(
                    f"Aggregating weather data over intervals for {len(target_timestamps_parsed)} target timestamps"
                )
                combined_df = _aggregate_weather_data_for_intervals(
                    combined_df, target_timestamps_parsed
                )

        if combined_df is None or combined_df.empty:
            logger.warning("No weather data available")
            return None

        # Use helper function to create MetadataDataFrame objects
        metadata_dataframes = _create_weatherstack_metadata_dataframes(
            combined_df,
            metadata_info,
            timezone_offset,
            target_timestamps_parsed,
            target_timezone,
            target_timestamps,  # Pass original timestamp strings
        )

        logger.info(
            f"Successfully created {len(metadata_dataframes)} MetadataDataFrame objects"
        )
        return metadata_dataframes

    except Exception as e:
        logger.error(f"Error processing WeatherStack metadata: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


async def _remove_failed_haver_dataset_from_s3(
    bucket_name: str,
    database_name: str,
    name: str,
    metadata_info: HaverMetadataAccessInfo,
) -> None:
    """Remove a failed Haver dataset from both flat json file and hierarchical path in S3.

    Args:
        bucket_name: Name of the S3 bucket
        database_name: Name of the Haver database
        name: Name of the dataset
        metadata_info: HaverMetadataAccessInfo object containing dataset metadata
    """
    s3_client = boto3.client("s3")
    flat_json_path = (
        f"univariate/haver/synthefy-flat/access_{database_name}.json"
    )

    try:
        data = await asyncio.to_thread(
            lambda: json.load(
                smart_open(
                    os.path.join("s3://", bucket_name, flat_json_path), "r"
                )
            )
        )

        target_key = (name, database_name)
        filtered_data = [
            item
            for item in data
            if (item["name"], item["database_name"]) != target_key
        ]

        await asyncio.to_thread(
            lambda: json.dump(
                filtered_data,
                smart_open(
                    os.path.join("s3://", bucket_name, flat_json_path), "w"
                ),
            )
        )

        logger.info(
            f"Removed failed dataset {name}@{database_name} from flat json file"
        )
    except Exception as e:
        logger.error(f"Error removing dataset from flat json file: {e}")

    try:
        description = metadata_info.description
        parts = description.split(":")
        clean_parts = [
            part.replace("/", "_").replace("\\", "_") for part in parts
        ]
        hierarchy_path = "/".join(
            ["univariate/haver", database_name, *clean_parts]
        )
        key = f"{hierarchy_path}_access.json"

        await asyncio.to_thread(delete_s3_objects, s3_client, bucket_name, key)
        logger.info(
            f"Removed failed dataset {name}@{database_name} from hierarchical path"
        )
    except Exception as e:
        logger.error(f"Error removing dataset from hierarchical path: {e}")


async def process_haver_metadata_access_info(
    metadata_info: HaverMetadataAccessInfo,
    bucket_name: str,
) -> Optional[List[MetadataDataFrame]]:
    """Process Haver metadata access info and return list of MetadataDataFrames.

    Args:
        metadata_info: HaverMetadataAccessInfo object
        bucket_name: Name of the S3 bucket that stores the metadata datasets

    Returns:
        List containing a single MetadataDataFrame for Haver data or None if error/not available
    """
    try:
        if metadata_info.file_name is not None:
            if metadata_info.db_path_info is None:
                raise ValueError(f"db_path_info is None for {metadata_info}")

            with smart_open(
                os.path.join("s3://", bucket_name, metadata_info.db_path_info),
                "r",
            ) as f:
                raw_data = json.load(f)
                haver_data = {
                    "data_source": "haver",
                    "description": raw_data.get("description", ""),
                    "database_name": raw_data.get("databaseName"),
                    "name": raw_data.get("name"),
                    "start_date": raw_data.get("startDate", 0),
                    "db_path_info": metadata_info.db_path_info,
                    "file_name": metadata_info.file_name,
                }
                loaded_metadata = HaverMetadataAccessInfo(**haver_data)
            database_name = loaded_metadata.database_name
            name = loaded_metadata.name
            description = loaded_metadata.description
            metadata_info = loaded_metadata

        elif (
            metadata_info.database_name is not None
            and metadata_info.name is not None
        ):
            database_name = metadata_info.database_name
            name = metadata_info.name
            description = metadata_info.description
        else:
            raise ValueError(
                f"Invalid metadata info configuration: {metadata_info}"
            )

        if database_name is None or name is None:
            raise ValueError(
                f"Missing required database_name or name after loading metadata: {metadata_info}"
            )

        df = await asyncio.to_thread(
            haver.read_df, haver_codes=[f"{name}@{database_name}"]
        )
        df = df[df.columns[df.columns.isin(["date", "value"])]]
        df = df.dropna()

        if df.empty:
            logger.error(f"df is empty for {name}@{database_name}")
            await _remove_failed_haver_dataset_from_s3(
                bucket_name, database_name, name, metadata_info
            )
            return None

    except KeyError as e:
        if str(e) == "'dataPoints'" or "KeyError" in str(type(e)):
            logger.error(
                f"Invalid Haver code {name}@{database_name}: The series does not exist or is not accessible"
            )
            assert database_name is not None
            assert name is not None
            await _remove_failed_haver_dataset_from_s3(
                bucket_name, database_name, name, metadata_info
            )
        else:
            logger.error(
                f"Unexpected KeyError reading Haver dataframe for {name}@{database_name}: {e}"
            )
        return None

    except Exception as e:
        logger.error(f"Error reading Haver dataframe: {e}")
        return None

    metadata_json = {
        "data_source": "haver",
        "timestamp_columns": ["date"],
        "columns": [
            {
                "id": "value",
                "title": description,
                "column_id": "value",
            }
        ],
    }

    ret = MetadataDataFrame(
        df=df,
        metadata_json=metadata_json,
        timestamp_key="date",
        description=description,
    )

    return [ret]


async def prepare_metadata_dataframes(
    metadata_info_combined: Optional[
        List[
            HaverMetadataAccessInfo
            | WeatherMetadataAccessInfo
            | PredictHQMetadataAccessInfo
        ]
    ],
    metadata_datasets_bucket: str,
    target_timestamps: List[str],
    leak_idxs: Optional[List[int]] = None,
) -> Tuple[List[MetadataDataFrame], List[int]]:
    """Prepare metadata dataframes if specified.

    Since not all metadata dataframes will be available, we need to update the leak_idxs
    to reflect the new indices of the metadata dataframes.

    Args:
        metadata_info_combined: List of metadata info (haver, weatherstack, or predicthq)
        metadata_datasets_bucket: Name of the S3 bucket
        target_timestamps: List of exact timestamps in ISO format to fetch data for.
        leak_idxs: List of indices to leak from the metadata dataframes

    Returns:
        Tuple containing:
            - List of metadata dataframes
            - List of indices of the metadata dataframes that are new leaks
    """
    metadata_dataframes: List[MetadataDataFrame] = []
    num_metadata_failed_to_load = 0
    new_leak_idxs: List[int] = []
    leak_idxs_set = set(leak_idxs) if leak_idxs else set()

    if metadata_info_combined:
        logger.debug(
            f"Metadata dataframe keys specified: {metadata_info_combined}"
        )
        for idx, metadata_info in enumerate(metadata_info_combined):
            try:
                if isinstance(metadata_info, HaverMetadataAccessInfo):
                    result = await process_haver_metadata_access_info(
                        metadata_info, metadata_datasets_bucket
                    )
                    if result is None:
                        logger.error(
                            f"Error processing Haver metadata info: {metadata_info}"
                        )
                        num_metadata_failed_to_load += 1
                        continue

                    prev_len = len(metadata_dataframes)
                    metadata_dataframes.extend(result)
                    result_len = len(result)

                    if leak_idxs and idx in leak_idxs_set:
                        new_leak_idxs.extend(
                            range(prev_len, prev_len + result_len)
                        )

                elif isinstance(metadata_info, WeatherMetadataAccessInfo):
                    result = await process_weatherstack_metadata_access_info(
                        metadata_info, target_timestamps
                    )
                    if result is None:
                        logger.error(
                            f"Error processing WeatherStack metadata info: {metadata_info}"
                        )
                        num_metadata_failed_to_load += 1
                        continue

                    prev_len = len(metadata_dataframes)
                    metadata_dataframes.extend(result)
                    result_len = len(result)

                    if leak_idxs and idx in leak_idxs_set:
                        new_leak_idxs.extend(
                            range(prev_len, prev_len + result_len)
                        )

                elif isinstance(metadata_info, PredictHQMetadataAccessInfo):
                    result = await process_predicthq_metadata_access_info(
                        metadata_info
                    )
                    if result is None:
                        logger.error(
                            f"Error processing PredictHQ metadata info: {metadata_info}"
                        )
                        num_metadata_failed_to_load += 1
                        continue

                    prev_len = len(metadata_dataframes)
                    metadata_dataframes.extend(result)
                    result_len = len(result)

                    if leak_idxs and idx in leak_idxs_set:
                        new_leak_idxs.extend(
                            range(prev_len, prev_len + result_len)
                        )

                else:
                    logger.error(
                        f"Invalid metadata info type: {type(metadata_info)}"
                    )
                    num_metadata_failed_to_load += 1
                    continue

            except Exception as e:
                logger.error(f"Error processing metadata info: {str(e)}")
                num_metadata_failed_to_load += 1
                continue

    else:
        logger.debug("No metadata dataframe keys specified")

    if num_metadata_failed_to_load > 0:
        logger.warning(
            f"Failed to load {num_metadata_failed_to_load} metadata dataframes"
        )

    return metadata_dataframes, new_leak_idxs


class LocationCacheManager:
    """Manages the location cache with expiration and automatic cleanup."""

    def __init__(self, ttl_minutes: int = 2880) -> None:
        """Initialize the location cache manager.

        Args:
            ttl_minutes: Time-to-live in minutes for cached items (default: 48 hours)
        """
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._last_access: Optional[datetime] = None
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task[None]] = None

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get value from cache and update last access time.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if exists, None otherwise
        """
        with self._lock:
            if key in self._cache:
                self._last_access = datetime.now()
                return self._cache[key]
        return None

    def set(self, key: str, value: List[Dict[str, Any]]) -> None:
        """Set value in cache and update last access time.

        Args:
            key: Cache key to set
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = value
            self._last_access = datetime.now()

    def is_expired(self) -> bool:
        """Check if cache has expired.

        Returns:
            True if cache has expired, False otherwise
        """
        if self._last_access is None:
            return False
        return datetime.now() - self._last_access > self._ttl

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._last_access = None

    async def start_cleanup_task(self) -> None:
        """Start the background task to clean up expired cache."""

        async def cleanup() -> None:
            while True:
                await asyncio.sleep(60)
                if self.is_expired():
                    logger.info("Cache expired, clearing locations from memory")
                    self.clear()

        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(cleanup())


async def load_locations_into_cache(
    s3_client,
    bucket_name: str,
    file_key: str,
    cache_manager: LocationCacheManager,
) -> None:
    """Load locations into memory cache for faster searching.

    Downloads and parses a JSON file containing location data from S3, then stores
    the parsed locations in a memory cache for efficient searching. Uses streaming
    JSON parsing to handle large files without loading everything into memory at once.

    The function implements early return if locations are already cached, making
    subsequent calls very fast. Each location record contains geographic information
    including name, country code, administrative codes, coordinates, and population.

    Args:
        s3_client: Boto3 S3 client instance for downloading the file from S3
        bucket_name: Name of the S3 bucket containing the locations file
        file_key: S3 object key path to the locations JSON file
        cache_manager: LocationCacheManager instance to store the cached data

    Returns:
        None: Function modifies the cache state as a side effect

    Raises:
        Exception: If there are issues downloading from S3, parsing JSON data,
                  or storing data in the cache. Specific errors are logged.

    Note:
        - Uses ijson for streaming JSON parsing to handle large files efficiently
        - Automatically starts a cache cleanup task after loading
        - Handles malformed or missing data gracefully with None values
        - Location records are expected to have: name, country_code, admin1_code,
          latitude, longitude, and population fields
    """
    logger.info(f"Loading locations into cache from {bucket_name}/{file_key}")
    if cache_manager.get("locations"):
        return

    try:
        response = await s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = await response["Body"].read()

        # Parse the entire JSON file once
        locations = []
        parser = ijson.parse(io.BytesIO(content))
        current_item = {}

        for prefix, event, value in parser:
            if prefix.endswith(".name"):
                if event == "null":
                    current_item = {}
                    continue
                current_item["name"] = value
            elif prefix.endswith(".country_code"):
                current_item["country_code"] = (
                    value if event != "null" else None
                )
            elif prefix.endswith(".admin1_code"):
                current_item["admin1_code"] = value if event != "null" else None
            elif prefix.endswith(".latitude"):
                try:
                    current_item["latitude"] = (
                        float(value) if event != "null" else None
                    )
                except (ValueError, TypeError):
                    current_item["latitude"] = None
            elif prefix.endswith(".longitude"):
                try:
                    current_item["longitude"] = (
                        float(value) if event != "null" else None
                    )
                except (ValueError, TypeError):
                    current_item["longitude"] = None
            elif prefix.endswith(".population"):
                try:
                    current_item["population"] = (
                        int(value) if event != "null" else None
                    )
                except (ValueError, TypeError):
                    current_item["population"] = None
                if current_item.get("name"):
                    locations.append(current_item.copy())
                current_item = {}

        # Store in cache and start cleanup task
        cache_manager.set("locations", locations)
        await cache_manager.start_cleanup_task()
        logger.info(f"Loaded {len(locations)} locations into cache")

    except Exception as e:
        logger.error(f"Error loading locations into cache: {str(e)}")
        raise


async def search_cached_locations(
    search_term: str, cache_manager: LocationCacheManager, limit: int = 100
) -> AsyncGenerator[bytes, None]:
    """
    Search through cached locations and yield matching results as NDJSON.

    Args:
        search_term: The search term to look for in location names
        cache_manager: LocationCacheManager instance containing cached locations
        limit: Maximum number of results to return (default: 100)

    Yields:
        bytes: JSON-encoded WeatherStackLocation objects as NDJSON lines

    Raises:
        HTTPException: If there's an error during the search process
    """
    try:
        search_term_lower = search_term.lower()
        count = 0

        # Search through cached locations
        for location in cache_manager.get("locations") or []:
            if location["name"].lower().find(search_term_lower) != -1:
                yield (
                    json.dumps(
                        WeatherStackLocation(**location).model_dump()
                    ).encode()
                    + b"\n"
                )
                count += 1
                if count >= limit:
                    break

    except Exception as e:
        logger.error(f"Error in location search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching locations: {str(e)}",
        )


def call_predicthq_features_api(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call PredictHQ Features API with given parameters.

    Args:
        params: API parameters dictionary for Features API

    Returns:
        JSON response from PredictHQ Features API
    """
    features_url = "https://api.predicthq.com/v1/features/"

    # Add API key to headers
    headers = {
        "Authorization": f"Bearer {PREDICTHQ_API_KEY}",
        "Accept": "application/json",
    }

    logger.info(
        f"PredictHQ Features API call to {features_url} with params: {params}"
    )

    try:
        response = requests.post(features_url, json=params, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"PredictHQ Features API error: HTTP {response.status_code}"
            )
            logger.error(f"Response: {response.text}")

            # Parse error response for more specific error messages
            try:
                error_data = response.json()
                if "errors" in error_data:
                    error_msg = error_data["errors"].get(
                        "error", "Unknown error"
                    )
                    if "subscription settings" in error_msg.lower():
                        logger.error(
                            "Features API subscription limitation detected - date range or features not available"
                        )
                    elif "invalid additional feature" in error_msg.lower():
                        logger.error(
                            f"PredictHQ Features API: Some requested features are not available with current subscription. "
                            f"Error details: {error_msg}. "
                            f"Please check your PredictHQ subscription plan and update PREDICTHQ_SUPPORTED_*_FEATURES constants."
                        )
                    elif "details" in error_data["errors"]:
                        details = error_data["errors"]["details"]
                        for detail in details:
                            field = detail.get("field", "unknown")
                            msg = detail.get("msg", "")
                            logger.error(f"Field '{field}': {msg}")

                            # Extract specific unsupported features for better logging
                            if "invalid additional feature" in msg.lower():
                                import re

                                feature_match = re.search(
                                    r"Invalid - \[(.*?)\]", msg
                                )
                                if feature_match:
                                    unsupported_features = feature_match.group(
                                        1
                                    )
                                    logger.error(
                                        f"Unsupported features detected: {unsupported_features}. "
                                        f"These features may not be available with your current PredictHQ subscription plan."
                                    )
            except Exception:
                pass  # Ignore JSON parsing errors, already logged raw response

            return {}

        json_response = response.json()
        return json_response

    except Exception as e:
        logger.error(f"Error calling PredictHQ Features API: {e}")
        return {}


def _build_features_api_params(
    metadata_info: PredictHQMetadataAccessInfo,
) -> Dict[str, Any]:
    """Build parameters for PredictHQ Features API call.

    Args:
        metadata_info: PredictHQ metadata access information

    Returns:
        Dictionary containing Features API parameters
    """
    location_config = metadata_info.location_config
    time_period = metadata_info.time_period
    event_categories = metadata_info.event_categories
    aggregation_config = metadata_info.aggregation_config

    # Normalize timestamps to have zero time components (midnight) as required by Features API
    start_date = pd.to_datetime(time_period.min_timestamp).normalize()
    end_date = pd.to_datetime(time_period.forecast_timestamp).normalize()

    # Build base parameters
    params: Dict[str, Any] = {
        "active": {
            "gte": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "lte": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    }

    # Add location parameters
    if (
        location_config.latitude
        and location_config.longitude
        and location_config.radius_km
    ):
        params["location"] = {
            "geo": {
                "lat": str(location_config.latitude),
                "lon": str(location_config.longitude),
                "radius": f"{location_config.radius_km}mi",  # Features API uses miles
            }
        }
    elif location_config.place_id:
        params["location"] = {"place": {"scope": [location_config.place_id]}}
    elif location_config.country_code:
        # Features API doesn't support country_code directly - need to use place or geo
        # For now, we'll raise an error suggesting the use of Events API instead
        raise ValueError(
            "Features API doesn't support country_code directly. Use latitude/longitude/radius or place_id instead, or the system will fall back to Events API."
        )
    else:
        raise ValueError(
            "Location configuration is missing latitude, longitude, and radius or place_id"
        )

    # Add category-specific features based on enabled categories
    active_categories = [
        category
        for category, enabled in event_categories.model_dump().items()
        if enabled and category in PREDICTHQ_CATEGORY_MAPPING
    ]

    # Determine which stats to request based on aggregation method
    if aggregation_config.method == "count":
        stats = ["count"]
    elif aggregation_config.method == "impact_score":
        if aggregation_config.include_attendance:
            stats = ["sum"]  # Sum of attendance
        else:
            # For impact_score without attendance, compute weighted impact using count and average rank
            stats = [
                "count",
                "avg",
            ]  # Count of events and average rank (if available)
    else:  # category_indicators
        stats = ["count"]  # We just need to know if events exist

    # Add features for each active category
    for category in active_categories:
        api_category = PREDICTHQ_CATEGORY_MAPPING[category]

        # Use attendance-based features for most categories
        if api_category in PREDICTHQ_SUPPORTED_ATTENDANCE_FEATURES:
            feature_key = f"phq_attendance_{api_category.replace('-', '_')}"
            feature_params: Dict[str, Any] = {"stats": stats}

            # Add rank filter if specified
            if aggregation_config.min_phq_rank is not None:
                feature_params["phq_rank"] = {
                    "gte": aggregation_config.min_phq_rank
                }

            params[feature_key] = feature_params

        # Use rank-based features for non-attendance categories
        elif api_category in PREDICTHQ_SUPPORTED_RANK_FEATURES:
            feature_key = f"phq_rank_{api_category.replace('-', '_')}"
            params[feature_key] = True  # Just presence/absence for these

            # Note: min_phq_rank filtering is built into these features

        # Handle academic events with specific subtypes
        elif api_category == "academic":
            # Academic events have specific subtypes instead of a generic feature
            for academic_feature in PREDICTHQ_ACADEMIC_RANK_FEATURES:
                params[academic_feature] = True

        else:
            # Log unsupported categories
            logger.warning(
                f"PredictHQ category '{api_category}' is not supported by the current subscription. Skipping this category."
            )

    return params


def _get_predicthq_rank_impact_from_levels(
    rank_levels: Dict[str, Any],
) -> float:
    """Calculate impact score from PredictHQ rank levels data.

    Args:
        rank_levels: Either a dict with rank level keys and count values,
                    or a list of dictionaries with min_rank and count

    Returns:
        Float representing the total impact score
    """
    total_impact = 0.0

    if isinstance(rank_levels, dict):
        # New format: rank_levels is a dict with rank level as key and count as value
        # e.g., {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        # These levels represent PHQ rank buckets based on PredictHQ rank system (0-100 scale)
        rank_level_weights = {
            "1": 90,  # High impact (approx. midpoint of 81-100)
            "2": 70,  # Medium-high impact (approx. midpoint of 61-80)
            "3": 50,  # Medium impact (approx. midpoint of 41-60)
            "4": 30,  # Medium-low impact (approx. midpoint of 21-40)
            "5": 10,  # Low impact (approx. midpoint of 0-20)
        }
        for rank_level_key, count in rank_levels.items():
            if isinstance(count, (int, float)) and count > 0:
                impact_weight = rank_level_weights.get(
                    rank_level_key, 50
                )  # Default to medium if unknown
                total_impact += count * impact_weight

    elif isinstance(rank_levels, list):
        # Legacy format: list of dictionaries with min_rank and count
        # e.g., [{"min_rank": 80, "count": 2}, {"min_rank": 60, "count": 1}]
        for rank_level in rank_levels:
            if isinstance(rank_level, dict):
                min_rank = rank_level.get("min_rank", 0)
                count = rank_level.get("count", 0)
                if min_rank > 0 and count > 0:
                    # Use min_rank directly as it represents the actual PHQ rank
                    total_impact += count * min_rank

    return total_impact


def _get_predicthq_rank_count_from_levels(rank_levels: Dict[str, Any]) -> int:
    """Calculate total event count from PredictHQ rank levels data.

    Args:
        rank_levels: Either a dict with rank level keys and count values,
                    or a list of dictionaries with min_rank and count

    Returns:
        Integer representing the total event count
    """
    total_count = 0

    if isinstance(rank_levels, dict):
        # New format: rank_levels is a dict with rank level as key and count as value
        # e.g., {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        for rank_level_key, count in rank_levels.items():
            if isinstance(count, (int, float)):
                total_count += int(count)

    elif isinstance(rank_levels, list):
        # Legacy format: list of dictionaries with min_rank and count
        # e.g., [{"min_rank": 80, "count": 2}, {"min_rank": 60, "count": 1}]
        for rank_level in rank_levels:
            if isinstance(rank_level, dict):
                count = rank_level.get("count", 0)
                total_count += count

    return total_count


def _aggregate_time_series_by_granularity(
    df: pd.DataFrame, time_granularity: str, aggregation_method: str = "sum"
) -> pd.DataFrame:
    """Aggregate daily time series data to the specified time granularity.

    Args:
        df: DataFrame with 'date' and 'value' columns containing daily data
        time_granularity: Target granularity ('daily', 'weekly', 'monthly')
        aggregation_method: How to aggregate values ('sum', 'mean', 'max')

    Returns:
        DataFrame aggregated to the specified granularity
    """
    if df.empty:
        return df

    # Convert date column to datetime if it's not already
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Return as-is for daily granularity but with proper date formatting
    if time_granularity == "daily":
        df_reset = df.reset_index()
        df_reset["date"] = df_reset["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return df_reset

    # Map granularity to pandas frequency
    freq_mapping = {
        "weekly": "W",  # Week ending Sunday
        "monthly": "M",  # Month end
    }

    if time_granularity not in freq_mapping:
        raise ValueError(f"Unsupported time granularity: {time_granularity}")

    freq = freq_mapping[time_granularity]

    # Aggregate based on method
    if aggregation_method == "sum":
        aggregated = df.resample(freq).sum()
    elif aggregation_method == "mean":
        aggregated = df.resample(freq).mean()
    elif aggregation_method == "max":
        aggregated = df.resample(freq).max()
    else:
        raise ValueError(
            f"Unsupported aggregation method: {aggregation_method}"
        )

    # Reset index and format date column
    aggregated = aggregated.reset_index()
    aggregated["date"] = aggregated["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return aggregated


def _process_features_api_response_for_aggregation(
    api_response: Dict[str, Any],
    aggregation_config: PredictHQAggregationConfig,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> List[pd.DataFrame]:
    """Process Features API response into time series DataFrames.

    Args:
        api_response: Response from Features API
        aggregation_config: Configuration for aggregation
        start_date: Start date for time series
        end_date: End date for time series

    Returns:
        List of DataFrames containing processed time series
    """
    if not api_response.get("results"):
        logger.warning("No results in Features API response")
        # Return empty DataFrame with the expected date range
        date_range = _create_date_range(
            start_date, end_date, aggregation_config.time_granularity
        )
        empty_df = pd.DataFrame(
            {
                "date": date_range.strftime("%Y-%m-%dT%H:%M:%S"),
                "value": [0] * len(date_range),
            }
        )
        return [empty_df]

    results = api_response["results"]

    if aggregation_config.method == "count":
        return _process_features_count_aggregation(results, aggregation_config)
    elif aggregation_config.method == "impact_score":
        return _process_features_impact_score_aggregation(
            results, aggregation_config
        )
    elif aggregation_config.method == "category_indicators":
        return _process_features_category_indicators_aggregation(
            results, aggregation_config
        )
    else:
        raise ValueError(
            f"Unsupported aggregation method for Features API: {aggregation_config.method}"
        )


def _process_features_count_aggregation(
    results: List[Dict[str, Any]],
    aggregation_config: PredictHQAggregationConfig,
) -> List[pd.DataFrame]:
    """Process Features API results for count aggregation.

    Args:
        results: Results from Features API
        aggregation_config: Configuration for aggregation

    Returns:
        List containing a single DataFrame with event counts
    """
    aggregated_data = []

    for result in results:
        date = result["date"]
        total_count = 0

        # Sum up counts from all attendance-based features
        for key, value in result.items():
            if key.startswith("phq_attendance_") and isinstance(value, dict):
                stats = value.get("stats", {})
                count = stats.get("count", 0)
                total_count += count
            elif (
                key.startswith("phq_rank_")
                or key in PREDICTHQ_ACADEMIC_RANK_FEATURES
            ) and value is not None:
                # For rank-based features (including academic subtypes), handle different response formats
                if isinstance(value, dict):
                    # Handle structured rank response with rank_levels
                    if value.get("rank_levels"):
                        count = _get_predicthq_rank_count_from_levels(
                            value["rank_levels"]
                        )
                        total_count += count
                    # Handle other dict structures - try to find count field
                    elif "count" in value:
                        total_count += value.get("count", 0)
                    else:
                        # For boolean-like responses where presence indicates events
                        total_count += 1
                elif isinstance(value, (int, float)):
                    # Direct count value
                    total_count += int(value)
                elif value is True:
                    # Boolean response - presence indicates at least 1 event
                    total_count += 1

        aggregated_data.append({"date": date, "value": total_count})

    # Create DataFrame and apply time granularity aggregation
    df = pd.DataFrame(aggregated_data)

    # Apply time granularity aggregation (sum counts for weekly/monthly)
    aggregated_df = _aggregate_time_series_by_granularity(
        df, aggregation_config.time_granularity, "sum"
    )

    return [aggregated_df]


def _process_features_impact_score_aggregation(
    results: List[Dict[str, Any]],
    aggregation_config: PredictHQAggregationConfig,
) -> List[pd.DataFrame]:
    """Process Features API results for impact score aggregation.

    Args:
        results: Results from Features API
        aggregation_config: Configuration for aggregation

    Returns:
        List containing a single DataFrame with impact scores
    """
    aggregated_data = []

    for result in results:
        date = result["date"]
        total_impact = 0.0

        if aggregation_config.include_attendance:
            # Sum up attendance from all attendance-based features
            for key, value in result.items():
                if key.startswith("phq_attendance_") and isinstance(
                    value, dict
                ):
                    stats = value.get("stats", {})
                    attendance_sum = stats.get("sum", 0)
                    total_impact += attendance_sum
        else:
            # For impact_score without attendance, compute weighted impact using count and average rank
            for key, value in result.items():
                if key.startswith("phq_attendance_") and isinstance(
                    value, dict
                ):
                    stats = value.get("stats", {})
                    count = stats.get("count", 0)
                    avg_rank = stats.get("avg", 0)  # Average rank of events

                    # Estimate total rank impact as count * average_rank
                    # This approximates the sum of ranks that we would get from Events API
                    if count > 0 and avg_rank > 0:
                        estimated_rank_impact = count * avg_rank
                        total_impact += estimated_rank_impact
                elif key.startswith("phq_rank_") and value is not None:
                    # For rank-based features, use the rank information if available
                    if isinstance(value, dict) and value.get("rank_levels"):
                        impact = _get_predicthq_rank_impact_from_levels(
                            value["rank_levels"]
                        )
                        total_impact += impact
                    elif isinstance(value, dict) and "count" in value:
                        # Handle other dict structures with count
                        count = value.get("count", 0)
                        # Use a default rank value for estimation
                        estimated_impact = (
                            count * 50
                        )  # Assume rank 50 as default
                        total_impact += estimated_impact
                    elif isinstance(value, (int, float)):
                        # Direct value - treat as impact score
                        total_impact += float(value)
                    elif value is True:
                        # Boolean response - assume some default impact
                        total_impact += 50.0  # Default impact value
                elif (
                    key in PREDICTHQ_ACADEMIC_RANK_FEATURES
                    and value is not None
                ):
                    # Handle academic features specifically
                    if isinstance(value, dict) and value.get("rank_levels"):
                        impact = _get_predicthq_rank_impact_from_levels(
                            value["rank_levels"]
                        )
                        total_impact += impact
                    elif isinstance(value, dict) and "count" in value:
                        count = value.get("count", 0)
                        estimated_impact = count * 50  # Default rank
                        total_impact += estimated_impact
                    elif isinstance(value, (int, float)):
                        total_impact += float(value)
                    elif value is True:
                        total_impact += (
                            50.0  # Default impact for academic events
                        )

        aggregated_data.append({"date": date, "value": float(total_impact)})

    # Create DataFrame and apply time granularity aggregation
    df = pd.DataFrame(aggregated_data)

    # Apply time granularity aggregation (sum impact scores for weekly/monthly)
    aggregated_df = _aggregate_time_series_by_granularity(
        df, aggregation_config.time_granularity, "sum"
    )

    return [aggregated_df]


def _process_features_category_indicators_aggregation(
    results: List[Dict[str, Any]],
    aggregation_config: PredictHQAggregationConfig,
) -> List[pd.DataFrame]:
    """Process Features API results for category indicators aggregation.

    Args:
        results: Results from Features API
        aggregation_config: Configuration for aggregation

    Returns:
        List of DataFrames, one for each category with binary indicators
    """
    # Collect all unique categories from the results
    all_categories = set()
    for result in results:
        for key in result.keys():
            if key.startswith("phq_attendance_") or key.startswith("phq_rank_"):
                # Extract category name from feature key
                if key.startswith("phq_attendance_"):
                    category = key.replace("phq_attendance_", "")
                else:
                    category = key.replace("phq_rank_", "")
                all_categories.add(category)

    dataframes = []

    for category in sorted(all_categories):
        aggregated_data = []

        for result in results:
            date = result["date"]
            has_events = 0

            # Check if this category has events on this date
            attendance_key = f"phq_attendance_{category}"
            rank_key = f"phq_rank_{category}"

            if attendance_key in result and isinstance(
                result[attendance_key], dict
            ):
                stats = result[attendance_key].get("stats", {})
                count = stats.get("count", 0)
                has_events = 1 if count > 0 else 0
            elif rank_key in result and result[rank_key] is not None:
                if isinstance(result[rank_key], dict) and result[rank_key].get(
                    "rank_levels"
                ):
                    # Check if any rank level has events
                    total_count = _get_predicthq_rank_count_from_levels(
                        result[rank_key]["rank_levels"]
                    )
                    has_events = 1 if total_count > 0 else 0
                elif (
                    isinstance(result[rank_key], dict)
                    and "count" in result[rank_key]
                ):
                    # Handle other dict structures with count
                    count = result[rank_key].get("count", 0)
                    has_events = 1 if count > 0 else 0
                elif isinstance(result[rank_key], (int, float)):
                    # Direct count value
                    has_events = 1 if result[rank_key] > 0 else 0
                elif result[rank_key] is True:
                    # Boolean response - presence indicates events
                    has_events = 1

            aggregated_data.append({"date": date, "value": has_events})

        # Create DataFrame and apply time granularity aggregation
        df = pd.DataFrame(aggregated_data)

        # For category indicators, use max aggregation (if any day in the period has events, the period has events)
        aggregated_df = _aggregate_time_series_by_granularity(
            df, aggregation_config.time_granularity, "max"
        )

        dataframes.append(aggregated_df)

    return dataframes


def _create_date_range(
    start_date: pd.Timestamp, end_date: pd.Timestamp, time_granularity: str
) -> pd.DatetimeIndex:
    """Create a date range based on the specified granularity.

    Args:
        start_date: Start date for the range
        end_date: End date for the range
        time_granularity: Granularity ('daily', 'weekly', 'monthly')

    Returns:
        DatetimeIndex with the specified granularity
    """
    # Ensure both dates are timezone-naive to avoid pandas timezone conflicts
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tz is not None:
        end_date = end_date.tz_localize(None)

    if time_granularity == "daily":
        return pd.date_range(start=start_date, end=end_date, freq="D")
    elif time_granularity == "weekly":
        return pd.date_range(start=start_date, end=end_date, freq="W")
    elif time_granularity == "monthly":
        return pd.date_range(start=start_date, end=end_date, freq="M")
    else:
        raise ValueError(f"Unsupported time granularity: {time_granularity}")


def _adjust_predicthq_start_date_for_subscription_limit(
    original_start_date: pd.Timestamp,
) -> pd.Timestamp:
    """Adjust PredictHQ start date to respect subscription historical limits.

    The current PredictHQ subscription only allows querying the last 90 days of historical data.
    This function ensures the start date doesn't exceed this limit and provides informative logging.
    The returned date is normalized to midnight (00:00:00) as required by PredictHQ Features API.

    Args:
        original_start_date: The originally requested start date

    Returns:
        Adjusted start date that respects the subscription limits, normalized to midnight
    """
    current_time = pd.Timestamp.now(
        tz=original_start_date.tz if original_start_date.tz else None
    )
    max_historical_date = current_time - pd.Timedelta(
        days=PREDICTHQ_MAX_HISTORICAL_DAYS
    )

    if original_start_date < max_historical_date:
        # Normalize to midnight as required by PredictHQ Features API
        adjusted_start_date = max_historical_date.normalize()

        days_requested = (current_time - original_start_date).days
        days_available = (current_time - adjusted_start_date).days

        logger.warning(
            f"PredictHQ API subscription limit applied. "
            f"Requested {days_requested} days of history, but subscription allows only {PREDICTHQ_MAX_HISTORICAL_DAYS} days. "
            f"Adjusting start date from {original_start_date.strftime('%Y-%m-%d')} to {adjusted_start_date.strftime('%Y-%m-%d')}. "
            f"Data window: {days_available} days of events will be retrieved"
        )

        return adjusted_start_date
    else:
        days_requested = (current_time - original_start_date).days
        logger.info(
            f"PredictHQ API: Requesting {days_requested} days of history (within {PREDICTHQ_MAX_HISTORICAL_DAYS}-day subscription limit)"
        )

        # Also normalize the original date to ensure consistency with API requirements
        return original_start_date.normalize()


async def process_predicthq_metadata_access_info(
    metadata_info: PredictHQMetadataAccessInfo,
) -> Optional[List[MetadataDataFrame]]:
    """Process PredictHQ metadata access info and return list of MetadataDataFrames.

    Args:
        metadata_info: PredictHQ metadata access information

    Returns:
        List of MetadataDataFrame objects or None if processing fails
    """
    try:
        location_config = metadata_info.location_config
        time_period = metadata_info.time_period
        aggregation_config = metadata_info.aggregation_config

        logger.info(f"Processing PredictHQ request for {metadata_info.name}")
        logger.info(
            f"Time period: {time_period.min_timestamp} to {time_period.forecast_timestamp}"
        )
        logger.info(f"Location: {location_config.location_name}")
        logger.info(
            f"Aggregation: {aggregation_config.method} ({aggregation_config.time_granularity})"
        )

        original_start_date = pd.to_datetime(time_period.min_timestamp)
        end_date = pd.to_datetime(time_period.forecast_timestamp)

        # Apply PredictHQ subscription historical limits
        start_date = _adjust_predicthq_start_date_for_subscription_limit(
            original_start_date
        )

        # Check if the adjusted date range is valid
        if start_date > end_date:
            logger.error(
                f"PredictHQ data unavailable: After applying subscription limits, "
                f"the adjusted start date ({start_date.strftime('%Y-%m-%d')}) is after the end date "
                f"({end_date.strftime('%Y-%m-%d')}). "
                f"This request requires historical data beyond the {PREDICTHQ_MAX_HISTORICAL_DAYS}-day subscription limit."
            )
            return None

        logger.info(
            f"Using PredictHQ Features API for {aggregation_config.method} method"
        )

        # Build Features API parameters
        features_params = _build_features_api_params(metadata_info)

        # Apply the adjusted start date to Features API params
        if features_params and "active" in features_params:
            features_params["active"]["gte"] = start_date.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

        if not features_params:
            logger.error("Failed to build Features API parameters")
            return None

        # Call Features API
        features_response = await asyncio.to_thread(
            call_predicthq_features_api, features_params
        )

        if not features_response:
            logger.error(
                "PredictHQ Features API failed. This might be due to subscription limitations or API issues."
            )
            return None

        logger.info("Successfully retrieved data from PredictHQ Features API")

        # Process Features API response into time series
        time_series_dfs = await asyncio.to_thread(
            _process_features_api_response_for_aggregation,
            features_response,
            aggregation_config,
            start_date,
            end_date,
        )

        if not time_series_dfs:
            logger.error("Failed to convert data to time series")
            return None

        logger.info(f"Generated {len(time_series_dfs)} time series DataFrames")

        # Create MetadataDataFrame objects
        metadata_dataframes = []

        for i, ts_df in enumerate(time_series_dfs):
            if aggregation_config.method == "category_indicators":
                # For category indicators, we need to determine which category this DataFrame represents
                # This is a simplified approach - in practice you might want to track this more carefully
                description = f"PredictHQ Events Category Indicator {i} - {metadata_info.name}"
                column_title = f"Events Category {i} Indicator"
            else:
                description = f"PredictHQ Events {aggregation_config.method.title()} - {metadata_info.name}"
                column_title = f"Events {aggregation_config.method.title()}"

            metadata_json = {
                "data_source": "predicthq",
                "timestamp_columns": ["date"],
                "columns": [
                    {
                        "id": "value",
                        "title": column_title,
                        "column_id": "value",
                        "description": description,
                    }
                ],
            }

            metadata_dataframes.append(
                MetadataDataFrame(
                    df=ts_df,
                    metadata_json=metadata_json,
                    description=description,
                    timestamp_key="date",
                )
            )

        logger.info(
            f"Successfully created {len(metadata_dataframes)} MetadataDataFrame objects"
        )
        return metadata_dataframes

    except Exception as e:
        logger.error(f"Error processing PredictHQ metadata: {str(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


def _detect_target_timezone(
    target_timestamps: List[str],
) -> Optional[str]:
    """Detect timezone from target_timestamps if they are timezone-aware.
    Args:
        target_timestamps: List of target timestamps to detect timezone from
    Returns:
        Timezone string if detected, None if timestamps are timezone-naive
    """
    # Parse first timestamp to check if it's timezone-aware
    try:
        first_ts = pd.to_datetime(target_timestamps[0])
        if first_ts.tz is not None:
            return str(first_ts.tz)
    except Exception as e:
        logger.debug(f"Could not parse timestamp for timezone detection: {e}")

    return None


def _convert_timestamps_to_location_timezone(
    timestamps: List[pd.Timestamp], location_timezone_offset: float
) -> List[pd.Timestamp]:
    """Convert timestamps to location timezone and remove timezone information.
    Args:
        timestamps: List of timestamps to convert
        location_timezone_offset: Timezone offset for the location (in hours)
    Returns:
        List of timestamps converted to target timezone
    Raises:
        ValueError: If target_timezone is invalid or conversion fails
    """
    converted_timestamps = []

    try:
        target_tz = pytz.timezone(
            f"Etc/GMT{-int(location_timezone_offset):+d}"  # GMT offset is inverted
        )

        for ts in timestamps:
            # Convert to target timezone
            converted_ts = ts.astimezone(target_tz)
            converted_ts = converted_ts.tz_localize(None)
            converted_timestamps.append(converted_ts)
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(f"Invalid target timezone: {target_tz}")
    except Exception as e:
        raise ValueError(f"Error converting timestamps: {e}")

    return converted_timestamps


def _convert_dataframe_timestamps_to_target_timezone(
    df: pd.DataFrame,
    location_timezone_offset: float,
    target_timezone: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Convert DataFrame timestamps from location timezone to target timezone.
    Args:
        df: DataFrame with timestamp column
        location_timezone_offset: Timezone offset for the location (in hours)
        target_timezone: Target timezone string
        timestamp_col: Name of the timestamp column
    Returns:
        DataFrame with timestamps converted to target timezone
    """
    if df.empty or timestamp_col not in df.columns:
        return df

    df = df.copy()

    # Create timezone object from offset
    location_tz = (
        f"Etc/GMT{-int(location_timezone_offset):+d}"  # GMT offset is inverted
    )

    try:
        # Localize timestamps to location timezone, then convert to target timezone
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(location_tz)
        df[timestamp_col] = df[timestamp_col].dt.tz_convert(target_timezone)

        # Keep timezone-aware for final output
        logger.info(
            f"Converted timestamps from location timezone (offset {location_timezone_offset}) to target timezone {target_timezone}"
        )

    except Exception as e:
        logger.warning(
            f"Failed to convert timestamps to target timezone: {e}. Using location timezone."
        )
        # Fallback: just localize to location timezone
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(location_tz)
        except Exception as e2:
            logger.error(f"Failed to localize to location timezone: {e2}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    return df


def _determine_start_end_times(
    target_timestamps: List[str],
    location_timezone_offset: Optional[float] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp, List[pd.Timestamp], Optional[str]]:
    """Determine start and end times based on target_timestamps.

    Args:
        target_timestamps: List of target timestamps in ISO format from the user's uploaded data
        location_timezone_offset: Optional timezone offset for the location (in hours)

    Returns:
        Tuple of (start_time, end_time, target_timestamps_parsed, target_timezone) as pd.Timestamp objects
    """
    target_timezone = None
    if len(target_timestamps) > 0:
        target_timezone = _detect_target_timezone(target_timestamps)
        # Parse target timestamps
        target_timestamps_parsed = [
            pd.to_datetime(ts) for ts in target_timestamps
        ]

        if target_timezone is not None and location_timezone_offset is not None:
            # Convert timezone-aware timestamps to location timezone for processing
            logger.info(
                f"Target timestamps are timezone-aware ({target_timezone}). Converting to location timezone for processing."
            )
            target_timestamps_converted = (
                _convert_timestamps_to_location_timezone(
                    target_timestamps_parsed, location_timezone_offset
                )
            )
            target_timestamps_parsed = target_timestamps_converted

        start_time = min(target_timestamps_parsed)
        end_time = max(target_timestamps_parsed)
        logger.info(
            f"Using target_timestamps range: {start_time} to {end_time}"
        )

    else:
        raise ValueError("target_timestamps must be provided")

    return start_time, end_time, target_timestamps_parsed, target_timezone
