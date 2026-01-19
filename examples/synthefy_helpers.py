import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------- Setup API ---------------------------------------------------
def setup_synthefy_client(
    api_key: str, base_url: str = "https://prod.synthefy.com"
):
    """Setup Synthefy API client with proper timeout configuration"""
    timeout_config = httpx.Timeout(
        connect=30.0,  # 30 seconds to establish connection
        read=300.0,  # 5 minutes to read response
        write=30.0,  # 30 seconds to send request
        pool=30.0,  # 30 seconds to get connection from pool
    )
    return httpx.Client(base_url=base_url, timeout=timeout_config)


# API endpoints
FORECAST_ENDPOINT = "api/foundation_models/forecast/stream"
ENRICH_ENDPOINT = "api/foundation_models/enrich_with_metadata"
LOCATIONS_ENDPOINT = "api/locations_search/"
METADATA_RECOMMENDATIONS_ENDPOINT = (
    "api/foundation_models/metadata_recommendations"
)

# Common weather parameter presets for different use cases
COMMON_WEATHER_PARAMETERS = {
    "basic": ["temperature", "humidity", "pressure"],
    "comprehensive": [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "precip",
        "visibility",
    ],
    "weather_events": [
        "chanceofrain",
        "chanceofsnow",
        "chanceofthunder",
        "chanceoffog",
    ],
    "comfort": [
        "temperature",
        "humidity",
        "feelslike",
        "heatindex",
        "windchill",
    ],
    "all": [
        "temperature",
        "uv_index",
        "wind_speed",
        "wind_degree",
        "windchill",
        "windgust",
        "precip",
        "humidity",
        "visibility",
        "pressure",
        "cloudcover",
        "heatindex",
        "dewpoint",
        "feelslike",
        "chanceofrain",
        "chanceofremdry",
        "chanceofwindy",
        "chanceofovercast",
        "chanceofsunshine",
        "chanceoffrost",
        "chanceofhightemp",
        "chanceoffog",
        "chanceofsnow",
        "chanceofthunder",
    ],
}

# Valid frequency options for weather data
WEATHER_FREQUENCY_OPTIONS = {
    "minute": "T",
    "hour": "H",
    "day": "D",
    "week": "W",
    "month": "M",
    "year": "Y",
}

# Weather units options
WEATHER_UNITS = {
    "m": "metric (°C, km/h, mm)",
    "s": "scientific (°C, m/s, mm)",
    "f": "fahrenheit (°F, mph, inches)",
}


def show_progress(operation_name: str, start_time: Optional[float] = None):
    """Show progress for long-running operations"""
    if start_time is None:
        start_time = time.time()
        print(f"🚀 Starting {operation_name}...")
        return start_time
    else:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            print(
                f"⏳ {operation_name} in progress... ({minutes}m {seconds}s elapsed)"
            )
        else:
            print(f"⏳ {operation_name} in progress... ({seconds}s elapsed)")


# --------------------------------------------------- Data Models ---------------------------------------------------


class WeatherLocation(BaseModel):
    name: str
    latitude: float  # Required - get from location search
    longitude: float  # Required - get from location search


class WeatherConfig(BaseModel):
    """Weather configuration"""

    name: str
    location: WeatherLocation
    weather_parameters: List[str] = [
        "temperature",
        "humidity",
        "precip",
        "wind_speed",
    ]  # Simple list of parameters
    frequency: str = "day"  # day, week, month
    units: str = "m"  # m=metric, s=scientific, f=fahrenheit
    start_time: str  # Simplified from min_timestamp
    end_time: str  # Simplified from forecast_timestamp


class HaverConfig(BaseModel):
    """Haver configuration"""

    database_name: str = Field(
        ..., description="Name of the Haver database (e.g., 'USECON')"
    )
    name: str = Field(..., description="Name of the Haver series (e.g., 'GDP')")
    description: Optional[str] = Field(
        default=None, description="Optional description of the series"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Optional start time to filter data (ISO format)",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Optional end time to filter data (ISO format)",
    )


class MetadataEnrichmentRequest(BaseModel):
    weather_metadata_info: Optional[WeatherConfig] = None
    haver_metadata_info: Optional[HaverConfig] = None
    user_dataframe_json: Optional[Any] = (
        None  # List of dicts from df.to_dict(orient='records')
    )
    user_timestamp_column: Optional[str] = None


class LocationSearchRequest(BaseModel):
    search: str


class MetadataRecommendationsRequest(BaseModel):
    user_prompt: str
    number_of_datasets_to_return: int = 3


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


# --------------------------------------------------- Helper Functions ---------------------------------------------------


def make_api_call(
    client: httpx.Client, endpoint: str, request_data: Any, api_key: str
):
    """Make the API call to specified endpoint"""
    response = client.post(
        endpoint,
        json=request_data.model_dump()
        if hasattr(request_data, "model_dump")
        else request_data,
        headers={"X-API-Key": api_key},
    )
    return response


def search_locations_detailed(
    client: httpx.Client, api_key: str, city_name: str
) -> List[Dict[str, Any]]:
    """Search for locations and return detailed raw data for interactive selection"""
    request = LocationSearchRequest(search=city_name)
    response = client.post(
        LOCATIONS_ENDPOINT,
        json=request.model_dump(),
        headers={"X-API-Key": api_key},
    )

    if response.status_code == 200:
        # Parse NDJSON response and keep raw data
        locations = []
        for line in response.text.strip().split("\n"):
            if line:
                location_data = json.loads(line)
                locations.append(location_data)
        return locations
    else:
        print(
            f"Error searching locations: {response.status_code} - {response.text}"
        )
        return []


def select_location_interactively(
    client: httpx.Client, api_key: str, location_name: str
) -> Optional[WeatherLocation]:
    """
    Interactive location selector that shows multiple options when available.

    Args:
        client: httpx client instance
        api_key: API key for authentication
        location_name: City name to search for

    Returns:
        Selected WeatherLocation or None if cancelled/not found
    """
    try:
        print(f"🔍 Searching for locations matching: '{location_name}'")
        raw_locations = search_locations_detailed(
            client, api_key, location_name
        )

        if not raw_locations:
            logger.error(f"❌ No locations found for '{location_name}'")
            print(
                "💡 Try a more specific search (e.g., 'New York, NY' or 'London, UK')"
            )
            return None

        if len(raw_locations) == 1:
            # Only one match - use it directly
            loc_data = raw_locations[0]
            location = WeatherLocation(
                name=loc_data["name"],
                latitude=loc_data["latitude"],
                longitude=loc_data["longitude"],
            )
            print(
                f"✅ Found unique location: {location.name} ({location.latitude}, {location.longitude})"
            )
            return location

        # Multiple matches - show interactive selection
        print(
            f"🎯 Found {len(raw_locations)} locations matching '{location_name}':"
        )
        print("=" * 80)

        for i, loc_data in enumerate(raw_locations, 1):
            # Create a more descriptive location string using raw data
            location_parts = []
            if loc_data.get("name"):
                location_parts.append(loc_data["name"])
            if loc_data.get("country_code"):
                location_parts.append(loc_data["country_code"])
            if loc_data.get("admin1_code"):
                location_parts.append(f"({loc_data['admin1_code']})")

            location_desc = " ".join(location_parts)
            if loc_data.get("population"):
                location_desc += f" - Pop: {loc_data['population']:,}"

            print(f"   {i}. {location_desc}")
            print(
                f"      📍 Coordinates: ({loc_data['latitude']}, {loc_data['longitude']})"
            )

        print("=" * 80)

        # Interactive selection
        try:
            while True:
                choice = input(
                    f"\n🤔 Please select a location (1-{len(raw_locations)}) or 'q' to quit: "
                ).strip()

                if choice.lower() in ["q", "quit", "cancel"]:
                    print("❌ Location selection cancelled")
                    return None

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(raw_locations):
                        selected_data = raw_locations[choice_idx]
                        selected = WeatherLocation(
                            name=selected_data["name"],
                            latitude=selected_data["latitude"],
                            longitude=selected_data["longitude"],
                        )
                        print(
                            f"✅ Selected: {selected.name} ({selected.latitude}, {selected.longitude})"
                        )
                        return selected
                    else:
                        logger.warning(
                            f"❌ Invalid choice. Please enter a number between 1 and {len(raw_locations)}"
                        )
                except ValueError:
                    logger.warning(
                        f"❌ Invalid input. Please enter a number between 1 and {len(raw_locations)}"
                    )

        except (EOFError, KeyboardInterrupt):
            # Handle non-interactive environments or user cancellation
            logger.warning("⚠️ Interactive selection not available or cancelled")
            first_loc = raw_locations[0]
            fallback_location = WeatherLocation(
                name=first_loc["name"],
                latitude=first_loc["latitude"],
                longitude=first_loc["longitude"],
            )
            print(
                f"🔄 Using first location as fallback: {fallback_location.name}"
            )
            return fallback_location

    except Exception as e:
        logger.error(f"❌ Error in location selection: {str(e)}")
        return None


def search_locations(
    client: httpx.Client, api_key: str, city_name: str
) -> List[WeatherLocation]:
    """Search for locations by city name"""
    request = LocationSearchRequest(search=city_name)
    response = client.post(
        LOCATIONS_ENDPOINT,
        json=request.model_dump(),
        headers={"X-API-Key": api_key},
    )

    if response.status_code == 200:
        # Parse NDJSON response
        locations = []
        for line in response.text.strip().split("\n"):
            if line:
                location_data = json.loads(line)
                simple_location = WeatherLocation(
                    name=location_data["name"],
                    latitude=location_data["latitude"],
                    longitude=location_data["longitude"],
                )
                locations.append(simple_location)
        return locations
    else:
        print(
            f"Error searching locations: {response.status_code} - {response.text}"
        )
        return []


def convert_datetime_columns_to_iso_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame datetime columns to ISO format strings for JSON serialization."""
    df_for_api = df.copy()
    for col in df_for_api.columns:
        if pd.api.types.is_datetime64_any_dtype(df_for_api[col]):
            df_for_api[col] = df_for_api[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df_for_api


def get_metadata_recommendations(
    client: httpx.Client,
    api_key: str,
    user_prompt: str,
    number_of_datasets: int = 3,
) -> List[Dict[str, Any]]:
    """Get metadata recommendations based on user prompt"""
    params = {
        "user_prompt": user_prompt,
        "number_of_datasets_to_return": number_of_datasets,
    }
    response = client.get(
        METADATA_RECOMMENDATIONS_ENDPOINT,
        params=params,
        headers={"X-API-Key": api_key},
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Error getting metadata recommendations: {response.status_code} - {response.text}"
        )
        return []


def get_weather_data(
    client: httpx.Client,
    api_key: str,
    location_name: str,
    weather_parameters: Union[str, List[str]] = "basic",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    frequency: str = "day",
    units: str = "m",
    user_dataframe: Optional[pd.DataFrame] = None,
    user_timestamp_column: Optional[str] = None,
    auto_select_location: bool = False,
) -> pd.DataFrame:
    """
    Simple helper to get weather data with minimal setup and interactive location selection.

    Args:
        client: httpx client instance
        api_key: API key for authentication
        location_name: City name to search for (e.g., "Springfield", "New York")
        weather_parameters: Either a preset name ("basic", "comprehensive", "weather_events",
                           "comfort", "all") or a custom list of parameter names
        start_time: Optional start time in ISO format (e.g., "2012-10-01T00:00:00")
        end_time: Optional end time in ISO format (e.g., "2012-10-26T23:59:59")
        frequency: Time frequency ("day", "week", "month")
        units: Units ("m" for metric, "s" for scientific, "f" for fahrenheit)
        user_dataframe: Optional DataFrame to enrich with weather data
        user_timestamp_column: Required if user_dataframe is provided
        auto_select_location: If True, automatically use first location (no interaction)

    Returns:
        Ready-to-use DataFrame with weather data (either standalone or enriched with user data)
    """
    try:
        # 1. Search for location with interactive selection
        if auto_select_location:
            # Non-interactive mode - just use first result
            print(
                f"🔍 Searching for location: {location_name} (auto-select mode)"
            )
            locations = search_locations(client, api_key, location_name)
            if not locations:
                raise ValueError(f"No locations found for '{location_name}'")
            location = locations[0]
            print(
                f"✅ Auto-selected: {location.name} ({location.latitude}, {location.longitude})"
            )
        else:
            # Interactive mode
            location = select_location_interactively(
                client, api_key, location_name
            )
            if not location:
                raise ValueError(f"No location found for '{location_name}'")

        # 2. Handle weather parameters
        if isinstance(weather_parameters, str):
            if weather_parameters not in COMMON_WEATHER_PARAMETERS:
                raise ValueError(
                    f"Unknown weather parameter preset: '{weather_parameters}'. Available: {list(COMMON_WEATHER_PARAMETERS.keys())}"
                )
            params_list = COMMON_WEATHER_PARAMETERS[weather_parameters]
            print(
                f"📊 Using '{weather_parameters}' parameter preset: {params_list}"
            )
        else:
            params_list = weather_parameters
            print(f"📊 Using custom parameters: {params_list}")

        # 3. Handle timestamps
        if user_dataframe is not None:
            if user_timestamp_column is None:
                raise ValueError(
                    "user_timestamp_column is required when user_dataframe is provided"
                )

            # Auto-detect start/end times from user data if not provided
            if start_time is None or end_time is None:
                df_timestamps = pd.to_datetime(
                    user_dataframe[user_timestamp_column]
                )
                auto_start = df_timestamps.min().strftime("%Y-%m-%dT%H:%M:%S")
                auto_end = df_timestamps.max().strftime("%Y-%m-%dT%H:%M:%S")

                if start_time is None:
                    start_time = auto_start
                    print(
                        f"🕐 Auto-detected start time from user data: {start_time}"
                    )
                if end_time is None:
                    end_time = auto_end
                    print(
                        f"🕐 Auto-detected end time from user data: {end_time}"
                    )

        if start_time is None or end_time is None:
            raise ValueError(
                "start_time and end_time are required when no user_dataframe is provided"
            )

        # 4. Create weather configuration
        weather_config = WeatherConfig(
            name=f"{location.name} Weather",
            location=WeatherLocation(
                name=location.name,
                latitude=location.latitude,
                longitude=location.longitude,
            ),
            weather_parameters=params_list,
            frequency=frequency,
            units=units,
            start_time=start_time,
            end_time=end_time,
        )

        # 5. Prepare user data for enrichment
        user_data_json = None
        if user_dataframe is not None:
            user_data_json = convert_datetime_columns_to_iso_strings(
                user_dataframe
            ).to_dict(orient="records")

        # 6. Create enrichment request
        enrichment_request = MetadataEnrichmentRequest(
            weather_metadata_info=weather_config,
            user_dataframe_json=user_data_json,
            user_timestamp_column=user_timestamp_column,
        )

        # 7. Make API call
        print("🌤️ Fetching weather data...")
        response = make_api_call(
            client, ENRICH_ENDPOINT, enrichment_request, api_key
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Weather data retrieved successfully!")
            print(
                f"   📊 Added weather columns: {result.get('metadata_columns', [])}"
            )
            if user_dataframe is not None:
                print(
                    f"   📈 Enriched {result.get('original_row_count', 0)} → {result.get('final_row_count', 0)} rows"
                )

            # Convert to DataFrame and return ready-to-use data
            if result.get("result_dataframe_json"):
                df = pd.DataFrame(result["result_dataframe_json"])
                # Convert timestamp columns to datetime
                if (
                    user_timestamp_column
                    and user_timestamp_column in df.columns
                ):
                    df[user_timestamp_column] = pd.to_datetime(
                        df[user_timestamp_column]
                    )
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            else:
                raise ValueError("No data found in response")
        else:
            logger.error(
                f"❌ Weather API error: {response.status_code} - {response.text}"
            )
            raise ValueError(
                f"Weather API error: {response.status_code} - {response.text}"
            )

    except Exception as e:
        logger.error(f"❌ Error getting weather data: {str(e)}")
        raise ValueError(f"Error getting weather data: {str(e)}")


def find_haver_data(
    client: httpx.Client,
    api_key: str,
    prompt: str,
    count: int = 5,
    auto_select: bool = False,
) -> Dict[str, Any]:
    """
    Find Haver data using AI-powered search and recommendations.

    Note: Searches Haver Analytics' comprehensive economic and financial datasets.

    Args:
        client: httpx client instance
        api_key: API key for authentication
        prompt: Description of what kind of data you're looking for
        count: Number of recommendations to return
        auto_select: If True, automatically select the first recommendation

    Returns:
        Selected dataset's access_info ready for use

    Raises:
        ValueError: If no recommendations found or selection cancelled
    """
    try:
        print(f"🔍 Finding Haver data for: '{prompt}'")
        recommendations = get_metadata_recommendations(
            client, api_key, prompt, count
        )

        if not recommendations:
            raise ValueError(
                "No Haver data recommendations found. Try a different search term (e.g., 'GDP inflation unemployment interest rates')"
            )

        if len(recommendations) == 1:
            # Only one match - use it directly
            access_info = recommendations[0].get("access_info", {})
            print(
                f"✅ Found unique dataset: {access_info.get('description', 'No description')}"
            )
            print(
                f"   📊 {access_info.get('name')}@{access_info.get('database_name')}"
            )
            return access_info

        # Auto-select mode
        if auto_select:
            access_info = recommendations[0].get("access_info", {})
            print(
                f"🤖 Auto-selected: {access_info.get('description', 'No description')}"
            )
            print(
                f"   📊 {access_info.get('name')}@{access_info.get('database_name')}"
            )
            return access_info

        # Multiple matches - show interactive selection
        print(f"🎯 Found {len(recommendations)} Haver datasets for '{prompt}':")
        print("=" * 80)

        for i, rec in enumerate(recommendations, 1):
            access_info = rec.get("access_info", {})
            description = access_info.get("description", "No description")
            name = access_info.get("name", "Unknown")
            database = access_info.get("database_name", "Unknown")

            print(f"   {i}. {description}")
            print(f"      📊 Series: {name}@{database}")

            # Add additional info if available
            if "frequency" in access_info:
                print(f"      📅 Frequency: {access_info['frequency']}")
            if "start_date" in access_info and "end_date" in access_info:
                print(
                    f"      📈 Range: {access_info['start_date']} to {access_info['end_date']}"
                )
            print()

        print("=" * 80)

        # Interactive selection
        try:
            while True:
                choice = input(
                    f"\n🤔 Please select a dataset (1-{len(recommendations)}) or 'q' to quit: "
                ).strip()

                if choice.lower() in ["q", "quit", "cancel"]:
                    raise ValueError("Haver data selection cancelled")

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(recommendations):
                        selected_info = recommendations[choice_idx].get(
                            "access_info", {}
                        )
                        print(
                            f"✅ Selected: {selected_info.get('description', 'No description')}"
                        )
                        print(
                            f"   📊 {selected_info.get('name')}@{selected_info.get('database_name')}"
                        )
                        return selected_info
                    else:
                        logger.warning(
                            f"❌ Invalid choice. Please enter a number between 1 and {len(recommendations)}"
                        )
                except ValueError:
                    logger.warning(
                        f"❌ Invalid input. Please enter a number between 1 and {len(recommendations)}"
                    )

        except (EOFError, KeyboardInterrupt):
            # Handle non-interactive environments or user cancellation
            logger.warning("⚠️ Interactive selection not available or cancelled")
            fallback_info = recommendations[0].get("access_info", {})
            print(
                f"🔄 Using first dataset as fallback: {fallback_info.get('description', 'No description')}"
            )
            return fallback_info

    except Exception as e:
        logger.error(f"❌ Error in Haver data search: {str(e)}")
        raise ValueError(f"Error in Haver data search: {str(e)}")


def get_haver_data(
    client: httpx.Client,
    api_key: str,
    database_name: Optional[str] = None,
    series_name: Optional[str] = None,
    description: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    user_dataframe: Optional[pd.DataFrame] = None,
    user_timestamp_column: Optional[str] = None,
    search_prompt: Optional[str] = None,
    auto_select: bool = False,
) -> pd.DataFrame:
    """
    Get Haver data with optional interactive search.

    Note: Haver Analytics provides comprehensive economic and financial time series data
    that can be valuable for forecasting models (GDP, unemployment, inflation, etc.).

    Args:
        client: httpx client instance
        api_key: API key for authentication
        database_name: Haver database name (e.g., "USECON", "EUROSTAT") - optional if using search
        series_name: Haver series name (e.g., "GDP", "RSXFS") - optional if using search
        description: Optional description of the series
        start_time: Optional start time to filter data (ISO format)
        end_time: Optional end time to filter data (ISO format)
        user_dataframe: Optional DataFrame to enrich with Haver data
        user_timestamp_column: Required if user_dataframe is provided
        search_prompt: If provided, will search for Haver data interactively instead of using database_name/series_name
        auto_select: If True and using search, automatically select the first recommendation

    Returns:
        Ready-to-use DataFrame with Haver data (either standalone or enriched with user data).
        DataFrame has .attrs containing metadata_columns, original_row_count, final_row_count.
        Returns None if failed.
    """
    try:
        # Interactive search mode
        if search_prompt:
            print(f"🔍 Finding Haver data with prompt: '{search_prompt}'")
            selected_data = find_haver_data(
                client,
                api_key,
                prompt=search_prompt,
                count=5,
                auto_select=auto_select,
            )

            # Use found data
            database_name = selected_data.get("database_name")
            series_name = selected_data.get("name")
            description = description or selected_data.get("description")

            print(f"📊 Using found data: {series_name}@{database_name}")

        # Validate required parameters
        if not database_name or not series_name:
            raise ValueError(
                "Either provide database_name and series_name directly, or use search_prompt for interactive selection"
            )

        print(f"🏦 Fetching Haver data: {series_name}@{database_name}")

        # 1. Handle timestamps from user data if provided
        if user_dataframe is not None:
            if user_timestamp_column is None:
                raise ValueError(
                    "user_timestamp_column is required when user_dataframe is provided"
                )

            # Auto-detect start/end times from user data if not provided
            if start_time is None or end_time is None:
                df_timestamps = pd.to_datetime(
                    user_dataframe[user_timestamp_column]
                )
                auto_start = df_timestamps.min().strftime("%Y-%m-%dT%H:%M:%S")
                auto_end = df_timestamps.max().strftime("%Y-%m-%dT%H:%M:%S")

                if start_time is None:
                    start_time = auto_start
                    print(
                        f"🕐 Auto-detected start time from user data: {start_time}"
                    )
                if end_time is None:
                    end_time = auto_end
                    print(
                        f"🕐 Auto-detected end time from user data: {end_time}"
                    )

        # 2. Create Haver configuration
        haver_config = HaverConfig(
            database_name=database_name,
            name=series_name,
            description=description or f"{series_name}@{database_name}",
            start_time=start_time,
            end_time=end_time,
        )

        # 3. Prepare user data for enrichment
        user_data_json = None
        if user_dataframe is not None:
            user_data_json = convert_datetime_columns_to_iso_strings(
                user_dataframe
            ).to_dict(orient="records")

        # 4. Create enrichment request
        enrichment_request = MetadataEnrichmentRequest(
            haver_metadata_info=haver_config,
            user_dataframe_json=user_data_json,
            user_timestamp_column=user_timestamp_column,
        )

        # 5. Make API call
        print("📈 Fetching Haver data...")
        response = make_api_call(
            client, ENRICH_ENDPOINT, enrichment_request, api_key
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Haver data retrieved successfully!")
            print(
                f"   📊 Added Haver columns: {result.get('metadata_columns', [])}"
            )
            if user_dataframe is not None:
                print(
                    f"   📈 Enriched {result.get('original_row_count', 0)} → {result.get('final_row_count', 0)} rows"
                )

            # Convert to DataFrame and return ready-to-use data
            if result.get("result_dataframe_json"):
                df = pd.DataFrame(result["result_dataframe_json"])

                # Convert timestamp columns to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                if (
                    user_timestamp_column
                    and user_timestamp_column in df.columns
                ):
                    df[user_timestamp_column] = pd.to_datetime(
                        df[user_timestamp_column]
                    )

                # Rename generic "value" column to meaningful name based on series info
                if "value" in df.columns:
                    # Create meaningful column name from series name
                    meaningful_name = series_name
                    # Clean up the name to be a valid column name
                    meaningful_name = meaningful_name.replace(" ", "_").replace(
                        "-", "_"
                    )
                    meaningful_name = "".join(
                        c for c in meaningful_name if c.isalnum() or c == "_"
                    )

                    # Rename the column
                    df = df.rename(columns={"value": meaningful_name})

                # Add metadata info as attributes for easy access
                df.attrs["metadata_columns"] = result.get(
                    "metadata_columns", []
                )
                df.attrs["original_row_count"] = result.get(
                    "original_row_count", 0
                )
                df.attrs["final_row_count"] = result.get("final_row_count", 0)
                df.attrs["series_name"] = series_name
                df.attrs["database_name"] = database_name

                return df
            else:
                raise ValueError("No data found in response")
        else:
            logger.error(
                f"❌ Haver API error: {response.status_code} - {response.text}"
            )
            raise ValueError(
                f"Haver API error: {response.status_code} - {response.text}"
            )

    except Exception as e:
        logger.error(f"❌ Error getting Haver data: {str(e)}")
        raise ValueError(f"Error getting Haver data: {str(e)}")


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
        covariate_cols: Additional columns to use as features (including metadata columns)

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


def generate_forecast(
    client: httpx.Client,
    api_key: str,
    df: pd.DataFrame,
    target_columns: Union[str, List[str]],
    forecast_timestamps: List[str],
    timestamp_column: Optional[str] = None,
    covariate_columns: Optional[List[str]] = None,
    future_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate forecasts for given target columns using the Synthefy API.

    Args:
        client: httpx client instance
        api_key: API key for authentication
        df: DataFrame with historical data (can be enriched with weather/Haver data)
        target_columns: Column name(s) to forecast
        forecast_timestamps: List of timestamps to forecast for (ISO format)
        timestamp_column: Column containing timestamps (auto-detected if None)
        covariate_columns: Additional columns to use as features
        future_df: Optional DataFrame with future known data

    Returns:
        Dictionary with 'forecast' and 'quantiles' DataFrames

    Raises:
        ValueError: If forecast fails or data is invalid
    """
    try:
        # Normalize target_columns to list
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        # Validate inputs
        if not target_columns:
            raise ValueError("target_columns cannot be empty")

        if not forecast_timestamps:
            raise ValueError("forecast_timestamps cannot be empty")

        # Auto-detect timestamp column if not provided
        if timestamp_column is None:
            datetime_cols = [
                col
                for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            ]
            if not datetime_cols:
                raise ValueError(
                    "No datetime column found. Please specify timestamp_column."
                )
            timestamp_column = datetime_cols[0]
            print(f"🕐 Auto-detected timestamp column: {timestamp_column}")

        # Validate timestamp column exists
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_column}' not found in DataFrame"
            )

        # Validate target columns exist
        missing_targets = [
            col for col in target_columns if col not in df.columns
        ]
        if missing_targets:
            raise ValueError(
                f"Target columns not found in DataFrame: {missing_targets}"
            )

        # Validate covariate columns exist
        covariate_columns = covariate_columns or []
        missing_covariates = [
            col for col in covariate_columns if col not in df.columns
        ]
        if missing_covariates:
            raise ValueError(
                f"Covariate columns not found in DataFrame: {missing_covariates}"
            )

        # Check for NaN values in the DataFrame
        all_columns_to_check = (
            [timestamp_column] + target_columns + covariate_columns
        )
        relevant_df = df[all_columns_to_check]

        if relevant_df.isnull().any().any():
            nan_summary = relevant_df.isnull().sum()
            nan_columns = nan_summary[nan_summary > 0]

            error_msg = "❌ DataFrame contains NaN/missing values that must be handled before forecasting:\n\n"
            for col, count in nan_columns.items():
                error_msg += f"   • {col}: {count} missing values\n"

            error_msg += "\n💡 Please handle missing values using one of these approaches:\n"
            error_msg += "   1. Remove rows with NaNs: df = df.dropna()\n"
            error_msg += "   2. Fill NaNs with appropriate values: df = df.fillna(method='forward') or df.fillna(0)\n"
            error_msg += (
                "   3. Interpolate missing values: df = df.interpolate()\n"
            )
            error_msg += "   4. Inspect and manually handle: df.isnull().sum() to see missing data patterns\n"

            raise ValueError(error_msg)

        print(f"🎯 Generating forecast for: {target_columns}")
        if covariate_columns:
            print(
                f"📊 Using {len(covariate_columns)} covariates: {covariate_columns}"
            )

        # Create forecast request
        request = convert_df_to_synthefy_request(
            df=df,
            future_df=future_df,
            target_cols=target_columns,
            forecast_timestamps=forecast_timestamps,
            timestamp_col=timestamp_column,
            covariate_cols=covariate_columns,
        )

        # Make forecast API call
        print("🚀 Making forecast API call...")
        response = make_api_call(client, FORECAST_ENDPOINT, request, api_key)

        if response.status_code == 200:
            forecast_df, quantiles_df = convert_response_to_df(response.json())
            print("✅ Forecast completed successfully!")
            print(
                f"📈 Generated forecasts for {len(forecast_timestamps)} timestamps"
            )
            return {"forecast": forecast_df, "quantiles": quantiles_df}
        else:
            error_msg = f"Forecast API failed with status {response.status_code}: {response.text}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Error generating forecast: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
