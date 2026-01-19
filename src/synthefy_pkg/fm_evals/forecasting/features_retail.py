import datetime
from enum import Enum
from typing import Callable, List, Optional

import holidays
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from synthefy_pkg.fm_evals.forecasting.utils import detect_date_column


class RetailTag(str, Enum):
    """Retail Tag Data Enume."""

    ONLINE = "online"
    PHYSICAL = "physical"
    GROCERY = "grocery"
    LUXURY = "luxury"
    CONVENIENCE = "convenience"
    ALWAYS_OPEN = "always_open"


class RetailCategory(str, Enum):
    """Retail category for context-based feature weighting."""

    GROCERY = "grocery"
    CONVENIENCE = "convenience"
    PHYSICAL_ONLY = "physical_only"
    ONLINE_ONLY = "online_only"
    HYBRID = "hybrid"


class RetailContext(BaseModel):
    """Retail Context Data Model."""

    tags: List[RetailTag] = Field(
        default=[RetailTag.ONLINE, RetailTag.PHYSICAL],
        description="List of retail tags describing the business",
    )

    def has(self, tag: RetailTag) -> bool:
        """Check if context has a specific tag."""
        # Handle both enum and string comparisons (due to use_enum_values)
        return tag in self.tags or tag.value in self.tags

    def has_any(self, *tags: RetailTag) -> bool:
        """Check if context has any of the specified tags."""
        return any(self.has(tag) for tag in tags)

    def has_all(self, *tags: RetailTag) -> bool:
        """Check if context has all of the specified tags."""
        return all(self.has(tag) for tag in tags)


class LocationConfig(BaseModel):
    """
    Geographic context for location-aware features.

    Used by:
    - Holiday features (country-specific holidays)
    - Weather features (lat/lon-based weather)
    - Event features (local events)
    - Seasonality features (hemisphere, climate)
    """

    country: str = Field(
        default="US",
        description="ISO 3166-1 alpha-2 country code (e.g., 'US', 'GB', 'CN')",
    )
    region_or_state: Optional[str] = Field(
        default=None, description="State/province code (e.g., 'CA', 'NY', 'ON')"
    )
    city: Optional[str] = Field(
        default=None, description="City name for weather/events"
    )
    timezone: str = Field(
        default="America/New_York",
        description="IANA timezone (e.g., 'America/Los_Angeles', 'Europe/London')",
    )

    class Config:
        frozen = True  # Immutable (can be used as dict key for caching)


# ============================================================================
# Holiday Keywords and Impact Weights
# ============================================================================
MAJOR_US_HOLIDAY_KEYWORDS = ["christmas", "thanksgiving", "new year"]
HIGH_US_IMPORTANCE_HOLIDAY_KEYWORDS = ["independence", "memorial", "labor day"]
MEDIUM_US_HIGH_IMPORTANCE_HOLIDAY_KEYWORDS = ["easter", "veterans", "columbus"]
MEDIUM_US_IMPORTANCE_HOLIDAY_KEYWORDS = [
    "martin luther king",
    "presidents",
    "washington",
]

HOLIDAY_IMPACT_WEIGHTS = {
    "major_holiday": {  # Christmas, Thanksgiving, New Year
        "physical_only": 1.0,  # Reduced foot traffic (people celebrating at home)
        "online_only": 0.3,  # Reduced browsing (people busy with family)
        "hybrid": 0.65,  # Blended impact
        "grocery": 0.5,  # Pre-holiday rush, holiday day slowdown
        "luxury": 1.3,  # Gifting season multiplier
        "convenience": 0.2,  # Open but slower traffic
    },
    "minor_holiday": {  # MLK Day, Presidents Day (schools closed, many stores open)
        "physical_only": 0.6,  # Some increased traffic (day off from work/school)
        "online_only": 0.1,  # Minimal impact
        "hybrid": 0.35,  # Blended
        "grocery": 0.2,  # Normal operations
        "luxury": 0.4,  # Some gifting (e.g., Valentine's proximity)
        "convenience": 0.05,  # Minimal change
    },
    "shopping_event": {  # Black Friday, Prime Day (major sales events)
        "physical_only": 2.5,  # Huge in-store traffic and sales
        "online_only": 3.0,  # Peak online shopping
        "hybrid": 2.75,  # High traffic both channels
        "grocery": 0.3,  # Not typically discount-driven
        "luxury": 1.5,  # Strong sales but less discount-focused
        "convenience": 0.1,  # Minimal participation in sales events
    },
}

# PHQ rank thresholds for holiday classification
MAJOR_HOLIDAY_THRESHOLD = 80  # phq_rank >= 80 is major
MINOR_HOLIDAY_THRESHOLD = 50  # 50 <= phq_rank < 80 is minor

# PHQ rank estimates for holidays
PHQ_RANK_MAJOR = 95.0  # Major holidays (Christmas, Thanksgiving, New Year)
PHQ_RANK_HIGH = 85.0  # High importance (Independence, Memorial, Labor Day)
PHQ_RANK_MEDIUM_HIGH = 75.0  # Medium-high (Easter, Veterans, Columbus)
PHQ_RANK_MEDIUM = 65.0  # Medium (MLK, Presidents, Washington)
PHQ_RANK_LOW = 50.0  # Lower importance (default)

# Default importance for holidays when PHQ rank is unavailable (normalized 0-1)
DEFAULT_HOLIDAY_IMPORTANCE = 0.5  # Mid-range importance

# Window for pre/post holiday binary flags (days)
HOLIDAY_WEEK_WINDOW = 7  # 1 week before/after holiday

HOLIDAY_MIN_YEAR_TO_FETCH = 1990
HOLIDAY_MAX_YEAR_TO_FETCH = 2050

# ============================================================================
# Helper Functions
# ============================================================================


def _get_context_key(context: RetailContext) -> RetailCategory:
    """Determine the context key for weight lookup."""
    has_physical = context.has(RetailTag.PHYSICAL)
    has_online = context.has(RetailTag.ONLINE)

    # Check business type tags first
    if context.has(RetailTag.GROCERY):
        return RetailCategory.GROCERY
    elif context.has(RetailTag.CONVENIENCE):
        return RetailCategory.CONVENIENCE

    # Then check operation model
    if has_physical and not has_online:
        return RetailCategory.PHYSICAL_ONLY
    elif has_online and not has_physical:
        return RetailCategory.ONLINE_ONLY
    elif has_physical and has_online:
        return RetailCategory.HYBRID
    else:
        return RetailCategory.HYBRID


def _calculate_holiday_weight(
    phq_rank: float, context: RetailContext, holiday_type: str = "major_holiday"
) -> float:
    """
    Calculate context-adjusted holiday weight.

    Args:
        phq_rank: PredictHQ rank (0-100)
        context: Retail context
        holiday_type: Type of holiday (major_holiday, minor_holiday, shopping_event)

    Returns:
        Adjusted weight (0.0 - 3.0+)
    """
    # Get base multiplier from context
    context_key = _get_context_key(context)
    base_multiplier = HOLIDAY_IMPACT_WEIGHTS[holiday_type].get(context_key, 1.0)

    # Apply luxury boost if applicable (multiplicative)
    if context.has(RetailTag.LUXURY):
        luxury_boost = HOLIDAY_IMPACT_WEIGHTS[holiday_type].get("luxury", 1.0)
        base_multiplier *= luxury_boost

    # Calculate final weight
    # weight = (phq_rank / 100) * multiplier
    weight = (phq_rank / 100.0) * base_multiplier

    return weight


def _fetch_holidays_fallback(
    location: LocationConfig, years: List[int]
) -> pd.DataFrame:
    """
    Fetch holidays using the holidays library.

    Args:
        location: Location configuration
        years: List of years to fetch

    Returns:
        DataFrame with columns: date, name, phq_rank, category
    """
    # Map country codes to holidays library
    country_map = {
        "US": "US",
        "GB": "UK",
        "CA": "CA",
        "CN": "CN",
        "DE": "DE",
        "FR": "FR",
        "AU": "AU",
    }

    country_code = country_map.get(location.country, "US")

    # Get holidays for the years
    all_holidays = []
    for year in years:
        country_holidays = holidays.country_holidays(country_code, years=year)

        for date, name in country_holidays.items():
            # Assign synthetic PHQ rank based on holiday name
            # (This is a simple heuristic)
            phq_rank = _estimate_phq_rank(name)
            all_holidays.append(
                {
                    "date": pd.Timestamp(date),
                    "name": name,
                    "phq_rank": phq_rank,
                    "category": "public-holidays",
                }
            )

    return pd.DataFrame(all_holidays)


def _estimate_phq_rank(holiday_name: str) -> float:
    """Estimate PHQ rank based on holiday name."""
    name_lower = holiday_name.lower()

    if any(keyword in name_lower for keyword in MAJOR_US_HOLIDAY_KEYWORDS):
        return PHQ_RANK_MAJOR

    if any(
        keyword in name_lower for keyword in HIGH_US_IMPORTANCE_HOLIDAY_KEYWORDS
    ):
        return PHQ_RANK_HIGH

    if any(
        keyword in name_lower
        for keyword in MEDIUM_US_HIGH_IMPORTANCE_HOLIDAY_KEYWORDS
    ):
        return PHQ_RANK_MEDIUM_HIGH

    if any(
        keyword in name_lower
        for keyword in MEDIUM_US_IMPORTANCE_HOLIDAY_KEYWORDS
    ):
        return PHQ_RANK_MEDIUM

    return PHQ_RANK_LOW


# ============================================================================
# Holiday Features
# ============================================================================
def make_holiday_features(
    location: LocationConfig,
    context: RetailContext,
    include_school_holidays: bool = True,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Generate holiday indicator features with context-aware weighting.

    Creates binary flags for different holiday types and applies business
    context-specific weights to capture differential impact across retail types.

    Args:
        location: Geographic location for holiday calendar
        context: Retail business context (tags like online, physical, luxury)
        include_school_holidays: Include school holiday indicators

    Returns:
        Feature generator function that adds holiday columns to DataFrame

    Features Generated:
        - is_holiday: Binary (1=public holiday, 0=regular day)
        - is_major_holiday: Binary (1=major holiday like Christmas, 0=otherwise)
        - is_school_holiday: Binary (1=school holiday/break, 0=otherwise)
        - holiday_weight: Float (context-adjusted impact, 0.0-3.0+)
        - holiday_name: String (name of holiday or empty)
        - holiday_phq_rank: Float (PredictHQ importance score 0-100)

    Example:
        >>> import pandas as pd
        >>> from synthefy_pkg.fm_evals.forecasting.features_retail import (
        ...     make_holiday_features, LocationConfig, RetailContext, RetailTag
        ... )
        >>>
        >>> # Input DataFrame
        >>> df = pd.DataFrame({
        ...     'date': ['2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26'],
        ...     'sales': [12000, 15000, 3000, 8000]
        ... })
        >>>
        >>> # Configure for luxury walk-in store
        >>> location = LocationConfig(country="US")
        >>> context = RetailContext(tags=[RetailTag.PHYSICAL, RetailTag.LUXURY])
        >>>
        >>> # Generate features
        >>> add_features = make_holiday_features(location, context)
        >>> result = add_features(df)
        >>>
        >>> # Output DataFrame
        >>> print(result[['date', 'sales', 'is_holiday', 'holiday_name', 'holiday_weight']])
              date  sales  is_holiday   holiday_name  holiday_weight
        0  2024-12-23  12000           0                          0.00
        1  2024-12-24  15000           0                          0.00
        2  2024-12-25   3000           1  Christmas Day           1.24
        3  2024-12-26   8000           0                          0.00

        Note:
            - Christmas (Dec 25) is detected with is_holiday=1
            - is_major_holiday=1 (phq_rank=95 >= 80)
            - holiday_weight=1.24 for luxury store (base 0.95 * luxury multiplier 1.3)
            - For online-only store, weight would be 0.29 (95 * 0.3 / 100)
            - For physical-only store, weight would be 0.95 (95 * 1.0 / 100)
    """

    def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features to DataFrame (inner function following pattern)."""
        # Do not change row order; just add columns
        result = df.copy()

        # Extract dates from DataFrame
        dates = None
        date_col = detect_date_column(result)

        if date_col is not None:
            dates = pd.to_datetime(result[date_col])
            # Validate dates: check if they're clearly invalid (e.g., all identical at date level)
            if len(dates) > 1 and dates.dt.date.nunique() == 1:
                # All dates are the same, likely not a real date column
                dates = None

        # Fall back to DatetimeIndex if column date check failed
        if dates is None and isinstance(result.index, pd.DatetimeIndex):
            dates = result.index

        # If no valid dates found, return with empty features
        if dates is None:
            result["is_holiday"] = 0
            result["is_major_holiday"] = 0
            result["is_school_holiday"] = 0
            result["holiday_weight"] = 0.0
            result["holiday_name"] = ""
            result["holiday_phq_rank"] = 0.0
            return result

        # Handle empty DataFrame
        if len(dates) == 0 or dates.isna().all():
            result["is_holiday"] = 0
            result["is_major_holiday"] = 0
            result["is_school_holiday"] = 0
            result["holiday_weight"] = 0.0
            result["holiday_name"] = ""
            result["holiday_phq_rank"] = 0.0
            return result

        # Determine year range from actual data
        min_year = dates.min().year
        max_year = dates.max().year
        # Add buffer of 1 year on each side to handle edge cases
        years_to_fetch = list(range(min_year - 1, max_year + 2))

        # Fetch holidays only for relevant years
        holidays_df = _fetch_holidays_fallback(location, years_to_fetch)

        # Create a dict for fast lookup: date -> holiday info
        holiday_lookup = {}
        for _, row in holidays_df.iterrows():
            date_key = row["date"].date()
            holiday_lookup[date_key] = {
                "name": row["name"],
                "phq_rank": row["phq_rank"],
                "category": row["category"],
            }

        # Initialize all dates as non-holidays (default values)
        n_dates = len(dates)
        is_holiday = [0] * n_dates
        is_major_holiday = [0] * n_dates
        is_school_holiday = [0] * n_dates
        holiday_weights = [0.0] * n_dates
        holiday_names = [""] * n_dates
        holiday_phq_ranks = [0.0] * n_dates

        # Create date to index mapping for fast lookup
        date_to_idx = {date.date(): idx for idx, date in enumerate(dates)}

        # Loop through holidays (much smaller set than all dates)
        for date_key, holiday_info in holiday_lookup.items():
            if date_key in date_to_idx:
                idx = date_to_idx[date_key]
                phq_rank = holiday_info["phq_rank"]

                # Classify holiday type
                if phq_rank >= MAJOR_HOLIDAY_THRESHOLD:
                    holiday_type = "major_holiday"
                    is_major = 1
                else:
                    holiday_type = "minor_holiday"
                    is_major = 0

                # Calculate weight
                weight = _calculate_holiday_weight(
                    phq_rank, context, holiday_type
                )

                # Update this specific date
                is_holiday[idx] = 1
                is_major_holiday[idx] = is_major
                is_school_holiday[idx] = 0  # TODO: Add school holiday detection
                holiday_weights[idx] = weight
                holiday_names[idx] = holiday_info["name"]
                holiday_phq_ranks[idx] = phq_rank

        # Add columns to result
        result["is_holiday"] = is_holiday
        result["is_major_holiday"] = is_major_holiday
        result["is_school_holiday"] = is_school_holiday
        result["holiday_weight"] = holiday_weights
        result["holiday_name"] = holiday_names
        result["holiday_phq_rank"] = holiday_phq_ranks

        return result

    return _add_holiday_features


def make_holiday_proximity_features(
    location: LocationConfig, horizons: List[int] = [7, 14]
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Generate holiday proximity features (distance to nearest holidays).

    Creates continuous features measuring distance to upcoming/past holidays,
    useful for capturing pre-holiday shopping buildup and post-holiday lulls.

    Args:
        location: Geographic location for holiday calendar
        horizons: Look-ahead/behind windows in days (default: [7, 14])

    Returns:
        Feature generator function that adds proximity columns to DataFrame

    Features Generated (per horizon):
        - days_to_next_holiday_{horizon}: Days until next holiday (0 to horizon)
        - days_since_last_holiday_{horizon}: Days since last holiday (0 to horizon)
        - holidays_in_window_{horizon}: Count of holidays within ±horizon window
        - holiday_cluster_score_{horizon}: Weighted score combining proximity & importance
        - is_pre_holiday_week: Binary (1=within 7 days before holiday)
        - is_post_holiday_week: Binary (1=within 7 days after holiday)

    Example:
        >>> import pandas as pd
        >>> from synthefy_pkg.fm_evals.forecasting.features_retail import (
        ...     make_holiday_proximity_features, LocationConfig
        ... )
        >>>
        >>> # Input DataFrame (Christmas & New Year period)
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-12-23', '2025-01-02', freq='D'),
        ...     'sales': [15000, 18000, 5000, 8000, 9000, 9500, 10000, 11000, 12000, 20000, 8000]
        ... })
        >>>
        >>> # Generate proximity features
        >>> location = LocationConfig(country="US")
        >>> add_features = make_holiday_proximity_features(location, horizons=[14])
        >>> result = add_features(df)
        >>>
        >>> # Output DataFrame (key columns shown)
        >>> print(result[['date', 'days_to_next_holiday_14', 'days_since_last_holiday_14',
        ...               'holidays_in_window_14', 'holiday_cluster_score_14']])
              date  days_to_next_holiday_14  days_since_last_holiday_14  holidays_in_window_14  holiday_cluster_score_14
        0  2024-12-23                       2                          14                      2                      1.85
        1  2024-12-24                       1                          14                      2                      1.90
        2  2024-12-25                       0                           0                      2                      1.90  # Christmas
        3  2024-12-26                       6                           1                      2                      1.20
        4  2024-12-27                       5                           2                      2                      0.95
        5  2024-12-28                       4                           3                      2                      0.70
        6  2024-12-29                       3                           4                      1                      0.55
        7  2024-12-30                       2                           5                      1                      0.75
        8  2024-12-31                       1                           6                      1                      0.80
        9  2025-01-01                       0                           0                      1                      0.85  # New Year
        10 2025-01-02                      14                           1                      1                      0.25

        Note:
            - days_to_next_holiday tracks nearest upcoming holiday (switches from Christmas to New Year)
            - days_since_last_holiday tracks days after most recent holiday (resets at each holiday)
            - holidays_in_window=2 on Dec 23-28 (both Christmas & New Year within 14 days)
            - holiday_cluster_score combines effects of nearby holidays:
              * Dec 24: High (1.90) - 1 day to Christmas + New Year also in window
              * Dec 25: Peak (1.90) - Christmas day + New Year approaching
              * Dec 26-28: Moderate (0.7-1.2) - between two holidays
              * Jan 1: (0.85) - New Year's day
            - is_pre_holiday_week=1 for dates 1-7 days before a holiday
            - is_post_holiday_week=1 for dates 1-7 days after a holiday
    """

    def _add_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday proximity features to DataFrame.

        Generates temporal distance features that capture pre/post-holiday patterns
        useful for forecasting shopping behavior (e.g., pre-holiday rush, post-holiday
        lull).

        Features Generated (per horizon):
            - days_to_next_holiday_{horizon}: Days until next holiday (0 to horizon)
            - days_since_last_holiday_{horizon}: Days since last holiday (0 to horizon)
            - holidays_in_window_{horizon}: Count of holidays within ±horizon days
            - holiday_cluster_score_{horizon}: Weighted score combining proximity & importance
                Formula: sum(proximity_weight * phq_rank/100) for holidays in window
                where proximity_weight = 1.0 - (distance/horizon)
            - is_pre_holiday_week: Binary (1 if 1-7 days before any holiday)
            - is_post_holiday_week: Binary (1 if 1-7 days after any holiday)

        Args:
            df: Input DataFrame with date column or DatetimeIndex

        Returns:
            DataFrame with added proximity feature columns (preserves row order)

        Note:
            - If no date column found, returns DataFrame with default values
            - Cluster score captures cumulative effect of multiple nearby holidays
            - Higher cluster scores indicate dates near important holidays
        """
        result = df.copy()

        # Extract dates
        dates = None
        date_col = detect_date_column(result)

        if date_col is not None:
            dates = pd.to_datetime(result[date_col])
            # Validate dates: check if they're clearly invalid (e.g., all identical at date level)
            if len(dates) > 1 and dates.dt.date.nunique() == 1:
                # All dates are the same, likely not a real date column
                dates = None

        # Fall back to DatetimeIndex if column date check failed
        if dates is None and isinstance(result.index, pd.DatetimeIndex):
            dates = result.index

        # If no valid dates found, return with default values
        if dates is None:
            for horizon in horizons:
                result[f"days_to_next_holiday_{horizon}"] = horizon
                result[f"days_since_last_holiday_{horizon}"] = horizon
                result[f"holidays_in_window_{horizon}"] = 0
                result[f"holiday_cluster_score_{horizon}"] = 0.0
            result["is_pre_holiday_week"] = 0
            result["is_post_holiday_week"] = 0
            return result

        # Handle empty DataFrame
        if len(dates) == 0 or dates.isna().all():
            for horizon in horizons:
                result[f"days_to_next_holiday_{horizon}"] = horizon
                result[f"days_since_last_holiday_{horizon}"] = horizon
                result[f"holidays_in_window_{horizon}"] = 0
                result[f"holiday_cluster_score_{horizon}"] = 0.0
            result["is_pre_holiday_week"] = 0
            result["is_post_holiday_week"] = 0
            return result

        # Determine year range from actual data (with buffer for proximity window)
        min_year = dates.min().year
        max_year = dates.max().year
        # Add buffer of 1 year on each side to handle edge cases
        years_to_fetch = list(range(min_year - 1, max_year + 2))

        # Fetch holidays only for relevant years
        holidays_df = _fetch_holidays_fallback(location, years_to_fetch)
        holidays_df = holidays_df.sort_values("date").reset_index(drop=True)

        # Convert to numpy array for vectorized operations
        holiday_dates_np = holidays_df["date"].values.astype("datetime64[D]")
        dates_np = pd.to_datetime(dates).values.astype("datetime64[D]")

        # Calculate proximity features for each horizon
        for horizon in horizons:
            # Vectorized calculation of days to/from holidays
            # Broadcast to create (n_dates, n_holidays) matrix
            dates_expanded = dates_np[:, np.newaxis]
            holidays_expanded = holiday_dates_np[np.newaxis, :]

            # Calculate day differences
            day_diffs = (holidays_expanded - dates_expanded) / np.timedelta64(
                1, "D"
            )
            day_diffs = day_diffs.astype(int)

            # Days to next holiday: minimum of positive differences (or horizon)
            future_diffs = np.where(day_diffs >= 0, day_diffs, horizon + 1)
            days_to_next = np.min(future_diffs, axis=1)
            days_to_next = np.clip(days_to_next, 0, horizon)

            # Days since last holiday: minimum of negative differences in absolute value (or horizon)
            past_diffs = np.where(day_diffs <= 0, -day_diffs, horizon + 1)
            days_since_last = np.min(past_diffs, axis=1)
            days_since_last = np.clip(days_since_last, 0, horizon)

            # Holidays in window: count holidays within ±horizon days
            abs_day_diffs = np.abs(day_diffs)
            in_window = abs_day_diffs <= horizon
            holidays_in_window = np.sum(in_window, axis=1)

            # Cluster scores: weighted by proximity and importance
            proximity_weights = np.where(
                in_window, 1.0 - (abs_day_diffs / float(horizon)), 0.0
            )
            holiday_ranks = holidays_df["phq_rank"].values.astype(float) / 100.0
            cluster_scores = np.sum(
                proximity_weights * holiday_ranks[np.newaxis, :], axis=1
            )

            result[f"days_to_next_holiday_{horizon}"] = days_to_next.tolist()
            result[f"days_since_last_holiday_{horizon}"] = (
                days_since_last.tolist()
            )
            result[f"holidays_in_window_{horizon}"] = (
                holidays_in_window.tolist()
            )
            result[f"holiday_cluster_score_{horizon}"] = cluster_scores.tolist()

        # Add binary flags (using HOLIDAY_WEEK_WINDOW)
        if HOLIDAY_WEEK_WINDOW in horizons or len(horizons) > 0:
            # Use the smallest horizon for binary flags, or default to HOLIDAY_WEEK_WINDOW
            days_to_week = result.get(
                f"days_to_next_holiday_{HOLIDAY_WEEK_WINDOW}",
                result[f"days_to_next_holiday_{horizons[0]}"],
            )
            days_since_week = result.get(
                f"days_since_last_holiday_{HOLIDAY_WEEK_WINDOW}",
                result[f"days_since_last_holiday_{horizons[0]}"],
            )

            result["is_pre_holiday_week"] = (days_to_week > 0) & (
                days_to_week <= HOLIDAY_WEEK_WINDOW
            )
            result["is_post_holiday_week"] = (days_since_week > 0) & (
                days_since_week <= HOLIDAY_WEEK_WINDOW
            )
        else:
            result["is_pre_holiday_week"] = 0
            result["is_post_holiday_week"] = 0

        # Convert boolean to int
        result["is_pre_holiday_week"] = result["is_pre_holiday_week"].astype(
            int
        )
        result["is_post_holiday_week"] = result["is_post_holiday_week"].astype(
            int
        )

        return result

    return _add_proximity_features
