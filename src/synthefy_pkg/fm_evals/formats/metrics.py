from dataclasses import dataclass
from typing import Any

SUPPORTED_METRICS = [
    "mae",
    "median_mae",
    "nmae",
    "median_nmae",
    "mape",
    "median_mape",
    "mse",
    "median_mse",
]


@dataclass
class ForecastMetrics:
    sample_id: Any

    mae: float
    median_mae: float

    nmae: float
    median_nmae: float

    mape: float
    median_mape: float

    mse: float
    median_mse: float
