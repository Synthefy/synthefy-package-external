#!/usr/bin/env python3
"""
Programmatic inference with synthefy models.

Usage:
    python inference_example.py --task synthesis
    python inference_example.py --task forecast --forecast-length 50
    python inference_example.py --task synthesis --seed 42 --output results.csv
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# Add the api directory to the path for imports
API_DIR = Path(__file__).parent / "api"
sys.path.insert(0, str(API_DIR))

from models import DataFrameModel, OneTimeSeries
from services.config_loader import get_config_loader
from services.demo_synthesis_service import DemoSynthesisService

# =============================================================================
# CONSTANTS (configurable via command line)
# =============================================================================

DEFAULT_DATASET = "oura_subset"
DEFAULT_MODEL_TYPE = "flexible"
DEFAULT_NUM_SAMPLES = 20
DEFAULT_FORECAST_LENGTH = 96
DEFAULT_GROUND_TRUTH_PREFIX = 0
DEFAULT_SEED: Optional[int] = 123
FAKE_DATA_PATH = Path(__file__).parent / "fake_oura_subset_data.parquet"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility. Returns the seed used."""
    if seed is None:
        seed = int(time.time() * 1000) % 100000 + random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from parquet file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_parquet(data_path)


def timeseries_to_dataframe(timeseries_list: List[OneTimeSeries]) -> pd.DataFrame:
    """Convert list of OneTimeSeries to DataFrame."""
    return pd.DataFrame({
        ts.name: [np.nan if v is None else v for v in ts.values]
        for ts in timeseries_list
    })


def keep_only_columns_for_forecast(
    df: pd.DataFrame,
    columns_to_keep: List[str],
    forecast_length: int,
    timeseries_columns: List[str],
) -> pd.DataFrame:
    """
    Keep only specified columns, zero out all other timeseries columns for the forecast horizon.

    Args:
        df: Input DataFrame
        columns_to_keep: List of column names to keep (not zero out)
        forecast_length: Number of rows from the end to zero out
        timeseries_columns: List of all timeseries column names

    Returns:
        DataFrame with only specified columns having values in the forecast horizon
    """
    df = df.copy()
    for col in timeseries_columns:
        if col not in columns_to_keep and col in df.columns:
            df.loc[df.index[-forecast_length:], col] = 0.0
    return df


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    task: str,
    dataset_name: str = DEFAULT_DATASET,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_path: Optional[str] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    forecast_length: int = DEFAULT_FORECAST_LENGTH,
    ground_truth_prefix: int = DEFAULT_GROUND_TRUTH_PREFIX,
    columns_to_keep: Optional[List[str]] = None,
    seed: Optional[int] = DEFAULT_SEED,
    data_path: Path = FAKE_DATA_PATH,
) -> pd.DataFrame:
    """
    Run synthesis or forecast inference.

    Args:
        task: "synthesis" or "forecast"
        dataset_name: Dataset name for config loading
        model_type: "flexible" or "standard"
        model_path: Optional custom path to model checkpoint
        num_samples: Number of synthesis runs to average
        forecast_length: For forecast, number of time steps to predict
        ground_truth_prefix: For synthesis, keep first N points from input
        columns_to_keep: For forecast, only keep these columns (zero out others)
        seed: Random seed (None = time-based)
        data_path: Path to input data file

    Returns:
        DataFrame with synthetic/forecasted time series
    """
    actual_seed = set_seed(seed)

    # Load config and data
    config_loader = get_config_loader(dataset_name)
    window_size = config_loader.get_window_size()
    timeseries_cols = config_loader.get_required_columns().timeseries

    df = load_data(data_path)
    if len(df) > window_size:
        df = df.head(window_size).copy()
    elif len(df) < window_size:
        raise ValueError(f"Data has {len(df)} rows but window_size is {window_size}")

    # Apply column masking if specified
    if columns_to_keep is not None and task == "forecast":
        df = keep_only_columns_for_forecast(df, columns_to_keep, forecast_length, timeseries_cols)

    # Initialize service
    service = DemoSynthesisService(
        dataset_name=dataset_name,
        model_type=model_type,
        task_type=task,
        model_path=model_path,
    )

    # Convert to DataFrameModel
    columns = {col: df[col].tolist() for col in df.columns}
    data_model = DataFrameModel(columns=columns)

    # Run inference
    info = f"Running {task} | samples={num_samples} | seed={actual_seed}"
    if task == "forecast":
        info += f" | forecast_length={forecast_length}"
        if columns_to_keep:
            info += f" | columns_to_keep={columns_to_keep}"
    print(info)

    if task == "forecast":
        result_timeseries = service.generate(
            data=data_model,
            num_samples=num_samples,
            forecast_length=forecast_length,
        )
    else:
        result_timeseries = service.generate(
            data=data_model,
            num_samples=num_samples,
            ground_truth_prefix_length=ground_truth_prefix,
        )

    return timeseries_to_dataframe(result_timeseries)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run synthesis/forecast inference")

    parser.add_argument("--task", "-t", choices=["synthesis", "forecast"], default="synthesis")
    parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET, choices=["oura", "oura_subset", "ppg"])
    parser.add_argument("--model-type", "-m", default=DEFAULT_MODEL_TYPE, choices=["flexible", "standard"])
    parser.add_argument("--model-path", "-p", default=None, help="Custom model checkpoint path")
    parser.add_argument("--num-samples", "-n", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--forecast-length", "-f", type=int, default=DEFAULT_FORECAST_LENGTH)
    parser.add_argument("--ground-truth-prefix", "-g", type=int, default=DEFAULT_GROUND_TRUTH_PREFIX)
    parser.add_argument("--seed", "-s", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-path", type=Path, default=FAKE_DATA_PATH)
    parser.add_argument("--columns-to-keep", "-c", nargs="+", default=None,
                        help="For forecast: only keep these columns (zero out others)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV/Parquet path")

    args = parser.parse_args()

    try:
        result = run_inference(
            task=args.task,
            dataset_name=args.dataset,
            model_type=args.model_type,
            model_path=args.model_path,
            num_samples=args.num_samples,
            forecast_length=args.forecast_length,
            ground_truth_prefix=args.ground_truth_prefix,
            columns_to_keep=args.columns_to_keep,
            seed=args.seed,
            data_path=args.data_path,
        )
        print(f"{result.shape=}")
        print(result.head())
        if args.output:
            if args.output.endswith(".parquet"):
                result.to_parquet(args.output, index=False)
            else:
                result.to_csv(args.output, index=False)
            print(f"Saved to {args.output}")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
