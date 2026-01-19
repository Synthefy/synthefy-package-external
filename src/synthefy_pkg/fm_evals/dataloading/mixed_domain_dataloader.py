import copy
import random
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)
from synthefy_pkg.fm_evals.dataloading.dataloader_utils import (
    add_noise,
    list_s3_files,
    lowpass_filter,
    resample_df,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler

# Generic parameter configurations grouped by common patterns
PARAMETER_CONFIGS = {
    # Short-term forecasting (daily/weekly patterns)
    "short_term": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 7,
        "forecast_window": 7,
        "num_target_rows": 7,
        "min_trajectory_length": 14,
        "resample_freq": None,
    },
    # Financial time series (monthly patterns)
    "financial_monthly": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 12,
        "forecast_window": 12,
        "num_target_rows": 96,
        "min_trajectory_length": 200,
        "resample_freq": None,
    },
    # Financial time series (long-term patterns)
    "financial_long": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 180,
        "forecast_window": 180,
        "num_target_rows": 1825,
        "min_trajectory_length": 2000,
        "resample_freq": None,
    },
    # Transportation/Weather (hourly patterns)
    "hourly_patterns": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 68,
        "forecast_window": 68,
        "num_target_rows": 68,
        "min_trajectory_length": 200,
        "resample_freq": None,
    },
    # Transportation (weekly patterns)
    "weekly_transport": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 168,
        "forecast_window": 168,
        "num_target_rows": 1000,
        "min_trajectory_length": 2000,
        "resample_freq": None,
    },
    # Transportation (large datasets)
    "large_transport": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 168,
        "forecast_window": 168,
        "num_target_rows": 4500,
        "min_trajectory_length": 5000,
        "resample_freq": None,
    },
    # Energy (daily patterns)
    "energy_daily": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 24,
        "forecast_window": 24,
        "num_target_rows": 192,
        "min_trajectory_length": 300,
        "resample_freq": "W",  # Weekly resampling
    },
    # Energy (large datasets)
    "energy_large": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 24,
        "forecast_window": 24,
        "num_target_rows": 720,  # 24 * 30
        "min_trajectory_length": 1000,
        "resample_freq": None,
    },
    # Air quality (large datasets)
    "air_quality": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 168,
        "forecast_window": 168,
        "num_target_rows": 8760,
        "min_trajectory_length": 10000,
        "resample_freq": None,
    },
    # Retail/Sales (monthly patterns)
    "retail_monthly": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 30,
        "forecast_window": 30,
        "num_target_rows": 30,
        "min_trajectory_length": 60,
        "resample_freq": None,
    },
    # Retail (bi-weekly patterns)
    "retail_biweekly": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 14,  # 2W
        "forecast_window": 14,  # 2W
        "num_target_rows": 20,
        "min_trajectory_length": 40,
        "resample_freq": None,
    },
    # Healthcare (high frequency)
    "healthcare_hf": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 288,  # 1 day of 5-minute data
        "forecast_window": 288,  # 1 day
        "num_target_rows": 576,  # 2 days
        "min_trajectory_length": 1000,
        "resample_freq": None,
    },
    # Healthcare (medium frequency)
    "healthcare_mf": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 200,
        "forecast_window": 200,
        "num_target_rows": 5000,
        "min_trajectory_length": 6000,
        "resample_freq": None,
    },
    # Sensors (medium frequency)
    "sensors": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 200,
        "forecast_window": 200,
        "num_target_rows": 2000,
        "min_trajectory_length": 3000,
        "resample_freq": None,
    },
    # Infrastructure (weekly patterns)
    "infrastructure": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 168,
        "forecast_window": 168,
        "num_target_rows": 1000,
        "min_trajectory_length": 2000,
        "resample_freq": None,
    },
    # Infrastructure (web traffic)
    "web_traffic": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 168,  # 1 week
        "forecast_window": 168,  # 1 week
        "num_target_rows": 672,  # 4 weeks
        "min_trajectory_length": 1000,
        "resample_freq": None,
    },
    # Reinforcement learning (with noise and filtering)
    "reinforcement_learning": {
        "noise_level": 0.10,
        "lowpass_window": 5,
        "stride": 128,
        "forecast_window": 128,
        "num_target_rows": 128,
        "min_trajectory_length": 500,
        "resample_freq": None,
    },
    # Computer vision/Language/Synthetic (generic)
    "generic_ml": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 100,
        "forecast_window": 100,
        "num_target_rows": 100,
        "min_trajectory_length": 200,
        "resample_freq": None,
    },
    # Economic data (monthly, long-term)
    "economic_monthly": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 60,  # 5 years
        "forecast_window": 60,  # 5 years
        "num_target_rows": 360,  # 30 years
        "min_trajectory_length": 500,
        "resample_freq": None,
    },
    # Economic data (high frequency)
    "economic_hf": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 192,  # 2 days
        "forecast_window": 192,  # 2 days
        "num_target_rows": 1024,
        "min_trajectory_length": 1500,
        "resample_freq": None,
    },
    # Economic data (daily)
    "economic_daily": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 30,
        "forecast_window": 30,
        "num_target_rows": 100,
        "min_trajectory_length": 200,
        "resample_freq": None,
    },
    # General domain - Default configuration
    "general": {
        "noise_level": 0.0,
        "lowpass_window": 0,
        "stride": 128,
        "forecast_window": 128,
        "num_target_rows": 128,
        "min_trajectory_length": 500,
        "resample_freq": None,
    },
}

# Dataset to parameter configuration mapping
DATASET_CONFIG_MAPPING = {
    # User behavior data
    "cursor_tabs": "short_term",
    # Financial time series
    "bitcoin_price": "financial_long",
    "gold_prices": "financial_monthly",
    "rice_prices": "financial_monthly",
    # Transportation data
    "traffic": "hourly_patterns",
    "blue_bikes": "large_transport",
    "paris_mobility": "weekly_transport",
    "rideshare_uber": "weekly_transport",
    "rideshare_lyft": "weekly_transport",
    # Energy market data
    "aus_electricity": "energy_daily",
    "spain_energy": "energy_large",
    "tetuan_power": "energy_large",
    "ercot_load": "energy_large",
    "mds_microgrid": "energy_large",
    # Weather data
    "weather_mpi": "hourly_patterns",
    "solar_alabama": "hourly_patterns",
    "oikolab_weather": "energy_large",
    # Air quality data
    "beijing_aq": "air_quality",
    "beijing_embassy": "air_quality",
    "open_aq": "air_quality",
    # Retail/Sales data
    "walmart_sales": "retail_monthly",
    "pasta_sales": "retail_monthly",
    "goodrx": "retail_biweekly",
    # Healthcare data
    "cgm": "healthcare_hf",
    "sleep_lab": "healthcare_mf",
    # Sensor data
    "gas_sensor": "sensors",
    "ev_sensors": "sensors",
    "voip": "sensors",
    # Manufacturing data
    "blow_molding": "sensors",
    "tac": "sensors",
    # Infrastructure data
    "austin_water": "infrastructure",
    "mta_ridership": "infrastructure",
    "web_visitors": "web_traffic",
    # Reinforcement learning data
    "mujoco_halfcheetah_v2": "reinforcement_learning",
    "mujoco_ant_v2": "reinforcement_learning",
    "mujoco_hopper_v2": "reinforcement_learning",
    "mujoco_walker2d_v2": "reinforcement_learning",
    # Computer vision data
    "cifar100": "generic_ml",
    "kitti": "generic_ml",
    # Language data
    "openwebtext": "generic_ml",
    "wikipedia": "generic_ml",
    # Synthetic data
    "spriteworld": "generic_ml",
    "gpt-synthetic": "generic_ml",
    "synthetic_medium_lag": "generic_ml",
    "complex_seasonal_timeseries": "generic_ml",
    # Economic data
    "fred_md1": "economic_monthly",
    "fred_md2": "economic_monthly",
    "fred_md3": "economic_monthly",
    "fred_md4": "economic_monthly",
    "fred_md5": "economic_monthly",
    "fred_md6": "economic_monthly",
    "fred_md7": "economic_monthly",
    "fred_md8": "economic_monthly",
    "ecl": "economic_hf",
    "stock_nasdaqtrader": "economic_daily",
    # Other datasets
    "causal_rivers": "infrastructure",
    "dynamic_data": "generic_ml",
    "mn_interstate": "infrastructure",
}

# Generic column configuration (all datasets use same generic names)
COLUMN_CONFIG = {
    "timestamp_col": "timestamp",
    "target_col": "target",
    "metadata_cols": "auto",
}

# Dataset-specific column mappings (maps original column names to generic names)
DATASET_COLUMN_MAPPINGS = {
    "cursor_tabs": {"timestamp": "Date", "target": "Tabs Accepted"},
    "bitcoin_price": {"timestamp": "date", "target": "send_usd"},
    "gold_prices": {"timestamp": "date", "target": "price"},
    "rice_prices": {"timestamp": "date", "target": "price"},
    "traffic": {
        "timestamp": "auto",
        "target": "auto",
    },  # Convert year/month/day/hour/minute to timestamp, last column as target
    "aus_electricity": {
        "timestamp": "auto",
        "target": "auto",
    },  # First column timestamp, second column target
    "weather_mpi": {"timestamp": "datetime", "target": "temperature"},
    "solar_alabama": {"timestamp": "datetime", "target": "solar_power"},
    "beijing_aq": {"timestamp": "datetime", "target": "PM2.5"},
    "walmart_sales": {"timestamp": "date", "target": "sales"},
    "cgm": {"timestamp": "timestamp", "target": "gl"},
    "gas_sensor": {
        "timestamp": "timestamp",
        "target": "auto",
    },  # Concentration columns
    "mujoco_halfcheetah_v2": {
        "timestamp": "auto",
        "target": "observations",
    },  # Last column timestamp
    "fred_md1": {"timestamp": "sasdate", "target": "RPI"},
    "fred_md2": {"timestamp": "sasdate", "target": "UNRATE"},
    "fred_md3": {"timestamp": "sasdate", "target": "HOUST"},
    "fred_md4": {"timestamp": "sasdate", "target": "UMCSENTx"},
    "fred_md5": {"timestamp": "sasdate", "target": "TOTRESNS"},
    "fred_md6": {"timestamp": "sasdate", "target": "FEDFUNDS"},
    "fred_md7": {"timestamp": "sasdate", "target": "CPIAUCSL"},
    "fred_md8": {"timestamp": "sasdate", "target": "S&P 500"},
    "ecl": {"timestamp": "datetime", "target": "MT_001"},
    # TODO: incomplete
}


# Path patterns to dataset types mapping
PATH_PATTERNS = {
    # User behavior data
    "cursor-tabs": "cursor_tabs",
    # Financial time series
    "bitcoin_price": "bitcoin_price",
    "gold_prices": "gold_prices",
    "rice_prices": "rice_prices",
    # Transportation data
    "traffic_PeMS": "traffic",
    "blue_bikes": "blue_bikes",
    "paris_mobility": "paris_mobility",
    "rideshare": "rideshare_uber",  # Default to uber for rideshare
    # Energy market data
    "aus_electricity": "aus_electricity",
    "spain_energy": "spain_energy",
    "tetuan_power": "tetuan_power",
    "ercot_load": "ercot_load",
    "mds_microgrid": "mds_microgrid",
    # Weather data
    "weather_mpi": "weather_mpi",
    "solar_alabama": "solar_alabama",
    "oikolab_weather": "oikolab_weather",
    # Air quality data
    "beijing_aq": "beijing_aq",
    "beijing_embassy": "beijing_embassy",
    "open_aq": "open_aq",
    # Retail/Sales data
    "walmart-sales": "walmart_sales",
    "pasta_sales": "pasta_sales",
    "goodrx": "goodrx",
    # Healthcare data
    "cgm": "cgm",
    "sleep_lab": "sleep_lab",
    # Sensor data
    "gas_sensor": "gas_sensor",
    "ev_sensors": "ev_sensors",
    "voip": "voip",
    # Manufacturing data
    "blow_molding": "blow_molding",
    "tac": "tac",
    # Infrastructure data
    "austin_water": "austin_water",
    "mta_ridership": "mta_ridership",
    "website_visitors": "web_visitors",
    # Reinforcement learning data
    "cheetah_csv_out": "mujoco_halfcheetah_v2",
    "ant_csv_out": "mujoco_ant_v2",
    "hopper_csv_out": "mujoco_hopper_v2",
    "walker2d_csv_out": "mujoco_walker2d_v2",
    # Computer vision data
    "cifar100": "cifar100",
    "kitti": "kitti",
    # Language data
    "openwebtext": "openwebtext",
    "wikipedia": "wikipedia",
    # Synthetic data
    "spriteworld": "spriteworld",
    "gpt-synthetic": "gpt-synthetic",
    "synthetic_medium_lag": "synthetic_medium_lag",
    "complex_seasonal_timeseries": "complex_seasonal_timeseries",
    # Economic data
    "fred_md": "fred_md1",  # Default to fred_md1
    "ecl": "ecl",
    "stock_nasdaqtrader": "stock_nasdaqtrader",
    # Other datasets
    "causal_rivers": "causal_rivers",
    "dynamic_data": "dynamic_data",
    "mn_interstate": "mn_interstate",
}

# All available dataset paths extracted from dataloader files and eval.py
ALL_DATASET_PATHS = {
    # User behavior data
    "cursor_tabs": ["s3://synthefy-fm-eval-datasets/cursor-tabs/"],
    # Financial time series
    "bitcoin_price": ["s3://synthefy-fm-eval-datasets/bitcoin_price/"],
    "gold_prices": ["s3://synthefy-fm-eval-datasets/gold_prices/"],
    "rice_prices": ["s3://synthefy-fm-eval-datasets/rice_prices/"],
    # Transportation data
    "traffic": ["s3://synthefy-fm-eval-datasets/traffic_PeMS/"],
    "blue_bikes": ["s3://synthefy-fm-eval-datasets/blue_bikes/"],
    "paris_mobility": ["s3://synthefy-fm-eval-datasets/paris_mobility/"],
    "rideshare_uber": ["s3://synthefy-fm-eval-datasets/rideshare/"],
    "rideshare_lyft": ["s3://synthefy-fm-eval-datasets/rideshare/"],
    # Energy market data
    "aus_electricity": ["s3://synthefy-fm-eval-datasets/aus_electricity/"],
    "spain_energy": ["s3://synthefy-fm-eval-datasets/spain_energy/"],
    "tetuan_power": ["s3://synthefy-fm-eval-datasets/tetuan_power/"],
    "ercot_load": ["s3://synthefy-fm-eval-datasets/ercot_load/"],
    "mds_microgrid": ["s3://synthefy-fm-eval-datasets/mds_microgrid/"],
    # Weather data
    "weather_mpi": ["s3://synthefy-fm-eval-datasets/weather_mpi/"],
    "solar_alabama": ["s3://synthefy-fm-eval-datasets/solar_alabama/"],
    "oikolab_weather": ["s3://synthefy-fm-eval-datasets/oikolab_weather/"],
    # Air quality data
    "beijing_aq": ["s3://synthefy-fm-eval-datasets/beijing_aq/"],
    "beijing_embassy": ["s3://synthefy-fm-eval-datasets/beijing_embassy/"],
    "open_aq": ["s3://synthefy-fm-eval-datasets/open_aq/"],
    # Retail/Sales data
    "walmart_sales": ["s3://synthefy-fm-eval-datasets/walmart-sales/"],
    "pasta_sales": ["s3://synthefy-fm-eval-datasets/pasta_sales/"],
    # Healthcare data
    "cgm": ["s3://synthefy-fm-eval-datasets/cgm/"],
    "sleep_lab": ["s3://synthefy-fm-eval-datasets/sleep_lab/"],
    # Sensor data
    "gas_sensor": ["s3://synthefy-fm-eval-datasets/gas_sensor/"],
    "ev_sensors": ["s3://synthefy-fm-eval-datasets/ev_sensors/"],
    "voip": ["s3://synthefy-fm-eval-datasets/voip/"],
    # Manufacturing data
    "blow_molding": ["s3://synthefy-fm-eval-datasets/blow_molding/"],
    "tac": ["s3://synthefy-fm-eval-datasets/tac/"],
    # Infrastructure data
    "austin_water": ["s3://synthefy-fm-eval-datasets/austin_water/"],
    "mta_ridership": ["s3://synthefy-fm-eval-datasets/mta_ridership/"],
    "web_visitors": ["s3://synthefy-fm-eval-datasets/web_visitors/"],
    # Reinforcement learning data
    "mujoco_halfcheetah_v2": [
        "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-random-v2/",
        "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-v2/",
        "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-replay-v2/",
        "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-medium-expert-v2/",
        "s3://synthefy-fm-eval-datasets/cheetah_csv_out/halfcheetah-expert-v2/",
    ],
    "mujoco_ant_v2": [
        "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-random-v2/",
        "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-v2/",
        "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-replay-v2/",
        "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-medium-expert-v2/",
        "s3://synthefy-fm-eval-datasets/ant_csv_out/ant-expert-v2/",
    ],
    "mujoco_hopper_v2": [
        "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-v2/",
        "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-replay-v2/",
        "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-medium-expert-v2/",
        "s3://synthefy-fm-eval-datasets/hopper_csv_out/hopper-expert-v2/",
    ],
    "mujoco_walker2d_v2": [
        "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-v2/",
        "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-replay-v2/",
        "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-medium-expert-v2/",
        "s3://synthefy-fm-eval-datasets/walker2d_csv_out/walker2d-expert-v2/",
    ],
    # Computer vision data
    "cifar100": ["s3://synthefy-fm-eval-datasets/cifar100/"],
    "kitti": ["s3://synthefy-fm-eval-datasets/kitti/"],
    # Language data
    "openwebtext": ["s3://synthefy-fm-eval-datasets/openwebtext/"],
    "wikipedia": ["s3://synthefy-fm-eval-datasets/wikipedia/"],
    # Synthetic data
    "spriteworld": ["s3://synthefy-fm-eval-datasets/spriteworld/"],
    "gpt-synthetic": ["s3://synthefy-fm-eval-datasets/gpt-synthetic/"],
    "synthetic_medium_lag": [
        "s3://synthefy-fm-eval-datasets/synthetic_medium_lag/"
    ],
    "complex_seasonal_timeseries": [
        "s3://synthefy-fm-eval-datasets/complex_seasonal_timeseries/"
    ],
    # Economic data
    "fred_md1": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md2": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md3": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md4": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md5": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md6": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md7": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "fred_md8": ["s3://synthefy-fm-eval-datasets/fred_md/"],
    "ecl": ["s3://synthefy-fm-eval-datasets/ecl/"],
    "stock_nasdaqtrader": [
        "s3://synthefy-fm-eval-datasets/stock_nasdaqtrader/"
    ],
    # Other datasets
    "goodrx": ["s3://synthefy-fm-eval-datasets/goodrx/"],
    "causal_rivers": ["s3://synthefy-fm-eval-datasets/causal_rivers/"],
    "dynamic_data": ["s3://synthefy-fm-eval-datasets/dynamic_data/"],
    "mn_interstate": ["s3://synthefy-fm-eval-datasets/mn_interstate/"],
}


class MixedDomainDataloader(BaseEvalDataloader):
    """
    A simplified dataloader that can handle mixed domain datasets.

    Similar to GeneralizedDomainDataloader but allows selecting CSV files
    from values that would have been in other dataloader init files.
    """

    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        paths: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        keep_cols: Optional[List[str]] = None,
        timestamp_col: Optional[str] = None,
        random_ordering: bool = False,
        noise_level: Optional[float] = None,
        resample_freq: Optional[str] = None,
        lowpass_window: Optional[int] = None,
        min_trajectory_length: Optional[int] = None,
        forecast_window: Optional[int] = None,
        stride: Optional[int] = None,
        num_target_rows: Optional[int] = None,
        generate_timestamps: bool = True,
        timestamp_frequency: str = "H",
        timestamp_start_date: str = "2020-01-01",
        use_all_datasets: bool = False,
        default_metadata_cols: int = 50,
        replace_metadata_with_random_ts: bool = False,
        random_ts_sampling: str = "mixed_simple",
        use_other_metadata: bool = False,
    ):
        """
        Initialize the MixedDomainDataloader.

        Parameters
        ----------
        datasets : Optional[List[str]]
            List of dataset names to load. If provided, will use presaved paths for these datasets.
            If None, paths parameter must be provided.
        paths : Optional[List[str]]
            List of S3 paths or local paths containing CSV files.
            If None, datasets parameter must be provided.
        target_cols : Optional[List[str]]
            Columns to use as targets. If None, will use dataset-specific defaults.
        keep_cols : Optional[List[str]]
            Additional columns to keep as metadata. If None, will use dataset-specific defaults.
        timestamp_col : Optional[str]
            Name of timestamp column. If None, will use dataset-specific defaults or generate.
        random_ordering : bool
            Whether to randomize file order.
        noise_level : Optional[float]
            Level of noise to add (0.0 = no noise). If None, will use dataset-specific defaults.
        resample_freq : Optional[str]
            Frequency for resampling (e.g., 'H', 'D', '5T'). If None, will use dataset-specific defaults.
        lowpass_window : Optional[int]
            Window size for low-pass filtering (0 = no filtering). If None, will use dataset-specific defaults.
        min_trajectory_length : Optional[int]
            Minimum number of rows required for a trajectory. If None, will use dataset-specific defaults.
        forecast_window : Optional[int]
            Number of rows to use for forecasting. If None, will use dataset-specific defaults.
        stride : Optional[int]
            Stride for creating evaluation batches. If None, will use dataset-specific defaults.
        num_target_rows : Optional[int]
            Number of target rows in each batch. If None, will use dataset-specific defaults.
        generate_timestamps : bool
            Whether to generate timestamps if none exist.
        timestamp_frequency : str
            Frequency for generated timestamps.
        timestamp_start_date : str
            Start date for generated timestamps.
        use_all_datasets : bool
            If True and both datasets and paths are None, all presaved dataset paths will be used.
        default_metadata_cols : int
            Default number of metadata columns to pad/truncate to. Default is 50.
        replace_metadata_with_random_ts : bool
            If True, replace metadata columns with randomly generated time series.
            Default is False.
        random_ts_sampling : str
            Type of random time series to generate. Options include:
            'mixed_simple', 'mixed_all', 'fourier', 'arima', 'wiener', etc.
            Default is 'mixed_simple'.
        use_other_metadata : bool
            If True, replace metadata columns with metadata from a different domain dataset.
            The other dataset is selected at runtime from the existing pool of CSVs.
            Default is False.
        """
        # Determine paths based on input parameters
        if datasets is not None:
            # Use presaved paths for specified datasets
            self.paths = []
            for dataset in datasets:
                if dataset in ALL_DATASET_PATHS:
                    self.paths.extend(ALL_DATASET_PATHS[dataset])
                    logger.info(
                        f"Added {len(ALL_DATASET_PATHS[dataset])} paths for dataset: {dataset}"
                    )
                else:
                    logger.warning(
                        f"Dataset '{dataset}' not found in presaved paths"
                    )
            if not self.paths:
                raise ValueError(
                    f"No valid paths found for datasets: {datasets}"
                )
        elif paths is not None:
            # Use provided paths
            self.paths = paths
        elif use_all_datasets:
            # Use all presaved paths
            self.paths = self._get_all_available_paths()
            logger.info(
                f"Using all available datasets: {len(self.paths)} paths"
            )
        else:
            raise ValueError(
                "Either 'datasets' or 'paths' must be provided, or 'use_all_datasets' must be True"
            )

        # Store user-provided columns (will be overridden by dataset-specific mappings)
        self.user_target_cols = target_cols
        self.user_keep_cols = keep_cols
        self.user_timestamp_col = timestamp_col

        # Initialize columns as None - will be set by dataset-specific mappings
        self.target_cols = None
        self.keep_cols = None
        self.timestamp_col = None
        self.random_ordering = random_ordering
        self.generate_timestamps = generate_timestamps
        self.timestamp_frequency = timestamp_frequency
        self.timestamp_start_date = timestamp_start_date
        self.default_metadata_cols = default_metadata_cols
        self.replace_metadata_with_random_ts = replace_metadata_with_random_ts
        self.random_ts_sampling = random_ts_sampling
        self.use_other_metadata = use_other_metadata

        # Store user-provided parameters (will override dataset-specific defaults)
        self.user_noise_level = noise_level
        self.user_resample_freq = resample_freq
        self.user_lowpass_window = lowpass_window
        self.user_min_trajectory_length = min_trajectory_length
        self.user_forecast_window = forecast_window
        self.user_stride = stride
        self.user_num_target_rows = num_target_rows

        # Initialize runtime parameters as None - will be set per file
        self.noise_level = None
        self.resample_freq = None
        self.lowpass_window = None
        self.min_trajectory_length = None
        self.forecast_window = None
        self.stride = None
        self.num_target_rows = None

        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)

    def _collect_files(self) -> List[str]:
        """Collect and sort all CSV files from the specified paths."""
        all_csv_files = []
        for path in self.paths:
            new_files = list_s3_files(path, file_extension=".csv")
            logger.info(f"Found {len(new_files)} CSV files in {path}")
            all_csv_files.extend(new_files)
        return sorted(all_csv_files)

    def _get_all_available_paths(self) -> List[str]:
        """Get all available dataset paths from ALL_DATASET_PATHS."""
        all_paths = []
        for dataset_name, paths in ALL_DATASET_PATHS.items():
            all_paths.extend(paths)
        return all_paths

    def _detect_dataset_type(self, file_path: str) -> str:
        """Detect dataset type based on file path patterns."""
        path_lower = file_path.lower()
        for pattern, dataset_type in PATH_PATTERNS.items():
            if pattern in path_lower:
                # logger.info(f"Detected dataset type: {dataset_type} from file: {file_path}")
                return dataset_type

        # logger.info(f"No specific dataset type detected for {file_path}, using general configuration")
        return "general"

    def _apply_dataset_config(self, dataset_type: str = "general"):
        """Apply dataset-specific configuration parameters."""
        # Get the parameter configuration name for this dataset type
        config_name = DATASET_CONFIG_MAPPING.get(dataset_type, "general")
        config = PARAMETER_CONFIGS[config_name]

        # Set parameters with dataset-specific values, user overrides take precedence
        self.noise_level = (
            self.user_noise_level
            if self.user_noise_level is not None
            else config["noise_level"]
        )
        self.lowpass_window = (
            self.user_lowpass_window
            if self.user_lowpass_window is not None
            else config["lowpass_window"]
        )
        self.stride = (
            self.user_stride
            if self.user_stride is not None
            else config["stride"]
        )
        self.forecast_window = (
            self.user_forecast_window
            if self.user_forecast_window is not None
            else config["forecast_window"]
        )
        self.num_target_rows = (
            self.user_num_target_rows
            if self.user_num_target_rows is not None
            else config["num_target_rows"]
        )
        self.min_trajectory_length = (
            self.user_min_trajectory_length
            if self.user_min_trajectory_length is not None
            else config["min_trajectory_length"]
        )
        self.resample_freq = (
            self.user_resample_freq
            if self.user_resample_freq is not None
            else config["resample_freq"]
        )

        logger.info(
            f"Applied {dataset_type} configuration (using {config_name} pattern):"
        )
        logger.info(f"  noise_level: {self.noise_level}")
        logger.info(f"  lowpass_window: {self.lowpass_window}")
        logger.info(f"  stride: {self.stride}")
        logger.info(f"  forecast_window: {self.forecast_window}")
        logger.info(f"  num_target_rows: {self.num_target_rows}")
        logger.info(f"  min_trajectory_length: {self.min_trajectory_length}")
        logger.info(f"  resample_freq: {self.resample_freq}")

    def _apply_column_config(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        loading_other_metadata: bool = False,
    ) -> pd.DataFrame:
        """Apply dataset-specific column configuration and rename to generic names."""
        # Get dataset-specific mappings
        mapping = DATASET_COLUMN_MAPPINGS.get(dataset_type, {})

        # Set timestamp column
        if self.user_timestamp_col:
            self.timestamp_col = self.user_timestamp_col
        else:
            timestamp_col = mapping.get(
                "timestamp", COLUMN_CONFIG["timestamp_col"]
            )
            if timestamp_col == "auto":
                self.timestamp_col = self._detect_timestamp_column(
                    df, dataset_type
                )
            else:
                self.timestamp_col = timestamp_col

        # Set target column (single target only)
        if self.user_target_cols:
            target_cols = self.user_target_cols
        else:
            target_col = mapping.get("target", "auto")
            if target_col == "auto":
                target_cols = self._detect_target_column(df, dataset_type)
            else:
                target_cols = [target_col]

        # Ensure only one target column
        if len(target_cols) > 1:
            logger.warning(
                f"Multiple target columns found: {target_cols}. Using first as target, moving others to metadata."
            )
            self.target_cols = [target_cols[0]]
            extra_targets = target_cols[1:]
        else:
            self.target_cols = target_cols
            extra_targets = []

        # Set metadata columns
        if self.user_keep_cols:
            self.keep_cols = self.user_keep_cols
        else:
            exclude_cols = [self.timestamp_col] + self.target_cols
            metadata_cols = [
                col for col in df.columns if col not in exclude_cols
            ]
            self.keep_cols = metadata_cols + extra_targets

        # Replace metadata with random time series if requested
        if self.replace_metadata_with_random_ts:
            logger.info(
                f"Replacing metadata with random time series using '{self.random_ts_sampling}' sampling"
            )
            self.keep_cols = self._generate_random_metadata_columns(
                df, dataset_type
            )
        elif self.use_other_metadata and not loading_other_metadata:
            logger.info(
                "Replacing metadata with columns from different domain dataset"
            )
            df = self._load_other_metadata_columns(df, dataset_type)
            self.keep_cols = [
                col
                for col in df.columns
                if col not in [self.timestamp_col, self.target_cols[0]]
            ]
        elif loading_other_metadata:
            # When loading other metadata, just use existing metadata columns
            exclude_cols = [self.timestamp_col] + self.target_cols
            self.keep_cols = [
                col for col in df.columns if col not in exclude_cols
            ]

        # Pad/truncate metadata columns to default length
        truncated = len(self.keep_cols) > self.default_metadata_cols
        if len(self.keep_cols) > self.default_metadata_cols:
            logger.warning(
                f"Metadata columns ({len(self.keep_cols)}) exceed default length ({self.default_metadata_cols}). Truncating to first {self.default_metadata_cols} columns."
            )
            self.keep_cols = self.keep_cols[: self.default_metadata_cols]
            print(df)
        elif len(self.keep_cols) < self.default_metadata_cols:
            # Pad with generic metadata column names
            padding_cols = [
                f"metadata_col_{i}"
                for i in range(len(self.keep_cols), self.default_metadata_cols)
            ]
            self.keep_cols.extend(padding_cols)
            logger.info(
                f"Padded metadata columns from {len(self.keep_cols) - len(padding_cols)} to {self.default_metadata_cols} with generic names"
            )

        # Rename columns to generic names
        df.rename(
            columns={self.timestamp_col: COLUMN_CONFIG["timestamp_col"]},
            inplace=True,
        )
        if self.target_cols:
            df.rename(
                columns={self.target_cols[0]: COLUMN_CONFIG["target_col"]},
                inplace=True,
            )

        # Rename metadata columns to generic names
        metadata_rename_map = {}
        existing_metadata_cols = [
            col
            for col in df.columns
            if col
            not in [COLUMN_CONFIG["timestamp_col"], COLUMN_CONFIG["target_col"]]
        ]
        for i, col in enumerate(existing_metadata_cols):
            if i < len(self.keep_cols):
                metadata_rename_map[self.keep_cols[i]] = f"metadata_col_{i}"
        self.keep_cols = [
            f"metadata_col_{i}" for i in range(len(self.keep_cols))
        ]

        if metadata_rename_map:
            df.rename(columns=metadata_rename_map, inplace=True)
            logger.info(
                f"Renamed metadata columns to generic names: {metadata_rename_map}"
            )

        # Add padding columns to dataframe if needed (only if not using random time series or other metadata)
        if len(self.keep_cols) > len(
            [
                col
                for col in df.columns
                if col
                not in [
                    COLUMN_CONFIG["timestamp_col"],
                    COLUMN_CONFIG["target_col"],
                ]
            ]
        ):
            existing_metadata_cols = [
                col
                for col in df.columns
                if col
                not in [
                    COLUMN_CONFIG["timestamp_col"],
                    COLUMN_CONFIG["target_col"],
                ]
            ]
            padding_cols = [
                col
                for col in self.keep_cols
                if col not in existing_metadata_cols
            ]
            for col in padding_cols:
                df[col] = 0.0  # Fill with zeros

        # Update column references
        self.timestamp_col = COLUMN_CONFIG["timestamp_col"]
        self.target_cols = [COLUMN_CONFIG["target_col"]]

        logger.info(f"Applied {dataset_type} column configuration:")
        logger.info(f"  timestamp_col: {self.timestamp_col}")
        logger.info(f"  target_cols: {self.target_cols}")
        logger.info(f"  metadata_cols: {len(self.keep_cols)} columns")
        if truncated:
            for col in df.columns:
                if col not in self.keep_cols + [
                    self.timestamp_col,
                    self.target_cols[0],
                ]:
                    df.drop(columns=[col], inplace=True)
        return df

    def _generate_random_metadata_columns(
        self, df: pd.DataFrame, dataset_type: str
    ) -> List[str]:
        """Generate random time series metadata columns using TSSampler."""
        seq_len = len(df)

        # Create TSSampler instance (limit to reasonable number for summarization)
        max_metadata_cols = min(
            self.default_metadata_cols, 50
        )  # Cap at 50 for summarization compatibility
        ts_sampler = TSSampler(
            seq_len=seq_len,
            num_features=max_metadata_cols,
            sampling=self.random_ts_sampling,
            device="cpu",
        )

        # Generate random time series
        random_ts_result = ts_sampler.sample(return_numpy=True)

        # Create metadata column names with generic naming
        metadata_cols = [f"metadata_col_{i}" for i in range(max_metadata_cols)]

        # Add the random time series data to the dataframe
        # Handle different return types from TSSampler
        import numpy as np

        if isinstance(random_ts_result, tuple):
            # If it returns a tuple, take the first element (the data)
            random_ts_data = random_ts_result[0]
        else:
            random_ts_data = random_ts_result

        # Ensure we have a 2D array
        if hasattr(random_ts_data, "shape") and len(random_ts_data.shape) == 2:
            for i, col_name in enumerate(metadata_cols):
                df[col_name] = random_ts_data[:, i]
        else:
            # Fallback: create simple random data
            for i, col_name in enumerate(metadata_cols):
                df[col_name] = np.random.randn(seq_len)

        logger.info(
            f"Generated {max_metadata_cols} random time series columns using '{self.random_ts_sampling}' sampling"
        )
        return metadata_cols

    def _load_other_metadata_columns(
        self, df: pd.DataFrame, current_dataset_type: str
    ) -> pd.DataFrame:
        """Load all metadata columns from a single different dataset file."""

        # Get files from different domains
        different_domain_files = [
            file_path
            for file_path in self.csv_files
            if self._detect_dataset_type(file_path) != current_dataset_type
        ]

        if not different_domain_files:
            logger.warning(
                "No files from different domains available for metadata. Using random time series instead."
            )
            self._generate_random_metadata_columns(df, current_dataset_type)
            return df

        # Select a random file from different domain files
        selected_file = random.choice(different_domain_files)
        selected_dataset_type = self._detect_dataset_type(selected_file)
        logger.info(
            f"Loading all metadata from file: {selected_file} (dataset: {selected_dataset_type})"
        )

        # Load the metadata file and apply dataset-specific column configuration
        metadata_df = pd.read_csv(selected_file)

        # Apply column configuration for the metadata dataset to properly identify columns
        # Pass loading_other_metadata=True to prevent recursion
        cur_ts_col, cur_target_cols = (
            copy.deepcopy(self.timestamp_col),
            copy.deepcopy(self.target_cols),
        )
        self._apply_column_config(
            metadata_df, selected_dataset_type, loading_other_metadata=True
        )
        self.timestamp_col, self.target_cols = cur_ts_col, cur_target_cols
        print("TARGET COLS", self.target_cols)

        # Get metadata columns (exclude timestamp and target columns that were identified)
        exclude_cols = [self.timestamp_col] + (self.target_cols or [])
        other_metadata_cols = [
            col for col in metadata_df.columns if col not in exclude_cols
        ]

        # Clear existing metadata columns from the current dataframe
        # Keep only timestamp and target columns
        current_exclude_cols = [self.timestamp_col] + (self.target_cols or [])
        current_metadata_cols = [
            col for col in df.columns if col not in current_exclude_cols
        ]
        for col in current_metadata_cols:
            df.drop(columns=[col], inplace=True)

        # Use all available metadata columns from the other dataset
        seq_len = len(df)
        generic_metadata_cols = [
            f"metadata_col_{i}" for i in range(len(other_metadata_cols))
        ]

        # Add metadata columns from the selected different dataset
        for i, generic_col_name in enumerate(generic_metadata_cols):
            original_col = other_metadata_cols[i]
            original_data = metadata_df[original_col]

            # Convert to numerical data
            numeric_data = pd.to_numeric(original_data, errors="coerce")

            # Skip column if all values are NaN
            if numeric_data.isna().all():
                continue

            numeric_values = np.array(numeric_data.values, dtype=float)

            # Resize data to match current dataframe length
            if len(numeric_values) >= seq_len:
                df[generic_col_name] = numeric_values[:seq_len]
            else:
                repeats = (seq_len // len(numeric_values)) + 1
                repeated_data = np.tile(numeric_values, repeats)
                df[generic_col_name] = repeated_data[:seq_len]

        logger.info(
            f"Replaced all {len(generic_metadata_cols)} metadata columns with data from dataset '{selected_dataset_type}'"
        )
        return df

    def _detect_timestamp_column(
        self, df: pd.DataFrame, dataset_type: str
    ) -> str:
        """Detect timestamp column based on dataset type and data structure."""
        if dataset_type == "traffic":
            # Traffic data has year, month, day, hour, minute columns that need to be converted
            time_cols = ["year", "month", "day", "hour", "minute"]
            if all(col in df.columns for col in time_cols):
                # Convert to timestamp
                df["timestamp"] = pd.to_datetime(df[time_cols])
                df.drop(columns=time_cols, inplace=True)
                return "timestamp"
            else:
                # Fallback to first column if time columns not found
                return df.columns[0]
        elif dataset_type == "aus_electricity":
            # First column is timestamp
            return df.columns[0]
        elif dataset_type == "mujoco_halfcheetah_v2":
            # Last column is timestamp
            return df.columns[-1]
        else:
            # Look for common timestamp column names
            timestamp_candidates = [
                "timestamp",
                "date",
                "time",
                "datetime",
                "Date",
                "Time",
            ]
            for col in timestamp_candidates:
                if col in df.columns:
                    return col
            # Default to first column
            return df.columns[0]

    def _detect_target_column(
        self, df: pd.DataFrame, dataset_type: str
    ) -> List[str]:
        """Detect single target column based on dataset type."""
        if dataset_type == "traffic":
            non_timestamp_cols = [
                col for col in df.columns if col != self.timestamp_col
            ]
            return [non_timestamp_cols[-1]] if non_timestamp_cols else []
        elif (
            dataset_type == "aus_electricity"
        ):  # TODO: aus electricity is broken
            return [df.columns[1]] if len(df.columns) > 1 else []
        elif dataset_type == "gas_sensor":
            concentration_cols = [
                col for col in df.columns if "conc" in col.lower()
            ]
            return (
                concentration_cols[:1]
                if concentration_cols
                else [df.columns[1]]
            )
        else:
            return [df.columns[1]] if len(df.columns) > 1 else []

    def _generate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic timestamps for dataframes without them."""
        start_date = pd.to_datetime(self.timestamp_start_date)
        timestamps = pd.date_range(
            start=start_date, periods=len(df), freq=self.timestamp_frequency
        )

        df = df.copy()
        df["timestamp"] = timestamps
        self.timestamp_col = "timestamp"

        logger.info(
            f"Generated timestamps from {timestamps[0]} to {timestamps[-1]}"
        )
        return df

    def _load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess a single CSV file."""
        # Load CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")

        if len(df) == 0:
            raise ValueError(f"Empty file: {file_path}")

        # Detect dataset type for this specific file
        dataset_type = self._detect_dataset_type(file_path)

        # Apply dataset-specific configuration
        self._apply_dataset_config(dataset_type)

        # Apply column configuration based on dataset type
        df = self._apply_column_config(df, dataset_type)

        # Generate timestamps if needed
        if self.generate_timestamps and not self.timestamp_col:
            df = self._generate_timestamps(df)

        # Handle timestamp column
        if self.timestamp_col and self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(
                df[self.timestamp_col], errors="coerce"
            )
            # Remove rows with invalid timestamps
            initial_rows = len(df)
            df = df.dropna(subset=[self.timestamp_col])
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.info(
                    f"Dropped {dropped_rows} rows with invalid timestamps"
                )

            # Sort by timestamp
            df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        elif self.generate_timestamps:
            df = self._generate_timestamps(df)

        # Reorder columns to put target first, then timestamp, then metadata columns
        target_col = (
            self.target_cols[0]
            if self.target_cols
            else COLUMN_CONFIG["target_col"]
        )
        timestamp_col = (
            self.timestamp_col
            if self.timestamp_col
            else COLUMN_CONFIG["timestamp_col"]
        )
        metadata_cols = [
            col for col in df.columns if col not in [target_col, timestamp_col]
        ]

        # Create new column order: target, timestamp, then metadata columns
        new_column_order = [target_col] + metadata_cols + [timestamp_col]
        print(df.columns, new_column_order)
        df = df[new_column_order]
        logger.info(
            f"Reordered columns: target first, then timestamp, then {len(metadata_cols)} metadata columns"
        )

        # Apply preprocessing
        if self.resample_freq and self.timestamp_col:
            df = resample_df(
                df,
                time_col=self.timestamp_col,
                freq=self.resample_freq,
                agg="mean",
                upsample_interpolate="linear",
                keep_non_numeric="first",
            )

        # Apply low-pass filtering
        if self.lowpass_window is not None and self.lowpass_window > 0:
            obs_cols = [
                c
                for c in df.columns
                if c in (self.keep_cols or []) + (self.target_cols or [])
            ]
            if obs_cols:
                df = lowpass_filter(
                    df,
                    columns=obs_cols,
                    method="moving_average",
                    window=self.lowpass_window,
                )

        # Add noise
        if self.noise_level is not None and self.noise_level > 0:
            obs_cols = [
                c
                for c in df.columns
                if c in (self.keep_cols or []) + (self.target_cols or [])
            ]
            if obs_cols:
                df = add_noise(
                    df,
                    columns=obs_cols,
                    dist="gaussian",
                    level=self.noise_level,
                    mode="relative",
                    random_state=42,
                )

        logger.info(
            f"Preprocessed data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def __len__(self) -> int:
        """Return the total number of files in the dataset."""
        return len(self.csv_files)

    def __iter__(self) -> Iterator[EvalBatchFormat]:
        """Yield EvalBatchFormat objects one at a time."""
        for file_path in self.csv_files:
            # Load and preprocess the data
            df = self._load_and_preprocess_data(file_path)

            if (
                self.min_trajectory_length is not None
                and len(df) < self.min_trajectory_length
            ):
                raise ValueError(
                    f"Trajectory too short: {len(df)} rows < {self.min_trajectory_length}"
                )

            # Ensure timestamp column exists
            if not self.timestamp_col or self.timestamp_col not in df.columns:
                if self.generate_timestamps:
                    df = self._generate_timestamps(df)
                else:
                    raise ValueError(
                        f"No timestamp column found in {file_path}"
                    )

            if self.timestamp_col is None:
                raise ValueError(
                    f"No timestamp column available for {file_path}"
                )

            # Ensure columns are set
            if self.target_cols is None:
                raise ValueError(f"Target columns not set for {file_path}")
            if self.keep_cols is None:
                self.keep_cols = []

            # Create evaluation batch
            eval_batch = EvalBatchFormat.from_dfs(
                dfs=[df],
                timestamp_col=self.timestamp_col,
                num_target_rows=self.num_target_rows,
                target_cols=self.target_cols,
                metadata_cols=self.keep_cols,
                leak_cols=[],
                forecast_window=self.forecast_window,
                stride=self.stride,
            )

            if eval_batch is None:
                raise RuntimeError(
                    f"Failed to create EvalBatchFormat from {file_path}"
                )

            yield eval_batch
