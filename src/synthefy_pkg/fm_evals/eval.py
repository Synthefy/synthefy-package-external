#!/usr/bin/env python3
"""
Forecasting Model Evaluation Script

This script provides argument parsing for evaluating forecasting models on specified datasets.
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from botocore.exceptions import ClientError
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.eval_utils import (
    ARGPARSE_EPILOG,
    download_checkpoint_from_s3,
    get_test_samples,
    parse_s3_url,
)
from synthefy_pkg.fm_evals.formats.dataset_result_format import (
    DatasetResultFormat,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import EvalBatchFormat
from synthefy_pkg.fm_evals.metrics.database_manager import DatabaseManager
from synthefy_pkg.fm_evals.visualizations.line_plot import plot_batch_forecasts
from synthefy_pkg.fm_evals.visualizations.plotter import Plotter, quick_analysis

COMPILE = True


def get_supported_models() -> List[str]:
    """Return list of supported model names."""
    return [
        "prophet",
        "causal_impact",
        "tabpfn_univariate",
        "tabpfn_multivariate",
        "tabpfn_future_leaked",
        "sfm_forecaster",
        "tabpfn_boosting",
        "mitra_boosting",
        "mitra_univariate",
        "mitra_multivariate",
        "mitra_future_leaked",
        "toto",
        "toto_univariate",
        "gridicl",
        "gridicl_univariate",
        "gridicl_multivariate",
        "gridicl_future_leaked",
        "chronos",
        "autoarima",
        "sarima",
        "sarimax_future_leaked",
        "mean",
        "llm",
        "gridicl_experts_univariate",
        "gridicl_experts_multivariate",
        "gridicl_experts_future_leaked",
        "routing",
        "api",
    ]


def get_supported_datasets() -> List[str]:
    """Return list of supported dataset names."""
    return [
        "fmv3",
        "traffic",
        "solar_alabama",
        "weather_mpi",
        "gift",
        "gift_sharded",
        "synthetic_medium_lag",
        "goodrx",
        "spain_energy",
        "gpt-synthetic",
        "beijing_embassy",
        "ercot_load",
        "open_aq",
        "beijing_aq",
        "cgm",
        "ppg",
        "ppg_sharded",
        "mn_interstate",
        "blow_molding",
        "tac",
        # "co2_monitor",
        "gas_sensor",
        # "news_sentiment",
        "tetuan_power",
        "paris_mobility",
        # "taiwan_aq",
        # "external",
        "aus_electricity",
        "cursor_tabs",
        "walmart_sales",
        "complex_seasonal_timeseries",
        "mta_ridership",
        "pasta_sales",
        "austin_water",
        "ny_electricity2025",
        "fl_electricity",
        "tn_electricity",
        "pa_electricity",
        "car_electricity",
        "cal_electricity",
        "tx_electricity",
        "se_electricity",
        "ne_electricity",
        "az_electricity",
        "id_electricity",
        "or_electricity",
        "central_electricity",
        "eastern_electricity",
        "western_electricity",
        "southern_electricity",
        "northern_electricity",
        "tx_daily",
        "ne_daily",
        "ny_daily",
        "az_daily",
        "cal_daily",
        "nm_daily",
        "pa_daily",
        "tn_daily",
        "co_daily",
        "car_daily",
        "al_daily",
        "rideshare_uber",
        "rideshare_lyft",
        "causal_rivers",
        "bitcoin_price",
        "oikolab_weather",
        "blue_bikes",
        "web_visitors",
        "fred_md1",
        "fred_md2",
        "fred_md3",
        "fred_md4",
        "fred_md5",
        "fred_md6",
        "fred_md7",
        "fred_md8",
        "ecl",
        "rice_prices",
        "gold_prices",
        "sleep_lab",
        "mds_microgrid",
        "voip",
        "ev_sensors",
        # "external",
        "mujoco_halfcheetah_v2",
        "mujoco_ant_v2",
        "mujoco_hopper_v2",
        "mujoco_walker2d_v2",
        "cifar100",
        "openwebtext",
        "spriteworld",
        "SCM_tiny_obslag_synin_ns",
        "SCM_tiny_convlag_synin_ns",
        "SCM_medium_obslag_synin_s",
        "SCM_medium_convlag_synin_s",
        "SCM_large_convlag_synin_s",
        "dynamic_data",
        "stock_nasdaqtrader",
        "kitti",
        "wikipedia",
        "mixed_domain",
        "periodic_anomaly_time_series_2",
        "long_duration_anomaly_time_series_59",
        "high_confidence_anomaly_time_series_3",
        "spread_anomaly_time_series_9",
    ]


def get_supported_output_types() -> List[str]:
    """Return list of supported output types."""
    return ["h5", "pkl", "csv"]


def generate_unique_id() -> str:
    """Generate a Docker-style unique ID with timestamp suffix."""
    # Generate a short UUID (first 8 characters)
    short_uuid = str(uuid.uuid4())[:8]

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{short_uuid}_{timestamp}"


def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from yaml file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}")
        sys.exit(1)


def save_run_info(
    args: argparse.Namespace,
    run_id: str,
    output_directory: str,
    model_name: str,
    dataset_result: DatasetResultFormat,
) -> None:
    """Save run information to a YAML file for reproducibility."""
    run_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "parameters": {},
    }

    # Convert args to dictionary, filtering out None values
    for key, value in vars(args).items():
        if value is not None:
            run_info["parameters"][key] = value

    # Add metrics if available
    if dataset_result.metrics is not None:
        run_info["metrics"] = {
            "mae": float(dataset_result.metrics.mae),
            "mape": float(dataset_result.metrics.mape),
        }
    else:
        run_info["metrics"] = None

    # Save to the output directory
    run_info_file = os.path.join(output_directory, f"run_info_{run_id}.yaml")
    try:
        with open(run_info_file, "w") as file:
            yaml.dump(run_info, file, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved run info to: {run_info_file}")
    except Exception as e:
        logger.error(f"Failed to save run info: {e}")


def validate_arguments(args: argparse.Namespace) -> None:
    errors = []

    supported_models = get_supported_models()
    supported_datasets = get_supported_datasets()
    supported_output_types = get_supported_output_types()

    # 1. Basic argument validation
    if not args.eval_config:
        if not args.dataset:
            errors.append(
                "Either --eval-config or --dataset (and --models, --output-directory) must be provided"
            )
        if not args.models:
            errors.append(
                "Either --eval-config or --models (and --dataset, --output-directory) must be provided"
            )
        if not args.output_directory:
            errors.append(
                "Either --eval-config or --output-directory (and --dataset, --models) must be provided"
            )

    # 2. Value validation
    if args.dataset:
        if args.dataset not in supported_datasets:
            errors.append(
                f"Unsupported dataset name: {args.dataset}. Supported datasets: {supported_datasets}"
            )
    if args.models:
        invalid_models = [
            model for model in args.models if model not in supported_models
        ]
        if invalid_models:
            errors.append(
                f"Unsupported model names: {invalid_models}. Supported models: {supported_models}"
            )
    if args.output_type:
        invalid_output_types = [
            output_type
            for output_type in args.output_type
            if output_type not in supported_output_types
        ]
        if invalid_output_types:
            errors.append(
                f"Unsupported output type(s): {invalid_output_types}. Supported output types: {supported_output_types}"
            )

    # 3. Dataset-specific dependency validation
    if args.dataset == "fmv3":
        if not args.config_file:
            errors.append("--config-file is required for fmv3 dataset")
    elif args.dataset == "traffic":
        # traffic dataset loads data directly, no additional parameters required
        pass
    elif args.dataset == "solar_alabama":
        # solar_alabama dataset loads data directly, no additional parameters required
        pass
    elif args.dataset == "weather_mpi":
        # weather_mpi dataset loads data directly, no additional parameters required
        pass
    elif args.dataset == "gift":
        if not args.data_path:
            errors.append("--data-path is required for gift dataset")
        if not args.forecast_length:
            errors.append("--forecast-length is required for gift dataset")
        if not args.sub_dataset:
            errors.append("--sub-dataset is required for gift dataset")
    elif args.dataset == "gift_sharded":
        if not args.data_path:
            errors.append("--data-path is required for gift_sharded dataset")
    elif args.dataset == "goodrx":
        # GoodRx dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "spain_energy":
        # Spain energy dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "gpt-synthetic":
        if not args.dataset_name:
            errors.append(
                "--dataset-name is required for gpt-synthetic dataset"
            )
    elif args.dataset == "stock_nasdaqtrader":
        # Stock nasdaqtrader dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "cifar100":
        # CIFAR100 dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "openwebtext":
        # OpenWebText dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "spriteworld":
        # Spriteworld dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "dynamic_data":
        # Dynamic data dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "SCM_tiny_obslag_synin_ns":
        # SCM tiny obslag synin ns dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "SCM_tiny_convlag_synin_ns":
        # SCM tiny convlag synin ns dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "SCM_medium_obslag_synin_s":
        # SCM medium obslag synin s dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "SCM_medium_convlag_synin_s":
        # SCM medium convlag synin s dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "SCM_large_convlag_synin_s":
        # SCM large convlag synin s dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "cursor_tabs":
        # Cursor tabs dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "walmart_sales":
        # Walmart sales dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "complex_seasonal_timeseries":
        # Complex seasonal timeseries dataset loads data directly from local CSV, no additional parameters required
        pass
    elif args.dataset == "kitti":
        # KITTI dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "mixed_domain":
        # Mixed domain dataset loads data directly from S3, no additional parameters required
        pass
    elif args.dataset == "external":
        if not args.external_dataloader_spec:
            errors.append(
                "--external-dataloader-spec is required for external dataset"
            )
        elif "::" not in args.external_dataloader_spec:
            errors.append(
                "--external-dataloader-spec must be in format 'path::class_name'"
            )

    # 4. Model-specific dependency validation
    if args.models and "sfm_forecaster" in args.models:
        if not args.model_checkpoint_path and not args.model_ckpt:
            errors.append(
                "--model-checkpoint-path OR --model-ckpt is required for sfm_forecaster model"
            )
        elif args.model_checkpoint_path and args.model_ckpt:
            errors.append(
                "Only one of --model-checkpoint-path OR --model-ckpt should be provided for sfm_forecaster model"
            )
        if not args.config_file:
            errors.append("--config-file is required for sfm_forecaster model")
        if not args.history_length:
            errors.append(
                "--history-length is required for sfm_forecaster model"
            )
        if not args.forecast_length:
            errors.append(
                "--forecast-length is required for sfm_forecaster model"
            )

    if args.models and (
        "sarima" in args.models or "sarimax_future_leaked" in args.models
    ):
        if not args.seasonal_period:
            errors.append(
                "--seasonal-period is required for sarima and sarimax_future_leaked models"
            )

    if args.models and "api" in args.models:
        if not args.api_model_names:
            errors.append(
                "--api-model-names is required for api model"
            )

    # 5. File path validation
    if args.config_file and not os.path.exists(args.config_file):
        errors.append(f"Config file not found: {args.config_file}")
    if args.model_checkpoint_path and not os.path.exists(
        args.model_checkpoint_path
    ):
        errors.append(
            f"Model checkpoint file not found: {args.model_checkpoint_path}"
        )

    if args.model_ckpt:
        # Validate S3 URL format
        try:
            bucket, key = parse_s3_url(args.model_ckpt)
            if not bucket or not key:
                errors.append(f"Invalid S3 URL format: {args.model_ckpt}")
        except Exception:
            errors.append(f"Invalid S3 URL format: {args.model_ckpt}")

    if args.data_path and not os.path.exists(args.data_path):
        errors.append(f"Data path not found: {args.data_path}")

    if args.output_directory:
        # Check if output directory can be created
        try:
            os.makedirs(args.output_directory, exist_ok=True)
        except (OSError, PermissionError) as e:
            errors.append(
                f"Cannot create output directory {args.output_directory}: {e}"
            )

    # Report errors and exit if any
    if errors:
        logger.error("Argument validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Forecasting Model Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=ARGPARSE_EPILOG,
    )

    parser.add_argument(
        "--eval-config",
        type=str,
        help="Path to evaluation configuration YAML file (alternative to individual arguments)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to evaluate on",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model names to evaluate",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file (required for fmv3 dataset and sfm_forecaster)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data location (required for gift dataset)",
    )

    parser.add_argument(
        "--forecast-length",
        type=int,
        help="Forecast length (required for gift dataset and sfm_forecaster)",
    )

    parser.add_argument(
        "--history-length",
        type=int,
        help="History length for gift dataset (optional for gift dataset, required for sfm_forecaster)",
    )

    parser.add_argument(
        "--sub-dataset",
        type=str,
        help="Sub-dataset name (required for gift dataset)",
    )

    parser.add_argument(
        "--filter-datasets",
        type=str,
        nargs="+",
        help="Filter by dataset names in metadata (for gift_sharded dataset)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name (required for gpt-synthetic dataset)",
    )

    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        help="Path to model checkpoint file (required for sfm_forecaster)",
    )

    parser.add_argument(
        "--model-ckpt",
        type=str,
        help="S3 URI for model checkpoint file (alternative to --model-checkpoint-path for sfm_forecaster)",
    )
    parser.add_argument(
        "--boosting-models",
        type=str,
        nargs="+",
        help="List of models to use for boosting (required when --models includes 'boosting')",
    )

    parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory to save evaluation results and plots",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        choices=get_supported_output_types(),
        nargs="+",
        default=["pkl"],
        help="Output format(s) for results (default: pkl). Can specify multiple formats like: --output-type pkl csv h5",
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Number of test samples to use for small scale evaluation",
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches to process (default: process all batches)",
    )

    # Removed --server-url: ToTo and Mitra forecasters now read URLs from env vars

    parser.add_argument(
        "--llm-model-names",
        type=str,
        nargs="+",
        default=["gemini-2.0-flash"],
        help="List of LLM model names for LLM forecaster (default: gemini-2.0-flash). Can specify multiple models like: --llm-model-names gemini-2.0-flash gemini-2.0-pro-preview-05-06",
    )

    parser.add_argument(
        "--external-dataloader-spec",
        type=str,
        help="External dataloader specification in format 'path::class_name' (required for external dataset)",
    )

    parser.add_argument(
        "--seasonal-period",
        type=int,
        help="Seasonal period for SARIMA/SARIMAX models (required for sarima and sarimax_future_leaked models)",
    )

    parser.add_argument(
        "--num-covariate-lags",
        type=int,
        default=5,
        help="Number of covariate lags to include as features for TabPFN multivariate models (default: 5)",
    )

    parser.add_argument(
        "--plot-results",
        action="store_true",
        default=False,
        help="Generate additional analysis plots (default: False)",
    )

    parser.add_argument(
        "--random-ordering",
        action="store_true",
        default=False,
        help="Randomly order the data (default: False)",
    )

    parser.add_argument(
        "--covariate-fields",
        type=str,
        nargs="+",
        default=None,
        help="Covariate field names to load from sharded dataset for multivariate evaluation. "
        "Only needed for fine-tuned Chronos models trained with covariates. "
        "Example: --covariate-fields ECG",
    )

    parser.add_argument(
        "--mixed-datasets",
        type=str,
        nargs="+",
        help="List of dataset names for mixed domain dataloader (required for mixed_domain dataset)",
    )

    parser.add_argument(
        "--replace-metadata",
        action="store_true",
        default=False,
        help="Replace metadata with random time series (default: False)",
    )

    parser.add_argument(
        "--use-other-metadata",
        action="store_true",
        default=False,
        help="Use metadata from different domain datasets (default: False)",
    )

    parser.add_argument(
        "--random-ts-sampling",
        type=str,
        default="mixed_simple",
        help="Type of random time series sampling for metadata (default: mixed_simple)",
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Base URL of the forecasting API (required for api model, e.g., http://localhost:8018)",
    )

    parser.add_argument(
        "--api-model-name",
        "--api-model-names",
        type=str,
        nargs="+",
        default=None,
        dest="api_model_names",
        help="Model alias(es) to use when calling the API (required for api model). "
        "Multiple names can be specified to compare models in the same plot. "
        "Example: --api-model-names",
    )

    args = parser.parse_args()

    if args.eval_config:
        logger.info(f"Loading configuration from: {args.eval_config}")
        config = load_eval_config(args.eval_config)
        for key, value in config.items():
            attr_name = key
            if hasattr(args, attr_name):
                setattr(args, attr_name, value)
            else:
                logger.warning(f"Unknown config key: {key}")

    validate_arguments(args)

    # Validate boosting models argument
    if (
        "tabpfn_boosting" in args.models or "mitra_boosting" in args.models
    ) and args.boosting_models:
        # Validate that all boosting models are supported
        supported_boosting_models = [
            "prophet",
            "tabpfn_multivariate",
            "tabpfn_univariate",
            "chronos",
            "causal_impact",
            "toto",
            "llm",
        ]
        invalid_boosting_models = [
            model
            for model in args.boosting_models
            if model not in supported_boosting_models
        ]
        if invalid_boosting_models:
            logger.error(
                f"Unsupported boosting model names: {invalid_boosting_models}"
            )
            logger.error(
                f"Supported boosting models: {supported_boosting_models}"
            )
            sys.exit(1)

    return args


def get_dataloader(args: argparse.Namespace):
    """Get the appropriate dataloader based on the dataset."""
    if args.dataset == "fmv3":
        from synthefy_pkg.fm_evals.dataloading.fmv3_eval_dataloader import (
            FMV3EvalDataloader,
        )

        config = Configuration(config_filepath=args.config_file)
        return FMV3EvalDataloader(config)

    elif args.dataset == "traffic":
        from synthefy_pkg.fm_evals.dataloading.traffic_pems_dataloader import (
            TrafficPEMSDataloader,
        )

        return TrafficPEMSDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "solar_alabama":
        from synthefy_pkg.fm_evals.dataloading.solar_alabama_dataloader import (
            SolarAlabamaDataloader,
        )

        return SolarAlabamaDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "weather_mpi":
        from synthefy_pkg.fm_evals.dataloading.weather_mpi_dataloader import (
            WeatherMPIDataloader,
        )

        return WeatherMPIDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "gift":
        from synthefy_pkg.fm_evals.dataloading.gift_eval_dataloader import (
            GIFTEvalUnivariateDataloader,
        )

        return GIFTEvalUnivariateDataloader(
            data_dir=args.data_path,
            forecast_length=args.forecast_length,
            history_length=args.history_length,
            random_ordering=args.random_ordering,
        )

    elif args.dataset == "gift_sharded":
        from synthefy_pkg.fm_evals.dataloading.gift_sharded_dataloader import (
            GIFTShardedDataloader,
        )

        return GIFTShardedDataloader(
            data_path=args.data_path,
            forecast_length=args.forecast_length,
            history_length=args.history_length,
            filter_datasets=args.filter_datasets
            if hasattr(args, "filter_datasets")
            else None,
            random_ordering=args.random_ordering,
            limit=args.num_batches,
        )

    elif args.dataset == "goodrx":
        from synthefy_pkg.fm_evals.dataloading.goodrx_dataloader import (
            GoodRxDataloader,
        )

        return GoodRxDataloader(random_ordering=args.random_ordering)
    elif args.dataset == "synthetic_medium_lag":
        from synthefy_pkg.fm_evals.dataloading.synthetic_medium_lag_dataloader import (
            SyntheticMediumLagDataloader,
        )

        return SyntheticMediumLagDataloader(
            args.data_path, random_ordering=args.random_ordering
        )

    elif args.dataset == "spain_energy":
        from synthefy_pkg.fm_evals.dataloading.spain_energy_dataloader import (
            SpainEnergyDataloader,
        )

        return SpainEnergyDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "gpt-synthetic":
        from synthefy_pkg.fm_evals.dataloading.gpt_synthetic_dataloader import (
            GPTSyntheticDataloader,
        )

        return GPTSyntheticDataloader(
            args.dataset_name, random_ordering=args.random_ordering
        )

    elif args.dataset == "beijing_embassy":
        from synthefy_pkg.fm_evals.dataloading.beijing_embassy_dataloader import (
            BeijingEmbassyDataloader,
        )

        return BeijingEmbassyDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "ercot_load":
        from synthefy_pkg.fm_evals.dataloading.ercot_dataloader import (
            ERCOTLoadDataloader,
        )

        return ERCOTLoadDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "open_aq":
        from synthefy_pkg.fm_evals.dataloading.openaq_dataloader import (
            OpenAQDataloader,
        )

        return OpenAQDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "beijing_aq":
        from synthefy_pkg.fm_evals.dataloading.beijing_aq_dataloader import (
            BeijingAQDataloader,
        )

        return BeijingAQDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "cgm":
        from synthefy_pkg.fm_evals.dataloading.cgm_dataloader import (
            CGMDataloader,
        )

        return CGMDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "ppg_sharded":
        from synthefy_pkg.fm_evals.dataloading.ppg_dataloader import (
            PPGShardedDataloader,
        )

        # Get covariate fields from command line (--covariate-fields)
        # Note: Only needed when evaluating fine-tuned Chronos models with covariates
        covariate_fields = getattr(args, "covariate_fields", ["ECG", "EMG", "WEIGHT", "HEIGHT", "AGE"])

        sharded_path = "~/data/ppg/sharded/processed_ppg_data_10_test"

        return PPGShardedDataloader(
            sharded_dataset_path=sharded_path,
            random_ordering=args.random_ordering,
            max_samples=args.num_batches if args.num_batches else 100,
            covariate_fields=covariate_fields,
        )

    elif args.dataset == "mn_interstate":
        from synthefy_pkg.fm_evals.dataloading.mn_interstate_dataloader import (
            MNInterstateDataloader,
        )

        return MNInterstateDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "blow_molding":
        from synthefy_pkg.fm_evals.dataloading.blow_molding_dataloader import (
            BlowMoldingDataloader,
        )

        return BlowMoldingDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "tac":
        from synthefy_pkg.fm_evals.dataloading.tac_dataloader import (
            TACDataloader,
        )

        return TACDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "gas_sensor":
        from synthefy_pkg.fm_evals.dataloading.gas_sensor_dataloader import (
            GasSensorDataloader,
        )

        return GasSensorDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "tetuan_power":
        from synthefy_pkg.fm_evals.dataloading.tetuan_power_dataloader import (
            TetuanPowerDataloader,
        )

        return TetuanPowerDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "paris_mobility":
        from synthefy_pkg.fm_evals.dataloading.paris_mobility_dataloader import (
            ParisMobilityDataloader,
        )

        return ParisMobilityDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "aus_electricity":
        from synthefy_pkg.fm_evals.dataloading.aus_electricity_dataloader import (
            AusElectricityDataloader,
        )

        return AusElectricityDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "cursor_tabs":
        from synthefy_pkg.fm_evals.dataloading.cursor_tabs_dataloader import (
            CursorTabsDataloader,
        )

        return CursorTabsDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "walmart_sales":
        from synthefy_pkg.fm_evals.dataloading.walmart_sales_dataloader import (
            WalmartSalesDataloader,
        )

        return WalmartSalesDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "complex_seasonal_timeseries":
        from synthefy_pkg.fm_evals.dataloading.complex_seasonal_timeseries_dataloader import (
            ComplexSeasonalTimeseriesDataloader,
        )

        return ComplexSeasonalTimeseriesDataloader(
            random_ordering=args.random_ordering
        )

    elif args.dataset == "mujoco_halfcheetah_v2":
        from synthefy_pkg.fm_evals.dataloading.mujoco_v2_dataloader import (
            MujocoHalfCheetahV2Dataloader,
        )

        return MujocoHalfCheetahV2Dataloader(
            random_ordering=args.random_ordering
        )
    elif args.dataset == "mujoco_ant_v2":
        from synthefy_pkg.fm_evals.dataloading.mujoco_v2_dataloader import (
            MujocoAntV2Dataloader,
        )

        return MujocoAntV2Dataloader(random_ordering=args.random_ordering)
    elif args.dataset == "mujoco_hopper_v2":
        from synthefy_pkg.fm_evals.dataloading.mujoco_v2_dataloader import (
            MujocoHopperV2Dataloader,
        )

        return MujocoHopperV2Dataloader(random_ordering=args.random_ordering)
    elif args.dataset == "mujoco_walker2d_v2":
        from synthefy_pkg.fm_evals.dataloading.mujoco_v2_dataloader import (
            MujocoWalker2dV2Dataloader,
        )

        return MujocoWalker2dV2Dataloader(random_ordering=args.random_ordering)

    elif args.dataset == "cifar100":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            CIFAR100Dataloader,
        )

        return CIFAR100Dataloader(random_ordering=args.random_ordering)
    elif args.dataset == "openwebtext":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            OpenWebTextDataloader,
        )

        return OpenWebTextDataloader(random_ordering=args.random_ordering)
    elif args.dataset == "spriteworld":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SpriteworldDataloader,
        )

        return SpriteworldDataloader(random_ordering=args.random_ordering)
    elif args.dataset == "SCM_tiny_obslag_synin_ns":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SCM_tiny_obslag_synin_nsDataloader,
        )

        return SCM_tiny_obslag_synin_nsDataloader(
            random_ordering=args.random_ordering
        )
    elif args.dataset == "SCM_tiny_convlag_synin_ns":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SCM_tiny_convlag_synin_nsDataloader,
        )

        return SCM_tiny_convlag_synin_nsDataloader(
            random_ordering=args.random_ordering
        )
    elif args.dataset == "SCM_medium_obslag_synin_s":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SCM_medium_obslag_synin_sDataloader,
        )

        return SCM_medium_obslag_synin_sDataloader(
            random_ordering=args.random_ordering
        )
    elif args.dataset == "SCM_medium_convlag_synin_s":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SCM_medium_convlag_synin_sDataloader,
        )

        return SCM_medium_convlag_synin_sDataloader(
            random_ordering=args.random_ordering
        )
    elif args.dataset == "SCM_large_convlag_synin_s":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            SCM_large_convlag_synin_sDataloader,
        )

        return SCM_large_convlag_synin_sDataloader(
            random_ordering=args.random_ordering
        )

    elif args.dataset == "dynamic_data":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            DynamicDataDataloader,
        )

        return DynamicDataDataloader(random_ordering=args.random_ordering)
    elif args.dataset == "external":
        from synthefy_pkg.fm_evals.dataloading.external_dataloader import (
            ExternalDataloader,
        )

        if not args.external_dataloader_spec:
            raise ValueError(
                "--external-dataloader-spec is required for external dataset"
            )

        config = Configuration()
        return ExternalDataloader(config, args.external_dataloader_spec)

    elif args.dataset == "aus_electricity":
        from synthefy_pkg.fm_evals.dataloading.aus_electricity_dataloader import (
            AusElectricityDataloader,
        )

        return AusElectricityDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "mta_ridership":
        from synthefy_pkg.fm_evals.dataloading.mta_ridership_dataloader import (
            MTARidershipDataloader,
        )

        return MTARidershipDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "pasta_sales":
        from synthefy_pkg.fm_evals.dataloading.pasta_sales_dataloader import (
            PastaSalesDataloader,
        )

        return PastaSalesDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "austin_water":
        from synthefy_pkg.fm_evals.dataloading.austin_water_dataloader import (
            AustinWaterDataloader,
        )

        return AustinWaterDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "ny_electricity2025":
        from synthefy_pkg.fm_evals.dataloading.ny_electricity_dataloader import (
            NYElectricityDataloader,
        )

        return NYElectricityDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "fl_electricity":
        from synthefy_pkg.fm_evals.dataloading.fl_electricity_dataloader import (
            FLElectricityDataloader,
        )

        return FLElectricityDataloader(random_ordering=args.random_ordering)
    elif args.dataset == "stock_nasdaqtrader":
        from synthefy_pkg.fm_evals.dataloading.general_domain_dataloader import (
            NasdaqTraderDataloader,
        )

        return NasdaqTraderDataloader(random_ordering=args.random_ordering)

    elif args.dataset in [
        "tn_electricity",
        "pa_electricity",
        "car_electricity",
        "cal_electricity",
        "tx_electricity",
        "se_electricity",
        "ne_electricity",
        "az_electricity",
        "id_electricity",
        "or_electricity",
    ]:
        from synthefy_pkg.fm_evals.dataloading.hourly_electricity_dataloader import (
            HourlyElectricityDataloader,
        )

        return HourlyElectricityDataloader(
            args.dataset, random_ordering=args.random_ordering
        )

    elif args.dataset in [
        "central_electricity",
        "eastern_electricity",
        "western_electricity",
        "southern_electricity",
        "northern_electricity",
    ]:
        from synthefy_pkg.fm_evals.dataloading.europe_electricity_dataloader import (
            EuropeElectricityDataloader,
        )

        return EuropeElectricityDataloader(
            args.dataset, random_ordering=args.random_ordering
        )

    elif args.dataset in [
        "tx_daily",
        "ne_daily",
        "ny_daily",
        "az_daily",
        "cal_daily",
        "nm_daily",
        "pa_daily",
        "tn_daily",
        "co_daily",
        "car_daily",
        "al_daily",
    ]:
        from synthefy_pkg.fm_evals.dataloading.daily_electricity_dataloader import (
            DailyElectricityDataloader,
        )

        return DailyElectricityDataloader(
            args.dataset, random_ordering=args.random_ordering
        )

    elif args.dataset == "tetuan_power":
        from synthefy_pkg.fm_evals.dataloading.tetuan_power_dataloader import (
            TetuanPowerDataloader,
        )

        return TetuanPowerDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "paris_mobility":
        from synthefy_pkg.fm_evals.dataloading.paris_mobility_dataloader import (
            ParisMobilityDataloader,
        )

        return ParisMobilityDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "rideshare_uber" or args.dataset == "rideshare_lyft":
        from synthefy_pkg.fm_evals.dataloading.rideshare_dataloader import (
            RideshareDataloader,
        )

        return RideshareDataloader(
            args.dataset, random_ordering=args.random_ordering
        )

    elif args.dataset == "causal_rivers":
        from synthefy_pkg.fm_evals.dataloading.causal_rivers_dataloader import (
            CausalRiversDataloader,
        )

        return CausalRiversDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "bitcoin_price":
        from synthefy_pkg.fm_evals.dataloading.bitcoin_price_dataloader import (
            BitcoinPriceDataloader,
        )

        return BitcoinPriceDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "oikolab_weather":
        from synthefy_pkg.fm_evals.dataloading.oikolab_weather_dataloader import (
            OikolabWeatherDataloader,
        )

        return OikolabWeatherDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "blue_bikes":
        from synthefy_pkg.fm_evals.dataloading.blue_bikes_dataloader import (
            BlueBikesDataloader,
        )

        return BlueBikesDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "web_visitors":
        from synthefy_pkg.fm_evals.dataloading.web_visitors_dataloader import (
            WebVisitorsDataloader,
        )

        return WebVisitorsDataloader(random_ordering=args.random_ordering)

    elif args.dataset in [
        "fred_md1",
        "fred_md2",
        "fred_md3",
        "fred_md4",
        "fred_md5",
        "fred_md6",
        "fred_md7",
        "fred_md8",
    ]:
        from synthefy_pkg.fm_evals.dataloading.fred_md_dataloader import (
            FredMdDataloader,
        )

        return FredMdDataloader(
            args.dataset, random_ordering=args.random_ordering
        )

    elif args.dataset == "ecl":
        from synthefy_pkg.fm_evals.dataloading.ecl_dataloader import (
            ECLDataloader,
        )

        return ECLDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "rice_prices":
        from synthefy_pkg.fm_evals.dataloading.rice_prices_dataloader import (
            RicePricesDataloader,
        )

        return RicePricesDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "gold_prices":
        from synthefy_pkg.fm_evals.dataloading.gold_prices_dataloader import (
            GoldPricesDataloader,
        )

        return GoldPricesDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "sleep_lab":
        from synthefy_pkg.fm_evals.dataloading.sleep_lab_dataloader import (
            SleepLabDataloader,
        )

        return SleepLabDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "mds_microgrid":
        from synthefy_pkg.fm_evals.dataloading.mds_microgrid_dataloader import (
            MdsMicrogridDataloader,
        )

        return MdsMicrogridDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "voip":
        from synthefy_pkg.fm_evals.dataloading.voip_dataloader import (
            VoIPDataloader,
        )

        return VoIPDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "ev_sensors":
        from synthefy_pkg.fm_evals.dataloading.ev_sensors_dataloaders import (
            EVSensorsDataloader,
        )

        return EVSensorsDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "kitti":
        from synthefy_pkg.fm_evals.dataloading.kitti_dataloader import (
            KITTIDataloader,
        )

        return KITTIDataloader(random_ordering=args.random_ordering)

    elif args.dataset == "wikipedia":
        from synthefy_pkg.fm_evals.dataloading.wikipedia_dataloader import (
            WikipediaDataloader,
        )

        return WikipediaDataloader()

    elif args.dataset == "mixed_domain":
        from synthefy_pkg.fm_evals.dataloading.mixed_domain_dataloader import (
            MixedDomainDataloader,
        )

        return MixedDomainDataloader(
            datasets=args.mixed_datasets,
            replace_metadata_with_random_ts=args.replace_metadata,
            use_other_metadata=args.use_other_metadata,
            random_ts_sampling=args.random_ts_sampling,
        )

    elif args.dataset == "periodic_anomaly_time_series_2":
        from synthefy_pkg.fm_evals.dataloading.cisco_anomaly_dataloader import (
            PeriodicAnomalyTimeSeries2Dataloader,
        )

        return PeriodicAnomalyTimeSeries2Dataloader(
            random_ordering=args.random_ordering,
            use_first_half=False,  # Use last 50% for inference
            history_length=args.history_length if args.history_length else 192,
            prediction_length=args.forecast_length
            if args.forecast_length
            else 1,
            stride=1,
        )

    elif args.dataset == "long_duration_anomaly_time_series_59":
        from synthefy_pkg.fm_evals.dataloading.cisco_anomaly_dataloader import (
            LongDurationAnomalyTimeSeries59Dataloader,
        )

        return LongDurationAnomalyTimeSeries59Dataloader(
            random_ordering=args.random_ordering,
            use_first_half=False,  # Use last 50% for inference
            history_length=args.history_length if args.history_length else 192,
            prediction_length=args.forecast_length
            if args.forecast_length
            else 1,
            stride=1,
        )

    elif args.dataset == "high_confidence_anomaly_time_series_3":
        from synthefy_pkg.fm_evals.dataloading.cisco_anomaly_dataloader import (
            HighConfidenceAnomalyTimeSeries3Dataloader,
        )

        return HighConfidenceAnomalyTimeSeries3Dataloader(
            random_ordering=args.random_ordering,
            use_first_half=False,  # Use last 50% for inference
            history_length=args.history_length if args.history_length else 192,
            prediction_length=args.forecast_length
            if args.forecast_length
            else 1,
            stride=1,
        )

    elif args.dataset == "spread_anomaly_time_series_9":
        from synthefy_pkg.fm_evals.dataloading.cisco_anomaly_dataloader import (
            SpreadAnomalyTimeSeries9Dataloader,
        )

        return SpreadAnomalyTimeSeries9Dataloader(
            random_ordering=args.random_ordering,
            use_first_half=False,  # Use last 50% for inference
            history_length=args.history_length if args.history_length else 192,
            prediction_length=args.forecast_length
            if args.forecast_length
            else 1,
            stride=1,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_forecasting_model(model_name: str, args: argparse.Namespace):
    if model_name == "prophet":
        from synthefy_pkg.fm_evals.forecasting.prophet_forecaster import (
            ProphetForecaster,
        )

        return ProphetForecaster()

    elif model_name == "causal_impact":
        from synthefy_pkg.fm_evals.forecasting.causal_impact_forecaster import (
            CausalImpactForecaster,
        )

        return CausalImpactForecaster()

    elif model_name == "tabpfn_univariate":
        from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
            TabPFNUnivariateForecaster,
        )

        return TabPFNUnivariateForecaster()

    elif model_name == "tabpfn_multivariate":
        from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
            TabPFNMultivariateForecaster,
        )

        return TabPFNMultivariateForecaster(
            future_leak=False,
            individual_correlate_timestamps=False,
            num_covariate_lags=args.num_covariate_lags,
            add_running_index=False,
            add_calendar_features=True,
        )

    elif model_name == "tabpfn_future_leaked":
        from synthefy_pkg.fm_evals.forecasting.tabpfn_forecaster import (
            TabPFNMultivariateForecaster,
        )

        return TabPFNMultivariateForecaster(
            future_leak=True,
            individual_correlate_timestamps=False,
            num_covariate_lags=args.num_covariate_lags,
            add_running_index=False,
            add_calendar_features=True,
        )

    elif model_name == "sfm_forecaster":
        from synthefy_pkg.fm_evals.forecasting.sfm_forecaster import (
            SFMForecaster,
        )

        model_checkpoint_path = args.model_checkpoint_path
        if args.model_ckpt:
            model_checkpoint_path = download_checkpoint_from_s3(args.model_ckpt)
            # Track downloaded checkpoint for cleanup
            args._downloaded_checkpoint = model_checkpoint_path

        return SFMForecaster(
            model_checkpoint_path=model_checkpoint_path,
            config_path=args.config_file,
            history_length=args.history_length,
            forecast_length=args.forecast_length,
        )
    elif model_name == "tabpfn_boosting":
        from synthefy_pkg.fm_evals.forecasting.boosting_forecaster import (
            TabPFNBoostingForecaster,
        )

        # Use default if no boosting_models provided
        if args.boosting_models is None:
            return TabPFNBoostingForecaster()
        else:
            return TabPFNBoostingForecaster(
                boosting_models=args.boosting_models
            )
    elif model_name == "mitra_boosting":
        from synthefy_pkg.fm_evals.forecasting.boosting_forecaster import (
            MitraBoostingForecaster,
        )

        # Use default if no boosting_models provided
        if args.boosting_models is None:
            return MitraBoostingForecaster()
        else:
            return MitraBoostingForecaster(boosting_models=args.boosting_models)
    elif model_name == "toto":
        from synthefy_pkg.fm_evals.forecasting.toto_forecaster import (
            TotoForecaster,
        )

        return TotoForecaster()
    elif model_name == "mitra_univariate":
        from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import (
            MitraForecaster,
        )

        return MitraForecaster(multivariate=False, future_leak=False)
    elif model_name == "mitra_multivariate":
        from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import (
            MitraForecaster,
        )

        return MitraForecaster(multivariate=True, future_leak=False)
    elif model_name == "mitra_future_leaked":
        from synthefy_pkg.fm_evals.forecasting.mitra_forecaster import (
            MitraForecaster,
        )

        return MitraForecaster(multivariate=True, future_leak=True)
    elif model_name == "toto_univariate":
        from synthefy_pkg.fm_evals.forecasting.toto_forecaster import (
            TotoUnivariateForecaster,
        )

        return TotoUnivariateForecaster()
    elif (
        model_name == "gridicl_univariate"
        or model_name == "gridicl_multivariate"
        or model_name == "gridicl_future_leaked"
    ):
        from synthefy_pkg.fm_evals.forecasting.gridicl_forecaster import (
            GridICLForecaster,
        )

        return GridICLForecaster(
            model_checkpoint_path=args.model_checkpoint_path,
            history_length=args.history_length,
            forecast_length=args.forecast_length,
            name=model_name,
        )
    elif (
        model_name == "gridicl_experts_univariate"
        or model_name == "gridicl_experts_multivariate"
        or model_name == "gridicl_experts_future_leaked"
    ):
        from synthefy_pkg.fm_evals.forecasting.gridicl_avec_experts_forecaster import (
            GridICLAvecExpertsForecaster,
        )

        return GridICLAvecExpertsForecaster(
            model_checkpoint_path=args.model_checkpoint_path,
            config=args.config_file,
            history_length=args.history_length,
            forecast_length=args.forecast_length,
            name=model_name,
            use_aux_features=True,
        )
    elif model_name == "chronos":
        from synthefy_pkg.fm_evals.forecasting.chronos_forecaster import (
            ChronosForecaster,
        )

        return ChronosForecaster()
    elif model_name == "autoarima":
        from synthefy_pkg.fm_evals.forecasting.autoarima_forecaster import (
            AutoARIMAForecaster,
        )

        return AutoARIMAForecaster()
    elif model_name == "sarima":
        from synthefy_pkg.fm_evals.forecasting.sarimax_forecaster import (
            SARIMAXForecaster,
        )

        if not args.seasonal_period:
            raise ValueError("--seasonal-period is required for sarima model")

        return SARIMAXForecaster(
            seasonal_period=args.seasonal_period, future_leaked=False
        )
    elif model_name == "sarimax_future_leaked":
        from synthefy_pkg.fm_evals.forecasting.sarimax_forecaster import (
            SARIMAXForecaster,
        )

        if not args.seasonal_period:
            raise ValueError(
                "--seasonal-period is required for sarimax_future_leaked model"
            )

        return SARIMAXForecaster(
            seasonal_period=args.seasonal_period, future_leaked=True
        )
    elif model_name == "mean":
        from synthefy_pkg.fm_evals.forecasting.mean_forecaster import (
            MeanForecaster,
        )

        return MeanForecaster()
    elif model_name == "llm":
        from synthefy_pkg.fm_evals.forecasting.llm_forecaster import (
            AzureOpenAIForecaster,
            GeminiForecaster,
        )

        # Get LLM model name if provided, otherwise use default
        llm_model_name = getattr(args, "llm_model_name", "gemini-2.0-flash")

        # Determine which concrete class to use based on model name
        model_lower = llm_model_name.lower()
        if "gemini" in model_lower:
            return GeminiForecaster(
                model_name=llm_model_name,
                temperature=0.0,  # Use default
                max_retries=2,  # Use default
            )
        elif "gpt" in model_lower or "openai" in model_lower:
            return AzureOpenAIForecaster(
                model_name=llm_model_name,
                temperature=0.0,  # Use default
                max_retries=2,  # Use default
            )
        else:
            raise ValueError(
                f"Model '{llm_model_name}' is not supported. Supported models include Gemini models (gemini-*) and OpenAI models (gpt-*)"
            )
    elif model_name == "routing":
        from synthefy_pkg.fm_evals.forecasting.routing_forecaster import (
            RoutingForecaster,
        )

        # Defaults inside RoutingForecaster handle model options and criterion
        return RoutingForecaster()
    elif model_name == "api":
        from synthefy_pkg.fm_evals.forecasting.api.api_forecaster import (
            APIForecaster,
        )

        if not args.api_model_names:
            raise ValueError(
                "--api-model-names is required for api model"
            )

        # For backwards compatibility, use the first model name
        return APIForecaster(
            model_name=args.api_model_names[0],
            server_url=args.server_url,
        )
    elif model_name.startswith("api_"):
        # Handle expanded API model names
        from synthefy_pkg.fm_evals.forecasting.api.api_forecaster import (
            APIForecaster,
        )

        # Extract the actual model name from api_<model_name>
        api_model_name = model_name[4:]  # Remove "api_" prefix
        return APIForecaster(
            model_name=api_model_name,
            server_url=args.server_url,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def compare_models(model_results: Dict[str, Dict], dataset_name: str):
    logger.info("METRICS")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of models evaluated: {len(model_results)}")
    logger.info("")

    # Print individual model results
    for model_name, result_info in model_results.items():
        dataset_result = result_info["dataset_result"]

        logger.info(f"Model: {model_name}")
        if dataset_result.metrics is not None:
            logger.info(str(dataset_result))
        else:
            logger.warning("No metrics available")
        logger.info("")

    # moved metric-specific logic to the DatasetResultFormat class
    best_models = DatasetResultFormat.find_best_models(model_results)

    if not best_models:
        logger.warning("No metrics available for any model")
        return

    for metric_name, best_info in best_models.items():
        logger.info(
            f"Best {metric_name.upper()}: {best_info['model_name']} ({metric_name.upper()}: {best_info['value']:.6f})"
        )


def get_effective_dataset_name(args: argparse.Namespace) -> str:
    """Get the effective dataset name for database storage and logging.

    For gift datasets, combines the main dataset with sub-dataset (e.g., "gift_traffic").
    For gpt-synthetic datasets, combines the main dataset with dataset-name (e.g., "gpt-synthetic/retail").
    For other datasets, returns the dataset name as-is.
    """
    if args.dataset == "gift" and args.sub_dataset:
        return f"{args.dataset}/{args.sub_dataset}"
    elif args.dataset == "gpt-synthetic" and args.dataset_name:
        return f"{args.dataset}/{args.dataset_name}"
    return args.dataset


def main():
    args = parse_arguments()

    # Generate unique ID for this run
    run_id = generate_unique_id()
    logger.info(f"Run ID: {run_id}")

    effective_dataset_name = get_effective_dataset_name(args)
    logger.info(f"Dataset: {effective_dataset_name}")
    logger.info(f"Original models: {args.models}")

    # Display conditional arguments based on dataset
    if args.dataset == "fmv3":
        logger.info(f"Config file: {args.config_file}")
    elif args.dataset == "traffic":
        logger.info("Using traffic dataset (loading data directly)")
    elif args.dataset == "solar_alabama":
        logger.info("Using solar_alabama dataset (loading data directly)")
    elif args.dataset == "weather_mpi":
        logger.info("Using weather_mpi dataset (loading data directly)")
    elif args.dataset == "gift":
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Sub-dataset: {args.sub_dataset}")
        logger.info(f"Forecast length: {args.forecast_length}")
        if args.history_length:
            logger.info(f"History length: {args.history_length}")
    elif args.dataset == "gift_sharded":
        logger.info(f"Using gift_sharded dataset from: {args.data_path}")
        if args.forecast_length:
            logger.info(f"Forecast length: {args.forecast_length}")
        if args.history_length:
            logger.info(f"History length: {args.history_length}")
        if args.filter_datasets:
            logger.info(f"Filtering datasets: {args.filter_datasets}")
    elif args.dataset == "goodrx":
        logger.info("Using GoodRx dataset (loading from S3)")
    elif args.dataset == "spain_energy":
        logger.info("Using Spain energy dataset (loading from S3)")
    elif args.dataset == "gpt-synthetic":
        logger.info(f"Using GPT synthetic dataset: {args.dataset_name}")
    elif args.dataset == "cursor_tabs":
        logger.info("Using cursor_tabs dataset (loading from S3)")
    elif args.dataset == "walmart_sales":
        logger.info("Using walmart_sales dataset (loading from S3)")
    elif args.dataset == "complex_seasonal_timeseries":
        logger.info(
            "Using complex_seasonal_timeseries dataset (loading from local CSV)"
        )
    elif args.dataset == "external":
        logger.info(
            f"Using external dataloader: {args.external_dataloader_spec}"
        )

    # Display checkpoint information
    if args.model_ckpt:
        logger.info(f"Model checkpoint S3 URL: {args.model_ckpt}")
    elif args.model_checkpoint_path:
        logger.info(f"Model checkpoint path: {args.model_checkpoint_path}")

    # Create output directory structure
    if args.dataset == "gift":
        base_output_dir = os.path.join(
            args.output_directory, args.dataset, args.sub_dataset
        )
    elif args.dataset == "gpt-synthetic":
        base_output_dir = os.path.join(
            args.output_directory, args.dataset, args.dataset_name
        )
    else:
        base_output_dir = os.path.join(args.output_directory, args.dataset)
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info(f"Output directory: {base_output_dir}")

    # Create run-specific directory
    run_output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    dataloader = get_dataloader(args)

    # Store results for each model for comparison
    model_results = {}

    # Initialize models and their results
    models = {}
    dataset_results = {}
    expanded_model_names = []

    for model_name in args.models:
        if model_name == "llm":
            # Expand "llm" into multiple models based on llm_model_names
            for llm_model_name in args.llm_model_names:
                expanded_name = (
                    f"llm_{llm_model_name.replace('-', '_').replace('.', '_')}"
                )
                expanded_model_names.append(expanded_name)
                logger.info(
                    f"Initializing LLM model: {expanded_name} (using {llm_model_name})"
                )

                # Create a temporary args object with the specific LLM model name
                temp_args = argparse.Namespace()
                for attr, value in vars(args).items():
                    setattr(temp_args, attr, value)
                temp_args.llm_model_name = llm_model_name  # Use the old attribute name for compatibility

                models[expanded_name] = get_forecasting_model("llm", temp_args)
                dataset_results[expanded_name] = DatasetResultFormat()

                # Create model-specific directory within run directory
                model_output_dir = os.path.join(run_output_dir, expanded_name)
                os.makedirs(model_output_dir, exist_ok=True)

                # Store results for this model
                model_results[expanded_name] = {
                    "dataset_result": dataset_results[expanded_name],
                    "output_dir": model_output_dir,
                }
        elif model_name == "api":
            # Expand "api" into multiple models based on api_model_names
            for api_model_name in args.api_model_names:
                expanded_name = f"api_{api_model_name}"
                expanded_model_names.append(expanded_name)
                logger.info(
                    f"Initializing API model: {expanded_name} (using {api_model_name})"
                )

                models[expanded_name] = get_forecasting_model(
                    expanded_name, args
                )
                dataset_results[expanded_name] = DatasetResultFormat()

                # Create model-specific directory within run directory
                model_output_dir = os.path.join(run_output_dir, expanded_name)
                os.makedirs(model_output_dir, exist_ok=True)

                # Store results for this model
                model_results[expanded_name] = {
                    "dataset_result": dataset_results[expanded_name],
                    "output_dir": model_output_dir,
                }
        else:
            # Handle other models as before
            expanded_model_names.append(model_name)
            logger.info(f"Initializing model: {model_name}")
            models[model_name] = get_forecasting_model(model_name, args)
            dataset_results[model_name] = DatasetResultFormat()

            # Create model-specific directory within run directory
            model_output_dir = os.path.join(run_output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Store results for this model
            model_results[model_name] = {
                "dataset_result": dataset_results[model_name],
                "output_dir": model_output_dir,
            }

    # Log the expanded model names
    logger.info(f"Expanded models: {expanded_model_names}")

    # Create plots directory within run directory
    plots_dir = os.path.join(run_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Determine total number of batches for progress bar
    total_batches = None
    if args.num_batches:
        total_batches = args.num_batches
    else:
        # Try to get total count from dataloader if available
        try:
            total_batches = len(dataloader)
        except (TypeError, AttributeError):
            # If dataloader doesn't support len(), we'll use None for indeterminate progress
            pass

    # Iterate over batches first, then run each model on that batch
    for batch_idx, batch in enumerate(
        tqdm(
            dataloader, total=total_batches, desc="eval.py: Processing batches"
        )
    ):
        if batch is None:
            logger.warning(f"Skipping batch {batch_idx} - no data available")
            continue

        if args.test_samples:
            batch = get_test_samples(batch, test_samples=args.test_samples)
        logger.info(f"Processing batch {batch_idx}")

        # Store predictions for all models on this batch
        batch_predictions = {}

        # Run each model on this batch
        for model_name, model in models.items():
            logger.info(f"Running {model_name} on batch {batch_idx}")
            model.fit(batch)
            predictions = model.predict(batch)
            dataset_results[model_name].add_batch(batch, predictions)
            batch_predictions[model_name] = predictions

        # Create plot for this batch with all model predictions
        plot_file = os.path.join(plots_dir, f"{batch_idx}.pdf")
        # Convert dictionary of predictions to list for plot_batch_forecasts
        predictions_list = list(batch_predictions.values())
        plot_batch_forecasts(batch, predictions_list, plot_file)
        logger.info(f"Saved plot to: {plot_file}")

        # Save results and print metrics after every batch
        logger.info(
            f"Batch {batch_idx} completed. Saving results and printing metrics..."
        )

        # Save results for each model after this batch
        for model_name in expanded_model_names:
            # Save results in all requested formats
            for output_type in args.output_type:
                if output_type == "pkl":
                    results_file = os.path.join(
                        model_results[model_name]["output_dir"],
                        f"{model_name}_dataset_results.pkl",
                    )
                    dataset_results[model_name].save_pkl(results_file)

                elif output_type == "h5":
                    h5_results_file = os.path.join(
                        model_results[model_name]["output_dir"],
                        f"{model_name}_dataset_results.h5",
                    )
                    dataset_results[model_name].save_h5(h5_results_file)

                elif output_type == "csv":
                    csv_results_file = os.path.join(
                        model_results[model_name]["output_dir"],
                        f"{model_name}_dataset_results.csv",
                    )
                    dataset_results[model_name].save_csv(csv_results_file)

            # Save run info for this model
            save_run_info(
                args,
                run_id,
                model_results[model_name]["output_dir"],
                model_name,
                dataset_results[model_name],
            )

        # Print current metrics comparison after this batch
        logger.info(f"Metrics after batch {batch_idx}:")
        compare_models(model_results, get_effective_dataset_name(args))
        logger.info("=" * 50)

        if args.num_batches and batch_idx >= args.num_batches - 1:
            logger.info(f"Processed {args.num_batches} batches, stopping")
            break

    # Final metrics comparison
    logger.info("Final metrics comparison:")
    compare_models(model_results, get_effective_dataset_name(args))

    # Save results to database
    logger.info("Saving results to database...")
    database_manager = DatabaseManager(args.output_directory)
    for model_name in expanded_model_names:
        database_manager.save_results(
            dataset_name=get_effective_dataset_name(args),
            model_name=model_name,
            dataset_result=dataset_results[model_name],
            run_id=run_id,
            results_path=model_results[model_name]["output_dir"],
            git_hash=None,  # TODO: implement git hash functionality
        )
    logger.info(
        f"Results saved to database: {database_manager.get_database_path()} (run_id: {run_id})"
    )

    if args.plot_results:
        dataset_results = []
        for model_name, result_info in model_results.items():
            dataset_results.append(result_info["dataset_result"])

        plt = Plotter.from_dataset_results(dataset_results)

        plt_output_dir = os.path.join(base_output_dir, run_id)
        os.makedirs(plt_output_dir, exist_ok=True)

        # Create quick analysis plots in the same output directory
        quick_analysis_plots = quick_analysis(
            dataframes=plt.dataframes, output_dir=plt_output_dir, metric="mae"
        )

        quick_analysis_plots = quick_analysis(
            dataframes=plt.dataframes, output_dir=plt_output_dir, metric="mape"
        )

        plt.create_sample_analysis_plots(
            output_path=plt_output_dir + "/mae_sample_analysis.pdf",
            figsize=(15, 10),
            seed=42,
            metric="mae",
            random_samples=2,
            median_samples=2,
            best_samples=2,
            worst_samples=2,
        )

        plt.create_sample_analysis_plots(
            output_path=plt_output_dir + "/mape_sample_analysis.pdf",
            figsize=(15, 10),
            seed=42,
            metric="mape",
            random_samples=2,
            median_samples=2,
            best_samples=2,
            worst_samples=2,
        )

        plt.create_metric_distribution_plot(
            output_path=plt_output_dir + "/mae_metric_distribution.png",
            figsize=(12, 8),
            metric="mae",
            bins=10,
        )

        plt.create_metric_distribution_plot(
            output_path=plt_output_dir + "/mae_metric_distribution.png",
            figsize=(12, 8),
            metric="mape",
            bins=10,
        )

        logger.info(
            f"Generated quick analysis plots: {list(quick_analysis_plots.keys())}"
        )

    # Cleanup downloaded checkpoint file if any
    if hasattr(args, "_downloaded_checkpoint"):
        try:
            if os.path.exists(args._downloaded_checkpoint):
                os.unlink(args._downloaded_checkpoint)
                logger.info(
                    f"Cleaned up downloaded checkpoint: {args._downloaded_checkpoint}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to cleanup checkpoint {args._downloaded_checkpoint}: {e}"
            )


if __name__ == "__main__":
    main()
