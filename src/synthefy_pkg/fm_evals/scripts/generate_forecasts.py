#!/usr/bin/env python3
"""
Forecast Generation Script

This script generates forecasts for specified datasets and models, storing results
in both S3 and locally for testing purposes. Same functionality and usage as eval.py.
Stored in s3://synthefy-fm-dataset-forecasts for future reference.

Usage:
python generate_forecasts.py --models prophet causal_impact --datasets aus_electricity traffic rideshare_uber
include model or dataset specific arguments as needed
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

# Import functions from eval.py to reuse existing code
from synthefy_pkg.fm_evals.eval import (
    generate_unique_id,
    get_dataloader,
    get_effective_dataset_name,
    get_forecasting_model,
    get_supported_datasets,
    get_supported_models,
)
from synthefy_pkg.fm_evals.eval_utils import parse_s3_url
from synthefy_pkg.fm_evals.formats.dataset_result_format import (
    DatasetResultFormat,
)
from synthefy_pkg.fm_evals.scripts.utils import (
    get_s3_client,
    read_existing_forecasts_from_s3,
    read_existing_forecasts_locally,
    save_forecasts_locally,
    write_forecasts_to_s3,
)


def load_seasonal_periods_mapping() -> Dict[str, int]:
    """Load the seasonal periods mapping from JSON file."""
    # Try to find the mapping file in the current directory or package directory
    mapping_files = [
        "dataset_seasonal_periods_mapping.json",
        os.path.join(
            os.path.dirname(__file__), "dataset_seasonal_periods_mapping.json"
        ),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "dataset_seasonal_periods_mapping.json",
        ),
    ]

    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    f"Could not load seasonal periods mapping from {mapping_file}: {e}"
                )
                continue

    logger.warning(
        "Could not find seasonal periods mapping file. SARIMAX models will require manual --seasonal-period specification."
    )
    return {}


def get_seasonal_period_for_dataset(
    dataset_name: str,
    seasonal_periods_mapping: Dict[str, int],
    args: argparse.Namespace,
) -> Optional[int]:
    """Get the seasonal period for a dataset, checking args first, then mapping."""
    # First check if seasonal_period is explicitly provided in args
    if hasattr(args, "seasonal_period") and args.seasonal_period is not None:
        return args.seasonal_period

    # Then check the mapping
    if dataset_name in seasonal_periods_mapping:
        return seasonal_periods_mapping[dataset_name]

    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate forecasts for datasets and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="List of dataset names to generate forecasts for",
    )
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Generate forecasts for all supported datasets",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to use for forecasting",
    )

    # Optional parameters for specific datasets/models
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
        "--llm-model-names",
        type=str,
        nargs="+",
        default=["gemini-2.0-flash"],
        help="List of LLM model names for LLM forecaster (default: gemini-2.0-flash)",
    )
    parser.add_argument(  # for testing
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

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    errors = []

    supported_models = get_supported_models()
    supported_datasets = get_supported_datasets()

    # Validate models
    invalid_models = [
        model for model in args.models if model not in supported_models
    ]
    if invalid_models:
        errors.append(
            f"Unsupported model names: {invalid_models}. Supported models: {supported_models}"
        )

    # Validate datasets
    if args.all_datasets:
        datasets_to_validate = supported_datasets
    else:
        datasets_to_validate = args.datasets
        invalid_datasets = [
            dataset
            for dataset in args.datasets
            if dataset not in supported_datasets
        ]
        if invalid_datasets:
            errors.append(
                f"Unsupported dataset names: {invalid_datasets}. Supported datasets: {supported_datasets}"
            )

    # Validate dataset-specific requirements
    for dataset in datasets_to_validate:
        if dataset == "fmv3" and not args.config_file:
            errors.append("--config-file is required for fmv3 dataset")
        elif dataset == "gift":
            if not args.data_path:
                errors.append("--data-path is required for gift dataset")
            if not args.forecast_length:
                errors.append("--forecast-length is required for gift dataset")
            if not args.sub_dataset:
                errors.append("--sub-dataset is required for gift dataset")
        elif dataset == "gpt-synthetic" and not args.dataset_name:
            errors.append(
                "--dataset-name is required for gpt-synthetic dataset"
            )
        elif dataset == "external":
            if not args.external_dataloader_spec:
                errors.append(
                    "--external-dataloader-spec is required for external dataset"
                )
            elif "::" not in args.external_dataloader_spec:
                errors.append(
                    "--external-dataloader-spec must be in format 'path::class_name'"
                )

    # Validate model-specific requirements
    if "sfm_forecaster" in args.models:
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

    # Note: We'll check for missing seasonal periods during execution instead of validation
    # to allow the script to continue and report missing ones at the end

    if errors:
        logger.error("Argument validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)


def generate_forecasts_for_dataset(
    dataset_name: str,
    models: List[str],
    args: argparse.Namespace,
    seasonal_periods_mapping: Dict[str, int],
) -> Dict[str, Any]:
    logger.info(f"Generating forecasts for dataset: {dataset_name}")

    temp_args = argparse.Namespace()
    for attr, value in vars(args).items():
        setattr(temp_args, attr, value)
    temp_args.dataset = dataset_name

    # Set seasonal period from mapping if not explicitly provided
    if (
        not hasattr(temp_args, "seasonal_period")
        or temp_args.seasonal_period is None
    ):
        seasonal_period = get_seasonal_period_for_dataset(
            dataset_name, seasonal_periods_mapping, args
        )
        if seasonal_period is not None:
            temp_args.seasonal_period = seasonal_period
            logger.info(
                f"Using seasonal period {seasonal_period} for dataset {dataset_name} from mapping"
            )

    try:
        dataloader = get_dataloader(temp_args)
    except Exception as e:
        logger.error(f"Failed to get dataloader for {dataset_name}: {e}")
        return {}

    # Extract covariate information from the dataloader
    covariate_info = {"num_covariates": 0, "target_variable_name": "unknown"}

    try:
        # Get the first batch to extract covariate information
        batch_iter = iter(dataloader)
        first_batch = next(batch_iter)

        if first_batch is not None:
            # Get target variable name from the first sample
            if (
                hasattr(first_batch, "samples")
                and first_batch.samples
                and len(first_batch.samples) > 0
            ):
                first_sample = first_batch.samples[0][
                    0
                ]  # First batch, first correlate
                if (
                    hasattr(first_sample, "column_name")
                    and first_sample.column_name
                ):
                    covariate_info["target_variable_name"] = (
                        first_sample.column_name
                    )

            # Count total correlates to determine number of covariates
            # The first correlate is typically the target, the rest are covariates
            num_correlates = getattr(first_batch, "num_correlates", 0)
            if num_correlates > 1:
                # Subtract 1 for the target variable, the rest are covariates
                covariate_info["num_covariates"] = num_correlates - 1

    except Exception as e:
        logger.warning(
            f"Could not extract covariate info for {dataset_name}: {e}"
        )

    forecasting_models = {}
    dataset_results = {}
    expanded_model_names = []

    for model_name in models:
        if model_name == "llm":
            for llm_model_name in args.llm_model_names:
                expanded_name = (
                    f"llm_{llm_model_name.replace('-', '_').replace('.', '_')}"
                )
                expanded_model_names.append(expanded_name)

                llm_temp_args = argparse.Namespace()
                for attr, value in vars(temp_args).items():
                    setattr(llm_temp_args, attr, value)
                llm_temp_args.llm_model_name = llm_model_name

                try:
                    forecasting_models[expanded_name] = get_forecasting_model(
                        "llm", llm_temp_args
                    )
                    dataset_results[expanded_name] = DatasetResultFormat()
                except Exception as e:
                    logger.error(
                        f"Failed to initialize model {expanded_name}: {e}"
                    )
        else:
            expanded_model_names.append(model_name)
            try:
                forecasting_models[model_name] = get_forecasting_model(
                    model_name, temp_args
                )
                dataset_results[model_name] = DatasetResultFormat()
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")

    forecasts = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_covariates": covariate_info["num_covariates"],
        "target_variable_name": covariate_info["target_variable_name"],
        "models": {},
    }

    total_batches = None
    if args.num_batches:
        total_batches = args.num_batches
    else:
        try:
            total_batches = len(dataloader)
        except (TypeError, AttributeError):
            pass

    # Process batches and store results directly
    batch_results = {}  # Store results by model and batch

    for batch_idx, batch in enumerate(
        tqdm(dataloader, total=total_batches, desc=f"Processing {dataset_name}")
    ):
        if batch is None:
            logger.warning(f"Skipping batch {batch_idx} - no data available")
            continue

        if args.test_samples:
            from synthefy_pkg.fm_evals.eval_utils import get_test_samples

            batch = get_test_samples(batch, test_samples=args.test_samples)

        for model_name, model in forecasting_models.items():
            try:
                model.fit(batch)
                predictions = model.predict(batch)
                dataset_results[model_name].add_batch(batch, predictions)

                # Store batch results directly for JSON generation
                if model_name not in batch_results:
                    batch_results[model_name] = []
                batch_results[model_name].append((batch, predictions))

            except Exception as e:
                logger.error(
                    f"Error running {model_name} on batch {batch_idx}: {e}"
                )

        if args.num_batches and batch_idx >= args.num_batches - 1:
            logger.info(f"Processed {args.num_batches} batches, stopping")
            break

    # Store results for each model
    for model_name in expanded_model_names:
        if model_name in dataset_results:
            result = dataset_results[model_name]

            # Extract detailed forecast data from direct batch results
            forecast_data = []
            if model_name in batch_results:
                for batch_idx, (eval_batch, forecast_batch) in enumerate(
                    batch_results[model_name]
                ):
                    # Iterate over sample rows in the batch
                    for sample_row_idx in range(eval_batch.batch_size):
                        # Iterate over correlates in each sample row
                        for correlate_idx in range(eval_batch.num_correlates):
                            eval_sample = eval_batch[
                                sample_row_idx, correlate_idx
                            ]
                            forecast_sample = forecast_batch[
                                sample_row_idx, correlate_idx
                            ]

                            # Skip samples with empty forecasts (covariates)
                            if len(forecast_sample.values) == 0:
                                continue

                            # Create forecast data with timestamps, target values, and forecast values as separate lists
                            target_timestamps = (
                                eval_sample.target_timestamps.astype(
                                    str
                                ).tolist()
                            )
                            target_values = eval_sample.target_values.tolist()
                            forecast_values = forecast_sample.values.tolist()

                            # Ensure all arrays have the same length
                            min_length = min(
                                len(target_timestamps),
                                len(target_values),
                                len(forecast_values),
                            )

                            sample_data = {
                                "batch_idx": batch_idx,
                                "sample_idx": sample_row_idx,
                                "correlate_idx": correlate_idx,
                                "sample_id": str(eval_sample.sample_id),
                                "history": {
                                    "timestamps": eval_sample.history_timestamps.astype(
                                        str
                                    ).tolist(),
                                    "values": eval_sample.history_values.tolist(),
                                },
                                "forecast": {
                                    "timestamps": target_timestamps[
                                        :min_length
                                    ],
                                    "target_values": target_values[:min_length],
                                    "forecast_values": forecast_values[
                                        :min_length
                                    ],
                                },
                                "metrics": {
                                    "mae": float(forecast_sample.metrics.mae)
                                    if forecast_sample.metrics
                                    else None,
                                    "mape": float(forecast_sample.metrics.mape)
                                    if forecast_sample.metrics
                                    else None,
                                }
                                if forecast_sample.metrics
                                else None,
                            }
                            forecast_data.append(sample_data)

            forecasts["models"][model_name] = {
                "metrics": {
                    "mae": float(result.metrics.mae)
                    if result.metrics
                    else None,
                    "mape": float(result.metrics.mape)
                    if result.metrics
                    else None,
                }
                if result.metrics
                else None,
                "num_batches_processed": len(result.eval_samples),
                "total_samples": len(result.eval_samples)
                * len(result.eval_samples[0])
                if len(result.eval_samples) > 0
                else 0,
                "forecast_data": forecast_data,
            }

    return forecasts


def main():
    """Main function to generate forecasts."""
    args = parse_arguments()
    validate_arguments(args)

    # Load seasonal periods mapping
    seasonal_periods_mapping = load_seasonal_periods_mapping()

    if args.all_datasets:
        datasets = get_supported_datasets()
        logger.info(f"Processing all {len(datasets)} supported datasets")
    else:
        datasets = args.datasets
        logger.info(f"Processing {len(datasets)} specified datasets")

    logger.info(f"Models: {args.models}")

    s3_bucket = "synthefy-fm-dataset-forecasts"

    # Track missing seasonal periods for SARIMAX models
    missing_seasonal_periods = []
    sarimax_models = [
        model
        for model in args.models
        if model in ["sarima", "sarimax_future_leaked"]
    ]

    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'=' * 50}")

        try:
            # Check for missing seasonal periods for SARIMAX models
            if sarimax_models:
                seasonal_period = get_seasonal_period_for_dataset(
                    dataset_name, seasonal_periods_mapping, args
                )
                if seasonal_period is None:
                    missing_seasonal_periods.append(dataset_name)
                    logger.warning(
                        f"Seasonal period not found for {dataset_name}. Skipping SARIMAX models for this dataset."
                    )
                    # Filter out SARIMAX models for this dataset
                    filtered_models = [
                        model
                        for model in args.models
                        if model not in sarimax_models
                    ]
                    if not filtered_models:
                        logger.warning(
                            f"No non-SARIMAX models specified. Skipping {dataset_name} entirely."
                        )
                        continue
                    models_to_use = filtered_models
                else:
                    models_to_use = args.models
            else:
                models_to_use = args.models

            # Generate forecasts for this dataset
            new_forecasts = generate_forecasts_for_dataset(
                dataset_name, models_to_use, args, seasonal_periods_mapping
            )

            if not new_forecasts.get("models"):
                logger.warning(
                    f"No forecasts generated for {dataset_name}, skipping"
                )
                continue

            s3_key = f"{dataset_name}_forecasts.json"

            s3_forecasts = {}
            try:
                s3_forecasts = read_existing_forecasts_from_s3(
                    s3_bucket, s3_key
                )
            except Exception as e:
                logger.warning(
                    f"Could not read existing forecasts from S3: {e}"
                )
            # local_forecasts = read_existing_forecasts_locally(dataset_name)

            existing_forecasts = (
                s3_forecasts  # if s3_forecasts else local_forecasts
            )

            if existing_forecasts:
                logger.info(
                    f"Merging with existing forecasts for {dataset_name}"
                )
                if "models" not in existing_forecasts:
                    existing_forecasts["models"] = {}

                # Check for model overwrites
                for model_name in new_forecasts["models"]:
                    if model_name in existing_forecasts["models"]:
                        logger.info(
                            f"Overwriting existing results for model '{model_name}' in dataset '{dataset_name}'"
                        )
                    else:
                        logger.info(
                            f"Adding new model '{model_name}' to dataset '{dataset_name}'"
                        )

                existing_forecasts["models"].update(new_forecasts["models"])
                existing_forecasts["last_updated"] = datetime.now().isoformat()
                merged_forecasts = existing_forecasts
            else:
                logger.info(f"Creating new forecasts file for {dataset_name}")
                merged_forecasts = new_forecasts

            try:
                write_forecasts_to_s3(s3_bucket, s3_key, merged_forecasts)
            except Exception as e:
                logger.warning(f"Could not write forecasts to S3: {e}")

            # local for testing
            # save_forecasts_locally(dataset_name, merged_forecasts)

            logger.info(f"Successfully processed {dataset_name}")

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            continue

    # Report missing seasonal periods at the end
    if missing_seasonal_periods:
        logger.info(f"\n{'=' * 60}")
        logger.info("MISSING SEASONAL PERIODS SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Datasets without seasonal period mapping ({len(missing_seasonal_periods)}):"
        )
        for dataset in missing_seasonal_periods:
            logger.info(f"  - {dataset}")
        logger.info("\nTo add seasonal periods for these datasets:")
        logger.info("1. Edit dataset_seasonal_periods_mapping.json")
        logger.info(f'2. Add entries like: "{missing_seasonal_periods[0]}": 24')
        logger.info("3. Or provide --seasonal-period when running the script")
        logger.info(f"{'=' * 60}")

    logger.info("\nForecast generation completed!")


if __name__ == "__main__":
    main()
