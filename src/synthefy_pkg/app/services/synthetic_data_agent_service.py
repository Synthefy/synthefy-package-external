import json
import os
import pickle
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import aioboto3
import boto3
import numpy as np
import pandas as pd
from fastapi import HTTPException
from loguru import logger

from synthefy_pkg.app.config import SyntheticDataAgentSettings
from synthefy_pkg.app.data_models import (
    GenerateCombinationsRequest,
    GridCombinations,
    MetaDataGrid,
    MetaDataGridSample,
    MetaDataRange,
    MetaDataVariation,
    OneContinuousMetaDataRange,
    OneDiscreteMetaDataRange,
    SyntheticDataGenerationRequest,
    TimeStampsRange,
)
from synthefy_pkg.app.utils.api_utils import (
    apply_metadata_variations,
    cleanup_local_directories,
    convert_str_to_isoformat,
    filter_window_dataframe_by_window_filters,
)
from synthefy_pkg.app.utils.synthetic_data_utils import (
    ensure_preprocessed_data_downloaded,
    ensure_synthesis_model_downloaded,
    get_files_to_download,
)
from synthefy_pkg.data.window_and_dataframe_utils import (
    convert_windows_to_dataframe,
)

DEFAULT_MAX_NUM_WINDOWS_TO_GENERATE = 1000


async def count_metadata_perturbations(
    metadata_grid: MetaDataGrid,
) -> List[List[MetaDataVariation]]:
    """
    Generate all possible combinations of the metadata perturbations from the metadata_grid.
    Uses itertools.product to generate all possible combinations.

    For example, if we have multiple variations for each parameter:
    - heart_rate: [5.0, 10.0] (two ContinuousVariation instances)
    - respiratory_rate: [1.1, 1.2] (two ContinuousVariation instances)

    Returns a list of MetaDataVariation objects containing all combinations like:
    [
        [
            MetaDataVariation(name="heart_rate", value=5.0, perturbation_or_exact_value="perturbation", perturbation_type=PerturbationType.ADD),
            MetaDataVariation(name="respiratory_rate", value=1.1, perturbation_or_exact_value="perturbation", perturbation_type=PerturbationType.MULTIPLY),
        ],
        [
            MetaDataVariation(name="heart_rate", value=5.0, perturbation_or_exact_value="perturbation", perturbation_type=PerturbationType.ADD),
            MetaDataVariation(name="respiratory_rate", value=1.2, perturbation_or_exact_value="perturbation", perturbation_type=PerturbationType.MULTIPLY),
        ],
        ...
    ]

    Args:
        metadata_grid: MetaDataGrid containing discrete, continuous and timestamp variations

    Returns:
        List[List[MetaDataVariation]]: List of metadata variations containing all possible combinations
    """
    param_variations = {}

    # Handle continuous conditions
    if metadata_grid.continuous_conditions_to_change:
        for continuous_var in metadata_grid.continuous_conditions_to_change:
            if (
                continuous_var.perturbation_type is not None
                and continuous_var.perturbation_value is not None
            ):
                param_variations.setdefault(continuous_var.name, []).append(
                    MetaDataVariation(
                        name=continuous_var.name,
                        value=continuous_var.perturbation_value,
                        perturbation_or_exact_value="perturbation",
                        perturbation_type=continuous_var.perturbation_type,
                    )
                )
            else:
                raise NotImplementedError(
                    "Continuous exact values not implemented yet."
                )

    # Handle discrete conditions
    if metadata_grid.discrete_conditions_to_change:
        for discrete_var in metadata_grid.discrete_conditions_to_change:
            variations = [
                MetaDataVariation(
                    name=discrete_var.name,
                    value=str(option),
                    perturbation_or_exact_value="exact_value",
                    perturbation_type=None,
                )
                for option in discrete_var.options
            ]
            param_variations[discrete_var.name] = variations

    # Handle timestamp conditions
    if metadata_grid.timestamps_conditions_to_change:
        for timestamp_var in metadata_grid.timestamps_conditions_to_change:
            if (
                timestamp_var.perturbation_type is not None
                and timestamp_var.perturbation_value is not None
            ):
                param_variations.setdefault(timestamp_var.name, []).append(
                    MetaDataVariation(
                        name=timestamp_var.name,
                        value=timestamp_var.perturbation_value,
                        perturbation_or_exact_value="perturbation",
                        perturbation_type=timestamp_var.perturbation_type,
                    )
                )
            else:
                raise NotImplementedError(
                    "Timestamp exact values not implemented"
                )

    # If no parameters, return empty list
    if not param_variations:
        return []

    # Get all parameter names and their possible variations
    param_names = list(param_variations.keys())
    param_value_lists = [param_variations[name] for name in param_names]

    # Use itertools.product to get all combinations
    all_combinations = list(product(*param_value_lists))

    # Convert tuples to lists before returning
    return [list(combo) for combo in all_combinations]


class SyntheticDataAgentService:
    def __init__(
        self,
        user_id: str,
        dataset_name: str,
        settings: SyntheticDataAgentSettings,
        aioboto3_session: aioboto3.Session,
    ):
        self.user_id = user_id
        self.dataset_name = dataset_name
        self.settings = settings
        self.aioboto3_session = aioboto3_session

    async def _ensure_preprocessed_data_downloaded(
        self, files_to_download: List[str]
    ) -> None:
        """Ensure all required preprocessed data files are downloaded."""
        await ensure_preprocessed_data_downloaded(
            self.settings,
            self.user_id,
            self.dataset_name,
            self.aioboto3_session,
            files_to_download,
        )

    async def _ensure_synthesis_model_downloaded(
        self, training_job_id: str
    ) -> None:
        """Ensure synthesis model and config are downloaded for the given training job ID."""

        await ensure_synthesis_model_downloaded(
            self.settings,
            self.user_id,
            self.dataset_name,
            self.aioboto3_session,
            training_job_id,
        )

    async def get_metadata_grid_sample(self) -> MetaDataGridSample:
        """
        Retrieve metadata grid sample for the dataset.
        format of labels_description.pkl:
            {
                "group_labels_combinations": {
                    "subject-device": ["S1-phone", "S10-watch", "S2-tablet"]
                },
                "discrete_labels": {
                    "category": {"A": 10, "B": 20, "C": 30},
                    "status": {"active": 10, "inactive": 50},
                },
                "continuous_labels": {
                    "temperature": {"min": 20.0, "max": 30.0, "mean": 25.0},
                    "humidity": {"min": 40.0, "max": 80.0, "mean": 60.0},
                },
                "time_labels": {
                    "@timestamp": {
                        "min": pd.Timestamp("2024-03-24 20:00:00+0000", tz="UTC"),
                        "max": pd.Timestamp("2024-04-23 23:30:00+0000", tz="UTC"),
                        "interval": pd.Timedelta("0 days 00:30:00"),
                    }
                },
            }
            The keys are always present, but empty values look like:

            {
                "discrete_labels": {},
                "continuous_labels": {},
                "time_labels": {},
                "group_labels_combinations": [],
            }

        """
        logger.info(
            f"Retrieving metadata grid sample for dataset: {self.dataset_name}"
        )

        # Always ensure necessary files are downloaded
        files_needed = ["labels_description.pkl"]
        await self._ensure_preprocessed_data_downloaded(files_needed)

        try:
            # Load labels description
            with open(
                os.path.join(
                    self.settings.preprocessed_data_path,
                    "labels_description.pkl",
                ),
                "rb",
            ) as f:
                labels_description = pickle.load(f)

            preprocess_config = json.load(
                open(self.settings.preprocess_config_path)
            )

            window_size = preprocess_config["window_size"]
            if len(labels_description["time_labels"]) > 0:
                default_time_labels = [
                    TimeStampsRange(
                        name=timestamp_col,
                        min_time=(
                            convert_str_to_isoformat(values["min"])
                            if isinstance(values["min"], str)
                            else str(values["min"])
                        ),
                        max_time=(
                            convert_str_to_isoformat(values["max"])
                            if isinstance(values["max"], str)
                            else str(values["max"])
                        ),
                        interval=(
                            convert_str_to_isoformat(values["interval"])
                            if isinstance(values["interval"], str)
                            else str(values["interval"])
                        ),
                        length=window_size,
                    )
                    for timestamp_col, values in labels_description[
                        "time_labels"
                    ].items()
                ][0]
            else:
                default_time_labels = None

            group_labels_combinations = labels_description.get(
                "group_labels_combinations"
            )
            if group_labels_combinations == [] or not preprocess_config.get(
                "use_label_col_as_discrete_metadata", False
            ):
                group_labels_combinations = None

            keys_to_remove = []
            if not preprocess_config.get(
                "use_label_col_as_discrete_metadata", False
            ):
                for key in labels_description["discrete_labels"].keys():
                    if key in labels_description.get("group_label_cols", []):
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del labels_description["discrete_labels"][key]

            return MetaDataGridSample(
                group_labels_combinations=group_labels_combinations,
                metadata_range=MetaDataRange(
                    discrete_conditions=[
                        OneDiscreteMetaDataRange(
                            name=key,
                            options=[str(o) for o in sorted(list(set(values)))],
                        )
                        for key, values in labels_description[
                            "discrete_labels"
                        ].items()
                    ],
                    continuous_conditions=[
                        OneContinuousMetaDataRange(
                            name=key,
                            min_val=values["min"],
                            max_val=values["max"],
                        )
                        for key, values in labels_description[
                            "continuous_labels"
                        ].items()
                    ],
                ),
                timestamps_range=default_time_labels,
            )

        except Exception as e:
            cleanup_local_directories([self.settings.preprocessed_data_path])
            raise e

    async def generate_combinations(self, request: GenerateCombinationsRequest):
        """
        Generate all possible combinations of the input parameters.
        1. Download the split data
        2. Convert it to a dataframe
        3. Filter the dataframe by the WindowFilters
        4. Generate all possible combinations of the metadata perturbations from the metadata_grid
        5. Return the combinations in some readable format for the UI

        Args:
            GenerateCombinationsRequest:
                split_type: Split type to generate combinations for ("train", "val", "test")
                metadata_grid: MetaDataGridSample

        Returns:
            List of dictionaries containing all possible combinations
        """

        # 1. Always ensure split data is downloaded
        files_to_download = get_files_to_download(request.split_type.value)
        await self._ensure_preprocessed_data_downloaded(files_to_download)

        # 2. Convert it to a dataframe
        df, _ = await convert_windows_to_dataframe(
            self.dataset_name,
            request.split_type.value,
        )

        # 3. Filter is by the WindowFilters
        df = await filter_window_dataframe_by_window_filters(
            df, request.window_filters
        )

        # 4. Count the possible combinations of the metadata perturbations from the metadata_grid
        combinations = await count_metadata_perturbations(request.metadata_grid)

        # Still generate windows if no metadata variations/combinations are provided
        total_num_windows_to_generate = (
            df.window_idx.nunique()
            if len(combinations) == 0
            else len(combinations) * df.window_idx.nunique()
        )

        return GridCombinations(
            combinations=combinations,
            num_windows_satisfying_conditions=df.window_idx.nunique(),
            num_windows_to_generate=total_num_windows_to_generate,
        )

    async def generate_synthetic_data(
        self, request: SyntheticDataGenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate synthetic data based on the provided combinations.
        Inputs:
            request: SyntheticDataGenerationRequest
        Outputs:
            presigned url to download the synthetic data as a dataframe

        Exceptions:
            ValueError: If the number of windows to generate is too large (> 100)

        """

        SYNTHEFY_DATASETS_BASE = str(os.getenv("SYNTHEFY_DATASETS_BASE"))
        assert SYNTHEFY_DATASETS_BASE is not None, (
            "SYNTHEFY_DATASETS_BASE must be set in the environment"
        )

        # TODO - check if the filters are exact same as labels description, and just return immediately

        if (
            request.previous_grid_combinations_output.num_windows_to_generate
            > DEFAULT_MAX_NUM_WINDOWS_TO_GENERATE
        ):
            num_combinations_per_window = len(
                request.previous_grid_combinations_output.combinations
            )

            if (
                (request.window_inclusive_end_idx - request.window_start_idx)
                * num_combinations_per_window
                > DEFAULT_MAX_NUM_WINDOWS_TO_GENERATE
            ):
                raise ValueError(
                    f"The number of windows to generate is too large (> {DEFAULT_MAX_NUM_WINDOWS_TO_GENERATE}) - "
                    "Reduce the number of windows or the number of combinations per window"
                )

        # 1. Always ensure split data is downloaded
        files_to_download = get_files_to_download(request.split_type.value)
        await self._ensure_preprocessed_data_downloaded(files_to_download)

        # 2. Always ensure synthesis model and config are downloaded for the specific training job
        await self._ensure_synthesis_model_downloaded(
            request.synthesis_training_job_id
        )

        if self.settings.bucket_name != "local":
            # Check if run already exists based on storage location
            s3_client = boto3.client("s3")
            s3_prefix = os.path.join(
                self.user_id,
                "generation_logs",
                self.dataset_name,
                request.run_name,
            )
            response = s3_client.list_objects_v2(
                Bucket=self.settings.bucket_name,
                Prefix=s3_prefix,
                MaxKeys=1,
            )
            if "Contents" in response and len(response["Contents"]) > 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"Run name '{request.run_name}' already exists in S3 for dataset '{self.dataset_name}' - change the run name to generate new data",
                )

        # 3. Convert it to a dataframe
        df, _ = await convert_windows_to_dataframe(
            self.dataset_name,
            request.split_type.value,
        )

        # 4. Filter is by the WindowFilters
        df = await filter_window_dataframe_by_window_filters(
            df, request.window_filters
        )

        # 5. Apply the given metadata variations to the dataframe
        preprocess_config = json.load(
            open(self.settings.preprocess_config_path)
        )
        window_size = preprocess_config["window_size"]

        df = await apply_metadata_variations(
            df,
            metadata_variations=request.previous_grid_combinations_output.combinations,
            window_start_idx=request.window_start_idx,
            window_inclusive_end_idx=request.window_inclusive_end_idx,
            window_size=window_size,
        )

        # Import the task here to avoid circular imports
        from synthefy_pkg.app.tasks import run_synthetic_data_generation

        # Launch the Celery task
        task = run_synthetic_data_generation.delay(
            config_path=self.settings.synthesis_config_path,
            model_checkpoint_path=self.settings.synthesis_model_path,
            metadata_for_synthesis=json.loads(df.to_json()),
            preprocess_config_path=self.settings.preprocess_config_path,
            run_name=request.run_name,
            user_id=self.user_id,
            dataset_name=self.dataset_name,
            split=request.split_type.value,
            settings_dict=self.settings.model_dump(),  # Convert to dict for JSON serialization
            synthesis_training_job_id=request.synthesis_training_job_id,
            bucket_name=self.settings.bucket_name
            if self.settings.bucket_name != "local"
            else None,
        )

        # Return the task ID for tracking
        return {
            "task_id": task.id,
            "message": "Synthetic Data Generation Started",
        }
