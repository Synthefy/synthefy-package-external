import json
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple

import aioboto3
import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from synthefy_pkg.app.config import SynthefySettings
from synthefy_pkg.app.data_models import (
    MetaData,
    MetaDataRange,
    OneContinuousMetaDataRange,
    OneDiscreteMetaDataRange,
    SynthefyAgentFilterRequest,
    SynthefyAgentSetupRequest,
    SynthefyDefaultSetupOptions,
    SynthefyResponse,
    SynthefyTimeSeriesWindow,
    TimeStamps,
    TimeStampsRange,
    WindowFilters,
)
from synthefy_pkg.app.utils.api_utils import (
    array_to_continuous,
    array_to_discrete,
    array_to_timeseries,
    array_to_timestamps,
    cleanup_local_directories,
    convert_discrete_metadata_range_to_label_tuple_range,
    convert_discrete_metadata_to_label_tuple,
    convert_str_to_isoformat,
    create_window_name_from_group_labels,
    filter_window_dataframe_by_window_filters,
    get_settings,
)
from synthefy_pkg.app.utils.s3_utils import (
    download_config_from_s3_async,
    download_model_from_s3_async,
    download_preprocessed_data_from_s3_async,
    download_training_config_from_s3_async,
    get_aioboto3_session,
)
from synthefy_pkg.data.window_and_dataframe_utils import (
    convert_windows_to_dataframe,
)
from synthefy_pkg.utils.scaling_utils import (
    inverse_transform_discrete,
    transform_using_scaler,
)

COMPILE = False
router = APIRouter(tags=["Synthefy Playground"])

SELECTION_METHOD = "random"


def cleanup_inference_files(settings: SynthefySettings):
    logger.info(
        f"Cleaning up all inference files: "
        f"{settings.preprocessed_data_path=}, "
        f"{settings.synthesis_model_path=}, "
        f"{settings.forecast_model_path=}, "
        f"{settings.synthesis_config_path=}, "
        f"{settings.forecast_config_path=}, "
        f"{settings.preprocess_config_path=}"
    )
    if settings.bucket_name != "local":
        cleanup_paths = [
            settings.preprocessed_data_path,
            settings.synthesis_model_path,
            settings.forecast_model_path,
            settings.synthesis_config_path,
            settings.forecast_config_path,
            settings.preprocess_config_path,
        ]
        cleanup_local_directories(cleanup_paths)


async def download_inference_files_from_s3(
    settings: SynthefySettings,
    user_id: str,
    dataset_name: str,
    preprocessed_data_path: str,
    aioboto3_session: aioboto3.Session,
    synthesis_training_job_id: Optional[str] = None,
    forecast_training_job_id: Optional[str] = None,
    synthesis_model_save_path: Optional[str] = None,
    forecast_model_save_path: Optional[str] = None,
    include_npy_files: bool = True,
) -> None:
    """Download preprocessed files and model checkpoints from S3."""
    async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
        try:
            # Check if at least one model was provided
            if not (synthesis_training_job_id or forecast_training_job_id):
                raise HTTPException(
                    status_code=400,
                    detail="At least one of synthesis_training_job_id or forecast_training_job_id must be provided",
                )

            # Download models
            synthesis_success = await download_model_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                model_save_path=synthesis_model_save_path,
                training_job_id=synthesis_training_job_id,
            )
            forecast_success = await download_model_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                model_save_path=forecast_model_save_path,
                training_job_id=forecast_training_job_id,
            )

            if synthesis_training_job_id and not synthesis_success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download synthesis model for job {synthesis_training_job_id}",
                )
            if forecast_training_job_id and not forecast_success:
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download forecast model for job {forecast_training_job_id}",
                )

            if synthesis_success:
                if not await download_training_config_from_s3_async(
                    s3_client=async_s3_client,
                    bucket=settings.bucket_name,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    task_type="synthesis",
                    config_file_path=settings.synthesis_config_path,
                    training_job_id=synthesis_training_job_id,
                    overwrite_if_exists=True,
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download synthesis training config for dataset: {dataset_name}",
                    )

            if forecast_success:
                if not await download_training_config_from_s3_async(
                    s3_client=async_s3_client,
                    bucket=settings.bucket_name,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    task_type="forecast",
                    config_file_path=settings.forecast_config_path,
                    training_job_id=forecast_training_job_id,
                    overwrite_if_exists=True,
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download forecast training config for dataset: {dataset_name}",
                    )

            preprocessed_file_names = [
                "timeseries_windows_columns.json",
                "continuous_windows_columns.json",
                "discrete_windows_columns.json",
                "encoders_dict.pkl",
                "timeseries_scalers.pkl",
                "continuous_scalers.pkl",
                "labels_description.pkl",
                "colnames.json",
            ]
            if include_npy_files:
                preprocessed_file_names.extend(
                    [
                        "train_timeseries.npy",
                        "train_continuous_conditions.npy",
                        "train_discrete_conditions.npy",
                        "train_timestamps_original.npy",
                    ]
                )

            if not await download_config_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                f"config_{dataset_name}_preprocessing.json",
                settings.preprocess_config_path,
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download preprocessing config for dataset: {dataset_name}",
                )

            if not await download_preprocessed_data_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                preprocessed_data_path,
                preprocessed_file_names,
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download preprocessed files of the dataset: {dataset_name}",
                )

        except HTTPException:
            # cleanup files if download fails
            cleanup_inference_files(settings)
            raise
        except Exception as e:
            cleanup_inference_files(settings)
            raise HTTPException(
                status_code=500,
                detail=f"Error downloading files from S3: {str(e)}",
            )


def load_json_from_data_dir(
    settings: SynthefySettings, filename: str
) -> Dict[str, Any]:
    return json.load(
        open(os.path.join(settings.preprocessed_data_path, filename))
    )


def load_dataset_requirements(
    settings: SynthefySettings,
    num_windows_to_return: int = 5,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    int,
    Dict,
    Dict,
]:
    with open(settings.preprocess_config_path, "r") as f:
        preprocess_config: Dict[str, Any] = yaml.safe_load(f)

    window_size = preprocess_config.get("window_size")
    if window_size is None:
        raise ValueError("window_size not found in the preprocess config")

    def load_npy_first_window(
        filename: str, indices: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        # gets only the first window.
        data = np.load(
            os.path.join(settings.preprocessed_data_path, filename),
            allow_pickle=True,
        )
        if indices is not None:
            return data[indices], indices

        # select 5 random idxs
        if SELECTION_METHOD == "random":
            indices = np.random.choice(  # type: ignore
                len(data),  # type: ignore
                min(num_windows_to_return, len(data)),  # type: ignore
                replace=False,  # type: ignore
            )
        elif SELECTION_METHOD == "top_unique":
            idx_to_unique_values = {}
            for window_idx in range(len(data)):
                num_unique_values = len(
                    np.unique(np.round(data[window_idx], 2))
                )
                idx_to_unique_values[window_idx] = num_unique_values

            sorted_idx_to_unique_values = sorted(
                idx_to_unique_values.items(), key=lambda x: x[1], reverse=True
            )
            indices = [
                idx
                for idx, _ in sorted_idx_to_unique_values[
                    : min(num_windows_to_return, len(data))
                ]
            ]
        else:
            # just select first 5
            indices = list(range(min(num_windows_to_return, len(data))))
        return data[indices], indices

    timeseries_cols = list(
        load_json_from_data_dir(settings, "timeseries_windows_columns.json")
    )
    continuous_cols = list(
        load_json_from_data_dir(settings, "continuous_windows_columns.json")
    )
    discrete_cols = list(
        load_json_from_data_dir(settings, "discrete_windows_columns.json")
    )
    timestamps_cols = preprocess_config.get("timestamps_col", [])
    group_label_cols = preprocess_config.get("group_labels", {}).get("cols", [])

    timeseries, indices = load_npy_first_window("train_timeseries.npy")
    continuous_conditions, _ = load_npy_first_window(
        "train_continuous_conditions.npy", indices
    )
    discrete_conditions, _ = load_npy_first_window(
        "train_discrete_conditions.npy", indices
    )
    if len(discrete_conditions.shape) == 2:
        discrete_conditions = np.repeat(
            discrete_conditions[:, np.newaxis, :], window_size, axis=1
        )

    timestamps, _ = load_npy_first_window(
        "train_timestamps_original.npy", indices
    )
    # TODO - comment in below to delete the npy files now
    # os.remove(
    #     os.path.join(settings.preprocessed_data_path, "train_timeseries.npy")
    # )
    # os.remove(
    #     os.path.join(
    #         settings.preprocessed_data_path, "train_continuous_conditions.npy"
    #     )
    # )
    # os.remove(
    #     os.path.join(
    #         settings.preprocessed_data_path, "train_discrete_conditions.npy"
    #     )
    # )

    def load_pickle(filename: str) -> Dict:
        if not os.path.exists(
            os.path.join(settings.preprocessed_data_path, filename)
        ):
            logger.error(f"File {filename} not found")
            return {}
        with open(
            os.path.join(settings.preprocessed_data_path, filename), "rb"
        ) as f:
            return pickle.load(f)

    encoders = load_pickle("encoders_dict.pkl")
    labels_description = load_pickle("labels_description.pkl")

    return (
        timeseries,
        continuous_conditions,
        discrete_conditions,
        timestamps,
        timeseries_cols,
        continuous_cols,
        discrete_cols,
        group_label_cols,
        timestamps_cols,
        window_size,
        labels_description,
        encoders,
    )


def _get_default_synthefy_request(
    settings: SynthefySettings,
    dataset_name: str,
) -> SynthefyDefaultSetupOptions:
    (
        timeseries,
        continuous_conditions,
        discrete_conditions,
        timestamps,
        timeseries_cols,
        continuous_cols,
        discrete_cols,
        group_label_cols,
        timestamps_cols,
        window_size,
        labels_description,
        encoders,
    ) = load_dataset_requirements(settings)

    # unscale the data
    decoded_discrete_col_names, decoded_discrete_conditions = (
        inverse_transform_discrete(
            windows_data=discrete_conditions, encoders=encoders
        )
    )
    timeseries = transform_using_scaler(
        windows=timeseries,
        timeseries_or_continuous="timeseries",
        original_discrete_windows=decoded_discrete_conditions,
        dataset_name=dataset_name,
        inverse_transform=True,
    )
    continuous_conditions = transform_using_scaler(
        windows=continuous_conditions,
        timeseries_or_continuous="continuous",
        original_discrete_windows=decoded_discrete_conditions,
        dataset_name=dataset_name,
        inverse_transform=True,
    )

    window_for_displays = []
    for window_idx in range(len(timeseries)):
        discrete_conditions = (
            array_to_discrete(
                decoded_discrete_conditions[window_idx],
                decoded_discrete_col_names,
            )
            if discrete_cols
            else []
        )
        window = SynthefyTimeSeriesWindow(
            id=window_idx,
            name=create_window_name_from_group_labels(
                discrete_conditions, group_label_cols, f"Window {window_idx}"
            ),
            timeseries_data=array_to_timeseries(
                timeseries[window_idx].T, timeseries_cols
            ),
            metadata=MetaData(
                discrete_conditions=discrete_conditions,
                continuous_conditions=(
                    array_to_continuous(
                        continuous_conditions[window_idx],
                        continuous_cols,
                    )
                    if continuous_cols
                    else []
                ),
            ),
            timestamps=(
                array_to_timestamps(timestamps[window_idx], timestamps_cols)
                if timestamps_cols
                else TimeStamps(name="index", values=list(range(window_size)))
            ),
        )

        # add label tuples
        window.metadata.discrete_conditions = (
            convert_discrete_metadata_to_label_tuple(
                window.metadata.discrete_conditions,
                labels_description.get("group_labels_combinations", {}),
                group_label_cols,
            )
        )

        window_for_displays.append(window)

    default_discrete_conditions = {
        key: sorted(list(set(values)))
        for key, values in labels_description["discrete_labels"].items()
    }
    default_continuous_conditions = {
        key: [values["min"], values["max"]]
        for key, values in labels_description["continuous_labels"].items()
    }

    # time labels
    # can only have 1 timestamp column for now
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
        for timestamp_col, values in labels_description["time_labels"].items()
    ]
    # default to index as timestamps
    if not default_time_labels:
        # TODO - max_time should be the max time in the dataset - we need a timestamp col
        # in preprocess.py so that we can get the max time properly
        default_time_labels = [
            TimeStampsRange(
                name="index",
                min_time="0",
                max_time=str(window_size * 2),
                interval="1",
                length=window_size,
            )
        ]

    discrete_metadata_range = (
        convert_discrete_metadata_range_to_label_tuple_range(
            (
                [
                    OneDiscreteMetaDataRange(name=key, options=values)
                    for key, values in default_discrete_conditions.items()
                ]
                if default_discrete_conditions
                else []
            ),
            labels_description.get("group_labels_combinations", {}),
            group_label_cols,
        )
    )

    ret = SynthefyDefaultSetupOptions(
        windows=window_for_displays,
        metadata_range=MetaDataRange(
            discrete_conditions=discrete_metadata_range,
            continuous_conditions=(
                [
                    OneContinuousMetaDataRange(
                        name=key, min_val=min_value, max_val=max_value
                    )
                    for key, (
                        min_value,
                        max_value,
                    ) in default_continuous_conditions.items()
                ]
                if default_continuous_conditions
                else []
            ),
        ),
        timestamps_range=default_time_labels[
            0
        ],  # only support 1 timestamp for now
        text=[],
    )

    return ret


@router.post("/api/default", response_model=SynthefyDefaultSetupOptions)
async def get_default_synthefy_request(
    setup_request: SynthefyAgentSetupRequest,
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> SynthefyDefaultSetupOptions:
    try:
        logger.info("Started setup_ui")

        # Extract values from request body
        user_id = setup_request.user_id
        dataset_name = setup_request.dataset_name
        synthesis_training_job_id = setup_request.synthesis_training_job_id
        forecast_training_job_id = setup_request.forecast_training_job_id

        settings: SynthefySettings = get_settings(
            SynthefySettings, dataset_name=dataset_name
        )

        # If using S3, check and download files
        if settings.bucket_name != "local":
            await download_inference_files_from_s3(
                settings,
                user_id=user_id,
                dataset_name=dataset_name,
                preprocessed_data_path=settings.preprocessed_data_path,
                aioboto3_session=aioboto3_session,
                synthesis_training_job_id=synthesis_training_job_id,
                forecast_training_job_id=forecast_training_job_id,
                synthesis_model_save_path=settings.synthesis_model_path,
                forecast_model_save_path=settings.forecast_model_path,
            )

        response = _get_default_synthefy_request(settings, dataset_name)
        # convert all timestamps to str
        copied = response.model_copy()
        for idx in range(len(copied.windows)):
            window = copied.windows[idx]
            if window.timestamps is not None:
                window.timestamps.values = [
                    (
                        convert_str_to_isoformat(value)
                        if not isinstance(value, (int, float))
                        else value
                    )
                    for value in window.timestamps.values
                ]
                copied.windows[idx] = window
        if copied.timestamps_range is not None:
            pass  # Placeholder, no direct .values attribute on TimeStampsRange
        with open(f"{settings.json_save_path}/default_response.json", "w") as f:
            json.dump(dict(copied.model_dump()), f)

        # TODO - comment this in - not working some how right now.
        # if settings.bucket_name != "local":
        #     cleanup_local_directories([settings.preprocessed_data_path])

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get default synthefy request: {str(e)}")
        # Clean up only preprocessed data path in case of error
        if "settings" in locals() and settings.bucket_name != "local":
            cleanup_inference_files(settings)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred while getting default setup: {str(e)}",
        )


@router.post(
    "/api/filter-synthefy-agent-windows",
    response_model=SynthefyResponse,
)
async def filter_synthefy_agent_windows(
    request: SynthefyAgentFilterRequest,
    aioboto3_session: aioboto3.Session = Depends(get_aioboto3_session),
) -> SynthefyResponse:
    """
    Filter windows for the Synthefy Agent based on various criteria.

    This endpoint filters windows based on metadata conditions, timestamps, and indexes
    specified in the request. It allows selecting multiple windows based on filter
    criteria and returns them in a format suitable for display in the UI.

    Args:
        request: Filter request containing user_id, dataset_name, window_filters, split, n_windows
        aioboto3_session: AWS S3 session for downloading files if needed

    Returns:
        SynthefyResponse: Response containing filtered windows

    Raises:
        HTTPException: If files cannot be downloaded or other errors occur during filtering
    """
    try:
        logger.info("Started filtering synthefy agent windows")

        # Extract values from request body
        user_id = request.user_id
        dataset_name = request.dataset_name
        window_filters = request.window_filters
        split = request.split
        n_windows = request.n_windows
        synthesis_training_job_id = request.synthesis_training_job_id
        forecast_training_job_id = request.forecast_training_job_id

        settings: SynthefySettings = get_settings(
            SynthefySettings, dataset_name=dataset_name
        )

        # Download required files if needed and verify they exist
        await _prepare_dataset_files(
            settings,
            user_id,
            dataset_name,
            aioboto3_session,
            split,
            synthesis_training_job_id,
            forecast_training_job_id,
        )

        # Load dataset and filter windows
        filtered_df, dataset_info = await _load_and_filter_windows(
            settings, dataset_name, split, window_filters, n_windows
        )

        # Convert filtered dataframe to window display format
        window_for_displays = _create_windows_for_display(
            filtered_df, dataset_info
        )

        # Create the response directly
        response = SynthefyResponse(
            windows=window_for_displays,
            anomaly_timestamps=[],
            forecast_timestamps=[],
            combined_text="",
        )

        # Save response to file
        with open(
            f"{settings.json_save_path}/filtered_response.json", "w"
        ) as f:
            json.dump(dict(response.model_dump()), f)

        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(
            f"Failed to filter synthefy agent windows: {str(e)}\n{error_traceback}"
        )
        # if error, remove all files
        if settings.bucket_name != "local":
            cleanup_inference_files(settings)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error occurred while filtering windows: {str(e)}\n{error_traceback}",
        )


async def _prepare_dataset_files(
    settings: SynthefySettings,
    user_id: str,
    dataset_name: str,
    aioboto3_session: aioboto3.Session,
    split: str = "train",
    synthesis_training_job_id: Optional[str] = None,
    forecast_training_job_id: Optional[str] = None,
) -> None:
    """
    Download and verify required dataset files for filtering.

    This function ensures all necessary files are available locally before proceeding.
    If settings.bucket_name is not "local", it downloads required files from S3.
    Then it verifies that all essential files exist locally.

    Args:
        settings: Application settings containing paths and configurations
        user_id: User identifier for S3 path construction
        dataset_name: Name of the dataset to process
        aioboto3_session: AWS S3 session for downloading files
        split: Dataset split to use (train/val/test), defaults to "train"
        synthesis_training_job_id: Optional ID for synthesis training job
        forecast_training_job_id: Optional ID for forecast training job

    Raises:
        HTTPException: If required files cannot be downloaded or verified
    """
    # Check if files need to be downloaded from S3
    if settings.bucket_name != "local":
        files_to_download = [
            f"{split}_timeseries.npy",
            f"{split}_timestamps_original.npy",
            f"{split}_original_discrete_windows.npy",
            f"{split}_continuous_conditions.npy",
            # original_text_conditions is optional for backwards compatibility - try download separately
            # f"{request.split_type.value}_original_text_conditions.npy",
            "encoders_dict.pkl",
            "timeseries_scalers.pkl",
            "continuous_scalers.pkl",
            "labels_description.pkl",
            "colnames.json",
        ]
        logger.info(
            f"Downloading inference files for dataset: {dataset_name}, split: {split}"
        )

        # Create S3 client from session
        async with aioboto3_session.client("s3") as async_s3_client:  # type: ignore
            # Download main files
            if not await download_preprocessed_data_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                settings.preprocessed_data_path,
                files_to_download,
            ):
                cleanup_local_directories([settings.preprocessed_data_path])
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download preprocessed files for dataset: {dataset_name}, split: {split}",
                )

            # Download preprocessing config
            if not await download_config_from_s3_async(
                s3_client=async_s3_client,
                bucket=settings.bucket_name,
                user_id=user_id,
                dataset_name=dataset_name,
                filename=f"config_{dataset_name}_preprocessing.json",
                config_file_path=settings.preprocess_config_path,
            ):
                cleanup_local_directories([settings.preprocessed_data_path])
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download preprocessing config for dataset: {dataset_name}",
                )

            # Try to download text conditions file separately (optional)
            if not await download_preprocessed_data_from_s3_async(
                async_s3_client,
                settings.bucket_name,
                user_id,
                dataset_name,
                settings.preprocessed_data_path,
                ["original_text_conditions.npy"],
            ):
                logger.warning(
                    f"Failed to download the {split}_original_text_conditions for dataset: {dataset_name}"
                )

            # Download training config files if training job IDs are provided
            if synthesis_training_job_id:
                if not await download_training_config_from_s3_async(
                    s3_client=async_s3_client,
                    bucket=settings.bucket_name,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    task_type="synthesis",
                    config_file_path=settings.synthesis_config_path,
                    training_job_id=synthesis_training_job_id,
                    overwrite_if_exists=False,
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download synthesis training config for dataset: {dataset_name}",
                    )

            if forecast_training_job_id:
                if not await download_training_config_from_s3_async(
                    s3_client=async_s3_client,
                    bucket=settings.bucket_name,
                    user_id=user_id,
                    dataset_name=dataset_name,
                    task_type="forecast",
                    config_file_path=settings.forecast_config_path,
                    training_job_id=forecast_training_job_id,
                    overwrite_if_exists=False,
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to download forecast training config for dataset: {dataset_name}",
                    )

    # Verify that essential files exist locally
    required_files = [
        "encoders_dict.pkl",
        "timeseries_scalers.pkl",
        "continuous_scalers.pkl",
        "labels_description.pkl",
        f"{split}_timeseries.npy",
        f"{split}_continuous_conditions.npy",
        f"{split}_original_discrete_windows.npy",
        f"{split}_timestamps_original.npy",
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(settings.preprocessed_data_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        error_msg = (
            f"Missing required files for filtering: {', '.join(missing_files)}"
        )
        logger.error(error_msg)
        raise HTTPException(status_code=404, detail=error_msg)

    logger.info(
        f"All required files for filtering verified for dataset: {dataset_name}, split: {split}"
    )


async def _load_and_filter_windows(
    settings: SynthefySettings,
    dataset_name: str,
    split: str,
    window_filters: WindowFilters,
    n_windows: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset and filter windows based on criteria.

    This function loads the dataset as a dataframe,
    applies filters according to the specified window_filters, and returns
    both the filtered dataframe and relevant dataset information.

    Args:
        settings: Application settings containing paths and configurations
        dataset_name: Name of the dataset to load and filter
        split: Dataset split to use (train/val/test)
        window_filters: Filter criteria for windows including metadata conditions, timestamps, etc.
        n_windows: Optional number of windows to limit the results to

    Returns:
        Tuple containing:
            - Filtered pandas DataFrame with window data
            - Dictionary with dataset information (column names, labels, window size, etc.)
    """
    logger.info(
        f"Converting windows to dataframe for dataset: {dataset_name}, split: {split}"
    )

    # Convert windows to dataframe for filtering - only pass dataset_name and split
    df, continuous_cols = await convert_windows_to_dataframe(
        dataset_name=dataset_name,
        split=split,  # type: ignore
    )

    logger.info(f"Applying window filters to dataset: {dataset_name}")
    # Apply window filters
    filtered_df = await filter_window_dataframe_by_window_filters(
        df, window_filters
    )

    logger.info(
        f"Found {filtered_df['window_idx'].nunique()} windows matching filters"
    )

    # Get the first n_windows if specified
    if n_windows is not None and n_windows > 0:
        unique_windows = filtered_df["window_idx"].unique()[:n_windows]
        filtered_df = filtered_df[
            filtered_df["window_idx"].isin(unique_windows)
        ]
        logger.info(f"Limiting to {len(unique_windows)} windows")

    # Load metadata information for response
    with open(
        os.path.join(settings.preprocessed_data_path, "labels_description.pkl"),
        "rb",
    ) as f:
        labels_description = pickle.load(f)

    with open(
        os.path.join(settings.preprocessed_data_path, "colnames.json"), "r"
    ) as f:
        colnames = json.load(f)

    # Get window size from the data
    window_size = int(len(df) / df["window_idx"].nunique())

    # Get column names from colnames.json
    timeseries_cols = colnames.get("timeseries_colnames", [])
    # already got continuous cols from convert_windows_to_dataframe
    original_discrete_cols = colnames.get("original_discrete_colnames", [])

    # Get group label columns from preprocess config
    with open(settings.preprocess_config_path, "r") as f:
        preprocess_config = json.load(f)

    group_label_cols = preprocess_config.get("group_labels", {}).get("cols", [])
    timestamps_cols = preprocess_config.get("timestamps_col", [])

    # Load encoders if needed
    with open(
        os.path.join(settings.preprocessed_data_path, "encoders_dict.pkl"), "rb"
    ) as f:
        encoders = pickle.load(f)

    # Return filtered dataframe and dataset info
    dataset_info = {
        "timeseries_cols": timeseries_cols,
        "continuous_cols": continuous_cols,
        "original_discrete_cols": original_discrete_cols,
        "group_label_cols": group_label_cols,
        "timestamps_cols": timestamps_cols,
        "window_size": window_size,
        "labels_description": labels_description,
        "encoders": encoders,
    }

    return filtered_df, dataset_info


def _create_windows_for_display(
    filtered_df: pd.DataFrame, dataset_info: Dict[str, Any]
) -> List[SynthefyTimeSeriesWindow]:
    """
    Convert filtered dataframe to window objects for display.

    This function transforms the filtered dataframe rows into SynthefyTimeSeriesWindow
    objects that can be displayed in the UI. It handles converting arrays to appropriate
    data structures and formats for display.

    Args:
        filtered_df: Pandas DataFrame containing filtered window data
        dataset_info: Dictionary with dataset information including column names and metadata

    Returns:
        List of SynthefyTimeSeriesWindow objects ready for display
    """
    window_for_displays = []

    window_size = dataset_info["window_size"]
    for i in range(0, len(filtered_df), window_size):
        window_data = filtered_df.iloc[i : i + window_size]
        window_idx = window_data["window_idx"].iloc[0]

        # Get dataset info
        original_discrete_cols = dataset_info["original_discrete_cols"]
        continuous_cols = dataset_info["continuous_cols"]
        timeseries_cols = dataset_info["timeseries_cols"]
        timestamps_cols = dataset_info["timestamps_cols"]
        group_label_cols = dataset_info["group_label_cols"]
        labels_description = dataset_info["labels_description"]

        # Convert arrays to appropriate data structures
        discrete_conditions = (
            array_to_discrete(
                window_data[original_discrete_cols].values,
                original_discrete_cols,
            )
            if original_discrete_cols
            else []
        )

        continuous_conditions = (
            array_to_continuous(
                window_data[continuous_cols].values,
                continuous_cols,
            )
            if continuous_cols
            else []
        )

        timeseries_data = array_to_timeseries(
            window_data[timeseries_cols].values,
            timeseries_cols,
        )

        window_timestamps = (
            array_to_timestamps(
                window_data[timestamps_cols].values, timestamps_cols
            )
            if timestamps_cols
            else TimeStamps(name="index", values=list(range(window_size)))
        )

        # Create window name from group labels
        window_name = create_window_name_from_group_labels(
            discrete_conditions, group_label_cols, f"Window {window_idx}"
        )

        # Add label tuples
        discrete_conditions = convert_discrete_metadata_to_label_tuple(
            discrete_conditions,
            labels_description.get("group_labels_combinations", {}),
            group_label_cols,
        )

        window = SynthefyTimeSeriesWindow(
            id=window_idx,
            name=window_name,
            timeseries_data=timeseries_data,
            metadata=MetaData(
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            ),
            timestamps=window_timestamps,
        )

        window_for_displays.append(window)

    return window_for_displays
