from fastapi import APIRouter, Body, Depends, HTTPException, Path

from synthefy_pkg.app.config import DataRetrievalSettings
from synthefy_pkg.app.data_models import (
    DeletePreprocessedDatasetRequest,
    DeletePreprocessedDatasetResponse,
    DeleteTrainingJobsRequest,
    DeleteTrainingJobsResponse,
    ListAllTrainingJobsRequest,
    ListAllTrainingJobsResponse,
    ListSyntheticDataAgentRunsRequest,
    ListSyntheticDataAgentRunsResponse,
    RetrievePostPreProcessReportRequest,
    RetrievePostPreProcessReportResponse,
    RetrievePostprocessReportRequest,
    RetrievePostprocessReportResponse,
    RetrievePreprocessedDatasetNamesRequest,
    RetrievePreprocessedDatasetNamesResponse,
    RetrievePreprocessingConfigRequest,
    RetrievePreprocessingConfigResponse,
    RetrieveSummaryReportRequest,
    RetrieveSummaryReportResponse,
    RetrieveSyntheticDataAgentRunRequest,
    RetrieveSyntheticDataAgentRunResponse,
    RetrieveTrainingConfigModelRequest,
    RetrieveTrainingConfigModelResponse,
    RetrieveTrainJobIDsRequest,
    RetrieveTrainJobIDsResponse,
)
from synthefy_pkg.app.services.data_retrieval_service import (
    DataRetrievalService,
)
from synthefy_pkg.app.utils.api_utils import get_settings

COMPILE = False
router = APIRouter(tags=["Data Retrieval"])


def get_data_retrieval_service() -> DataRetrievalService:
    settings = get_settings(DataRetrievalSettings)
    return DataRetrievalService(settings=settings)  # type: ignore


@router.post(
    "/api/list-preprocessed-datasets",
    response_model=RetrievePreprocessedDatasetNamesResponse,
)
async def list_preprocessed_datasets(
    request: RetrievePreprocessedDatasetNamesRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrievePreprocessedDatasetNamesResponse:
    """List all preprocessed dataset names for a given user_id."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")

    try:
        dataset_names = await service.list_preprocessed_datasets(
            user_id=request.user_id.strip()
        )
        response = RetrievePreprocessedDatasetNamesResponse(
            dataset_names=dataset_names
        )
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/list-training-jobs", response_model=RetrieveTrainJobIDsResponse
)
async def list_training_jobs(
    request: RetrieveTrainJobIDsRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrieveTrainJobIDsResponse:
    """List training jobs for a given user_id and preprocessed dataset."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")
    if not request.dataset_name or not request.dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty."
        )

    try:
        response = await service.list_training_jobs(
            user_id=request.user_id.strip(),
            dataset_name=request.dataset_name.strip(),
        )
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/api/preprocessed-datasets",
    response_model=DeletePreprocessedDatasetResponse,
)
async def delete_preprocessed_dataset(
    request: DeletePreprocessedDatasetRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> DeletePreprocessedDatasetResponse:
    """Delete a preprocessed dataset."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")
    if not request.dataset_name or not request.dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty."
        )

    try:
        response = await service.delete_preprocessed_dataset(
            user_id=request.user_id.strip(),
            dataset_name=request.dataset_name.strip(),
        )
        return response
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.delete("/api/training-jobs", response_model=DeleteTrainingJobsResponse)
async def delete_training_jobs(
    request: DeleteTrainingJobsRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> DeleteTrainingJobsResponse:
    """Delete specific training jobs for a preprocessed dataset."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not request.dataset_name or not request.dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )
    if not request.training_job_ids:
        raise HTTPException(
            status_code=400, detail="training_job_ids cannot be empty"
        )

    return await service.delete_training_jobs(
        user_id=request.user_id.strip(),
        dataset_name=request.dataset_name.strip(),
        training_job_ids=request.training_job_ids,
    )


@router.post(
    "/api/list-all-training-jobs", response_model=ListAllTrainingJobsResponse
)
async def list_all_training_jobs(
    request: ListAllTrainingJobsRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> ListAllTrainingJobsResponse:
    """List all training jobs for a given user_id across all datasets."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")

    try:
        response = await service.list_all_training_jobs(
            user_id=request.user_id.strip()
        )
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/dataset-summary-report/{dataset_name}",
    response_model=RetrieveSummaryReportResponse,  # type: ignore
)
async def get_dataset_summary(
    dataset_name: str = Path(
        ..., description="Name of the dataset to fetch details for"
    ),
    request: RetrieveSummaryReportRequest = Body(...),
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrieveSummaryReportResponse:
    """Get dataset details with summary HTML.

    Returns a presigned URL to download the summary HTML from S3.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not dataset_name or not dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )

    try:
        return await service.get_dataset_summary(
            user_id=request.user_id.strip(), dataset_name=dataset_name.strip()
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/postprocess-report/{dataset_name}",
    response_model=RetrievePostprocessReportResponse,
)
async def retrieve_postprocess_report(
    dataset_name: str = Path(..., description="Name of the dataset"),
    request: RetrievePostprocessReportRequest = Body(...),
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrievePostprocessReportResponse:
    """Retrieve postprocess report HTML for a specific training job.
    Returns a presigned URL to download the postprocess report HTML from S3.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not request.job_id or not request.job_id.strip():
        raise HTTPException(status_code=400, detail="job_id cannot be empty")
    if not dataset_name or not dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )

    try:
        return await service.retrieve_postprocess_report(
            user_id=request.user_id.strip(),
            dataset_name=dataset_name.strip(),
            job_id=request.job_id.strip(),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/postpreprocess-report/{dataset_name}",
    response_model=RetrievePostPreProcessReportResponse,
)
async def retrieve_postpreprocess_report(
    dataset_name: str = Path(..., description="Name of the dataset"),
    request: RetrievePostPreProcessReportRequest = Body(...),
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrievePostPreProcessReportResponse:
    """Retrieve postpreprocess report HTML for a specific training job.
    Returns a presigned URL to download the postprocess report PDF from S3.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not dataset_name or not dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )

    try:
        return await service.retrieve_postpreprocess_report(
            user_id=request.user_id.strip(),
            dataset_name=dataset_name.strip(),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/list-synthetic-data",
    response_model=ListSyntheticDataAgentRunsResponse,
)
async def list_synthetic_data(
    request: ListSyntheticDataAgentRunsRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> ListSyntheticDataAgentRunsResponse:
    """List all synthetic data runs for a given user_id and dataset."""
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not request.dataset_name or not request.dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )

    try:
        response = await service.list_synthetic_data(
            user_id=request.user_id.strip(),
            dataset_name=request.dataset_name.strip(),
        )
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/synthetic-data/{dataset_name}/{run_name}",
    response_model=RetrieveSyntheticDataAgentRunResponse,
)
async def retrieve_synthetic_data(
    dataset_name: str = Path(..., description="Name of the dataset"),
    run_name: str = Path(..., description="Name of the synthetic data run"),
    request: RetrieveSyntheticDataAgentRunRequest = Body(...),
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrieveSyntheticDataAgentRunResponse:
    """Retrieve synthetic data for a specific run.
    Returns a presigned URL to download the synthetic data from S3.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")
    if not dataset_name or not dataset_name.strip():
        raise HTTPException(
            status_code=400, detail="dataset_name cannot be empty"
        )
    if not run_name or not run_name.strip():
        raise HTTPException(status_code=400, detail="run_name cannot be empty")

    try:
        return await service.retrieve_synthetic_data(
            user_id=request.user_id.strip(),
            dataset_name=dataset_name.strip(),
            run_name=run_name.strip(),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/training-config",
    response_model=RetrieveTrainingConfigModelResponse,
)
async def retrieve_training_config_model(
    request: RetrieveTrainingConfigModelRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrieveTrainingConfigModelResponse:
    """Retrieve the training config model for a specific training job.
    Returns a presigned URL to download the config.yaml file from S3.
    """
    try:
        return await service.retrieve_training_config_model(
            user_id=request.user_id.strip(),
            dataset_name=request.dataset_name.strip(),
            training_job_id=request.training_job_id.strip(),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/preprocessing-config",
    response_model=RetrievePreprocessingConfigResponse,
)
async def retrieve_preprocessing_config(
    request: RetrievePreprocessingConfigRequest,
    service: DataRetrievalService = Depends(get_data_retrieval_service),
) -> RetrievePreprocessingConfigResponse:
    """Retrieve the preprocessing config for a specific dataset.
    Returns a presigned URL to download the config_{dataset_name}_preprocessing.json file from S3.
    """
    try:
        return await service.retrieve_preprocessing_config(
            user_id=request.user_id.strip(),
            dataset_name=request.dataset_name.strip(),
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
