from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from synthefy_pkg.app.config import DataRetrievalSettings
from synthefy_pkg.app.data_models import (
    DatasetInfo,
    DeletePreprocessedDatasetResponse,
    DeleteTrainingJobsResponse,
    RetrieveTrainJobIDsResponse,
)
from synthefy_pkg.app.services.data_retrieval_service import (
    DataRetrievalService,
    get_data_retrieval_service,
)


@pytest.fixture
def mock_settings():
    return DataRetrievalSettings(
        bucket_name="test-bucket",
        dataset_path="/tmp/test_dataset",
        json_save_path="/tmp/test_json_save.json",
    )


@pytest.fixture
def mock_boto3_client():
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.boto3.client"
    ) as mock_client:
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        # Mock paginator.paginate to return predefined CommonPrefixes
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "user-1/dataset1/"},
                    {"Prefix": "user-1/dataset2/"},
                    {"Prefix": "user-1/training_logs/"},
                ]
            }
        ]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_client.return_value = mock_s3
        yield mock_client


@pytest.fixture
def service(mock_settings, mock_boto3_client):
    return DataRetrievalService(settings=mock_settings)


@pytest.mark.asyncio
async def test_list_preprocessed_datasets_success(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    test_time = datetime.now(timezone.utc)

    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "user-1/dataset1/file1.txt", "LastModified": test_time},
                {
                    "Key": "user-1/dataset2/file1.txt",
                    "LastModified": test_time - timedelta(days=1),
                },
                {
                    "Key": "user-1/training_logs/should_be_ignored.txt",
                    "LastModified": test_time,
                },
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Act
    result = await service.list_preprocessed_datasets("user-1")

    # Assert
    assert len(result) == 2
    assert isinstance(result[0], DatasetInfo)
    assert result[0].name == "dataset1"  # Most recent first
    assert result[0].last_modified == test_time.isoformat()
    assert result[1].name == "dataset2"
    assert (
        result[1].last_modified == (test_time - timedelta(days=1)).isoformat()
    )


@pytest.mark.asyncio
async def test_list_preprocessed_datasets_empty_common_prefixes(
    service, mock_boto3_client
):
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.boto3.client"
    ) as mock_boto3:
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto3.return_value = mock_s3

        result = await service.list_preprocessed_datasets("user-1")
        assert result == []


@pytest.mark.asyncio
async def test_list_training_jobs_basic_success(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {
                    "Prefix": "user-1/training_logs/dataset1/customer-0-forecast-123456789/"
                },
                {
                    "Prefix": "user-1/training_logs/dataset1/customer-0-synthesis-987654321/"
                },
                {
                    "Prefix": "user-1/training_logs/dataset1/customer-0-forecast-1122334455/"
                },
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_client.list_objects_v2 to simulate dataset exists
    mock_boto3_client().list_objects_v2.return_value = {
        "Contents": [{"Key": "test/key"}]
    }

    # Act
    response = await service.list_training_jobs("user-1", "dataset1")

    # Assert
    assert isinstance(response, RetrieveTrainJobIDsResponse)
    assert response.synthesis_train_job_ids == [
        "customer-0-synthesis-987654321"
    ]
    assert sorted(response.forecast_train_job_ids) == sorted(
        [
            "customer-0-forecast-123456789",
            "customer-0-forecast-1122334455",
        ]
    )


@pytest.mark.asyncio
async def test_delete_training_jobs_success(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "user-1/training_logs/dataset1/job-1/"},
                {"Prefix": "user-1/training_logs/dataset1/job-2/"},
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to return True
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=True,
    ):
        # Act
        response = await service.delete_training_jobs(
            "user-1", "dataset1", ["job-1", "job-2"]
        )

        # Assert
        assert isinstance(response, DeleteTrainingJobsResponse)
        assert response.status == "success"
        assert "Successfully deleted 2 training jobs" in response.message


@pytest.mark.asyncio
async def test_delete_training_jobs_nonexistent_directory(
    service, mock_boto3_client
):
    # Mock s3_prefix_exists to return False
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=False,
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_training_jobs("user-1", "dataset1", ["job-1"])

        assert exc_info.value.status_code == 404
        assert "No training jobs found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_delete_training_jobs_invalid_job_ids(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "user-1/training_logs/dataset1/job-1/"},
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to return True
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=True,
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_training_jobs(
                "user-1", "dataset1", ["job-1", "nonexistent-job"]
            )

        assert exc_info.value.status_code == 404
        assert "Training jobs not found: nonexistent-job" in str(
            exc_info.value.detail
        )


@pytest.mark.asyncio
async def test_delete_training_jobs_s3_error(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "user-1/training_logs/dataset1/job-1/"},
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to return True and delete_s3_objects to raise an exception
    with (
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
            return_value=True,
        ),
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.delete_s3_objects",
            side_effect=Exception("S3 Error"),
        ),
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_training_jobs("user-1", "dataset1", ["job-1"])

        assert exc_info.value.status_code == 500
        assert "Failed to delete training jobs" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_preprocessed_datasets_empty_paginator(
    service, mock_boto3_client
):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Act
    result = await service.list_preprocessed_datasets("user-1")

    # Assert
    assert result == []


@pytest.mark.asyncio
async def test_list_preprocessed_datasets_error(service, mock_boto3_client):
    # Arrange
    mock_boto3_client().get_paginator.side_effect = Exception("S3 Error")

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await service.list_preprocessed_datasets("user-1")

    assert exc_info.value.status_code == 500
    assert "Failed to list datasets" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_delete_preprocessed_dataset_success(service, mock_boto3_client):
    # Arrange
    with (
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
            return_value=True,
        ),
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.delete_s3_objects",
            return_value=None,
        ),
    ):
        # Act
        response = await service.delete_preprocessed_dataset(
            "user-1", "dataset1"
        )

        # Assert
        assert isinstance(response, DeletePreprocessedDatasetResponse)
        assert response.status == "success"
        assert "Successfully deleted preprocessed dataset" in response.message
        assert response.deleted_dataset_path == "user-1/dataset1/"


@pytest.mark.asyncio
async def test_delete_preprocessed_dataset_not_found(
    service, mock_boto3_client
):
    # Arrange
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=False,
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_preprocessed_dataset(
                "user-1", "nonexistent-dataset"
            )

        assert exc_info.value.status_code == 404
        assert "Dataset nonexistent-dataset not found" in str(
            exc_info.value.detail
        )


@pytest.mark.asyncio
async def test_delete_preprocessed_dataset_error(service, mock_boto3_client):
    # Arrange
    with (
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
            return_value=True,
        ),
        patch(
            "synthefy_pkg.app.services.data_retrieval_service.delete_s3_objects",
            side_effect=Exception("S3 Error"),
        ),
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_preprocessed_dataset("user-1", "dataset1")

        assert exc_info.value.status_code == 500
        assert "Failed to delete dataset" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_training_jobs_with_checkpoints(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "user-1/training_logs/dataset1/synthesis-job-1/"},
                {"Prefix": "user-1/training_logs/dataset1/forecast-job-1/"},
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock head_object to simulate model checkpoint exists
    mock_boto3_client().head_object.return_value = {}

    # Mock s3_prefix_exists to return True
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=True,
    ):
        # Act
        response = await service.list_training_jobs("user-1", "dataset1")

        # Assert
        assert isinstance(response, RetrieveTrainJobIDsResponse)
        assert len(response.synthesis_train_job_ids) == 1
        assert len(response.forecast_train_job_ids) == 1
        assert "synthesis-job-1" in response.synthesis_train_job_ids
        assert "forecast-job-1" in response.forecast_train_job_ids


@pytest.mark.asyncio
async def test_list_training_jobs_dataset_not_found(service, mock_boto3_client):
    # Arrange
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=False,
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.list_training_jobs("user-1", "nonexistent-dataset")

        assert exc_info.value.status_code == 404
        assert "Dataset nonexistent-dataset not found" in str(
            exc_info.value.detail
        )


@pytest.mark.asyncio
async def test_list_training_jobs_no_checkpoints(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "CommonPrefixes": [
                {"Prefix": "user-1/training_logs/dataset1/synthesis-job-1/"},
            ]
        }
    ]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock head_object to simulate no model checkpoint exists
    mock_boto3_client().head_object.side_effect = Exception("Not found")

    # Mock s3_prefix_exists to return True
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=True,
    ):
        # Act
        response = await service.list_training_jobs("user-1", "dataset1")

        # Assert
        assert isinstance(response, RetrieveTrainJobIDsResponse)
        assert len(response.synthesis_train_job_ids) == 0
        assert len(response.forecast_train_job_ids) == 0


@pytest.mark.asyncio
async def test_list_all_training_jobs_success(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()

    # Create a custom side effect function to handle different prefix cases
    def mock_paginate(**kwargs):
        prefix = kwargs.get("Prefix", "")
        if prefix == "user-1/training_logs/":
            return iter(
                [
                    {
                        "CommonPrefixes": [
                            {"Prefix": "user-1/training_logs/dataset1/"},
                            {"Prefix": "user-1/training_logs/dataset2/"},
                        ]
                    }
                ]
            )
        elif prefix == "user-1/training_logs/dataset1/":
            return iter(
                [
                    {
                        "CommonPrefixes": [
                            {
                                "Prefix": "user-1/training_logs/dataset1/synthesis-job-1/"
                            },
                            {
                                "Prefix": "user-1/training_logs/dataset1/forecast-job-1/"
                            },
                        ]
                    }
                ]
            )
        elif prefix == "user-1/training_logs/dataset2/":
            return iter(
                [
                    {
                        "CommonPrefixes": [
                            {
                                "Prefix": "user-1/training_logs/dataset2/synthesis-job-2/"
                            }
                        ]
                    }
                ]
            )
        return iter([{"CommonPrefixes": []}])

    mock_paginator.paginate = MagicMock(side_effect=mock_paginate)
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to return True
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=True,
    ):
        # Act
        result = await service.list_all_training_jobs("user-1")

        # Assert
        assert len(result.training_jobs) == 3

        # Check first job from dataset1
        assert result.training_jobs[0].training_job_id == "synthesis-job-1"
        assert result.training_jobs[0].job_type == "synthesis"
        assert result.training_jobs[0].dataset_name == "dataset1"

        # Check second job from dataset1
        assert result.training_jobs[1].training_job_id == "forecast-job-1"
        assert result.training_jobs[1].job_type == "forecast"
        assert result.training_jobs[1].dataset_name == "dataset1"

        # Check job from dataset2
        assert result.training_jobs[2].training_job_id == "synthesis-job-2"
        assert result.training_jobs[2].job_type == "synthesis"
        assert result.training_jobs[2].dataset_name == "dataset2"


@pytest.mark.asyncio
async def test_list_all_training_jobs_empty(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to return True (since we want to test empty results, not 404)
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        return_value=False,
    ):
        # Act
        result = await service.list_all_training_jobs(
            "user-1"
        )  # Make sure this is awaited in the service

        # Assert
        assert len(result.training_jobs) == 0


@pytest.mark.asyncio
async def test_list_all_training_jobs_error(service, mock_boto3_client):
    # Arrange
    mock_paginator = MagicMock()
    mock_paginator.paginate.side_effect = Exception("S3 Error")
    mock_boto3_client().get_paginator.return_value = mock_paginator

    # Mock s3_prefix_exists to raise an exception
    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.s3_prefix_exists",
        side_effect=Exception("S3 Error"),
    ):
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await service.list_all_training_jobs("user-1")

        assert exc_info.value.status_code == 500
        assert "Failed to list all training jobs" in str(exc_info.value.detail)


def test_dataset_exists_when_found(service, mock_boto3_client):
    # Arrange
    mock_boto3_client().list_objects_v2.return_value = {
        "Contents": [{"Key": "user-1/dataset1_user-1/file.txt"}]
    }

    # Act
    result = service.dataset_exists("user-1", "dataset1")

    # Assert
    assert result is True
    mock_boto3_client().list_objects_v2.assert_called_once_with(
        Bucket="test-bucket", Prefix="user-1/dataset1_user-1/", MaxKeys=1
    )


def test_dataset_exists_when_not_found(service, mock_boto3_client):
    # Arrange
    mock_boto3_client().list_objects_v2.return_value = {}

    # Act
    result = service.dataset_exists("user-1", "nonexistent-dataset")

    # Assert
    assert result is False
    mock_boto3_client().list_objects_v2.assert_called_once_with(
        Bucket="test-bucket",
        Prefix="user-1/nonexistent-dataset_user-1/",
        MaxKeys=1,
    )


def test_dataset_exists_handles_whitespace(service, mock_boto3_client):
    # Arrange
    mock_boto3_client().list_objects_v2.return_value = {
        "Contents": [{"Key": "user-1/dataset1_user-1/file.txt"}]
    }

    # Act
    result = service.dataset_exists(" user-1 ", " dataset1 ")

    # Assert
    assert result is True
    mock_boto3_client().list_objects_v2.assert_called_once_with(
        Bucket="test-bucket", Prefix="user-1/dataset1_user-1/", MaxKeys=1
    )


def test_dataset_exists_handles_s3_error(service, mock_boto3_client):
    # Arrange
    mock_boto3_client().list_objects_v2.side_effect = Exception("S3 Error")

    # Act
    result = service.dataset_exists("user-1", "dataset1")

    # Assert
    assert result is False
    mock_boto3_client().list_objects_v2.assert_called_once_with(
        Bucket="test-bucket", Prefix="user-1/dataset1_user-1/", MaxKeys=1
    )


@pytest.mark.asyncio
async def test_get_data_retrieval_service():
    # Mock the Request object
    mock_request = MagicMock()

    # Mock get_settings to return our test settings
    test_settings = DataRetrievalSettings(
        bucket_name="test-bucket",
        dataset_path="/tmp/test_dataset",
        json_save_path="/tmp/test_json_save.json",
    )

    with patch(
        "synthefy_pkg.app.services.data_retrieval_service.get_settings",
        return_value=test_settings,
    ):
        # Call the dependency function
        service = await get_data_retrieval_service(mock_request)

        # Assert we got a properly configured service
        assert isinstance(service, DataRetrievalService)
        assert service.settings == test_settings
        assert service.settings.bucket_name == "test-bucket"
        assert service.settings.dataset_path == "/tmp/test_dataset"
        assert service.settings.json_save_path == "/tmp/test_json_save.json"
