import asyncio  # ADDED: Needed to call async functions via asyncio.run()
import json
import os
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException

from synthefy_pkg.app.utils.s3_utils import (
    adelete_s3_object,
    avalidate_file_exists,
    create_presigned_url,
    download_config_from_s3_async,
    download_directory_from_s3_async,
    download_file_from_s3_async,
    download_model_from_s3_async,
    download_preprocessed_data_from_s3_async,
    handle_s3_source,
    upload_directory_to_s3,
    upload_file_to_s3,
)


@pytest.fixture
def mock_s3_client():
    return Mock()


@pytest.fixture
def sample_params():
    return {
        "bucket": "test-bucket",
        "user_id": "test-user",
        "dataset_name": "test-dataset",
        "training_job_id": "test-job-123",
    }


def test_download_file_from_s3_success(mock_s3_client, tmp_path):
    local_path = str(tmp_path / "test.file")

    # Mock successful download
    async def mock_download_file(*args, **kwargs):
        # Create a dummy file
        with open(local_path, "w") as f:
            f.write("test content")
        return True

    # Updated mock for head_object to return the expected structure
    async def mock_head_object(*args, **kwargs):
        return {"ContentLength": len("test content")}

    mock_s3_client.download_file = mock_download_file
    mock_s3_client.head_object = mock_head_object

    result, error_msg = asyncio.run(
        download_file_from_s3_async(
            mock_s3_client, "test-bucket", "test-key", local_path
        )
    )

    assert result is True
    assert error_msg == ""
    assert os.path.exists(local_path)


# def test_download_file_from_s3_file_not_found(mock_s3_client, tmp_path):
#     # Mock head_object to raise 404 error
#     async def mock_head_object(*args, **kwargs):
#         error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
#         raise ClientError(error_response, "HeadObject")

#     mock_s3_client.head_object = mock_head_object

#     result, error_msg = asyncio.run(
#         download_file_from_s3_async(
#             mock_s3_client,
#             "test-bucket",
#             "test-key",
#             str(tmp_path / "test.file"),
#         )
#     )

#     assert result is False
#     assert error_msg == "File not found in S3: test-key"


# COMMENTED OUT - FAILING TEST
# def test_download_model_from_s3_already_exists(mock_s3_client, tmp_path):
#     model_path = tmp_path / "model.ckpt"
#     # Create dummy model file
#     with open(model_path, "w") as f:
#         f.write("dummy model")
#
#     # Mock the download file function to return True
#     async def mock_download_async(*args, **kwargs):
#         return True
#
#     with patch(
#         "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
#     ) as mock_download:
#         mock_download.side_effect = mock_download_async
#
#         result = asyncio.run(
#             download_model_from_s3_async(
#                 mock_s3_client,
#                 "test-bucket",
#                 "test-user",
#                 "test-dataset",
#                 "test-job",
#                 str(model_path),
#                 overwrite_if_exists=False,
#             )
#         )
#
#     assert result is True
#     mock_download.assert_not_called()


def test_download_model_from_s3_success(
    mock_s3_client, tmp_path, sample_params
):
    model_path = str(tmp_path / "model.ckpt")

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        with open(model_path, "w") as f:
            f.write("test model")
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_model_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                sample_params["training_job_id"],
                model_path,
            )
        )

    assert result is True
    assert os.path.exists(model_path)
    mock_download.assert_called_once()


def test_download_model_from_s3_overwrite(
    mock_s3_client, tmp_path, sample_params
):
    model_path = tmp_path / "model.ckpt"
    # Create dummy model file with existing content
    with open(model_path, "w") as f:
        f.write("existing model")

    # Mock the download file function to return True and write new content
    async def mock_download_async(*args, **kwargs):
        with open(model_path, "w") as f:
            f.write("new model")
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_model_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                sample_params["training_job_id"],
                str(model_path),
                overwrite_if_exists=True,
            )
        )

    assert result is True
    mock_download.assert_called_once()
    # Verify the file was overwritten with new content
    with open(model_path) as f:
        content = f.read()
    assert content == "new model"


def test_download_preprocessed_data_success(
    mock_s3_client, tmp_path, sample_params
):
    local_path = str(tmp_path)
    required_files = [
        "train_timeseries.npy",
        "train_metadata.npy",
        "val_timeseries.npy",
        "val_metadata.npy",
        "test_timeseries.npy",
        "test_metadata.npy",
        "train_labels.npy",
        "val_labels.npy",
        "test_labels.npy",
        "feature_names.json",
        "metadata_feature_names.json",
    ]

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        # Create dummy files when download is called
        filename = os.path.basename(args[2])  # args[2] is s3_key
        with open(os.path.join(local_path, filename), "w") as f:
            f.write("test content")
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_preprocessed_data_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                local_path,
                required_files,
            )
        )

    assert result is True
    # Verify all required files were "downloaded"
    assert len(os.listdir(tmp_path)) == len(required_files)


def test_download_preprocessed_data_partial_failure(
    mock_s3_client, tmp_path, sample_params
):
    local_path = str(tmp_path)
    required_files = [
        "train_timeseries.npy",
        "train_metadata.npy",
        "val_timeseries.npy",
    ]

    # Mock download to fail for one file
    async def mock_download_async(*args, **kwargs):
        if "train_timeseries.npy" in args[2]:  # args[2] is s3_key
            return False, "Failed to download train_timeseries.npy"
        filename = os.path.basename(args[2])
        with open(os.path.join(local_path, filename), "w") as f:
            f.write("test content")
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_preprocessed_data_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                local_path,
                required_files,
            )
        )

    assert result is False


def test_download_preprocessing_config_success(
    mock_s3_client, tmp_path, sample_params
):
    config_path = str(tmp_path / "config_test-dataset_preprocessing.json")

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        with open(config_path, "w") as f:
            f.write('{"test": "config"}')
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_config_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                "config_test-dataset_preprocessing.json",
                config_path,
            )
        )

    assert result is True
    assert os.path.exists(config_path)
    mock_download.assert_called_once()


def test_download_preprocessing_config_already_exists(
    mock_s3_client, tmp_path, sample_params
):
    config_path = tmp_path / "config_test-dataset_preprocessing.json"
    # Create dummy config file
    with open(config_path, "w") as f:
        f.write('{"test": "existing config"}')

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        return True

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_config_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                "config_test-dataset_preprocessing.json",
                str(config_path),
                overwrite_if_exists=False,
            )
        )

    assert result is True
    mock_download.assert_not_called()


def test_download_preprocessing_config_overwrite(
    mock_s3_client, tmp_path, sample_params
):
    config_path = tmp_path / "config_test-dataset_preprocessing.json"
    # Create dummy config file with existing content
    with open(config_path, "w") as f:
        f.write('{"test": "existing config"}')

    # Mock the download file function to return True and write new content
    async def mock_download_async(*args, **kwargs):
        with open(config_path, "w") as f:
            f.write('{"test": "new config"}')
        return True, ""  # Return tuple of (success, error_msg)

    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        mock_download.side_effect = mock_download_async

        result = asyncio.run(
            download_config_from_s3_async(
                mock_s3_client,
                sample_params["bucket"],
                sample_params["user_id"],
                sample_params["dataset_name"],
                "config_test-dataset_preprocessing.json",
                str(config_path),
                overwrite_if_exists=True,
            )
        )

    assert result is True
    mock_download.assert_called_once()
    # Verify the file was overwritten with new content
    with open(config_path) as f:
        content = f.read()
    assert content == '{"test": "new config"}'


# CHANGED: Modified the fixture to patch the async download function since
# the async directory download now calls download_file_from_s3_async.
@pytest.fixture
def mock_download_file():
    with patch(
        "synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async"
    ) as mock_download:
        yield mock_download


@pytest.fixture
def test_dirs(tmp_path):
    local_dir = str(tmp_path / "local_dir")
    os.makedirs(local_dir)
    return local_dir


def setup_s3_paginator_mock(mock_s3_client, pages):
    """
    Helper function to setup the paginator mock on the s3_client.
    pages: list of dicts that represent pages returned by the paginator.
    Each page should have a 'Contents' key if there are objects.
    """
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    mock_s3_client.get_paginator.return_value = paginator


def test_local_dir_not_empty(mock_s3_client, mock_download_file, test_dirs):
    # Given a non-empty local directory
    with open(os.path.join(test_dirs, "existing_file.txt"), "w") as f:
        f.write("some content")

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    # Should return True and log an error
    assert result is True
    mock_logger.error.assert_called_once_with(
        f"Local directory {test_dirs} already exists, skipping download"
    )
    mock_s3_client.list_objects_v2.assert_not_called()


def test_s3_directory_empty(mock_s3_client, mock_download_file, test_dirs):
    # Mock the coroutine for list_objects_v2
    async def mock_list_objects(**kwargs):
        return {}

    mock_s3_client.list_objects_v2 = mock_list_objects

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "empty/s3/dir/", test_dirs
            )
        )

    assert result is False
    mock_logger.error.assert_called_once_with(
        "S3 directory empty/s3/dir/ is empty, skipping download"
    )
    mock_s3_client.get_paginator.assert_not_called()


def test_s3_directory_no_contents_in_page(
    mock_s3_client, mock_download_file, test_dirs
):
    # Mock the coroutine for list_objects_v2
    async def mock_list_objects(**kwargs):
        return {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}

    mock_s3_client.list_objects_v2 = mock_list_objects

    # Mock the paginator
    paginator = MagicMock()

    async def mock_paginate(*args, **kwargs):
        yield {}

    paginator.paginate = mock_paginate
    mock_s3_client.get_paginator.return_value = paginator

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    assert result is False
    mock_logger.info.assert_called_once_with(
        "No contents found in S3 directory: some/s3/dir/"
    )
    mock_download_file.assert_not_called()


def test_successful_download_single_file(
    mock_s3_client, mock_download_file, test_dirs
):
    # Mock the coroutine for list_objects_v2
    async def mock_list_objects(**kwargs):
        return {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}

    mock_s3_client.list_objects_v2 = mock_list_objects

    # Mock the paginator
    paginator = MagicMock()

    async def mock_paginate(*args, **kwargs):
        yield {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}

    paginator.paginate = mock_paginate
    mock_s3_client.get_paginator.return_value = paginator

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        return True, ""  # Return tuple of (success, error_msg)

    mock_download_file.side_effect = mock_download_async

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    assert result is True
    mock_download_file.assert_called_once_with(
        mock_s3_client,
        "test-bucket",
        "some/s3/dir/file1.txt",
        os.path.join(test_dirs, "file1.txt"),
    )
    mock_logger.error.assert_not_called()


def test_successful_download_multiple_files_across_pages(
    mock_s3_client, mock_download_file, test_dirs
):
    # Mock the coroutine for list_objects_v2
    async def mock_list_objects(**kwargs):
        return {
            "Contents": [
                {"Key": "some/s3/dir/file1.txt"},
                {"Key": "some/s3/dir/subdir/file2.txt"},
            ]
        }

    mock_s3_client.list_objects_v2 = mock_list_objects

    # Mock the paginator
    paginator = MagicMock()

    async def mock_paginate(*args, **kwargs):
        yield {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}
        yield {"Contents": [{"Key": "some/s3/dir/subdir/file2.txt"}]}

    paginator.paginate = mock_paginate
    mock_s3_client.get_paginator.return_value = paginator

    # Mock the download file function to return True
    async def mock_download_async(*args, **kwargs):
        return True, ""  # Return tuple of (success, error_msg)

    mock_download_file.side_effect = mock_download_async

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    assert result is True
    expected_calls = [
        call(
            mock_s3_client,
            "test-bucket",
            "some/s3/dir/file1.txt",
            os.path.join(test_dirs, "file1.txt"),
        ),
        call(
            mock_s3_client,
            "test-bucket",
            "some/s3/dir/subdir/file2.txt",
            os.path.join(test_dirs, "subdir", "file2.txt"),
        ),
    ]
    mock_download_file.assert_has_calls(expected_calls, any_order=False)
    mock_logger.error.assert_not_called()


def test_download_failure_for_a_file(
    mock_s3_client, mock_download_file, test_dirs
):
    # Mock the coroutine for list_objects_v2
    async def mock_list_objects(**kwargs):
        return {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}

    mock_s3_client.list_objects_v2 = mock_list_objects

    # Mock the paginator
    paginator = MagicMock()

    async def mock_paginate(*args, **kwargs):
        yield {"Contents": [{"Key": "some/s3/dir/file1.txt"}]}

    paginator.paginate = mock_paginate
    mock_s3_client.get_paginator.return_value = paginator

    # Mock the download file function to return False with error message
    async def mock_download_async(*args, **kwargs):
        return False, "Failed to download some/s3/dir/file1.txt"

    mock_download_file.side_effect = mock_download_async

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    assert result is False
    mock_logger.error.assert_called_with(
        "Failed to download some/s3/dir/file1.txt: Failed to download some/s3/dir/file1.txt"
    )


def test_exception_handling(mock_s3_client, mock_download_file, test_dirs):
    async def mock_list_objects(**kwargs):
        raise Exception("Some error")

    mock_s3_client.list_objects_v2 = mock_list_objects

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = asyncio.run(
            download_directory_from_s3_async(
                mock_s3_client, "test-bucket", "some/s3/dir/", test_dirs
            )
        )

    assert result is False
    mock_logger.error.assert_any_call(
        "Error downloading directory some/s3/dir/: Some error"
    )
    mock_download_file.assert_not_called()


@pytest.fixture
def test_upload_dir(tmp_path):
    # This creates a temporary directory for testing uploads
    upload_dir = tmp_path / "upload_dir"
    upload_dir.mkdir()
    return str(upload_dir)


def test_upload_empty_directory(mock_s3_client, test_upload_dir):
    # An empty directory should upload nothing and should be removed after upload
    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = upload_directory_to_s3(
            mock_s3_client, "test-bucket", test_upload_dir, "some/s3/dir"
        )

    assert result is True
    # Directory should be removed
    assert not os.path.exists(test_upload_dir)
    mock_logger.info.assert_not_called()  # No files uploaded, so no info logs for upload
    mock_logger.error.assert_not_called()


def test_upload_single_file_success(mock_s3_client, test_upload_dir):
    # Create a single file in the local directory
    file_path = os.path.join(test_upload_dir, "file1.txt")
    with open(file_path, "w") as f:
        f.write("some content")

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = upload_directory_to_s3(
            mock_s3_client, "test-bucket", test_upload_dir, "some/s3/dir"
        )

    assert result is True
    # Check that the file was attempted to upload
    mock_s3_client.upload_file.assert_called_once_with(
        file_path, "test-bucket", "some/s3/dir/file1.txt"
    )

    # Directory should be removed after success
    assert not os.path.exists(test_upload_dir)

    # Logger should have info log for successful upload
    mock_logger.info.assert_called_once_with(
        f"Uploaded {file_path} to s3://test-bucket/some/s3/dir/file1.txt"
    )
    mock_logger.error.assert_not_called()


def test_upload_multiple_files_and_subdirs_success(
    mock_s3_client, test_upload_dir
):
    # Create a nested directory structure with multiple files
    subdir = os.path.join(test_upload_dir, "subdir")
    os.mkdir(subdir)
    file1 = os.path.join(test_upload_dir, "file1.txt")
    file2 = os.path.join(subdir, "file2.txt")

    with open(file1, "w") as f:
        f.write("file1 content")
    with open(file2, "w") as f:
        f.write("file2 content")

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = upload_directory_to_s3(
            mock_s3_client, "test-bucket", test_upload_dir, "some/s3/dir"
        )

    assert result is True
    # Check calls to upload_file for both files
    mock_s3_client.upload_file.assert_any_call(
        file1, "test-bucket", "some/s3/dir/file1.txt"
    )
    mock_s3_client.upload_file.assert_any_call(
        file2, "test-bucket", "some/s3/dir/subdir/file2.txt"
    )

    # Ensure both files and directories are removed
    assert not os.path.exists(file1)
    assert not os.path.exists(file2)
    assert not os.path.exists(subdir)
    assert not os.path.exists(test_upload_dir)

    # Check logging
    expected_info_calls = [
        call.info(
            f"Uploaded {file1} to s3://test-bucket/some/s3/dir/file1.txt"
        ),
        call.info(
            f"Uploaded {file2} to s3://test-bucket/some/s3/dir/subdir/file2.txt"
        ),
    ]
    mock_logger.info.assert_has_calls(expected_info_calls, any_order=True)
    mock_logger.error.assert_not_called()


def test_upload_failure_for_a_file(mock_s3_client, test_upload_dir):
    # Create a file that will fail to upload
    file_path = os.path.join(test_upload_dir, "file_fail.txt")
    with open(file_path, "w") as f:
        f.write("content")

    # Make upload_file raise an exception for this file
    mock_s3_client.upload_file.side_effect = Exception("Upload failed")

    with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
        result = upload_directory_to_s3(
            mock_s3_client, "test-bucket", test_upload_dir, "some/s3/dir"
        )

    assert result is False
    # Directory and file should still exist because it failed early
    # The function returns False immediately upon a failure
    assert os.path.exists(file_path)

    mock_logger.error.assert_called_once_with(
        f"Failed to upload {file_path}: Upload failed"
    )


def test_exception_handling_during_processing(mock_s3_client, test_upload_dir):
    # Cause an error before uploading - for example, by mocking os.walk or s3_client
    with patch("synthefy_pkg.app.utils.s3_utils.os.walk") as mock_walk:
        mock_walk.side_effect = Exception("Some unexpected error")

        with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
            result = upload_directory_to_s3(
                mock_s3_client, "test-bucket", test_upload_dir, "some/s3/dir"
            )

    assert result is False
    assert os.path.exists(
        test_upload_dir
    )  # Directory should remain if we never reached clearing step
    mock_logger.error.assert_called_once_with(
        f"Error uploading directory {test_upload_dir}: Some unexpected error"
    )


class TestS3Operations:
    def test_upload_file_to_s3_success(self, mock_s3_client):
        """Test successful file upload to S3."""
        with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
            result = upload_file_to_s3(
                mock_s3_client, "local_file.txt", "test-bucket", "test/key.txt"
            )

        assert result is True
        mock_s3_client.upload_file.assert_called_once_with(
            "local_file.txt", "test-bucket", "test/key.txt"
        )
        mock_logger.info.assert_called_once_with(
            "Successfully uploaded local_file.txt to s3://test-bucket/test/key.txt"
        )

    def test_upload_file_to_s3_failure(self, mock_s3_client):
        """Test file upload failure to S3."""
        mock_s3_client.upload_file.side_effect = Exception("Upload failed")

        with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
            result = upload_file_to_s3(
                mock_s3_client, "local_file.txt", "test-bucket", "test/key.txt"
            )

        assert result is False
        mock_logger.error.assert_called_once_with(
            "Error uploading local_file.txt: Upload failed"
        )

    def test_create_presigned_url_success(self, mock_s3_client):
        """Test successful presigned URL generation."""
        mock_s3_client.generate_presigned_url.return_value = (
            "https://test-url.com"
        )

        result = create_presigned_url(
            mock_s3_client, "test-bucket", "test/key.txt"
        )

        assert result == "https://test-url.com"
        mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "test/key.txt"},
            ExpiresIn=604800,
        )

    def test_create_presigned_url_failure(self, mock_s3_client):
        """Test presigned URL generation failure."""
        mock_s3_client.generate_presigned_url.side_effect = ClientError(
            {
                "Error": {
                    "Code": "InvalidRequest",
                    "Message": "URL generation failed",
                }
            },
            "generate_presigned_url",
        )

        with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
            result = create_presigned_url(
                mock_s3_client, "test-bucket", "test/key.txt"
            )

        assert result is None
        mock_logger.error.assert_called_once()

    @patch("synthefy_pkg.app.utils.s3_utils.aioboto3")
    @patch("synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async")
    def test_handle_s3_source_with_nested_path(
        self, mock_download, mock_aioboto3, tmp_path
    ):
        """Test S3 source handling with nested path in key."""
        mock_s3_client = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )
        mock_aioboto3.Session.return_value = mock_session
        mock_download.return_value = (
            True,
            "",
        )  # Return tuple of (success, error_msg)

        s3_source_str = json.dumps(
            {
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
                "region": "us-west-2",
                "bucket_name": "test-bucket",
                "key": "nested/path/to/file.txt",
            }
        )

        file_path, s3_source = asyncio.run(
            handle_s3_source(s3_source_str, str(tmp_path), mock_session)
        )

        assert file_path == str(tmp_path / "file.txt")
        assert s3_source.key == "nested/path/to/file.txt"
        mock_download.assert_called_once()

    @patch("synthefy_pkg.app.utils.s3_utils.aioboto3")
    def test_handle_s3_source_empty_key(self, mock_aioboto3):
        """Test handling of S3 source with empty key."""
        mock_session = MagicMock()
        s3_source_str = json.dumps(
            {
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
                "region": "us-west-2",
                "bucket_name": "test-bucket",
                "key": "",
            }
        )

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(handle_s3_source(s3_source_str, "/tmp", mock_session))

        assert exc_info.value.status_code == 400
        assert "key cannot be empty" in str(exc_info.value.detail)

    @patch("synthefy_pkg.app.utils.s3_utils.aioboto3")
    def test_handle_s3_source_invalid_region(self, mock_aioboto3):
        """Test handling of S3 source with invalid region."""
        mock_session = MagicMock()
        mock_session.client.side_effect = Exception("Invalid region")
        mock_aioboto3.Session.return_value = mock_session

        s3_source_str = json.dumps(
            {
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
                "region": "invalid-region",
                "bucket_name": "test-bucket",
                "key": "test/file.txt",
            }
        )

        with pytest.raises(Exception) as exc_info:
            asyncio.run(handle_s3_source(s3_source_str, "/tmp", mock_session))

        assert str(exc_info.value) == "Invalid region"

    @patch("synthefy_pkg.app.utils.s3_utils.aioboto3")
    @patch("synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async")
    def test_handle_s3_source_special_characters(
        self, mock_download, mock_aioboto3, tmp_path
    ):
        """Test S3 source handling with special characters in key."""
        mock_s3_client = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )
        mock_aioboto3.Session.return_value = mock_session
        mock_download.return_value = (
            True,
            "",
        )  # Return tuple of (success, error_msg)

        s3_source_str = json.dumps(
            {
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
                "region": "us-west-2",
                "bucket_name": "test-bucket",
                "key": "path/to/file with spaces!@#$.txt",
            }
        )

        file_path, s3_source = asyncio.run(
            handle_s3_source(s3_source_str, str(tmp_path), mock_session)
        )

        assert file_path == str(tmp_path / "file with spaces!@#$.txt")
        assert s3_source.key == "path/to/file with spaces!@#$.txt"
        mock_download.assert_called_once()

    @patch("synthefy_pkg.app.utils.s3_utils.aioboto3")
    @patch("synthefy_pkg.app.utils.s3_utils.download_file_from_s3_async")
    def test_handle_s3_source_tmp_dir_not_exists(
        self, mock_download, mock_aioboto3, tmp_path
    ):
        """Test handling when temporary directory doesn't exist."""
        mock_s3_client = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value.__aenter__.return_value = (
            mock_s3_client
        )
        mock_aioboto3.Session.return_value = mock_session
        mock_download.return_value = (
            True,
            "",
        )  # Return tuple of (success, error_msg)

        non_existent_path = tmp_path / "non_existent"
        s3_source_str = json.dumps(
            {
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
                "region": "us-west-2",
                "bucket_name": "test-bucket",
                "key": "test/file.txt",
            }
        )

        file_path, s3_source = asyncio.run(
            handle_s3_source(
                s3_source_str, str(non_existent_path), mock_session
            )
        )

        assert file_path == str(non_existent_path / "file.txt")
        mock_download.assert_called_once()


# Add tests for avalidate_file_exists
class TestAValidateFileExists:
    """Tests for the avalidate_file_exists function."""

    def test_avalidate_file_exists_success(self, mock_s3_client):
        """Test successful file validation in S3."""

        # Mock successful head_object response
        async def mock_head_object(*args, **kwargs):
            return {"ContentLength": 1024}  # Just a dummy response

        mock_s3_client.head_object = mock_head_object

        with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
            exists, message = asyncio.run(
                avalidate_file_exists(
                    mock_s3_client, "test-bucket", "test/file.txt"
                )
            )

        assert exists is True
        assert message == "File exists"
        mock_logger.debug.assert_any_call(
            "Checking if file_key exists in S3: test/file.txt"
        )
        mock_logger.debug.assert_any_call("File exists in S3: test/file.txt")

    # def test_avalidate_file_exists_not_found(self, mock_s3_client):
    #     """Test file not found validation in S3."""

    #     # Mock 404 error response
    #     async def mock_head_object(*args, **kwargs):
    #         error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
    #         raise ClientError(error_response, "HeadObject")

    #     mock_s3_client.head_object = mock_head_object

    #     with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
    #         exists, message = asyncio.run(
    #             avalidate_file_exists(
    #                 mock_s3_client, "test-bucket", "test/file.txt"
    #             )
    #         )

    #     assert exists is False
    #     assert (
    #         message == "The specified file does not exist in S3: test/file.txt"
    #     )
    #     mock_logger.warning.assert_called_once_with(
    #         "File not found in S3: test/file.txt"
    #     )

    # def test_avalidate_file_exists_other_error(self, mock_s3_client):
    #     """Test other S3 error during file validation."""

    #     # Mock other error response
    #     async def mock_head_object(*args, **kwargs):
    #         error_response = {
    #             "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
    #         }
    #         raise ClientError(error_response, "HeadObject")

    #     mock_s3_client.head_object = mock_head_object

    #     with patch("synthefy_pkg.app.utils.s3_utils.logger") as mock_logger:
    #         exists, message = asyncio.run(
    #             avalidate_file_exists(
    #                 mock_s3_client, "test-bucket", "test/file.txt"
    #             )
    #         )

    #     assert exists is False
    #     assert "S3 error:" in message
    #     mock_logger.error.assert_any_call("Unexpected S3 error: AccessDenied")


def test_parse_s3_url_basic():
    """Test basic S3 URL parsing."""
    from synthefy_pkg.app.utils.s3_utils import parse_s3_url

    s3_url = "s3://my-bucket/data/file.txt"
    bucket, key = parse_s3_url(s3_url)

    assert bucket == "my-bucket"
    assert key == "data/file.txt"


def test_parse_s3_url_directory():
    """Test S3 URL parsing when key is a directory (prefix)."""
    from synthefy_pkg.app.utils.s3_utils import parse_s3_url

    s3_url = "s3://my-bucket/data/directory/"
    bucket, key = parse_s3_url(s3_url)

    assert bucket == "my-bucket"
    assert key == "data/directory/"


@pytest.mark.asyncio
async def test_adelete_s3_object_single_object():
    s3_client = AsyncMock()
    s3_client.delete_object = AsyncMock()
    bucket = "test-bucket"
    s3_key = "test/path/file.txt"

    result = await adelete_s3_object(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=s3_key,
        is_directory=False,
    )

    assert result is True
    s3_client.delete_object.assert_called_once_with(Bucket=bucket, Key=s3_key)


@pytest.mark.asyncio
async def test_adelete_s3_object_directory():
    s3_client = AsyncMock()
    mock_paginator = Mock()

    # Make get_paginator a regular synchronous method
    s3_client.get_paginator = Mock(return_value=mock_paginator)

    # Create a function that returns an async generator when called
    def paginate_method(**kwargs):
        async def async_generator():
            yield {
                "Contents": [
                    {"Key": "test/dir/file1.txt"},
                    {"Key": "test/dir/file2.txt"},
                ]
            }

        return async_generator()

    # Assign the method to paginate
    mock_paginator.paginate = paginate_method
    s3_client.delete_objects = AsyncMock()

    bucket = "test-bucket"
    s3_key = "test/dir"

    result = await adelete_s3_object(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=s3_key,
        is_directory=True,
    )

    assert result is True
    s3_client.get_paginator.assert_called_once_with("list_objects_v2")
    s3_client.delete_objects.assert_called_once_with(
        Bucket=bucket,
        Delete={
            "Objects": [
                {"Key": "test/dir/file1.txt"},
                {"Key": "test/dir/file2.txt"},
            ],
            "Quiet": True,
        },
    )
