import os
import pytest
from unittest.mock import Mock, patch, call
from fastapi import HTTPException

from synthefy_pkg.app.services.preprocess_service import PreprocessService
from synthefy_pkg.app.data_models import AWSInfo
from synthefy_pkg.app.config import PreprocessSettings

@pytest.fixture
def preprocess_service():
    settings = PreprocessSettings(
        dataset_name = "mock-dataset",
        dataset_path="/tmp/test",
        max_file_size=1000000,
        bucket_name="test-bucket",
        preprocess_config_path="/tmp/config.json",
        json_save_path="/tmp/save.json"
    )
    return PreprocessService(settings)

@pytest.fixture
def aws_info():
    return AWSInfo(
        bucket_name="test-bucket",
        user_id="test-customer",
        dataset_name="test-dataset"
    )

@pytest.fixture
def mock_directory_structure(tmp_path):
    # Create a temporary directory structure
    base_dir = tmp_path / "test_output"
    subdir = base_dir / "subdir"
    subdir.mkdir(parents=True)
    
    # Create some test files
    (base_dir / "file1.txt").write_text("test1")
    (subdir / "file2.txt").write_text("test2")
    
    return str(base_dir)

def test_upload_to_s3_success(preprocess_service, aws_info, mock_directory_structure):
    with patch('boto3.client') as mock_boto3:
        mock_s3 = Mock()
        mock_boto3.return_value = mock_s3
        
        preprocess_service._upload_to_s3(mock_directory_structure, aws_info)
        
        # Verify S3 upload calls
        expected_calls = [
            call(
                os.path.join(mock_directory_structure, 'file1.txt'),
                'test-bucket',
                'test-customer/test-dataset/file1.txt'
            ),
            call(
                os.path.join(mock_directory_structure, 'subdir/file2.txt'),
                'test-bucket',
                'test-customer/test-dataset/subdir/file2.txt'
            )
        ]
        mock_s3.upload_file.assert_has_calls(expected_calls, any_order=True)
        
        # Verify directory cleanup
        assert not os.path.exists(mock_directory_structure)

def test_upload_to_s3_upload_failure(preprocess_service, aws_info, mock_directory_structure):
    with patch('boto3.client') as mock_boto3:
        mock_s3 = Mock()
        mock_s3.upload_file.side_effect = Exception("Upload failed")
        mock_boto3.return_value = mock_s3
        
        with pytest.raises(HTTPException) as exc_info:
            preprocess_service._upload_to_s3(mock_directory_structure, aws_info)
        
        assert exc_info.value.status_code == 500
        assert "Failed to upload files to S3" in str(exc_info.value.detail)
        
        # Verify files still exist (weren't deleted due to upload failure)
        assert os.path.exists(mock_directory_structure)

def test_upload_to_s3_empty_directory(preprocess_service, aws_info, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    with patch('boto3.client') as mock_boto3:
        mock_s3 = Mock()
        mock_boto3.return_value = mock_s3
        
        preprocess_service._upload_to_s3(str(empty_dir), aws_info)
        
        # Verify no uploads were attempted
        mock_s3.upload_file.assert_not_called()
        
        # Verify directory was removed
        assert not os.path.exists(empty_dir) 

