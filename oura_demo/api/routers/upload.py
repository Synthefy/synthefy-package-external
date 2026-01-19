"""
Upload router - Endpoints for file upload and validation.
"""

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from models import (
    ColumnValidationResult,
    DataFrameModel,
    DatasetName,
    FileType,
    UploadResponse,
)
from services.config_loader import get_config_loader

router = APIRouter(prefix="/api/upload", tags=["Upload"])


def detect_file_type(filename: str) -> FileType:
    """Detect file type from filename extension.

    Args:
        filename: Name of the uploaded file

    Returns:
        FileType enum value

    Raises:
        ValueError: If file type is not supported
    """
    if filename.endswith(".parquet"):
        return FileType.PARQUET
    elif filename.endswith(".csv"):
        return FileType.CSV
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Use .parquet or .csv"
        )


def read_uploaded_file(
    file_content: bytes, file_type: FileType
) -> pd.DataFrame:
    """Read uploaded file content into a DataFrame.

    Args:
        file_content: Raw file content as bytes
        file_type: Type of file (parquet or csv)

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file cannot be read
    """
    try:
        if file_type == FileType.PARQUET:
            return pd.read_parquet(io.BytesIO(file_content))
        else:  # CSV
            return pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


@router.post("/{dataset_name}", response_model=UploadResponse)
async def upload_file(
    dataset_name: DatasetName,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a parquet or CSV file and validate columns.

    Args:
        dataset_name: Dataset to validate against (e.g., oura, oura_subset, ppg)
        file: The uploaded file (parquet or csv)

    Returns:
        UploadResponse with parsed data and validation results

    Raises:
        HTTPException: If file type is invalid or validation fails
    """
    # Validate file name
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Detect file type
    try:
        file_type = detect_file_type(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Read file content
    content = await file.read()
    logger.info(f"Uploaded file: {file.filename}, size: {len(content)} bytes")

    # Parse file
    try:
        df = read_uploaded_file(content, file_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Parsed DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Load config and validate
    try:
        loader = get_config_loader(dataset_name.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    validation_result = loader.validate_columns(list(df.columns))

    # Convert to DataFrameModel
    data = DataFrameModel.from_dataframe(df)

    return UploadResponse(
        data=data,
        window_size=loader.get_window_size(),
        validation=ColumnValidationResult(
            valid=validation_result["valid"],
            missing_columns=validation_result["missing_columns"],
            extra_columns=validation_result["extra_columns"],
        ),
        file_type=file_type,
    )
