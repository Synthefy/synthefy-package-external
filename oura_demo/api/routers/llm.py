"""
LLM router - Endpoints for LLM-powered data modifications.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models import (
    DataFrameModel,
    LLMModifyRequest,
    LLMModifyResponse,
)
from services.config_loader import get_config_loader
from services.llm_service import get_llm_service

router = APIRouter(prefix="/api/llm", tags=["LLM"])


@router.post("/modify", response_model=LLMModifyResponse)
async def modify_data(request: LLMModifyRequest) -> LLMModifyResponse:
    """Modify data using LLM-generated code.

    The LLM will analyze the user's request and generate Python code
    to modify the DataFrame. The code is then executed and the
    modified data is returned.

    Args:
        request: LLMModifyRequest with dataset_name, data, and user_query

    Returns:
        LLMModifyResponse with modified data, executed code, and explanation

    Raises:
        HTTPException: If LLM service fails or code execution fails
    """
    logger.info(f"LLM modify request for dataset: {request.dataset_name.value}")
    logger.info(f"User query: {request.user_query}")

    # Get config for column context
    try:
        loader = get_config_loader(request.dataset_name.value)
        required_columns = loader.get_required_columns()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert to DataFrame
    df = request.data.to_dataframe()
    logger.info(f"Input DataFrame shape: {df.shape}")

    # Get LLM service and modify
    llm_service = get_llm_service()

    try:
        modified_df, code_executed, explanation = llm_service.modify_data(
            df=df,
            required_columns=required_columns,
            user_query=request.user_query,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("LLM modification failed")
        raise HTTPException(
            status_code=500,
            detail=f"LLM modification failed: {str(e)}",
        )

    # Convert back to DataFrameModel
    modified_data = DataFrameModel.from_dataframe(modified_df)

    logger.info(f"Modified DataFrame shape: {modified_df.shape}")

    return LLMModifyResponse(
        modified_data=modified_data,
        code_executed=code_executed,
        explanation=explanation,
    )
