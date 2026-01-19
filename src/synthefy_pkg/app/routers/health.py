from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    message: str


router = APIRouter(tags=["Health"])


@router.api_route(
    "/api/health", methods=["GET", "HEAD"], response_model=HealthResponse
)
async def health_check() -> HealthResponse:
    """
    Check the health of the API (supports GET and HEAD).
    """
    return HealthResponse(message="OK")
