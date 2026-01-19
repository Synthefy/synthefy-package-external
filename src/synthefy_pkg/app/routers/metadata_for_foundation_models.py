from typing import List

from fastapi import APIRouter

from synthefy_pkg.app.data_models import (
    HaverDatasetMatch,
)
from synthefy_pkg.app.metadata.metadata_rag_service import (
    get_metadata_recommendations,
)
from synthefy_pkg.app.utils.s3_utils import get_aioboto3_session

router = APIRouter(tags=["Metadata For Foundation Models"])

NUMBER_OF_METADATASETS_TO_GENERATE = 3


@router.get(
    "/api/foundation_models/metadata_recommendations",
    response_model=List[HaverDatasetMatch],
)
async def get_metadata_recommendations_endpoint(
    user_prompt: str,
    number_of_datasets_to_return: int = NUMBER_OF_METADATASETS_TO_GENERATE,
) -> List[HaverDatasetMatch]:
    """
    Get metadata recommendations based on user prompt.

    Args:
        user_prompt: The user's prompt to generate recommendations for
        number_of_datasets_to_return: Number of dataset recommendations to return

    Returns:
        List of HaverDatasetMatch objects containing the recommended metadata
    """
    return await get_metadata_recommendations(
        user_prompt=user_prompt,
        number_of_metadatasets_to_generate=number_of_datasets_to_return,
    )


if __name__ == "__main__":

    async def main():
        recommendations = await get_metadata_recommendations_endpoint(
            user_prompt="I want to build a model that predicts the stock market",
        )
        print(recommendations)

    import asyncio

    asyncio.run(main())
