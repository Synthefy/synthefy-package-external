import asyncio
import calendar
import json  # To pretty-print metadata for the prompt
import os
import re
from base64 import b64decode
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from loguru import logger
from pinecone import PineconeAsyncio
from pinecone.core.openapi.db_data.models import QueryResponse
from pydantic import SecretStr

from synthefy_pkg.app.config import (
    MetadataEmbeddingSettings,
    MetadataRagSettings,
)
from synthefy_pkg.app.data_models import (
    HaverDatasetMatch,
    HaverMetadataAccessInfo,
    ReducedMetadata,
)
from synthefy_pkg.app.utils.api_utils import get_settings

INDEX_NAME = "metadata-embeddings"
INDEX_NAME_V2 = "metadata-embeddings-v2"

_metadata_embedding_index_host = None


def get_metadata_embedding_index_host():
    global _metadata_embedding_index_host
    if _metadata_embedding_index_host is None:
        _metadata_embedding_index_host = get_settings(
            MetadataEmbeddingSettings
        ).metadata_index_host
    return _metadata_embedding_index_host


# Singleton for the LLM chain
_llm_chain = None


class FlexibleJsonOutputParser(BaseOutputParser):
    """
    A custom JSON output parser that can handle responses with explanatory text
    followed by JSON objects. Falls back to extracting JSON from mixed text.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the text, trying direct JSON first, then extracting from mixed text."""
        try:
            # First try to parse as pure JSON
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from mixed text
            return self._extract_json_from_text(text)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON object from text that may contain explanatory content."""
        # Look for the last occurrence of { and match it with }
        # This handles cases where JSON is at the end of explanatory text

        # Find all potential JSON objects (text between { and })
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if not matches:
            # If no braces found, try to find just the last line that might be JSON
            lines = text.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

            raise ValueError(f"No valid JSON found in text: {text[:200]}...")

        # Try to parse each match, starting from the last (most likely to be the final JSON)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If all else fails, try a more aggressive approach
        # Look for the last { and find its matching }
        last_brace_start = text.rfind("{")
        if last_brace_start != -1:
            # Find the matching closing brace
            brace_count = 0
            for i in range(last_brace_start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_substring = text[last_brace_start : i + 1]
                        try:
                            return json.loads(json_substring)
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"Could not extract valid JSON from text: {text[:200]}..."
        )


def get_llm_chain() -> Optional[Any]:
    global _llm_chain
    if _llm_chain is None:
        # Define the prompt template using PromptTemplate
        prompt_template_string = """
        You are tasked with selecting the top {k} metadata entries from a list of retrieved metadata entries. You need to carefully evaluate which metadata entries are most directly related to and highly correlated with the given covariates.

        **Covariates (variables to consider):** {array_of_covariates}

        **Instructions:**
        1. Analyze each metadata entry and determine how strongly it correlates with or is influenced by the given covariates.
        2. Select the {k} entries that have the strongest relationship, highest correlation, or most direct influence from the covariates.
        3. Prioritize entries where the covariates would be a significant predictor or influencing factor.
        4. Consider both direct relationships (e.g., if covariate is "temperature" and entry is "heating costs") and strong indirect relationships (e.g., if covariate is "fuel prices" and entry is "consumer price index").

        **Retrieved Metadata Entries:**
        {formatted_metadata_list}

        **Output Requirements:**
        - Return a JSON object with two keys:
          - "selected_indices": A list of integers representing the 0-based indices of the selected entries (e.g., [0, 2, 5])
          - "explanation": A brief explanation of why these entries were selected and how they relate to the covariates

        **Example Output:**
        {{
            "selected_indices": [0, 2, 5],
            "explanation": "Selected entries show strong correlation with the given covariates because..."
        }}
        """

        prompt = PromptTemplate(
            input_variables=[
                "k",
                "array_of_covariates",
                "formatted_metadata_list",
            ],
            template=prompt_template_string,
        )

        llm = get_llm()
        if llm:
            # Using LCEL: prompt template | llm | custom flexible output parser
            _llm_chain = prompt | llm | FlexibleJsonOutputParser()
            logger.info(
                "LCEL chain created successfully with FlexibleJsonOutputParser."
            )
        else:
            logger.error(
                "LLM chain not created due to LLM initialization failure."
            )
            _llm_chain = None

    return _llm_chain


def get_embeddings_model():
    """Initialize and return a Gemini embeddings model using LangChain."""
    # Decode the base64 encoded API key
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(get_gemini_api_key()),
        task_type="retrieval_document",
    )


def get_pinecone_api_key():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set")
    return pinecone_api_key


def get_gemini_api_key():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return gemini_api_key


def get_llm():
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash-lite",
            google_api_key=get_gemini_api_key(),
        )
        logger.info("Successfully initialized AI service with primary model")
    except Exception as e:
        logger.error(f"Error initializing ChatGoogleGenerativeAI: {e}")
        logger.error("Please ensure:")
        logger.error(
            "1. Your AI service configuration is correctly set and active."
        )
        logger.error(
            "2. The model configuration is correct and available for your service."
        )
        llm = None  # Set llm to None if initialization fails
    return llm


class MatchObject:
    def __init__(self, id, metadata):
        self.id = id
        self.metadata = metadata


# 4. Prepare your data and run the chain
def _process_selected_indices_to_reduced_metadata(
    selected_indices: List[int], retrieved_matches: List[MatchObject]
) -> List[ReducedMetadata]:
    """
    Process selected indices from LLM response and convert to ReducedMetadata objects.

    Args:
        selected_indices: List of indices selected by the LLM
        retrieved_matches: List of retrieved match objects

    Returns:
        List of ReducedMetadata objects for valid indices
    """
    top_k_metadata = []
    logger.info(f"Selected indices: {selected_indices}")
    for i in selected_indices:
        if i < len(retrieved_matches) and i >= 0:
            reduced_metadata = ReducedMetadata(
                description=retrieved_matches[i].metadata.get(
                    "description", None
                ),
                name=retrieved_matches[i].metadata.get("name", None),
                dataType=retrieved_matches[i].metadata.get("dataType", None),
                startDate=str(
                    int(retrieved_matches[i].metadata.get("startDate", -1))
                ),
                databaseName=retrieved_matches[i].metadata.get(
                    "databaseName", None
                ),
                frequency=str(
                    int(retrieved_matches[i].metadata.get("frequency", -1))
                ),
            )
            top_k_metadata.append(reduced_metadata)
            logger.info(f"Selected index: {i}")
            logger.info(f"Retrieved matches: {retrieved_matches[i].metadata}")
        else:
            logger.error(
                f"The LLM returned an index that is out of range, {i}, and does not actually exist in the list of retrieved matches"
            )

    return top_k_metadata


async def get_best_k_metadata_by_llm(
    retrieved_matches: List[MatchObject], array_of_covariates: list, k: int
) -> List[ReducedMetadata]:
    """
    Uses an LLM to select the k best metadata entries from a list of retrieved matches.
    """
    chain = get_llm_chain()
    if not chain:
        logger.error("LLM chain is not initialized. Cannot proceed.")
        return []
    if not retrieved_matches:
        return []

    formatted_items = []
    for i, match in enumerate(retrieved_matches):
        reduced_metadata = {
            "description": match.metadata["description"],
            "name": match.metadata["name"],
            "dataType": match.metadata["dataType"],
            "startDate": match.metadata["startDate"],
            "databaseName": match.metadata["databaseName"],
            "frequency": match.metadata["frequency"],
        }
        metadata_str = json.dumps(reduced_metadata, indent=4)
        formatted_items.append(
            f"Entry {i + 1} (ID: {match.id}):\n{metadata_str}"
        )

    formatted_metadata_list_str = "\n\n".join(formatted_items)
    covariates_str = ", ".join(array_of_covariates)

    try:
        # Use ainvoke for async invocation
        llm_response = await chain.ainvoke(
            {
                "k": k,
                "array_of_covariates": covariates_str,
                "formatted_metadata_list": formatted_metadata_list_str,
            }
        )

        # Parse the response
        if (
            isinstance(llm_response, dict)
            and "selected_indices" in llm_response
        ):
            selected_indices = llm_response["selected_indices"]
            return _process_selected_indices_to_reduced_metadata(
                selected_indices, retrieved_matches
            )
        else:
            logger.error(f"Unexpected response format: {llm_response}")
            return []

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(
            f"Inputs to chain.run were: k={k}, covariates='{covariates_str}', metadata_list_length={len(formatted_metadata_list_str)}"
        )
        return []


# --- Example Usage ---
# (This remains the same as the previous version)


async def get_top_k_metadata_embeddings(
    query: str, k: int
) -> List[MatchObject]:
    """
    Retrieves the top k most similar metadata embeddings from Pinecone for a given query string.

    Args:
        query (str): The query string to search for similar metadata
        k (int): Number of top matches to return

    Returns:
        List[MatchObject]: List of MatchObject instances containing the top k matches
    """
    # Get the embeddings model
    embeddings_model = get_embeddings_model()

    # Generate embedding for the query
    query_vector = embeddings_model.embed_query(query)

    # Use async context manager for Pinecone client
    async with PineconeAsyncio(api_key=get_pinecone_api_key()) as pc:
        # Get the index

        async with pc.IndexAsyncio(
            get_metadata_embedding_index_host()
        ) as index:
            # Query Pinecone for similar vectors
            results: QueryResponse = await index.query(
                vector=query_vector, top_k=k, include_metadata=True
            )

            # Convert results to MatchObject instances
            matches = []
            # Type ignore because we know matches exists and is iterable from Pinecone's API
            for match in results.matches:  # type: ignore
                matches.append(
                    MatchObject(id=str(match.id), metadata=dict(match.metadata))
                )

            return matches


# TODO: Need to confirm the structure of startDate with haver
def haver_to_unix(start_date_str: str, frequency: int) -> int:
    """
    Converts a Haver startDate string and frequency into a Unix timestamp.

    Parameters:
    - start_date_str (str or int): Haver-formatted startDate (e.g. "9801", "981", or "1998")
    - frequency (int): Frequency code used by Haver

    Returns:
    - int: Unix timestamp (seconds since 1970-01-01 UTC)
    """
    logger.info(
        f"Converting Haver startDate: {start_date_str} and frequency: {frequency} to Unix timestamp"
    )
    start_date_str = str(start_date_str).zfill(4)

    if frequency == 40:  # Monthly: YYMM
        yy = int(start_date_str[:2])
        mm = int(start_date_str[2:])
        year = 1900 + yy if yy >= 30 else 2000 + yy  # assume 1930–2029 window
        dt = datetime(year, mm, 1)

    elif frequency == 50:  # Quarterly: YYQ
        yy = int(start_date_str[:2])
        q = int(start_date_str[2:])
        year = 1900 + yy if yy >= 30 else 2000 + yy
        month = 3 * (q - 1) + 1
        dt = datetime(year, month, 1)

    elif frequency == 10:  # Annual: YYYY
        year = int(start_date_str)
        dt = datetime(year, 1, 1)

    else:
        raise ValueError(
            "Unsupported or ambiguous frequency — daily/weekly formats need external tools or context."
        )

    return calendar.timegm(dt.timetuple())  # convert to Unix timestamp in UTC


async def get_metadata_recommendations(
    user_prompt: str,
    number_of_metadatasets_to_generate: int = 3,
) -> List[HaverDatasetMatch]:
    """
    Get metadata recommendations based on user prompt.

    Args:
        user_prompt: The user's prompt to generate recommendations for
        number_of_metadatasets_to_generate: Number of metadata recommendations to generate

    Returns:
        List of HaverDatasetMatch objects containing the recommended metadata
    """
    # Get settings
    metadata_rag_settings = get_settings(MetadataRagSettings)

    # Determine number of vectors to retrieve based on settings
    if (
        metadata_rag_settings.use_llm_generation_in_metadata_recommendations
        == 1
    ):
        if number_of_metadatasets_to_generate <= 20:
            number_of_vectors_to_retrieve = (
                number_of_metadatasets_to_generate
                * metadata_rag_settings.number_of_vectors_to_retrieve_before_generation_scale_factor
            )
        else:
            number_of_vectors_to_retrieve = number_of_metadatasets_to_generate
    else:
        number_of_vectors_to_retrieve = number_of_metadatasets_to_generate

    # Get top k matches from vector store
    top_k_matches: List[MatchObject] = await get_top_k_metadata_embeddings(
        user_prompt, number_of_vectors_to_retrieve
    )

    # If LLM generation is disabled or number of datasets > 20, return direct matches
    if (
        metadata_rag_settings.use_llm_generation_in_metadata_recommendations
        != 1
        or number_of_metadatasets_to_generate > 20
    ):
        # Convert matches directly to HaverDatasetMatch objects
        metadata_recommendations = []
        for i, match in enumerate(top_k_matches):
            if i >= number_of_metadatasets_to_generate:
                break
            metadata_recommendations.append(
                HaverDatasetMatch(
                    access_info=HaverMetadataAccessInfo(
                        description=match.metadata.get("description", None),
                        name=match.metadata.get("name", None),
                        start_date=int(match.metadata.get("startDate", 0)),
                        database_name=match.metadata.get("databaseName", None),
                        data_source="haver",
                        file_name=None,
                    )
                )
            )
        return metadata_recommendations

    # Use LLM generation for small requests
    # Get best k metadata recommendations using LLM
    top_k_metadata: List[ReducedMetadata] = await get_best_k_metadata_by_llm(
        top_k_matches,
        user_prompt.split(),
        number_of_metadatasets_to_generate,
    )

    # Convert to HaverDatasetMatch objects
    metadata_recommendations = [
        HaverDatasetMatch(
            access_info=HaverMetadataAccessInfo(
                description=metadata.description,
                name=metadata.name,
                start_date=int(metadata.startDate),
                database_name=metadata.databaseName,
                data_source="haver",
                file_name=None,
            )
        )
        for metadata in top_k_metadata
    ]

    true_length = min(
        number_of_metadatasets_to_generate, len(metadata_recommendations)
    )
    return metadata_recommendations[:true_length]


if __name__ == "__main__":

    async def main():
        recommendations = await get_metadata_recommendations(
            user_prompt="I want to build a model that predicts the stock market",
        )
        print(recommendations)

    asyncio.run(main())
