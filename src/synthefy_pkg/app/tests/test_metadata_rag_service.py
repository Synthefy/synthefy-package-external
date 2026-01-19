# src/synthefy_pkg/app/tests/test_metadata_rag_service.py

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from synthefy_pkg.app.metadata.metadata_rag_service import (
    FlexibleJsonOutputParser,
    MatchObject,
    _process_selected_indices_to_reduced_metadata,
    get_best_k_metadata_by_llm,
    get_embeddings_model,
    get_gemini_api_key,
    get_llm,
    get_llm_chain,
    get_metadata_embedding_index_host,
    get_pinecone_api_key,
    get_top_k_metadata_embeddings,
)

# Test data
SAMPLE_METADATA = {
    "name": "Test Dataset",
    "databaseName": "TestDB",
    "s3_path": "s3://test/path",
    "description": "A test dataset",
    "shortSourceName": "TEST",
    "sourceName": "Test Source",
    "dataType": "float",
    "startDate": "2023",
    "frequency": "40",
}

SAMPLE_MATCHES = [
    MatchObject(id="1", metadata=SAMPLE_METADATA),
    MatchObject(id="2", metadata={**SAMPLE_METADATA, "name": "Test Dataset 2"}),
]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("PINECONE_API_KEY", "test_pinecone_key")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv(
        "SYNTHEFY_CONFIG_PATH",
        "src/synthefy_pkg/app/services/configs/api_config_general_local.yaml",
    )


@pytest.fixture
def mock_settings():
    """Mock settings for metadata embedding."""
    with patch(
        "synthefy_pkg.app.metadata.metadata_rag_service.get_settings"
    ) as mock:
        mock.return_value.metadata_index_host = "https://test-index.pinecone.io"
        yield mock


@pytest.fixture
def mock_embeddings_model():
    """Mock the embeddings model."""
    with patch(
        "synthefy_pkg.app.metadata.metadata_rag_service.GoogleGenerativeAIEmbeddings"
    ) as mock:
        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = [
            0.1,
            0.2,
            0.3,
        ]  # Mock embedding vector
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone client and index."""
    with patch(
        "synthefy_pkg.app.metadata.metadata_rag_service.PineconeAsyncio"
    ) as mock_pc:
        # Create mock index
        mock_index = AsyncMock()
        mock_index.query.return_value = MagicMock(
            matches=[
                MagicMock(id="1", metadata=SAMPLE_METADATA),
                MagicMock(
                    id="2",
                    metadata={**SAMPLE_METADATA, "name": "Test Dataset 2"},
                ),
            ]
        )

        # Create mock index context manager
        mock_index_cm = AsyncMock()
        mock_index_cm.__aenter__.return_value = mock_index

        # Create mock Pinecone client
        mock_pc_instance = AsyncMock()
        mock_pc_instance.IndexAsyncio = AsyncMock(return_value=mock_index_cm)

        # Set up the Pinecone client context manager
        mock_pc.return_value = AsyncMock()
        mock_pc.return_value.__aenter__.return_value = mock_pc_instance

        yield mock_pc


@pytest.fixture
def mock_llm():
    """Mock the LLM chain."""
    with patch(
        "synthefy_pkg.app.metadata.metadata_rag_service.ChatGoogleGenerativeAI"
    ) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


class TestMetadataEmbeddingService:
    """Test suite for metadata embedding service."""

    def test_get_metadata_embedding_index_host(self, mock_settings):
        """Test getting the metadata embedding index host."""
        host = get_metadata_embedding_index_host()
        assert host == "https://test-index.pinecone.io"
        mock_settings.assert_called_once()

    def test_get_embeddings_model(self, mock_env_vars):
        """Test getting the embeddings model."""
        model = get_embeddings_model()
        assert model is not None
        assert model.model == "models/embedding-001"
        assert model.task_type == "retrieval_document"

    @pytest.mark.asyncio
    async def test_get_best_k_metadata_by_llm(self, mock_llm):
        """Test getting the best k metadata entries using LLM."""
        # Mock the chain's ainvoke method
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = {
            "selected_indices": [0, 1],
            "explanation": "These entries are most relevant to the query",
        }

        with patch(
            "synthefy_pkg.app.metadata.metadata_rag_service.get_llm_chain"
        ) as mock_get_chain:
            mock_get_chain.return_value = mock_chain

            result = await get_best_k_metadata_by_llm(
                SAMPLE_MATCHES, ["temperature", "precipitation"], k=2
            )

            assert len(result) == 2
            assert result[0].name == "Test Dataset"
            assert result[1].name == "Test Dataset 2"
            mock_chain.ainvoke.assert_called_once()

    async def test_get_best_k_metadata_by_llm_no_chain(self):
        """Test getting best k metadata when LLM chain is not initialized."""
        with patch(
            "synthefy_pkg.app.metadata.metadata_rag_service.get_llm_chain"
        ) as mock_get_chain:
            mock_get_chain.return_value = None

            result = await get_best_k_metadata_by_llm(
                SAMPLE_MATCHES, ["temperature", "precipitation"], k=1
            )

            assert result == []

    async def test_get_best_k_metadata_by_llm_no_matches(self):
        """Test getting best k metadata when no matches are provided."""
        result = await get_best_k_metadata_by_llm(
            [], ["temperature", "precipitation"], k=1
        )
        assert result == []

    # TODO: Fix this test
    # @pytest.mark.asyncio
    # async def test_get_top_k_metadata_embeddings(self, mock_env_vars, mock_embeddings_model, mock_pinecone):
    #     """Test getting top k metadata embeddings."""
    #     result = await get_top_k_metadata_embeddings("test query", k=2)

    #     assert len(result) == 2
    #     assert isinstance(result[0], MatchObject)
    #     assert result[0].id == "1"
    #     assert result[0].metadata == SAMPLE_METADATA
    #     assert result[1].id == "2"
    #     assert result[1].metadata["name"] == "Test Dataset 2"

    @pytest.mark.asyncio
    async def test_get_top_k_metadata_embeddings_error(
        self, mock_env_vars, mock_embeddings_model, mock_pinecone
    ):
        """Test error handling in get_top_k_metadata_embeddings."""
        mock_pinecone.return_value.__aenter__.return_value.IndexAsyncio.return_value.__aenter__.return_value.query.side_effect = Exception(
            "Test error"
        )

        with pytest.raises(Exception):
            await get_top_k_metadata_embeddings("test query", k=2)

    def test_match_object(self):
        """Test MatchObject class."""
        match = MatchObject(id="1", metadata=SAMPLE_METADATA)
        assert match.id == "1"
        assert match.metadata == SAMPLE_METADATA

    def test_get_pinecone_api_key(self, mock_env_vars):
        """Test getting Pinecone API key."""
        key = get_pinecone_api_key()
        assert key == "test_pinecone_key"

    def test_get_pinecone_api_key_missing(self, monkeypatch):
        """Test getting Pinecone API key when not set."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="PINECONE_API_KEY is not set"):
            get_pinecone_api_key()

    def test_get_gemini_api_key(self, mock_env_vars):
        """Test getting Gemini API key."""
        key = get_gemini_api_key()
        assert key == "test_gemini_key"

    def test_get_gemini_api_key_missing(self, monkeypatch):
        """Test getting Gemini API key when not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            get_gemini_api_key()

    def test_get_llm_success(self, mock_env_vars):
        """Test successful LLM initialization."""
        with patch(
            "synthefy_pkg.app.metadata.metadata_rag_service.ChatGoogleGenerativeAI"
        ) as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            llm = get_llm()
            assert llm is not None
            mock.assert_called_once()

    def test_get_llm_failure(self, mock_env_vars):
        """Test LLM initialization failure."""
        with patch(
            "synthefy_pkg.app.metadata.metadata_rag_service.ChatGoogleGenerativeAI"
        ) as mock_llm:
            # Mock the LLM to raise an exception
            mock_llm.side_effect = Exception("API key invalid")

            llm = get_llm()
            assert llm is None

    def test_flexible_json_output_parser_pure_json(self):
        """Test FlexibleJsonOutputParser with pure JSON input."""
        parser = FlexibleJsonOutputParser()

        json_input = (
            '{"selected_indices": [0, 1, 2], "explanation": "Test explanation"}'
        )
        result = parser.parse(json_input)

        assert result == {
            "selected_indices": [0, 1, 2],
            "explanation": "Test explanation",
        }

    def test_flexible_json_output_parser_mixed_text_with_json(self):
        """Test FlexibleJsonOutputParser with explanatory text followed by JSON."""
        parser = FlexibleJsonOutputParser()

        mixed_input = """
        The covariates provided ("fuel prices") indicate that we are looking for datasets 
        that are highly correlated with or influenced by the cost of fuel. Fuel prices have 
        a significant impact on overall inflation, transportation costs, and the cost of goods 
        and services.

        Based on the analysis, here are the selected entries:

        {
            "selected_indices": [0, 1, 6],
            "explanation": "Fuel prices are a major determinant of overall inflation"
        }
        """

        result = parser.parse(mixed_input)

        assert result == {
            "selected_indices": [0, 1, 6],
            "explanation": "Fuel prices are a major determinant of overall inflation",
        }

    def test_flexible_json_output_parser_json_at_end_of_line(self):
        """Test FlexibleJsonOutputParser with JSON on the last line."""
        parser = FlexibleJsonOutputParser()

        input_text = """
        Here is some explanatory text.
        More explanation here.
        {"selected_indices": [2, 4], "explanation": "Final result"}
        """

        result = parser.parse(input_text)

        assert result == {
            "selected_indices": [2, 4],
            "explanation": "Final result",
        }

    def test_flexible_json_output_parser_nested_json(self):
        """Test FlexibleJsonOutputParser with nested JSON objects."""
        parser = FlexibleJsonOutputParser()

        input_text = """
        Analysis complete. Here are the results:
        
        {
            "selected_indices": [1, 3],
            "explanation": "Selected based on correlation",
            "metadata": {
                "confidence": 0.95,
                "method": "llm_analysis"
            }
        }
        """

        result = parser.parse(input_text)

        assert result == {
            "selected_indices": [1, 3],
            "explanation": "Selected based on correlation",
            "metadata": {"confidence": 0.95, "method": "llm_analysis"},
        }

    def test_flexible_json_output_parser_multiple_json_objects(self):
        """Test FlexibleJsonOutputParser chooses the last JSON object when multiple exist."""
        parser = FlexibleJsonOutputParser()

        input_text = """
        First analysis: {"temp": "value"}
        
        After further consideration:
        
        {
            "selected_indices": [0, 2, 5],
            "explanation": "Final selection after analysis"
        }
        """

        result = parser.parse(input_text)

        # Should pick the last (most complete) JSON object
        assert result == {
            "selected_indices": [0, 2, 5],
            "explanation": "Final selection after analysis",
        }

    def test_flexible_json_output_parser_json_with_whitespace(self):
        """Test FlexibleJsonOutputParser handles JSON with extra whitespace."""
        parser = FlexibleJsonOutputParser()

        input_text = """
        The analysis shows:
        
        
        {
            "selected_indices": [ 1, 2, 3 ],
            "explanation": "These are the best matches"
        }
        
        
        """

        result = parser.parse(input_text)

        assert result == {
            "selected_indices": [1, 2, 3],
            "explanation": "These are the best matches",
        }

    def test_flexible_json_output_parser_complex_real_example(self):
        """Test FlexibleJsonOutputParser with the actual example from user query."""
        parser = FlexibleJsonOutputParser()

        real_example = """The covariates provided ("fuel prices") indicate that we are looking for datasets that are highly correlated with or influenced by the cost of fuel. Fuel prices have a significant impact on overall inflation, transportation costs, and the cost of goods and services. Therefore, broad measures of price changes and inflation are the most relevant.

Let's evaluate the available entries:

*   **Entries 1, 2, 3, 5, 6, 9 (Various Consumer Price Indices - CPIs):** Consumer Price Indices (CPI) measure the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. Fuel (gasoline, energy) is a direct component of this basket, and its price also indirectly affects the cost of other goods and services (e.g., transportation costs for goods). Therefore, CPIs are highly correlated with and influenced by fuel prices.
*   **Entry 7 (US: Price Deflator Final Demand):** A price deflator for final demand is a comprehensive measure of price changes across all final goods and services in an economy. This would include energy costs and the impact of fuel prices on production and distribution costs throughout various sectors. This is a very strong candidate as it reflects the broader economic impact of fuel prices.
*   **Entry 4 (US: Domestic Demand Incl Stks):** This measures economic demand, which is indirectly affected by fuel prices, but it's not a direct measure of prices or inflation.
*   **Entry 8 (Portugal: EU Import: Vehicle excl Railway/Tramway Rolling Stock):** This describes vehicle imports, which has a very indirect and weak relationship to fuel prices.

Based on the strong correlation and direct influence, the Consumer Price Indices and the Price Deflator for Final Demand are the most appropriate choices. We need to select 3. To provide a comprehensive view of how fuel prices impact the economy, we will select key US-based price indicators.

1.  **Entry 2 (US: National Consumer Price Index):** This is a primary and widely recognized indicator of inflation in the US, directly reflecting the impact of fuel prices on household expenditures.
2.  **Entry 1 (US: Harmonised Consumer Price Index):** Similar to the National CPI, this is another crucial measure of consumer inflation in the US, also heavily influenced by fuel costs.
3.  **Entry 7 (US: Price Deflator Final Demand):** This is a broader measure of price changes across all final goods and services in the US economy. It captures the comprehensive impact of fuel prices not only on direct consumer purchases but also on the costs of production and distribution across various industries, making it highly correlated.

The selected entries represent the most direct and significant relationships with fuel prices by measuring overall price levels and inflation, which are profoundly affected by energy costs.

{
    "selected_indices": [0, 1, 6],
    "explanation": "Fuel prices are a major determinant of overall inflation and the cost of goods and services. The selected entries (US Harmonised Consumer Price Index, US National Consumer Price Index, and US Price Deflator Final Demand) are all comprehensive measures of price changes and inflation within the economy. These indices directly incorporate or are heavily influenced by energy and transportation costs, which are driven by fuel prices, thus demonstrating a high correlation and direct influence."

}"""

        result = parser.parse(real_example)

        assert result == {
            "selected_indices": [0, 1, 6],
            "explanation": "Fuel prices are a major determinant of overall inflation and the cost of goods and services. The selected entries (US Harmonised Consumer Price Index, US National Consumer Price Index, and US Price Deflator Final Demand) are all comprehensive measures of price changes and inflation within the economy. These indices directly incorporate or are heavily influenced by energy and transportation costs, which are driven by fuel prices, thus demonstrating a high correlation and direct influence.",
        }

    def test_flexible_json_output_parser_no_json_found(self):
        """Test FlexibleJsonOutputParser raises error when no JSON is found."""
        parser = FlexibleJsonOutputParser()

        input_text = "This is just plain text with no JSON objects at all."

        with pytest.raises(ValueError, match="No valid JSON found in text"):
            parser.parse(input_text)

    def test_flexible_json_output_parser_invalid_json(self):
        """Test FlexibleJsonOutputParser raises error when JSON is malformed."""
        parser = FlexibleJsonOutputParser()

        input_text = """
        Here's some analysis:
        
        {
            "selected_indices": [1, 2, 3
            "explanation": "Missing closing brace and comma"
        """

        with pytest.raises(ValueError, match="No valid JSON found in text"):
            parser.parse(input_text)

    def test_flexible_json_output_parser_empty_input(self):
        """Test FlexibleJsonOutputParser handles empty input."""
        parser = FlexibleJsonOutputParser()

        with pytest.raises(ValueError, match="No valid JSON found in text"):
            parser.parse("")

    def test_flexible_json_output_parser_whitespace_only(self):
        """Test FlexibleJsonOutputParser handles whitespace-only input."""
        parser = FlexibleJsonOutputParser()

        with pytest.raises(ValueError, match="No valid JSON found in text"):
            parser.parse("   \n\t   \n   ")

    def test_process_selected_indices_to_reduced_metadata_valid_indices(self):
        """Test _process_selected_indices_to_reduced_metadata with valid indices."""
        # Create test data with multiple matches
        test_metadata_1 = {
            "name": "Test Dataset 1",
            "description": "First test dataset",
            "dataType": "float",
            "startDate": "2023",
            "databaseName": "TestDB1",
            "frequency": "40",
        }
        test_metadata_2 = {
            "name": "Test Dataset 2",
            "description": "Second test dataset",
            "dataType": "int",
            "startDate": "2022",
            "databaseName": "TestDB2",
            "frequency": "50",
        }
        test_metadata_3 = {
            "name": "Test Dataset 3",
            "description": "Third test dataset",
            "dataType": "string",
            "startDate": "2021",
            "databaseName": "TestDB3",
            "frequency": "10",
        }

        matches = [
            MatchObject(id="1", metadata=test_metadata_1),
            MatchObject(id="2", metadata=test_metadata_2),
            MatchObject(id="3", metadata=test_metadata_3),
        ]

        selected_indices = [0, 2]  # Select first and third match

        result = _process_selected_indices_to_reduced_metadata(
            selected_indices, matches
        )

        assert len(result) == 2
        assert result[0].name == "Test Dataset 1"
        assert result[0].description == "First test dataset"
        assert result[0].dataType == "float"
        assert result[0].startDate == "2023"
        assert result[0].databaseName == "TestDB1"
        assert result[0].frequency == "40"

        assert result[1].name == "Test Dataset 3"
        assert result[1].description == "Third test dataset"
        assert result[1].dataType == "string"
        assert result[1].startDate == "2021"
        assert result[1].databaseName == "TestDB3"
        assert result[1].frequency == "10"

    def test_process_selected_indices_to_reduced_metadata_out_of_range_indices(
        self, caplog
    ):
        """Test _process_selected_indices_to_reduced_metadata with out-of-range indices."""
        test_metadata = {
            "name": "Test Dataset",
            "description": "Test description",
            "dataType": "float",
            "startDate": "2023",
            "databaseName": "TestDB",
            "frequency": "40",
        }

        matches = [MatchObject(id="1", metadata=test_metadata)]

        # Include indices that are out of range
        selected_indices = [0, 5, -1, 10]  # Only index 0 is valid

        with caplog.at_level("ERROR"):
            result = _process_selected_indices_to_reduced_metadata(
                selected_indices, matches
            )

        # Should only return the valid match
        assert len(result) == 1
        assert result[0].name == "Test Dataset"

    def test_process_selected_indices_to_reduced_metadata_empty_indices(self):
        """Test _process_selected_indices_to_reduced_metadata with empty indices list."""
        test_metadata = {
            "name": "Test Dataset",
            "description": "Test description",
            "dataType": "float",
            "startDate": "2023",
            "databaseName": "TestDB",
            "frequency": "40",
        }

        matches = [MatchObject(id="1", metadata=test_metadata)]
        selected_indices = []

        result = _process_selected_indices_to_reduced_metadata(
            selected_indices, matches
        )

        assert len(result) == 0
        assert result == []

    def test_process_selected_indices_to_reduced_metadata_empty_matches(
        self, caplog
    ):
        """Test _process_selected_indices_to_reduced_metadata with empty matches list."""
        matches = []
        selected_indices = [0, 1, 2]

        with caplog.at_level("ERROR"):
            result = _process_selected_indices_to_reduced_metadata(
                selected_indices, matches
            )

        assert len(result) == 0
        assert result == []

    def test_process_selected_indices_to_reduced_metadata_handles_string_numbers(
        self,
    ):
        """Test _process_selected_indices_to_reduced_metadata handles string numbers correctly."""
        metadata_with_strings = {
            "name": "String Numbers Dataset",
            "description": "Dataset with string number values",
            "dataType": "float",
            "startDate": "2023",  # String that can be converted to int
            "databaseName": "TestDB",
            "frequency": "40",  # String that can be converted to int
        }

        matches = [MatchObject(id="1", metadata=metadata_with_strings)]
        selected_indices = [0]

        result = _process_selected_indices_to_reduced_metadata(
            selected_indices, matches
        )

        assert len(result) == 1
        assert result[0].startDate == "2023"
        assert result[0].frequency == "40"
