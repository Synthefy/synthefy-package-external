from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Constants for supported models and defaults
SUPPORTED_TEXT_EMBEDDERS = {
    "openai/clip-vit-base-patch32": {
        "embedding_dim": 512,
        "max_length": 77,
        "memory_multiplier": 1.5,  # GB per 1000 samples (approximate)
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "embedding_dim": 384,
        "max_length": 256,
        "memory_multiplier": 1.0,  # GB per 1000 samples (approximate)
    },
}

DEFAULT_TEXT_EMBEDDER = "openai/clip-vit-base-patch32"
EMBEDDED_COL_NAMES_PREFIX = "text_embedding_"
PAD_TOKEN = "[PAD]"  # New constant for padding
DEFAULT_CHUNK_SIZE = 1000
MB_PER_GB = 1000  # Conversion factor from GB to MB
BYTES_TO_GB = 1024**3  # Conversion factor from bytes to GB
GPU_MEMORY_UTILIZATION_FACTOR = (
    0.9  # Reserve 10% of GPU memory for other operations
)
MIN_CHUNK_SIZE = 100  # Minimum chunk size for processing
MAX_CHUNK_SIZE = 10000  # Maximum chunk size for processing


class BaseTextEmbedder(ABC):
    """Abstract base class for text embedders.

    Example config:
    {
        "text_col_name": "text_column",  # Required: name of text column to embed
        "model": {  # Optional: model configuration
            "name": "openai/clip-vit-base-patch32",  # Optional: defaults to DEFAULT_TEXT_EMBEDDER
            "max_length": 77,  # Optional: max token length, defaults to model-specific value
            "embedding_dim": 512  # Optional: embedding dimension, defaults to model-specific value
        }
    }
    """

    def __init__(self, config: dict):
        """Initialize base text embedder.

        Args:
            config: Dictionary containing embedding configuration. See class docstring for example.
        """
        self.config = config
        self.text_column = config["text_col_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up model configuration
        model_config = config.get("model", {})
        self.model_name = model_config.get("name", DEFAULT_TEXT_EMBEDDER)

        if self.model_name not in SUPPORTED_TEXT_EMBEDDERS:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                f"Supported models are: {list(SUPPORTED_TEXT_EMBEDDERS.keys())}"
            )

        # Get defaults for this model
        defaults = SUPPORTED_TEXT_EMBEDDERS[self.model_name]

        # Set common attributes with defaults
        self.max_length = model_config.get("max_length", defaults["max_length"])
        self.embedding_dim = model_config.get(
            "embedding_dim", defaults["embedding_dim"]
        )

        # Get GPU memory information
        self.total_gpu_memory = self._get_gpu_memory_gb()
        # Calculate dynamic chunk size based on available GPU memory
        self.chunk_size = self._calculate_chunk_size()
        logger.info(
            f"Using chunk size of {self.chunk_size} based on available GPU memory"
        )

    def _get_gpu_memory_gb(self) -> float:
        """Get total GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / BYTES_TO_GB

    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on GPU memory."""
        if self.device == "cpu":
            return DEFAULT_CHUNK_SIZE

        # Get memory multiplier for the model
        memory_multiplier = SUPPORTED_TEXT_EMBEDDERS[self.model_name][
            "memory_multiplier"
        ]

        # Reserve portion of GPU memory for other operations
        usable_memory = self.total_gpu_memory * GPU_MEMORY_UTILIZATION_FACTOR

        # Calculate chunk size: (usable_memory_gb / memory_multiplier_per_1000) * 1000
        calculated_chunk_size = int(
            (usable_memory / memory_multiplier) * MB_PER_GB
        )

        # Set minimum and maximum bounds
        return max(MIN_CHUNK_SIZE, min(calculated_chunk_size, MAX_CHUNK_SIZE))

    def transform(
        self, df: pd.DataFrame, chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Transform text column to embeddings.

        Args:
            df: Input DataFrame
            chunk_size: Optional override for calculated chunk size

        Returns:
            DataFrame with embedded features with column names: "text_embedding{i}"
        """
        # Use provided chunk_size if specified, otherwise use calculated one
        chunk_size = chunk_size or self.chunk_size

        # Validate input first
        self._validate_input_df(df)

        # Initialize model and dimension reducer
        model = self._initialize_model()
        dimension_reducer = self._initialize_dimension_reducer(model)

        # Process chunks
        all_embeddings = []
        for start_idx in range(0, len(df), chunk_size):
            chunk_df = df.iloc[start_idx : start_idx + chunk_size]
            chunk_embeddings = self._process_chunk(
                chunk_df, model, dimension_reducer, chunk_size
            )
            all_embeddings.append(chunk_embeddings)

        # Clean up
        self._cleanup(model, dimension_reducer)

        # Create output DataFrame
        embeddings = np.concatenate(all_embeddings, axis=0)
        embedding_cols = [
            f"{EMBEDDED_COL_NAMES_PREFIX}{i}" for i in range(self.embedding_dim)
        ]
        return pd.DataFrame(embeddings, columns=embedding_cols)

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if df.empty:
            logger.error("Received empty DataFrame for text embedding")
            raise ValueError("Cannot process an empty DataFrame")

        if self.text_column not in df.columns:
            logger.error(
                f"Column '{self.text_column}' not found in DataFrame. Available columns: {df.columns.tolist()}"
            )
            raise ValueError(
                f"The column '{self.text_column}' is missing from DataFrame"
            )

        if df[self.text_column].notna().sum() == 0:
            logger.error("All values in text column are null")
            raise ValueError("All values in text column are null")

    @abstractmethod
    def _initialize_model(self) -> Any:
        """Initialize the model."""
        pass

    def _initialize_dimension_reducer(
        self, model: Any
    ) -> Optional[torch.nn.Module]:
        """Initialize dimension reducer if needed.

        Uses a fixed seed for reproducible dimension reduction transformation.
        The reducer parameters are frozen after initialization.
        """
        model_dim = self._get_model_dimension(model)
        if self.embedding_dim != model_dim:
            # Use fixed seed for reproducible dimension reduction
            torch.manual_seed(42)
            dimension_reducer = torch.nn.Linear(
                model_dim, self.embedding_dim
            ).to(self.device)
            # Freeze the parameters since this is a fixed transformation
            for param in dimension_reducer.parameters():
                param.requires_grad = False
            return dimension_reducer
        return None

    @abstractmethod
    def _get_model_dimension(self, model: Any) -> int:
        """Get the model's output dimension."""
        pass

    @abstractmethod
    def _process_chunk(
        self,
        chunk_df: pd.DataFrame,
        model: Any,
        dimension_reducer: Optional[torch.nn.Module],
        chunk_size: int,
    ) -> np.ndarray:
        """Process a chunk of the DataFrame."""
        pass

    def _cleanup(
        self, model: Any, dimension_reducer: Optional[torch.nn.Module]
    ) -> None:
        """Clean up resources."""
        del model
        if dimension_reducer is not None:
            del dimension_reducer
        torch.cuda.empty_cache()

    @abstractmethod
    def _preprocess_df(
        self, df: pd.DataFrame, tokenizer: Any = None
    ) -> Dict[str, torch.Tensor]:
        """Preprocess DataFrame text column into model inputs.

        Args:
            df: Input DataFrame
            tokenizer: Optional tokenizer for models that require it

        Returns:
            Dictionary containing preprocessed inputs as tensors
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseTextEmbedder":
        """Factory method to create appropriate embedder from config."""
        # Lazy import to avoid circular dependency
        from synthefy_pkg.preprocessing.text_embed_registry import \
            EmbedderRegistry

        # Validate config
        if "text_col_name" not in config:
            raise ValueError("'text_col_name' must be provided in config")
        if not isinstance(config["text_col_name"], str):
            raise ValueError("'text_col_name' must be a string")

        # Set up model config with defaults
        config = config.copy()
        if "model" not in config:
            logger.info(
                f"Using default configuration for model: {DEFAULT_TEXT_EMBEDDER}"
            )
            config["model"] = {}

        model_name = config["model"].get("name", DEFAULT_TEXT_EMBEDDER)

        # First check if the model is supported
        if model_name not in SUPPORTED_TEXT_EMBEDDERS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models are: {list(SUPPORTED_TEXT_EMBEDDERS.keys())}"
            )

        # Get embedder class directly by model name
        embedder_cls = EmbedderRegistry.get_embedder(model_name)
        return embedder_cls(config)
