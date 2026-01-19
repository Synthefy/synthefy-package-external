from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from synthefy_pkg.preprocessing.base_text_embedder import (PAD_TOKEN,
                                                           BaseTextEmbedder)
from synthefy_pkg.preprocessing.text_embed_registry import EmbedderRegistry


@EmbedderRegistry.register("sentence-transformers/all-MiniLM-L6-v2")
class SentenceTransformerEmbedder(BaseTextEmbedder):
    def __init__(self, config: dict):
        """Initialize Sentence Transformer embedder.

        Args:
            config: Dictionary containing embedding configuration
                Example config:
                {
                    "text_col_name": "text_column",  # Required: name of text column to embed
                    "model": {  # Optional: model configuration
                        "name": "sentence-transformers/all-MiniLM-L6-v2",  # Optional: defaults to DEFAULT_TEXT_EMBEDDER
                        "max_length": 256,  # Optional: max token length, defaults to model-specific value
                        "embedding_dim": 384  # Optional: embedding dimension, defaults to model-specific value
                    }
                }
        """
        super().__init__(config)

    def _initialize_model(self) -> SentenceTransformer:
        model = SentenceTransformer(self.model_name)
        model.to(self.device)
        model.eval()
        return model

    def _get_model_dimension(self, model: SentenceTransformer) -> int:
        sentence_embedding_dimension = model.get_sentence_embedding_dimension()
        assert isinstance(sentence_embedding_dimension, int)
        return sentence_embedding_dimension

    def _process_chunk(
        self,
        chunk_df: pd.DataFrame,
        model: SentenceTransformer,
        dimension_reducer: Optional[torch.nn.Module],
        chunk_size: int,
    ) -> np.ndarray:
        inputs = self._preprocess_df(chunk_df)

        with torch.no_grad():
            chunk_embeddings = model.encode(
                inputs["texts"],
                batch_size=chunk_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            if dimension_reducer is not None:
                chunk_embeddings = dimension_reducer(chunk_embeddings)

            return chunk_embeddings.cpu().numpy()

    def _preprocess_df(
        self, df: pd.DataFrame, tokenizer: Any = None
    ) -> Dict[str, Any]:
        texts = df[self.text_column].fillna(PAD_TOKEN).astype(str).tolist()
        return {"texts": texts}
