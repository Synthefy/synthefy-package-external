from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from synthefy_pkg.preprocessing.base_text_embedder import (PAD_TOKEN,
                                                           BaseTextEmbedder)
from synthefy_pkg.preprocessing.text_embed_registry import EmbedderRegistry

# CLIP model's fixed output embedding dimension
CLIP_EMBEDDING_DIM = 512


@EmbedderRegistry.register("openai/clip-vit-base-patch32")
class CLIPTextEmbedder(BaseTextEmbedder):
    def __init__(self, config: dict):
        """Initialize CLIP text embedder.

        Args:
            config: Dictionary containing embedding configuration
                Example config:
                {
                    "text_col_name": "text_column",  # Required: name of text column to embed
                    "model": {  # Optional: model configuration
                        "name": "openai/clip-vit-base-patch32",  # Optional: defaults to DEFAULT_TEXT_EMBEDDER
                        "max_length": 77,  # Optional: max token length, defaults to model-specific value
                        "embedding_dim": 512,  # Optional: embedding dimension, defaults to model-specific value
                        "device": "cuda"  # Optional: device to run model on, defaults to "cuda" if available else "cpu"
                    }
                }
        """
        super().__init__(config)
        self.device = config.get("model", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _initialize_model(self) -> Tuple[CLIPTokenizer, CLIPTextModel]:
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        model = CLIPTextModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        return (tokenizer, model)

    def _get_model_dimension(
        self, model: Tuple[CLIPTokenizer, CLIPTextModel]
    ) -> int:
        return CLIP_EMBEDDING_DIM

    def _process_chunk(
        self,
        chunk_df: pd.DataFrame,
        model: Tuple[CLIPTokenizer, CLIPTextModel],
        dimension_reducer: Optional[torch.nn.Module],
        chunk_size: int,
    ) -> np.ndarray:
        tokenizer, clip_model = model
        inputs = self._preprocess_df(chunk_df, tokenizer)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            chunk_embeddings = outputs.pooler_output

            if dimension_reducer is not None:
                chunk_embeddings = dimension_reducer(chunk_embeddings)

            return chunk_embeddings.cpu().numpy()

    def _preprocess_df(
        self, df: pd.DataFrame, tokenizer: Any = None
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(tokenizer, CLIPTokenizer):
            raise ValueError("CLIPTextEmbedder requires a CLIPTokenizer")
        texts = df[self.text_column].fillna(PAD_TOKEN).astype(str)
        return tokenizer(
            texts.tolist(),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
