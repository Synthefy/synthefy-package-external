from typing import Dict, Type

from synthefy_pkg.preprocessing.base_text_embedder import BaseTextEmbedder


class EmbedderRegistry:
    """Registry for text embedder implementations."""

    _embedders: Dict[str, Type[BaseTextEmbedder]] = {}

    @classmethod
    def register(cls, model_name: str):
        """Decorator to register a text embedder implementation."""

        def wrapper(
            embedder_cls: Type[BaseTextEmbedder],
        ) -> Type[BaseTextEmbedder]:
            cls._embedders[model_name] = embedder_cls
            return embedder_cls

        return wrapper

    @classmethod
    def get_embedder(cls, model_name: str) -> Type[BaseTextEmbedder]:
        """Get embedder class by model name."""
        from synthefy_pkg.preprocessing import clip_text_embedder  # noqa: F401
        from synthefy_pkg.preprocessing import \
            sentence_transformer_embedder  # noqa: F401

        if model_name not in cls._embedders:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {list(cls._embedders.keys())}"
            )
        return cls._embedders[model_name]
