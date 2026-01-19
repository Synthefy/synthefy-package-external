import os
from collections import Counter
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from loguru import logger

from synthefy_pkg.model.utils.synthefy_foundation_model_analysis import (
    SynthefyFoundationModelAnalyzer,
)


class AttentionAnalyzer:
    def __init__(self, model_analyzer):
        """
        Initialize the attention analyzer with a SynthefyFoundationModelAnalyzer instance.

        Args:
            model_analyzer: An instance of SynthefyFoundationModelAnalyzer
        """
        self.model_analyzer = model_analyzer
        self.num_correlates = model_analyzer.num_correlates

    def process_attention_weights(
        self,
        attention_weights: List[torch.Tensor],
        batch_idx: int = 0,
        head_idx: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process attention weights for analysis or visualization.

        Args:
            attention_weights: List of attention weight tensors
            batch_idx: Index of the batch to process (default: 0)
            head_idx: Optional index of specific attention head to process
            layer_idx: Optional index of specific layer to process

        Returns:
            Dictionary containing processed attention weights:
            - 'all_weights': All attention weights reshaped to [batch, heads, seq_len, seq_len]
            - 'layer_weights': Weights for specific layer if layer_idx provided
            - 'head_weights': Weights for specific head if head_idx provided
            - 'mean_weights': Mean attention weights across heads
        """
        processed = {}

        # Process all weights
        import time

        start_time = time.time()
        all_weights = [
            rearrange(w, "b h n m -> b h n m") for w in attention_weights
        ]
        processed["all_weights"] = all_weights

        # Process specific layer if requested
        if layer_idx is not None:
            if layer_idx >= len(all_weights):
                raise ValueError(f"Layer index {layer_idx} out of range")
            layer_weights = all_weights[layer_idx]
            processed["layer_weights"] = layer_weights

            # Process specific head if requested
            if head_idx is not None:
                if head_idx >= layer_weights.shape[1]:
                    raise ValueError(f"Head index {head_idx} out of range")
                processed["head_weights"] = layer_weights[batch_idx, head_idx]

            # Compute mean attention across heads
            processed["mean_weights"] = layer_weights[batch_idx].mean(dim=0)

        end_time = time.time()
        logger.info(
            f"Time taken to process attention weights: {end_time - start_time} seconds"
        )
        return processed

    def visualize_attention(
        self,
        attention_weights: List[torch.Tensor],
        layer_idx: int = 0,
        head_idx: int = 0,
        batch_idx: int = 0,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Prepare and visualize attention weights.

        Args:
            attention_weights: List of attention weight tensors
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            batch_idx: Index of the batch to visualize
            save_path: Optional path to save the visualization
            title: Optional title for the plot

        Returns:
            Attention weight matrix for the specified layer, head, and batch
        """
        import time

        start_time = time.time()
        processed = self.process_attention_weights(
            attention_weights,
            batch_idx=batch_idx,
            head_idx=head_idx,
            layer_idx=layer_idx,
        )
        weights = processed["head_weights"]

        # Convert to numpy for plotting
        weights_np = weights.detach().cpu().numpy()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_np, cmap="viridis")

        if title:
            plt.title(title)
        else:
            plt.title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")

        plt.xlabel("Key Position")
        plt.ylabel("Query Position")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        end_time = time.time()
        logger.info(
            f"Time taken to visualize attention: {end_time - start_time} seconds"
        )
        logger.info(f"Shape of weights: {weights.shape}, {weights_np.shape}")
        return weights

    def analyze_attention_patterns(
        self,
        attention_weights: List[torch.Tensor],
        num_heads: int = 8,
        plot: bool = False,
        save_path: Optional[str] = None,
        batch_idx: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze attention patterns across heads and layers.

        Args:
            attention_weights: List of attention weight tensors
            num_heads: Number of attention heads in the model
            plot: Whether to create visualization plots
            save_path: Optional path to save the plots

        Returns:
            Dictionary containing various attention statistics
        """
        patterns = {}

        for layer_idx in range(len(attention_weights)):
            processed = self.process_attention_weights(
                attention_weights, layer_idx=layer_idx
            )
            layer_weights = processed["layer_weights"]

            # create a histogram of the indexes of the top 10% of attention
            top_10_attention, top_10_attention_indexes = layer_weights.topk(
                10, dim=3
            )
            # generate the count dictionary for the top 10 attention indexes
            top_10_attention_indexes_count = Counter(
                top_10_attention_indexes.flatten().cpu().numpy()
            )
            top_10_attention_indexes_count = {
                k: v
                for k, v in sorted(
                    top_10_attention_indexes_count.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
            logger.info(
                f"Top 10 attention indexes count: {top_10_attention_indexes_count}"
            )

            # Compute statistics per head
            head_stats = {
                "mean_attention": layer_weights.mean(
                    dim=(0, 2, 3)
                ),  # Average attention per head
                "max_attention": layer_weights.max(dim=3)[0].mean(
                    dim=(0, 2)
                ),  # Maximum attention per head
                "min_attention": layer_weights.min(dim=3)[0].mean(
                    dim=(0, 2)
                ),  # Minimum attention per head
                "entropy": -(layer_weights * torch.log(layer_weights + 1e-10))
                .sum(dim=3)
                .mean(dim=(0, 2)),  # Attention entropy
                "top_10_attention_indexes": top_10_attention_indexes.cpu().numpy(),
            }

            patterns[f"layer_{layer_idx}"] = head_stats

            if plot:
                # Create subplots for each statistic
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))

                # Plot mean attention
                sns.barplot(
                    x=range(num_heads),
                    y=head_stats["mean_attention"].cpu().numpy(),
                    ax=axes[0],
                )
                axes[0].set_title(f"Mean Attention - Batch {batch_idx}")
                axes[0].set_xlabel("Head")
                axes[0].set_ylabel("Mean Attention")

                # Plot min attention
                sns.barplot(
                    x=range(num_heads),
                    y=head_stats["min_attention"].cpu().numpy(),
                    ax=axes[1],
                )
                axes[1].set_title(f"Min Attention - Batch {batch_idx}")
                axes[1].set_xlabel("Head")
                axes[1].set_ylabel("Min Attention")

                # Plot max attention
                sns.barplot(
                    x=range(num_heads),
                    y=head_stats["max_attention"].cpu().numpy(),
                    ax=axes[2],
                )
                axes[2].set_title(f"Max Attention - Batch {batch_idx}")
                axes[2].set_xlabel("Head")
                axes[2].set_ylabel("Max Attention")

                # Plot entropy
                sns.barplot(
                    x=range(num_heads),
                    y=head_stats["entropy"].cpu().numpy(),
                    ax=axes[3],
                )
                axes[3].set_title(f"Attention Entropy - Batch {batch_idx}")
                axes[3].set_xlabel("Head")
                axes[3].set_ylabel("Entropy")

                # Plot top 10 attention indexes histogram
                sns.histplot(
                    head_stats["top_10_attention_indexes"].flatten(),
                    ax=axes[4],
                )
                axes[4].set_title(
                    f"Top 10 Attention Indexes - Batch {batch_idx}"
                )
                axes[4].set_xlabel("Index")
                axes[4].set_ylabel("Count")

                plt.suptitle(f"Attention Statistics - Layer {layer_idx}")
                plt.tight_layout()

                if save_path:
                    plt.savefig(
                        f"{save_path}_layer_{layer_idx}_batch_{batch_idx}.png"
                    )
                    logger.info(
                        f"Saved attention plot to {save_path}_layer_{layer_idx}_batch_{batch_idx}.png"
                    )
                    plt.close()
                else:
                    plt.show()

        return patterns
