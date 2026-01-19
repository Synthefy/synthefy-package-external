import os
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


class InputWeightAnalyzer:
    def __init__(self, model_analyzer):
        """
        Initialize the attention analyzer with a SynthefyFoundationModelAnalyzer instance.

        Args:
            model_analyzer: An instance of SynthefyFoundationModelAnalyzer
        """
        self.model_analyzer = model_analyzer
        self.num_correlates = model_analyzer.num_correlates

    def process_input_weights(
        self,
        input_weights: torch.Tensor,
        correlate_idx: Optional[int] = None,
        value_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input weights for analysis or visualization.

        Args:
            input_weights: List of input weight tensors
            batch_idx: Index of the batch to process (default: 0)
            correlate_idx: Optional index of specific correlate to process
            value_idx: Optional index of specific value within correlate to process

        Returns:
            Dictionary containing processed input weights:
            - 'all_weights': All input weights reshaped to [batch, correlates, seq_len, value_dims]
            - 'correlate_weights': Weights for specific correlate if correlate_idx provided
            - 'value_weights': Weights for specific value if value_idx provided
            - 'mean_weights': Mean weights across correlates
        """
        processed = {}

        # Process all weights
        import time

        start_time = time.time()

        print(input_weights.shape)
        input_weights = input_weights[..., 0]
        # Reshape weights already [batch, correlates * seq_len]
        processed["all_weights"] = input_weights
        # Reshape weights to [batch, correlates, seq_len]
        all_weights = rearrange(
            input_weights, "b (c s) -> b c s", c=self.num_correlates
        )  # shape: b c s
        processed["correlate_weights"] = all_weights.mean(
            dim=-1
        )  # shape: b c where it is assumed b = c or b = ( c*s )

        # Process specific correlate if requested
        if correlate_idx is not None:
            if correlate_idx >= self.num_correlates:
                raise ValueError(
                    f"Correlate index {correlate_idx} out of range"
                )

            correlate_weights = all_weights[
                correlate_idx
            ]  # Shape: [num inputs]
            processed["single_correlate_weights"] = correlate_weights

            # Process specific value if requested
            if value_idx is not None:
                if value_idx >= correlate_weights.shape[1]:
                    raise ValueError(f"Value index {value_idx} out of range")
                processed["value_weights"] = correlate_weights[:, value_idx]

            # Compute mean weights across correlates
            processed["mean_weights"] = all_weights.mean(
                dim=0
            )  # Shape: [seq_len, num inputs]

        end_time = time.time()
        logger.info(
            f"Time taken to process input weights: {end_time - start_time} seconds"
        )
        return processed

    def visualize_inputs(
        self,
        input_weights: torch.Tensor,
        correlate_idx: int = -1,
        visualize_correlates: bool = False,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Prepare and visualize attention weights.

        Args:
            input_weights: List of input weight tensors
            correlate_idx: Index of the correlate to visualize
            batch_idx: Index of the batch to visualize
            save_path: Optional path to save the visualization
            title: Optional title for the plot

        Returns:
            Attention weight matrix for the specified layer, head, and batch
        """
        import time

        start_time = time.time()
        processed = self.process_input_weights(
            input_weights,
            correlate_idx=correlate_idx,
        )
        if visualize_correlates:
            weights = processed["correlate_weights"]
        else:
            if correlate_idx == -1:
                weights = processed["all_weights"]
            else:
                weights = processed["single_correlate_weights"]

        # Convert to numpy for plotting
        weights_np = weights.detach().cpu().numpy()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_np, cmap="viridis")

        if title:
            plt.title(title)
        else:
            cidx = correlate_idx if correlate_idx != -1 else "all"
            plt.title(f"Input Weights - Correlate {cidx}")

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

    def analyze_input_patterns(
        self,
        input_weights: torch.Tensor,
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

        processed = self.process_input_weights(input_weights)
        input_weights = processed["all_weights"]
        correlate_weights = processed["correlate_weights"]

        # create a histogram of the indexes of the top 10% of attention
        print(input_weights.shape)
        top_10_weight, top_10_weight_indexes = input_weights.topk(10, dim=1)
        top_2_correlate_weight, top_2_correlate_weight_indexes = (
            correlate_weights.topk(2, dim=-1)
        )

        # Compute statistics per head
        head_stats = {
            "input_mean_weight": input_weights.mean(
                dim=(-1)
            ),  # Average weight per input
            "input_max_weight": input_weights.max(dim=-1)[
                0
            ],  # Maximum weight per input
            "input_min_weight": input_weights.min(dim=-1)[
                0
            ],  # Minimum weight per input
            "input_std": input_weights.std(dim=-1),  # weight entropy
            "input_top_10_weight_indexes": top_10_weight_indexes.cpu().numpy(),
            "correlate_mean_weight": correlate_weights.mean(
                dim=-1
            ),  # Average weight per correlate
            "correlate_max_weight": correlate_weights.max(dim=-1)[
                0
            ],  # Maximum weight per correlate
            "correlate_min_weight": correlate_weights.min(dim=-1)[
                0
            ],  # Minimum weight per correlate
            "correlate_std": correlate_weights.std(
                dim=-1
            ),  # correlate weight entropy
            "correlate_top_2_weight_indexes": top_2_correlate_weight_indexes.cpu().numpy(),
        }

        if plot:
            # Create subplots for each statistic
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # Plot mean attention
            print(head_stats["input_mean_weight"].shape, input_weights.shape)
            sns.barplot(
                x=range(input_weights.shape[0]),
                y=head_stats["input_mean_weight"].cpu().numpy(),
                ax=axes[0],
            )
            axes[0].set_title(f"Mean Input Weight - Batch {batch_idx}")
            axes[0].set_xlabel("Input")
            axes[0].set_ylabel("Mean Input Weight")

            # Plot min attention
            sns.barplot(
                x=range(input_weights.shape[0]),
                y=head_stats["input_min_weight"].cpu().numpy(),
                ax=axes[1],
            )
            axes[1].set_title(f"Min Input Weight - Batch {batch_idx}")
            axes[1].set_xlabel("Input")
            axes[1].set_ylabel("Min Input Weight")

            # Plot max attention
            sns.barplot(
                x=range(input_weights.shape[0]),
                y=head_stats["input_max_weight"].cpu().numpy(),
                ax=axes[2],
            )
            axes[2].set_title(f"Max Input Weight - Batch {batch_idx}")
            axes[2].set_xlabel("Input")
            axes[2].set_ylabel("Max Input Weight")

            # Plot entropy
            sns.barplot(
                x=range(input_weights.shape[0]),
                y=head_stats["input_std"].cpu().numpy(),
                ax=axes[3],
            )
            axes[3].set_title(f"Input Weight St.D. - Batch {batch_idx}")
            axes[3].set_xlabel("Input")
            axes[3].set_ylabel("St.D.")

            # Plot top 10 attention indexes histogram
            sns.histplot(
                head_stats["input_top_10_weight_indexes"].flatten(),
                ax=axes[4],
            )
            axes[4].set_title(
                f"Top 10 Input Weight Indexes - Batch {batch_idx}"
            )
            axes[4].set_xlabel("Index")
            axes[4].set_ylabel("Count")

            plt.suptitle(f"Input Weight Statistics - Batch {batch_idx}")
            plt.tight_layout()

            if save_path:
                plt.savefig(
                    f"{save_path}_batch_{batch_idx}_input_weight_statistics.png"
                )
                logger.info(
                    f"Saved input weight statistics plot to {save_path}_batch_{batch_idx}_input_weight_statistics.png"
                )
                plt.close()
            else:
                plt.show()

            # Create subplots for each statistic
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # Plot mean attention
            sns.barplot(
                x=range(correlate_weights.shape[0]),
                y=head_stats["correlate_mean_weight"].cpu().numpy(),
                ax=axes[0],
            )
            axes[0].set_title(f"Mean Correlate Weight - Batch {batch_idx}")
            axes[0].set_xlabel("Correlate")
            axes[0].set_ylabel("Mean Correlate Weight")

            # Plot min attention
            sns.barplot(
                x=range(correlate_weights.shape[0]),
                y=head_stats["correlate_min_weight"].cpu().numpy(),
                ax=axes[1],
            )
            axes[1].set_title(f"Min Correlate Weight - Batch {batch_idx}")
            axes[1].set_xlabel("Correlate")
            axes[1].set_ylabel("Min Correlate Weight")

            # Plot max attention
            sns.barplot(
                x=range(correlate_weights.shape[0]),
                y=head_stats["correlate_max_weight"].cpu().numpy(),
                ax=axes[2],
            )
            axes[2].set_title(f"Max Correlate Weight - Batch {batch_idx}")
            axes[2].set_xlabel("Correlate")
            axes[2].set_ylabel("Max Correlate Weight")

            # Plot entropy
            sns.barplot(
                x=range(correlate_weights.shape[0]),
                y=head_stats["correlate_std"].cpu().numpy(),
                ax=axes[3],
            )
            axes[3].set_title(f"Correlate Weight St.D. - Batch {batch_idx}")
            axes[3].set_xlabel("Correlate")
            axes[3].set_ylabel("St.D.")

            # Plot top 10 attention indexes histogram
            sns.histplot(
                head_stats["correlate_top_2_weight_indexes"].flatten(),
                ax=axes[4],
            )
            axes[4].set_title(
                f"Top 2 Correlate Weight Indexes - Batch {batch_idx}"
            )
            axes[4].set_xlabel("Index")
            axes[4].set_ylabel("Count")

            plt.suptitle(f"Correlate Weight Statistics - Batch {batch_idx}")
            plt.tight_layout()

            if save_path:
                plt.savefig(
                    f"{save_path}_batch_{batch_idx}_correlate_weight_statistics.png"
                )
                logger.info(
                    f"Saved correlate weight statistics plot to {save_path}_batch_{batch_idx}_correlate_weight_statistics.png"
                )
                plt.close()
            else:
                plt.show()

        return head_stats
