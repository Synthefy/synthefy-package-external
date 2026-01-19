from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class SynthefyFoundationModelAnalyzer:
    def __init__(self, model: nn.Module):
        """
        Initialize the analyzer with a SynthefyFoundationForecastingModelV3 instance.

        Args:
            model: An instance of SynthefyFoundationForecastingModelV3
        """
        self.model = model
        self.attention_weights = []
        self.gradients = []
        self.num_correlates = model.num_correlates
        # Register hooks for attention weights
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights from all transformer layers."""

        def get_attention_weights(module, input, output):
            # The attention weights are typically in the output tuple
            if isinstance(output, tuple):
                scores, _ = output
                # Don't detach here to allow gradient flow
                self.attention_weights.append(scores)
            else:
                # Don't detach here to allow gradient flow
                self.attention_weights.append(output)

        # Register hooks for each transformer layer
        if hasattr(self.model, "decoder_model"):
            for layer in self.model.decoder_model.stacked_transformer.layers:
                layer.self_attn.register_forward_hook(get_attention_weights)
        elif hasattr(self.model, "stacked_transformer"):
            for layer in self.model.stacked_transformer.layers:
                layer.self_attn.register_forward_hook(get_attention_weights)
        else:
            raise ValueError(
                "Model has no stacked_transformer or decoder_model"
            )

    def get_attention_weights(
        self, decoder_input: Dict[str, torch.Tensor], use_no_grad: bool = True
    ) -> List[torch.Tensor]:
        """
        Get attention weights for a given input.

        Args:
            decoder_input: Dictionary containing model inputs (timestamps, descriptions, continuous_tokens, etc.)
            use_no_grad: Whether to use torch.no_grad() context. Set to False if you want to compute gradients
                        through the attention weights.

        Returns:
            List of attention weight tensors from each transformer layer. If use_no_grad is False,
            these tensors will have gradients enabled and can be used for loss computation.
        """
        self.attention_weights = []  # Clear previous weights

        if use_no_grad:
            with torch.no_grad():
                self.model.forward(decoder_input, keep_attention_weights=True)
        else:
            self.model.forward(decoder_input, keep_attention_weights=True)

        return self.attention_weights

    def get_input_attention_weights(
        self,
        decoder_input: Dict[str, torch.Tensor],
        use_no_grad: bool = True,
    ) -> List[torch.Tensor]:
        """
        Get attention weights for each input, averaging over heads and layers.

        Args:
            decoder_input: Dictionary containing model inputs
            use_no_grad: Whether to use torch.no_grad() context
        """
        attention_weights = self.get_attention_weights(
            decoder_input, use_no_grad
        )
        return [w.mean(dim=0).mean(dim=0) for w in attention_weights]

    def create_counterfactual_batches(
        self, batch: Dict[str, torch.Tensor], mask_type: str = "correlate"
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Get counterfactual batches for a given input.
        This is represented as dict of batched values,
        and a set of mask indices
        Even though it takes in a batch, it assumes that batch size is 1

        Returns:
            Tuple containing:
            - List of dictionaries containing masked batches for each correlate
            - Dictionary containing token information for each correlate
        """
        indices = list()
        correlates = list()
        if mask_type == "correlate":
            input_batch = self.model.prepare_training_input(batch)
            num_correlates = self.num_correlates
            tokens_per_correlate = (
                input_batch["continuous"].shape[1] // num_correlates
            )
            correlate_counts = np.repeat(
                np.arange(num_correlates), tokens_per_correlate
            )
            print(
                "correlate_counts",
                correlate_counts,
                input_batch["continuous"].shape,
                input_batch["descriptions"].shape,
                input_batch["timestamps"].shape,
            )
            for i in range(self.num_correlates):
                correlate_index = np.where(correlate_counts == i)[0]

                indices.append(correlate_index)
                correlates.append(
                    {
                        k: input_batch[k][0].detach().clone()
                        for k in input_batch.keys()
                    }
                )
        elif mask_type == "input_value":
            input_batch = self.model.prepare_training_input(batch)
            indices = np.arange(input_batch["continuous"].shape[1])
            for i in range(input_batch["continuous"].shape[1]):
                indices.append(np.array([i]))
                correlates.append(
                    {
                        k: input_batch[k].detach().clone()
                        for k in input_batch.keys()
                    }
                )
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        correlates = {
            k: torch.stack([c[k] for c in correlates])
            for k in correlates[0].keys()
        }
        indices = np.stack(indices, axis=0)
        print(
            [correlates[k].shape for k in correlates.keys()],
            indices.shape,
            indices[0],
        )
        return correlates, indices

    def compute_counterfactual_masking(
        self,
        decoder_input: Dict[str, torch.Tensor],
        mask_indices: Union[np.ndarray, torch.Tensor],
        target_indices: Optional[List[int]] = None,
        use_no_grad: bool = True,
        mask_type: str = "input_value",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the effect of masking specific tokens on the model's output.

        Args:
            decoder_input: Dictionary containing model inputs
            mask_indices: a set of indicies to mask for each example in the batch
            target_indices: Optional list of target indices to compute effect on
            use_no_grad: Whether to use torch.no_grad() context
            mask_type: Type of masking to apply ('input_value', 'input_text', 'input_both', 'input_all', 'attention', or 'both')
            attention_mask_indices: Optional list of (layer_idx, head_idx) tuples to mask attention for

        Returns:
            Dictionary containing original and masked predictions
        """
        # Store original input
        original_input = {
            k: v.detach().clone() for k, v in decoder_input.items()
        }
        device = next(iter(original_input.values())).device

        # Create masked input
        masked_input = {k: v.detach().clone() for k, v in decoder_input.items()}

        # Convert mask_indices to tensor and move to correct device
        if isinstance(mask_indices, np.ndarray):
            mask_indices = torch.from_numpy(mask_indices).to(device)

        # Apply input masking if requested
        if (
            mask_type in ["input_value", "input_both", "input_all", "both"]
            and "continuous_tokens" in masked_input
        ):
            if len(mask_indices.shape) > 0:
                masked_input["continuous_tokens"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN
            else:
                masked_input["continuous_tokens"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN

        if mask_type in ["input_text", "input_both", "input_all", "both"]:
            if len(mask_indices.shape) > 0:
                masked_input["descriptions"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN
            else:
                masked_input["descriptions"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN

        if mask_type in ["input_all"]:
            if len(mask_indices.shape) > 0:
                masked_input["timestamps"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN
            else:
                masked_input["timestamps"][
                    torch.arange(len(mask_indices)).unsqueeze(1), mask_indices
                ] = -1  # NULL_TOKEN

        # Get original predictions
        if use_no_grad:
            with torch.no_grad():
                original_output = self.model.forward(original_input)
                masked_output = self.model.forward(masked_input)
        else:
            original_output = self.model.forward(original_input)
            masked_output = self.model.forward(masked_input)

        # If target indices specified, only return those predictions
        if target_indices is not None:
            original_output["prediction"] = original_output["prediction"][
                target_indices
            ]
            masked_output["prediction"] = masked_output["prediction"][
                target_indices
            ]

        return {
            "original": original_output,
            "masked": masked_output,
            "difference": original_output["prediction"]
            - masked_output["prediction"],
        }

    def compute_input_gradients(
        self,
        decoder_input: Dict[str, torch.Tensor],
        target_indices: Optional[List[int]] = None,
        create_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of the output with respect to input tokens.

        Args:
            decoder_input: Dictionary containing model inputs
            target_indices: Optional list of target indices to compute gradients for
            create_graph: Whether to create a graph for second-order derivatives

        Returns:
            Dictionary containing gradients for each input type
        """
        # Enable gradient computation
        for k, v in decoder_input.items():
            if isinstance(v, torch.Tensor):
                v.requires_grad_(True)

        # Forward pass
        output = self.model.forward(decoder_input)

        # If target indices specified, only compute gradients for those predictions
        if target_indices is not None:
            predictions = output["prediction"][target_indices]
        else:
            predictions = output["prediction"]

        # Compute gradients
        gradients = {}
        for k, v in decoder_input.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                grad = torch.autograd.grad(
                    predictions.sum(),
                    v,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                if not create_graph:
                    grad = grad.detach()
                gradients[k] = grad

        return gradients

    def compute_regularization_loss(
        self,
        decoder_input: Dict[str, torch.Tensor],
        reg_type: str = "gradient",
        target_indices: Optional[List[int]] = None,
        reg_weight: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute regularization loss based on gradients or counterfactual differences.

        Args:
            decoder_input: Dictionary containing model inputs
            reg_type: Type of regularization ('gradient' or 'counterfactual')
            target_indices: Optional list of target indices to compute regularization for
            reg_weight: Weight for the regularization term
            **kwargs: Additional arguments for specific regularization types
                - For 'gradient':
                    - norm_type: Type of norm to use ('l1', 'l2', 'inf')
                - For 'counterfactual':
                    - mask_indices: List of indices to mask
                    - smoothness_weight: Weight for smoothness regularization

        Returns:
            Regularization loss tensor
        """
        if reg_type == "gradient":
            # Get gradients
            gradients = self.compute_input_gradients(
                decoder_input,
                target_indices=target_indices,
                create_graph=True,  # Need to create graph for gradient regularization
            )

            # Compute gradient norm
            norm_type = kwargs.get("norm_type", "l2")
            grad_norm = torch.tensor(
                0.0, device=decoder_input["continuous_tokens"].device
            )
            for grad in gradients.values():
                if norm_type == "l1":
                    grad_norm = grad_norm + grad.abs().mean()
                elif norm_type == "l2":
                    grad_norm = grad_norm + (grad**2).mean()
                elif norm_type == "inf":
                    grad_norm = grad_norm + grad.abs().max()

            return reg_weight * grad_norm

        elif reg_type == "counterfactual":
            # Get counterfactual differences
            mask_indices = kwargs.get("mask_indices", [])
            if not mask_indices:
                raise ValueError(
                    "mask_indices must be provided for counterfactual regularization"
                )

            counterfactual_results = self.compute_counterfactual_masking(
                decoder_input,
                mask_indices=mask_indices,
                target_indices=target_indices,
                use_no_grad=False,  # Need gradients for regularization
            )

            # Compute difference norm
            differences = counterfactual_results["difference"]
            diff_norm = (differences**2).mean()

            # Add smoothness regularization if requested
            smoothness_weight = kwargs.get("smoothness_weight", 0.0)
            if smoothness_weight > 0:
                # Compute smoothness as squared difference between adjacent predictions
                smoothness = (
                    (differences[:, 1:] - differences[:, :-1]) ** 2
                ).mean()
                diff_norm = diff_norm + smoothness_weight * smoothness

            return reg_weight * diff_norm

        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

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

        return processed

    def compute_attention_regularization(
        self,
        attention_weights: List[torch.Tensor],
        reg_type: str = "sparsity",
        reg_weight: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute regularization loss for attention weights.

        Args:
            attention_weights: List of attention weight tensors
            reg_type: Type of regularization ('sparsity', 'diversity', or 'smoothness')
            reg_weight: Weight for the regularization term
            **kwargs: Additional arguments for specific regularization types
                - For 'sparsity':
                    - target_sparsity: Target sparsity level (default: 0.5)
                - For 'diversity':
                    - temperature: Temperature for softmax (default: 1.0)
                - For 'smoothness':
                    - window_size: Size of smoothing window (default: 3)

        Returns:
            Regularization loss tensor
        """
        total_loss = torch.tensor(0.0, device=attention_weights[0].device)

        for layer_idx in range(len(attention_weights)):
            processed = self.process_attention_weights(
                attention_weights, layer_idx=layer_idx
            )
            weights = processed["layer_weights"]

            if reg_type == "sparsity":
                # L1 regularization to encourage sparsity
                target_sparsity = kwargs.get("target_sparsity", 0.5)
                current_sparsity = weights.mean()
                sparsity_loss = (
                    (current_sparsity - target_sparsity) ** 2
                ).mean()
                total_loss = total_loss + sparsity_loss

            elif reg_type == "diversity":
                # Encourage diverse attention patterns across heads
                temperature = kwargs.get("temperature", 1.0)
                # Compute cosine similarity between attention patterns
                weights_flat = weights.reshape(
                    weights.shape[0], weights.shape[1], -1
                )
                similarity = torch.matmul(
                    weights_flat, weights_flat.transpose(-2, -1)
                )
                # Remove self-similarity
                similarity = similarity - torch.eye(
                    similarity.shape[-1], device=similarity.device
                )
                diversity_loss = similarity.abs().mean()
                total_loss = total_loss + diversity_loss * temperature

            elif reg_type == "smoothness":
                # Encourage smooth attention patterns using sliding window
                window_size = kwargs.get("window_size", 3)
                if window_size < 2:
                    raise ValueError(
                        "Window size must be at least 2 for smoothness regularization"
                    )

                # Compute differences between attention weights within the window
                # Shape: [batch, heads, seq_len, seq_len]
                smoothness_loss = torch.tensor(0.0, device=weights.device)

                # For each position in the sequence
                for i in range(weights.shape[2] - window_size + 1):
                    # Get the window of attention weights
                    window = weights[:, :, i : i + window_size, :]

                    # Compute differences between adjacent positions in the window
                    # Shape: [batch, heads, window_size-1, seq_len]
                    window_diff = window[:, :, 1:] - window[:, :, :-1]

                    # Add squared differences to loss
                    smoothness_loss = smoothness_loss + (window_diff**2).mean()

                # Normalize by number of windows
                smoothness_loss = smoothness_loss / (
                    weights.shape[2] - window_size + 1
                )
                total_loss = total_loss + smoothness_loss

            else:
                raise ValueError(
                    f"Unknown attention regularization type: {reg_type}"
                )

        return reg_weight * total_loss

    def compute_attention_counterfactual(
        self,
        decoder_input: Dict[str, torch.Tensor],
        mask_indices: np.ndarray,
        target_indices: Optional[List[int]] = None,
        use_no_grad: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Compute the effect of masking specific attention heads on the model's output.

        Args:
            decoder_input: Dictionary containing model inputs
            attention_mask_indices: List of (layer_idx, head_idx) tuples to mask
            target_indices: Optional list of target indices to compute effect on
            use_no_grad: Whether to use torch.no_grad() context

        Returns:
            Dictionary containing:
            - 'original': Original model output
            - 'masked': Output with masked attention
            - 'difference': Difference between original and masked predictions
            - 'original_attention': List of original attention weights
        """
        # Store original attention weights
        original_attention_weights = self.get_attention_weights(
            decoder_input, use_no_grad=use_no_grad
        )

        # Create a copy of the model for masking
        masked_model = type(self.model)(self.model.config)
        masked_model.load_state_dict(self.model.state_dict())
        masked_model.to(self.model.device)

        # Register hooks to mask specific attention heads
        def mask_attention_weights(module, input, output, layer_idx, head_idx):
            scores, _ = output
            # Zero out the attention weights for the specified input
            scores[:, :, mask_indices] = 0.0  # skip batch, head dimensions
            return scores, _

        # Register hooks for each transformer layer
        hooks = []
        if hasattr(masked_model, "decoder_model"):
            for layer_idx, layer in enumerate(
                masked_model.decoder_model.stacked_transformer.layers
            ):
                hook = layer.self_attn.register_forward_hook(
                    lambda m, i, o, layer_idx=layer_idx: mask_attention_weights(
                        m, i, o, layer_idx, None
                    )
                )
                hooks.append(hook)
        elif hasattr(masked_model, "stacked_transformer"):
            for layer_idx, layer in enumerate(
                masked_model.stacked_transformer.layers
            ):
                hook = layer.self_attn.register_forward_hook(
                    lambda m, i, o, layer_idx=layer_idx: mask_attention_weights(
                        m, i, o, layer_idx, None
                    )
                )
                hooks.append(hook)
        else:
            raise ValueError(
                "Model has no stacked_transformer or decoder_model"
            )

        # Get predictions with masked attention
        if use_no_grad:
            with torch.no_grad():
                masked_output = masked_model.forward(decoder_input)
        else:
            masked_output = masked_model.forward(decoder_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Get original predictions
        if use_no_grad:
            with torch.no_grad():
                original_output = self.model.forward(decoder_input)
        else:
            original_output = self.model.forward(decoder_input)

        # If target indices specified, only return those predictions
        if target_indices is not None:
            original_output["prediction"] = original_output["prediction"][
                target_indices
            ]
            masked_output["prediction"] = masked_output["prediction"][
                target_indices
            ]

        return {
            "original": original_output,
            "masked": masked_output,
            "difference": original_output["prediction"]
            - masked_output["prediction"],
            "original_attention": original_attention_weights,
        }
