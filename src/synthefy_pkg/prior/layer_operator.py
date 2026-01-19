import copy
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from synthefy_pkg.prior.utils import apply_counterfactual_perturbation
from synthefy_pkg.utils.logging_utils import configure_logging

SUPPORTED_KERNEL_TYPES = [
    "gaussian",
    "laplacian",
    "sobel",
    "recent_exponent",
    "uniform",
    "butterworth",
    "exponential",
    "triangular",
    "box",
    "mixed",
    "random",
]
MIXED_KERNEL_TYPES = copy.deepcopy(SUPPORTED_KERNEL_TYPES)
MIXED_KERNEL_TYPES.remove("mixed")


class SmoothingLagFilter(nn.Module):
    def __init__(
        self,
        kernel_size_min: int,
        kernel_size_max: int,
        sigma: float,
        hidden_dim: int = 1,
        num_timesteps: int = 1,
        kernel_direction: str = "history",
        kernel_type: str = "gaussian",
        functional_lags: Optional[List[int]] = None,
        functional_lag_variance: float = 0.0,
        mixed_kernel_types: Optional[List[str]] = None,
        random_lags: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        self.sigma = sigma  # only used for gaussian skernels
        self.kernel_direction = kernel_direction
        self.kernel_type = kernel_type
        self.hidden_dim = hidden_dim
        self.device = device
        self.functional_lags = functional_lags
        self.max_functional_lag = (
            max(self.functional_lags) if self.functional_lags is not None else 0
        )
        self.functional_lag_variance = (
            functional_lag_variance  # only used for random kernels
        )
        self.random_lags = random_lags
        self.num_timesteps = num_timesteps
        self.supported_kernel_types = SUPPORTED_KERNEL_TYPES
        self.used_kernel_size = self.kernel_size_max
        self.used_kernel_size += (
            self.max_functional_lag
            + self.max_functional_lag * (self.kernel_direction in ["both"])
        )
        add_sided_kernel = self.used_kernel_size * int(
            self.kernel_direction in ["history", "future"]
        )
        self.used_kernel_size += add_sided_kernel
        self.mixed_kernel_types = (
            mixed_kernel_types
            if mixed_kernel_types is not None
            else MIXED_KERNEL_TYPES
        )
        self._init_filters()

    def _init_filters(self):
        """Initialize learnable convolutional filters and other dynamic components."""
        if self.random_lags:
            self.constructed_filters = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=self.num_timesteps,
                        out_features=self.num_timesteps,
                        bias=False,
                    ).to(self.device)
                    for k in range(self.hidden_dim)
                ]
            )
        else:
            self.constructed_filters = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=self.used_kernel_size,
                        padding=self.used_kernel_size,
                        bias=False,
                    ).to(self.device)
                    for k in range(self.hidden_dim)
                ]
            )

        # set the weights according to the kernel type
        self.kernel_endpoints = []
        self.kernel_sizes = []

        for i in range(self.hidden_dim):
            kernel_type = (
                self.kernel_type
                if self.kernel_type != "mixed"
                else np.random.choice(
                    [
                        "gaussian",
                        "laplacian",
                        "sobel",
                        "butterworth",
                        "exponential",
                        "triangular",
                        "box",
                    ]
                )
            )
            effective_kernel_size = int(
                self.kernel_size_min
                + (self.kernel_size_max - self.kernel_size_min)
                * np.random.rand()
            )
            # Ensure kernel size is odd
            if effective_kernel_size % 2 == 0:
                effective_kernel_size += 1
            effective_functional_lag = (
                self.functional_lags[i]
                if self.functional_lags is not None
                else 0
            )
            self.constructed_filters[i].weight.data.fill_(
                0
            )  # reset the weights
            self.constructed_filters[i], kernel_index = self._create_kernel(
                kernel_type,
                effective_kernel_size,
                self.used_kernel_size,
                self.constructed_filters[i],
                effective_functional_lag,
            )

            self.kernel_endpoints.append(kernel_index)
            self.kernel_sizes.append(effective_kernel_size)

    def _add_padding_and_lag(
        self,
        kernel: torch.Tensor,
        layer: nn.Module,
        functional_lag: float,
        kernel_direction: str,
        effective_kernel_size: int,
        used_kernel_size: int,
    ) -> Tuple[nn.Module, float]:
        apply_at_offset = (
            int(((used_kernel_size - effective_kernel_size) / 2)) + 1
        )
        direction = 1
        layer_kernel_size = layer.weight.data.shape[-1]

        # pad the left or right with zeros (should already be zero though)
        if kernel_direction == "future":
            logger.debug(
                f"history: {layer.weight.data.shape}, effective_kernel_size: {effective_kernel_size}, used_kernel_size: {used_kernel_size}"
            )
            layer.weight.data[..., -effective_kernel_size:] = 0
            apply_at_offset = int(used_kernel_size / 2)
            direction = -1
        elif kernel_direction == "history":
            layer.weight.data[..., :effective_kernel_size] = 0
            apply_at_offset = int(used_kernel_size / 2)
            direction = 1
        # if kernel_direction == "both":
        #     apply_at_offset += int(self.max_functional_lag / 2)

        # put the kernel in the desired spot to produce lags
        if self.random_lags:
            offsets = torch.arange(self.num_timesteps, device=self.device)
            used_lags = (
                functional_lag
                + self.functional_lag_variance
                * torch.rand(self.num_timesteps, device=self.device)
            )
            start_locations = offsets + used_lags * direction + 1
            start_locations = torch.round(start_locations).long()
            end_locations = start_locations.clone()
            start_locations = start_locations - effective_kernel_size

            idxes = torch.arange(self.num_timesteps, device=self.device)[
                (start_locations > 0) & (end_locations < self.num_timesteps)
            ]
            logger.debug(
                f"used bools: {(start_locations > 0) & (end_locations < self.num_timesteps)}"
            )
            logger.debug(
                f"idxes: {idxes}, used_lags: {used_lags}, start_locations: {start_locations}, end_locations: {end_locations}, effective_kernel_size: {effective_kernel_size}"
            )
            for i in idxes:
                use_start = start_locations[i]
                use_kernel = kernel
                if start_locations[i] < 0:
                    use_start = 0
                    use_kernel = kernel[
                        : end_locations[i]
                    ]  # TODO: only works with lagging
                layer.weight.data[
                    i,
                    use_start : end_locations[i],
                ] = use_kernel
            logger.debug("layer.weight.data: ", layer.weight.data)
            use_lag = (
                used_lags.mean().item()
            )  # give back an approximate value for record keeping

        else:
            if functional_lag >= 0:
                use_lag = int(functional_lag * direction)
                logger.debug(
                    f"use_lag: {use_lag}, apply_at_offset: {apply_at_offset}, effective_kernel_size: {effective_kernel_size}, used_kernel_size: {used_kernel_size}"
                )
                if (
                    apply_at_offset + use_lag - 1 >= 0
                    and apply_at_offset + use_lag + effective_kernel_size - 1
                    <= layer_kernel_size
                ):
                    logger.debug(
                        f"apply_at_offset: {apply_at_offset}, use_lag: {use_lag}, effective_kernel_size: {effective_kernel_size}, layer.weight.data.shape: {layer.weight.data.shape}"
                    )
                    logger.debug(
                        f"kernel: {kernel.shape}, layer.weight.offsetdata: {layer.weight.data[..., apply_at_offset + use_lag - 1 : apply_at_offset + use_lag + effective_kernel_size - 1].shape}, start: {apply_at_offset + use_lag - effective_kernel_size}, end: {apply_at_offset + use_lag}"
                    )
                    logger.debug(f"layer.weight.data: {layer.weight.data}")
                    logger.debug(f"kernel_direction: {kernel_direction}")
                    layer.weight.data[
                        ...,
                        apply_at_offset + use_lag - 1 : apply_at_offset
                        + use_lag
                        + effective_kernel_size
                        - 1,
                    ] = kernel
                else:
                    logger.error(
                        f"functional lag does not contain kernel: {layer.weight.shape} and {kernel.shape}, use_lag: {use_lag}, apply_at_offset: {apply_at_offset}, effective_kernel_size: {effective_kernel_size}, layer_kernel_size: {layer_kernel_size}"
                    )
                    assert False, (
                        f"functional lag does not contain kernel: {layer.weight.shape} and {kernel.shape}, use_lag: {use_lag}, apply_at_offset: {apply_at_offset}, effective_kernel_size: {effective_kernel_size}, layer_kernel_size: {layer_kernel_size}"
                    )
            else:
                use_lag = 0  # even though not necessary for usage because it's already embedded
                logger.debug(
                    layer.weight.data.shape,
                    kernel.shape,
                    apply_at_offset,
                    apply_at_offset + effective_kernel_size,
                    layer.weight.data[
                        ...,
                        apply_at_offset : apply_at_offset
                        + effective_kernel_size,
                    ].shape,
                )
                layer.weight.data[
                    :,
                    :,
                    apply_at_offset - 1 : apply_at_offset
                    + effective_kernel_size
                    - 1,
                ] = kernel  # default to no lag

        return layer, float(apply_at_offset + use_lag)

    def _get_kernel(
        self,
        kernel_type: str,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        if kernel_type == "gaussian":
            kernel = self._create_gaussian_kernel(
                kernel_size,
                used_kernel_size,
                self.sigma,
                layer,
                effective_functional_lag,
            )
        elif kernel_type == "laplacian":
            kernel = self._create_laplacian_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "sobel":
            kernel = self._create_sobel_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "recent_exponent":
            kernel = self._create_recent_exponent_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "uniform":
            kernel = self._create_uniform_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "butterworth":
            kernel = self._create_butterworth_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "exponential":
            kernel = self._create_exponential_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "triangular":
            kernel = self._create_triangular_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "box":
            kernel = self._create_box_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        elif kernel_type == "mixed":
            kernel = self._create_mixed_kernel(
                kernel_size, used_kernel_size, layer, effective_functional_lag
            )
        else:
            assert False, f"kernel_type: {kernel_type} not supported"
        return kernel

    def _create_kernel(
        self,
        kernel_type: str,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> Tuple[nn.Module, float]:
        """Create a base kernel."""

        kernel = self._get_kernel(
            kernel_type,
            kernel_size,
            used_kernel_size,
            layer,
            effective_functional_lag,
        )

        layer, apply_at_offset = self._add_padding_and_lag(
            kernel,
            layer,
            effective_functional_lag,
            self.kernel_direction,
            kernel_size,
            used_kernel_size,
        )
        return layer, float(apply_at_offset)

    def _create_gaussian_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        sigma: float,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Apply Gaussian smoothing filter."""
        half_kernel_size = kernel_size // 2
        x = torch.arange(
            -half_kernel_size,
            half_kernel_size + 1,
            dtype=torch.float32,
            device=self.device,
        )
        logger.debug(f"half_kernel_size: {half_kernel_size}, x: {x.shape}")
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(self.device)
        return kernel

    def _create_laplacian_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Apply 1D Laplacian edge detection filter."""
        kernel = torch.ones(
            kernel_size, dtype=torch.float32, device=self.device
        )
        center_idx = kernel_size // 2
        kernel[center_idx] = -kernel_size + 1
        return kernel

    def _create_sobel_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Apply 1D edge detection filter."""
        half_size = kernel_size // 2
        kernel = torch.arange(
            -half_size, half_size + 1, dtype=torch.float32, device=self.device
        )
        kernel = kernel.to(self.device)
        return kernel

    def _create_recent_exponent_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a kernel with strong bias for recent values using exponential decay."""
        # Create exponential decay kernel that heavily weights recent values
        # The kernel is flipped so recent values (right side) get higher weights
        r_x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x = torch.flip(r_x, dims=[0])
        # Exponential decay: recent values (higher indices) get exponentially higher weights
        decay_rate = 2.0  # Controls how strong the bias is
        kernel = torch.exp(decay_rate * x / kernel_size)
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        kernel = kernel.to(self.device)
        return kernel

    def _create_uniform_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a uniform kernel (equal weights for all positions)."""
        kernel = torch.ones(
            kernel_size, dtype=torch.float32, device=self.device
        )
        kernel = kernel / kernel.sum()  # Normalize
        kernel = kernel.to(self.device)
        return kernel

    def _create_butterworth_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a Butterworth low-pass filter kernel."""
        # Create a time-domain Butterworth-like kernel using window functions
        # This is a simplified approximation that provides smooth low-pass characteristics

        # Create a window-based kernel that approximates Butterworth behavior
        center = kernel_size // 2
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)

        # Use a raised cosine window with additional smoothing
        # This provides similar characteristics to a Butterworth filter
        window = 0.5 * (
            1 + torch.cos(2 * torch.pi * (x - center) / kernel_size)
        )

        # Apply additional smoothing to create low-pass characteristics
        smoothing_factor = 0.8
        kernel = window * smoothing_factor + (
            1 - smoothing_factor
        ) * torch.ones_like(window)

        # Normalize the kernel
        kernel = kernel / kernel.sum()
        kernel = kernel.to(self.device)

        return kernel

    def _create_exponential_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create an exponential moving average kernel."""
        # Create exponential moving average kernel
        alpha = 0.3  # Smoothing factor (0 < alpha < 1)
        kernel = torch.zeros(
            kernel_size, dtype=torch.float32, device=self.device
        )

        # Fill kernel with exponential weights
        for i in range(kernel_size):
            kernel[i] = alpha * (1 - alpha) ** (kernel_size - 1 - i)

        # Normalize the kernel
        kernel = kernel / kernel.sum()
        kernel = kernel.to(self.device)

        return kernel

    def _create_triangular_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a triangular kernel (increasing weights towards the center)."""
        # Create a triangular kernel that increases towards the center
        # The kernel is flipped so recent values (right side) get higher weights
        r_x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        x = torch.flip(r_x, dims=[0])
        # Triangular decay: recent values (higher indices) get linearly higher weights
        kernel = 1 - (x / kernel_size)
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        kernel = kernel.to(self.device)

        return kernel

    def _create_box_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a box (rectangular) kernel for simple averaging."""
        # Create a box kernel - this is essentially the same as uniform
        # but kept separate for clarity and potential future modifications
        kernel = torch.ones(
            kernel_size, dtype=torch.float32, device=self.device
        )
        kernel = kernel / kernel.sum()  # Normalize
        kernel = kernel.to(self.device)

        return kernel

    def _create_mixed_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
    ) -> torch.Tensor:
        """Create a mixed kernel that combines multiple kernel types with random weights."""
        # Create kernels of different types
        sampled_num_mixed_kernels = np.random.randint(
            self.num_mixed_kernels_min, self.num_mixed_kernels_max + 1
        )
        selected_kernels = np.random.choice(
            self.mixed_kernel_types,
            size=sampled_num_mixed_kernels,
            replace=False,
        )
        weights = np.random.rand(sampled_num_mixed_kernels)
        weights = weights / weights.sum()
        base_kernel = torch.zeros(
            kernel_size, dtype=torch.float32, device=self.device
        )
        for kernel, weight in zip(selected_kernels, weights):
            base_kernel += weight * self._get_kernel(
                kernel,
                kernel_size,
                used_kernel_size,
                layer,
                effective_functional_lag,
            )
        return base_kernel

    def _create_random_kernel(
        self,
        kernel_size: int,
        used_kernel_size: int,
        layer: nn.Module,
        effective_functional_lag: float,
        is_2d: bool = False,
        is_positive: bool = False,
    ) -> torch.Tensor:
        """Create a random kernel (weights are random)."""
        if is_positive:
            if is_2d:
                kernel = torch.rand(
                    kernel_size,
                    kernel_size,
                    dtype=torch.float32,
                    device=self.device,
                )
                kernel = kernel / kernel.sum(dim=1, keepdim=True)  # Normalize
            else:
                kernel = torch.rand(
                    kernel_size, dtype=torch.float32, device=self.device
                )
                kernel = kernel / kernel.sum()  # Normalizse
        else:
            if is_2d:
                kernel = (
                    torch.rand(
                        kernel_size,
                        kernel_size,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    - 0.5
                ) * 2
            else:
                kernel = (
                    torch.rand(
                        kernel_size, dtype=torch.float32, device=self.device
                    )
                    - 0.5
                ) * 2
        kernel = kernel.to(self.device)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the smoothing lag filter to the input tensor.
        use a different filter for each feature
        """
        # Ensure input tensor is on the same device as the filters
        if x.device != next(self.constructed_filters.parameters()).device:
            x = x.to(next(self.constructed_filters.parameters()).device)

        changed_dims = False
        if x.dim() == 2:
            x = x.unsqueeze(
                0
            )  # dimensions should be (batch_size, seq_len, hidden_dim)
            changed_dims = True

        skip_padding = 0
        is_linear = isinstance(self.constructed_filters[0], nn.Linear)
        if is_linear:
            filter_shape = self.constructed_filters[0].weight.shape[1]
        else:
            filter_shape = self.constructed_filters[0].weight.shape[2]
        skip_padding = filter_shape // 2

        logger.debug(f"x.shape: {x.shape}")
        logger.debug(self.constructed_filters[0].weight)
        logger.debug(x[0:1, :, 0])

        # Precompute slicing indices based on kernel direction
        if self.kernel_direction == "history":
            start_idx = skip_padding + 1
            end_idx = -skip_padding
        elif self.kernel_direction == "future":
            start_idx = skip_padding
            end_idx = -skip_padding - 1
        else:  # centered
            start_idx = skip_padding + 1
            end_idx = -skip_padding - 1

        # ff = (
        #     self.constructed_filters[0](x[:, :, 0])
        #     if is_linear
        #     else self.constructed_filters[0](x[:, :, 0].unsqueeze(1))[:, 0, :]
        # )
        # weight_data = self.constructed_filters[0].weight.data
        # logger.debug(
        #     f"functional_lag: {self.functional_lags}, kernel_endpoints: {self.kernel_endpoints}, weight_shape: {weight_data.shape}, kernel_sizes: {self.kernel_sizes}, kernel_direction: {self.kernel_direction}"
        # )
        # logger.debug(
        #     f"start_idx: {start_idx}, end_idx: {end_idx}, is_linear: {is_linear}, weight_data: {weight_data}, filtered_feature: {ff}"
        # )

        # Efficient implementation
        if is_linear:
            # For linear layers: process all features sequentially
            filtered_features = torch.stack(
                [
                    self.constructed_filters[i](x[:, :, i])
                    for i in range(x.shape[2])
                ],
                dim=2,
            )
        else:
            # For conv layers: use grouped convolution for maximum efficiency
            # Reshape input to (batch, features, seq_len, 1) for conv1d
            x_reshaped = x.transpose(1, 2).unsqueeze(
                -1
            )  # (batch, features, seq_len, 1)

            # Method 1: Use grouped convolution if all filters have same kernel size
            # if self._can_use_grouped_conv():
            # filtered_features_gc = self._apply_grouped_conv(x_reshaped, start_idx, end_idx)
            # else:
            # Method 2: Fallback to sequential processing
            filtered_list = []
            for i in range(x.shape[2]):
                filtered = self.constructed_filters[i](
                    x_reshaped[:, i : i + 1, :, 0]
                )
                filtered = filtered[:, 0, start_idx:end_idx]
                filtered_list.append(filtered)
            filtered_features = torch.stack(filtered_list, dim=2)

        # Visualize layers pre/post filtering and filters
        # self._visualize_filtering_process(x, [filtered_features[:, :, i] for i in range(filtered_features.shape[2])], save_dir="/tmp/filter_visualizations")
        if changed_dims:
            filtered_features = filtered_features.squeeze(0)
        return filtered_features

    def _can_use_grouped_conv(self) -> bool:
        """Check if all filters have the same kernel size for grouped convolution."""
        if not self.constructed_filters:
            return False

        first_kernel_size = self.constructed_filters[0].weight.shape[-1]
        return all(
            filter_layer.weight.shape[-1] == first_kernel_size
            for filter_layer in self.constructed_filters
        )

    def _apply_grouped_conv(
        self, x_reshaped: torch.Tensor, start_idx: int, end_idx: int
    ) -> torch.Tensor:
        """Apply grouped convolution for maximum efficiency when all filters have same kernel size."""
        batch_size, n_features, seq_len, _ = x_reshaped.shape

        # Create a single grouped convolution layer
        kernel_size = self.constructed_filters[0].weight.shape[-1]

        # Combine all filter weights into a single tensor
        combined_weights = torch.stack(
            [
                self.constructed_filters[i].weight[0, 0, :]  # (kernel_size,)
                for i in range(n_features)
            ],
            dim=0,
        )  # (n_features, kernel_size)

        # Reshape for grouped conv: (out_channels, in_channels, kernel_size)
        grouped_weights = combined_weights.unsqueeze(
            1
        )  # (n_features, 1, kernel_size)

        # Create temporary grouped conv layer
        grouped_conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            groups=n_features,  # This makes it grouped convolution
            padding=kernel_size // 2,
            bias=False,
        ).to(x_reshaped.device)

        # Set the weights
        grouped_conv.weight.data = grouped_weights

        # Apply grouped convolution
        # Input shape: (batch, n_features, seq_len, 1) -> (batch, n_features, seq_len)
        x_for_conv = x_reshaped.squeeze(-1)  # (batch, n_features, seq_len)
        filtered = grouped_conv(x_for_conv)  # (batch, n_features, seq_len)

        # Slice and transpose back to match expected output format
        filtered = filtered[
            :, :, start_idx:end_idx
        ]  # (batch, n_features, filtered_seq_len)
        filtered = filtered.transpose(
            1, 2
        )  # (batch, filtered_seq_len, n_features)

        return filtered

    def _visualize_filtering_process(
        self,
        x: torch.Tensor,
        filtered_features: list,
        save_dir: str = "/tmp/filter_visualizations",
    ):
        """Visualize the filtering process including pre/post filtering and filter weights."""
        import os
        from pathlib import Path

        import matplotlib.pyplot as plt

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        n_features = x.shape[2]
        n_samples = min(100, x.shape[1])

        # Helper functions
        def _plot_signal(ax, data, title, color="blue", label=None, lag=None):
            ax.plot(data, color=color, alpha=0.7, linewidth=2, label=label)
            if lag is not None and lag > 0:
                ax.axvline(
                    x=lag,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Lag: {lag}",
                )
            ax.set_title(title)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if label or (lag is not None and lag > 0):
                ax.legend()

        def _plot_weights(ax, weights, title, is_linear=True):
            if is_linear:
                ax.bar(range(len(weights)), weights, alpha=0.7)
                ax.set_xlabel("Weight Index")
            else:
                ax.plot(weights, "o-", linewidth=2, markersize=6)
                ax.set_xlabel("Kernel Position")
                ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.set_title(title)
            ax.set_ylabel("Weight Value")
            ax.grid(True, alpha=0.3)

        def _get_filter_weights(i):
            weights = (
                self.constructed_filters[i].weight.data.detach().cpu().numpy()
            )
            is_linear = len(weights.shape) == 1
            if not is_linear:
                weights = (
                    weights[0, 0, :]
                    if len(weights.shape) == 3
                    else weights[0, :]
                )
            return weights, is_linear

        def _get_filter_lag(i):
            """Get the lag for filter i, accounting for kernel direction and padding."""
            lag = 0

            # Try to get functional lag first
            if (
                hasattr(self, "functional_lags")
                and self.functional_lags is not None
                and i < len(self.functional_lags)
            ):
                lag = int(self.functional_lags[i])
                logger.info(f"Filter {i}: Using functional_lags = {lag}")
            # Try kernel sizes as fallback
            elif hasattr(self, "kernel_sizes") and i < len(self.kernel_sizes):
                kernel_size = self.kernel_sizes[i]
                if hasattr(self, "kernel_direction"):
                    if self.kernel_direction == "history":
                        lag = kernel_size - 1
                    elif self.kernel_direction == "future":
                        lag = 0
                    else:  # centered
                        lag = kernel_size // 2
                else:
                    lag = kernel_size // 2
                logger.debug(
                    f"Filter {i}: Using kernel_size = {kernel_size}, direction = {getattr(self, 'kernel_direction', 'unknown')}, lag = {lag}"
                )
            # Try to estimate from filter weights
            else:
                weights = self.constructed_filters[i].weight.data
                if len(weights.shape) > 1:
                    # For conv layers, estimate lag from kernel size
                    kernel_size = weights.shape[-1]
                    lag = kernel_size // 2
                    logger.debug(
                        f"Filter {i}: Estimated from weights, kernel_size = {kernel_size}, lag = {lag}"
                    )
                else:
                    logger.debug(
                        f"Filter {i}: No lag information available, using 0"
                    )

            return lag

        # Overview plot
        fig, axes = plt.subplots(3, n_features, figsize=(4 * n_features, 12))
        if n_features == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_features):
            # Get lag for this filter
            lag = _get_filter_lag(i)

            # Pre/post filtering with lag indicators
            _plot_signal(
                axes[0, i],
                x[0, :n_samples, i].detach().cpu().numpy(),
                f"Pre-filtering Feature {i}",
                lag=lag,
            )
            _plot_signal(
                axes[1, i],
                filtered_features[i][0, :n_samples].detach().cpu().numpy(),
                f"Post-filtering Feature {i}",
                lag=lag,
            )

            # Filter weights
            weights, is_linear = _get_filter_weights(i)
            _plot_weights(
                axes[2, i],
                weights,
                f"Filter {i} Weights ({'Linear' if is_linear else 'Conv'})",
                is_linear,
            )

        plt.tight_layout()
        plt.savefig(
            save_path / "filtering_process_overview.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Input vs output comparison
        fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 8))
        if n_features == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_features):
            _plot_signal(
                axes[0, i],
                x[0, :n_samples, i].detach().cpu().numpy(),
                f"Feature {i} - Input Signal",
                label="Input",
            )
            _plot_signal(
                axes[1, i],
                filtered_features[i][0, :n_samples].detach().cpu().numpy(),
                f"Feature {i} - Filtered Output",
                "red",
                "Filtered Output",
            )

        plt.tight_layout()
        plt.savefig(
            save_path / "input_vs_output_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Filter visualizations saved to {save_path}")


class LayerOperator(nn.Module):
    """
    A comprehensive layer operator for time series data that handles:
    - Convolutional filters (static and dynamic)
    - Lag operations with varying amounts
    - Counterfactual perturbations
    - Normalization operations

    Parameters
    ----------
    seq_len : int
        Length of the time series sequences

    hidden_dim : int
        Number of features in the time series

    device : str, default='cpu'
        Device to perform operations on

    filter_config : dict, optional
        Configuration for convolutional filters

    lag_config : dict, optional
        Configuration for lag operations

    perturbation_config : dict, optional
        Configuration for counterfactual perturbations

    normalization_config : dict, optional
        Configuration for normalization operations
    """

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        device: str = "cpu",
        filter_config: Optional[Dict] = None,
        lag_config: Optional[Dict] = None,
        perturbation_config: Optional[Dict] = None,
        normalization_config: Optional[Dict] = None,
        logging_level: str = "INFO",
        disable_logging: bool = False,
    ):
        super().__init__()  # Initialize the nn.Module base class

        # Configure logging
        configure_logging(level=logging_level, disable_logging=disable_logging)

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.device = device

        self.perturbation_config = perturbation_config or {
            "enabled": False,
            "counterfactual_type": "additive_uniform",
            "noise_std": 0.1,
            "apply_probability": 0.4,
        }

        self.normalization_config = normalization_config or {
            "enabled": True,
            "minimum_magnitude": 0.1,
            "maximum_magnitude": 20.0,
            "normalization_types": [
                "z_score",
                "min_max",
                "robust",
                "batch_norm",
            ],
            "apply_probability": 0.8,
        }

        # Initialize learnable components
        assert filter_config is not None, "filter_config must be provided"
        assert lag_config is not None, "lag_config must be provided"
        self.filter_config = filter_config
        self.lag_config = lag_config
        self.smoothing_lag_filter = SmoothingLagFilter(
            sigma=self.filter_config["sigma"],
            kernel_direction=self.filter_config["kernel_direction"],
            kernel_type=self.filter_config["kernel_type"],
            kernel_size_min=self.filter_config["kernel_size_min"],
            kernel_size_max=self.filter_config["kernel_size_max"],
            hidden_dim=self.hidden_dim,
            functional_lags=self.lag_config["functional_lags"],
            functional_lag_variance=self.lag_config["functional_lag_variance"],
            random_lags=self.lag_config["random_lags"],
            num_timesteps=self.seq_len,
            # TODO: mixed kernel types not well supported
        )
        self.used_kernel_size = self.smoothing_lag_filter.used_kernel_size

    def apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a kernel to the input tensor."""
        return self.smoothing_lag_filter(x)

    def apply_counterfactual_perturbations(
        self,
        x: torch.Tensor,
        perturbation_types: Optional[List[str]] = None,
        counterfactual_indices: Optional[List[int]] = None,
        start_idx_at: Optional[int] = None,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply counterfactual perturbations to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim)
        perturbation_types : list, optional
            List of perturbation types to apply. If None, uses configured types.

        Returns
        -------
        torch.Tensor
            Perturbed tensor of same shape as input
        """
        if not self.perturbation_config["enabled"]:
            return x

        apply_counterfactual_perturbation(
            self.perturbation_config["counterfactual_type"],
            x,
            counterfactual_indices,
            start_idx_at,
            noise_std,
        )
        return x

    def apply_normalization(
        self, x: torch.Tensor, normalization_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Apply normalization operations to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim)
        normalization_types : list, optional
            List of normalization types to apply. If None, uses configured types.

        Returns
        -------
        torch.Tensor
            Normalized tensor of same shape as input
        """
        if not self.normalization_config["enabled"]:
            return x

        if normalization_types is None:
            normalization_types = self.normalization_config[
                "normalization_types"
            ]

        normalized_x = x.clone()

        assert normalization_types is not None, (
            "normalization_types must be provided"
        )
        for norm_type in normalization_types:
            if random.random() > self.normalization_config["apply_probability"]:
                continue

            # only apply nroamlziation to values with magnitude less than minimum_magnitude and greater than maximum_magnitude
            # use negative values as flag
            if self.normalization_config["minimum_magnitude"] >= 0:
                if (
                    torch.mean(normalized_x.abs())
                    > self.normalization_config["minimum_magnitude"]
                ):
                    continue
            if self.normalization_config["maximum_magnitude"] != -1:
                if (
                    torch.mean(normalized_x.abs())
                    < self.normalization_config["maximum_magnitude"]
                ):
                    continue

            if norm_type == "z_score":
                normalized_x = self._apply_z_score_normalization(normalized_x)
            elif norm_type == "min_max":
                normalized_x = self._apply_min_max_normalization(normalized_x)
            elif norm_type == "robust":
                normalized_x = self._apply_robust_normalization(normalized_x)
            elif norm_type == "batch_norm":
                normalized_x = self._apply_batch_normalization(normalized_x)

        return normalized_x

    def _apply_z_score_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        return (x - mean) / std

    def _apply_min_max_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization."""
        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)

    def _apply_robust_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply robust normalization using median and IQR."""
        median = torch.median(x, dim=1, keepdim=True)[0]
        q75 = torch.quantile(x, 0.75, dim=1, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=1, keepdim=True)
        iqr = q75 - q25 + 1e-8
        return (x - median) / iqr

    def _apply_batch_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization."""
        # Reshape for batch norm: (batch, features, seq_len)
        x_reshaped = x.transpose(1, 2)

        # Apply batch norm along feature dimension
        batch_norm = nn.BatchNorm1d(self.hidden_dim).to(self.device)
        normalized = batch_norm(x_reshaped)

        return normalized.transpose(1, 2)  # Back to (batch, seq_len, features)

    def apply_all_operations(
        self,
        x: torch.Tensor,
        operation_order: Optional[List[str]] = None,
        counterfactual_indices: Optional[List[int]] = None,
        start_idx_at: Optional[int] = None,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply all configured operations in the specified order.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim)
        operation_order : list, optional
            Order of operations to apply. If None, uses default order.

        Returns
        -------
        torch.Tensor
            Processed tensor of same shape as input
        """
        if operation_order is None:
            operation_order = [
                "normalization",
                "convolutional_filters",
                "perturbations",
            ]

        processed_x = x.clone()

        for operation in operation_order:
            if operation == "normalization":
                processed_x = self.apply_normalization(processed_x)
            elif operation == "convolutional_filters":
                processed_x = self.apply_filter(processed_x)
            elif operation == "perturbations":
                processed_x = self.apply_counterfactual_perturbations(
                    processed_x,
                    counterfactual_indices=counterfactual_indices,
                    start_idx_at=start_idx_at,
                    noise_std=noise_std,
                )

        return processed_x
