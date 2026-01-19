from __future__ import annotations

import bisect
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from loguru import logger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from synthefy_pkg.prior.activations import get_activations_by_name
from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler
from synthefy_pkg.prior.input_sampling.mixed_sampler import MixedSampler
from synthefy_pkg.prior.input_sampling.real_ts_sampling import RealTSSampler
from synthefy_pkg.prior.input_sampling.stl_sampling import STLSampler
from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler
from synthefy_pkg.prior.layer_operator import LayerOperator
from synthefy_pkg.prior.ts_noise import MixedTSNoise, TSNoise
from synthefy_pkg.prior.utils import (
    GaussianNoise,
    apply_counterfactual_perturbation,
)


class MLPSCM(nn.Module):
    """Generates synthetic tabular datasets using a Multi-Layer Perceptron (MLP) based Structural Causal Model (SCM).

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate hidden states of the MLP transformation applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the MLP becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final MLP layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate MLP outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate MLP outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=10
        The total number of layers in the MLP transformation network. Must be >= 2.
        Includes the initial linear layer and subsequent blocks of
        (Activation -> Linear -> Noise).

    hidden_dim : int, default=20
        The dimensionality of the hidden representations within the MLP layers.
        If `is_causal=True`, this is automatically increased if it's smaller than
        `num_outputs + 2 * num_features` to ensure enough intermediate variables
        are generated for sampling `X` and `y`.

    mlp_activations : default=nn.Tanh
        The activation function to be used after each linear transformation
        in the MLP layers (except the first).

    init_std : float, default=1.0
        The standard deviation of the normal distribution used for initializing
        the weights of the MLP's linear layers.

    block_wise_dropout : bool, default=True
        Specifies the weight initialization strategy.
        - If `True`, uses a 'block-wise dropout' initialization where only random
          blocks within the weight matrix are initialized with values drawn from
          a normal distribution (scaled by `init_std` and potentially dropout),
          while the rest are zero. This encourages sparsity.
        - If `False`, uses standard normal initialization for all weights, followed
          by applying dropout mask based on `mlp_dropout_prob`.

    mlp_dropout_prob : float, default=0.1
        The dropout probability applied to weights during *standard* initialization
        (i.e., when `block_wise_dropout=False`). Ignored if
        `block_wise_dropout=True`. The probability is clamped between 0 and 0.99.

    scale_init_std_by_dropout : bool, default=True
        Whether to scale the `init_std` during weight initialization to compensate
        for the variance reduction caused by dropout. If `True`, `init_std` is
        divided by `sqrt(1 - dropout_prob)` or `sqrt(keep_prob)` depending on the
        initialization method.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each MLP
        layer's linear transformation (except the first layer).

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.

    use_ts_noise : bool, default=False
        Whether to use time series based noise instead of Gaussian noise.
        If True, the noise will be generated using TSSampler with various time series patterns.

    ts_noise_sampling : str, default='mixed_simple'
        The type of time series to generate for noise when use_ts_noise=True.
        Options include: 'fourier', 'wiener', 'arima', 'mixed_simple', 'mixed_all',
        'mixed_subset', 'mixed_periodic', 'mixed_both'.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_outputs: int = 1,
        is_causal: bool = True,
        num_causes: int = 10,
        y_is_effect: bool = True,
        in_clique: bool = False,
        sort_features: bool = True,
        num_layers: int = 10,
        hidden_dim: int = 20,
        mlp_activations: Any = nn.Tanh,
        init_std: float = 1.0,
        block_wise_dropout: bool = True,
        mlp_dropout_prob: float = 0.1,
        scale_init_std_by_dropout: bool = True,
        sampling: str = "normal",
        ts_noise_sampling: str = "mixed_simple",
        used_sampler: str = "ts",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        noise_type: str = "gaussian",
        mixed_noise_ratio: float = 0.5,
        device: str = "cpu",
        yaml_path: str = "",
        return_single_output: bool = False,
        max_features: int = 0,
        sampling_mixed_names: list[str] = [],
        sigmoid_mixed_sampling_rate: int = -1,
        real_data_loader: Optional[DataLoader] = None,
        try_regenerate_unreachable_target: bool = False,
        counterfactual_type: str = "random_uniform",
        counterfactual_enabled: bool = False,
        counterfactual_variance: float = 0.1,
        counterfactual_probability: float = 0.4,
        kernel_size_min: int = 3,
        kernel_size_max: int = 5,
        sigma: float = 1.0,
        kernel_direction: str = "history",
        kernel_type: str = "gaussian",
        use_layer_operator: bool = False,
        # functional_lags: Optional[List[int]] = None,
        functional_lag_min: float = 0.0,
        functional_lag_max: float = 0.0,
        functional_lag_rate: float = 0.0,
        functional_lag_variance: float = 0.1,
        layer_has_functional_lag_rate: float = 0.0,
        node_has_functional_lag_rate: float = 0.0,
        random_lags: bool = False,
        normalization_enabled: bool = True,
        normalization_minimum_magnitude: float = 0.1,
        normalization_maximum_magnitude: float = 20.0,
        normalization_types: List[str] = [
            "z_score",
            "min_max",
            "robust",
            "batch_norm",
        ],
        normalization_apply_probability: float = 0.8,
        **kwargs: Dict[str, Any],
    ):
        super(MLPSCM, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = False  # in_clique
        self.sort_features = sort_features
        self.max_features = max_features
        self.use_layer_operator = use_layer_operator
        self.num_layers = int(num_layers)
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        self.kernel_sigma = sigma
        self.kernel_direction = kernel_direction
        self.kernel_type = kernel_type
        self.functional_lag_min = functional_lag_min
        self.functional_lag_max = functional_lag_max
        self.functional_lag_rate = functional_lag_rate
        self.functional_lag_variance = functional_lag_variance
        self.layer_has_functional_lag_rate = layer_has_functional_lag_rate
        self.node_has_functional_lag_rate = node_has_functional_lag_rate
        self.random_lags = random_lags

        self.hidden_dim = hidden_dim
        self.mlp_activations = mlp_activations
        self.init_std = init_std
        self.block_wise_dropout = block_wise_dropout
        self.mlp_dropout_prob = mlp_dropout_prob
        self.scale_init_std_by_dropout = scale_init_std_by_dropout
        self.sampling = sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.noise_type = noise_type
        self.mixed_noise_ratio = mixed_noise_ratio
        self.device = device
        self.used_sampler = used_sampler
        self.ts_noise_sampling = ts_noise_sampling
        self.return_single_output = return_single_output
        self.sampling_mixed_names = sampling_mixed_names
        self.sigmoid_mixed_sampling_rate = sigmoid_mixed_sampling_rate
        self.try_regenerate_unreachable_target = (
            try_regenerate_unreachable_target
        )
        self.counterfactual_enabled = counterfactual_enabled
        self.counterfactual_type = counterfactual_type
        self.counterfactual_variance = counterfactual_variance
        self.counterfactual_probability = counterfactual_probability
        self.normalization_enabled = normalization_enabled
        self.normalization_minimum_magnitude = normalization_minimum_magnitude
        self.normalization_maximum_magnitude = normalization_maximum_magnitude
        self.normalization_types = normalization_types
        self.normalization_apply_probability = normalization_apply_probability

        self.real_data_loader = real_data_loader
        self.block_size = -1  # store the block dropout size

        if yaml_path:
            self.block_wise_dropout = False
            self.mlp_dropout_prob = 0.0
            self.init_from_yaml(yaml_path)
        else:
            if self.is_causal:
                # Ensure enough intermediate variables for sampling X and y
                # self.hidden_dim = max(
                #     self.hidden_dim, self.num_outputs + 2 * self.num_features
                # )

                self.hidden_dim = max(
                    self.hidden_dim,
                    int(
                        (self.num_outputs + self.num_features)
                        // max(self.num_layers, 1)
                    ),
                )
                # TODO: debugging switch back
                # self.hidden_dim = self.hidden_dim
            else:
                # In non-causal mode, features are the causes
                self.num_causes = self.num_features

            # Define the input sampler
            if self.used_sampler == "ts":
                self.xsampler = TSSampler(
                    self.seq_len,
                    self.num_causes,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    mixed_names=self.sampling_mixed_names,
                    sigmoid_rate=self.sigmoid_mixed_sampling_rate,
                    effective_seq_len=self.seq_len,
                )
            elif self.used_sampler == "real":
                self.xsampler = RealTSSampler(
                    self.seq_len,
                    self.num_causes,
                    real_data_loader=self.real_data_loader,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    effective_seq_len=self.seq_len,
                )
            elif self.used_sampler == "tabular":
                self.xsampler = XSampler(
                    self.seq_len,
                    self.num_causes,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    effective_seq_len=self.seq_len,
                )
            elif self.used_sampler == "mixed":
                self.xsampler = MixedSampler(
                    self.seq_len,
                    self.num_causes,
                    real_data_loader=self.real_data_loader,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    effective_seq_len=self.seq_len,
                    sampling_forms=["synthetic", "real", "stl"],
                    real_ratio=0.2,
                    stl_ratio=0.2,
                    synthetic_ratio=0.6,
                    use_random_ratios=True,
                    ratio_variation=0.2,
                )
            elif self.used_sampler == "stl":
                self.xsampler = STLSampler(
                    self.seq_len,
                    self.num_causes,
                    real_data_loader=self.real_data_loader,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    effective_seq_len=self.seq_len,
                    sampling_forms=["synthetic", "real", "stl"],
                )
            else:
                raise ValueError(
                    f"Invalid sampler: {self.used_sampler}, options are 'ts', 'real', 'tabular'"
                )

            # Build layers
            layers = list()
            assert num_layers >= 0, "Number of layers must be at least 0."
            if num_layers == 0:
                # TODO: this can crash things if input and output values are not compatible
                layers.append(
                    self.generate_layer_modules(
                        in_dim=self.num_causes, is_output_layer=True
                    )
                )
                self.layers = nn.Sequential(*layers).to(device)
                self.weight_matrices = [layer[1].weight for layer in layers]
            elif num_layers == 1:
                layers.append(
                    self.generate_layer_modules(in_dim=self.num_causes)
                )

                layers.append(self.generate_layer_modules(is_output_layer=True))
                self.layers = nn.Sequential(*layers).to(device)
                self.weight_matrices = [layer[1].weight for layer in layers]
            else:
                # layers.append(nn.Linear(self.num_causes, self.hidden_dim))
                layers.append(
                    self.generate_layer_modules(in_dim=self.num_causes)
                )
                for _ in range(self.num_layers - 1):
                    layers.append(self.generate_layer_modules())
                if not self.is_causal:
                    layers.append(
                        self.generate_layer_modules(is_output_layer=True)
                    )
                self.layers = nn.Sequential(*layers).to(device)
                self.weight_matrices = (
                    [layers[0][1].weight]
                    if isinstance(layers[0], nn.Sequential)
                    else [layers[0].weight]
                ) + [layer[1].weight for layer in layers[1:]]

            # Calculate cumulative sum of nodes for each layer
            self.node_cumsum = self.calculate_node_cumsum()

            # Initialize layers
            self.initialize_parameters()

            # Initialize LayerOperator for counterfactual perturbations
            self.layer_operators = list()
            self.functional_lags = (
                list()
            )  # initialize functional lags for each layer and append
            # TODO: add assertion preventing functional lags
            if self.use_layer_operator:
                # initialize layer operator for each layer, including the initial layer
                # TODO: we could consider applying "lag" to the first layer
                # TODO: implement proper ordering so that downstream layers cannot also have lag
                # self.layer_operators.append(
                #     self._init_layer_operator(self.num_causes)
                # )
                for layer in self.layers:
                    # TODO: does not account for parent lag at the moment
                    size = (
                        layer[1].weight.shape[0]
                        if isinstance(layer, nn.Sequential)
                        else layer.weight.shape[0]
                    )
                    self.layer_operators.append(self._init_layer_operator(size))

    def _init_layer_operator(self, size: int, no_smoothing: bool = False):
        layer_has_functional_lag = np.random.binomial(
            1, self.layer_has_functional_lag_rate
        )
        layer_lags = np.round(
            np.random.uniform(
                self.functional_lag_min, self.functional_lag_max, size=size
            )
            * self.seq_len
            * np.random.binomial(1, self.functional_lag_rate, size=size)
            * layer_has_functional_lag
        ).astype(int)
        self.functional_lags.append(layer_lags)

        filter_config = {
            "kernel_size_min": self.kernel_size_min if not no_smoothing else 1,
            "kernel_size_max": self.kernel_size_max if not no_smoothing else 1,
            "sigma": self.kernel_sigma,
            "kernel_direction": self.kernel_direction,
            "kernel_type": self.kernel_type,
            "num_features": size,
            "num_timesteps": self.seq_len,
        }
        lag_config = {
            "functional_lags": layer_lags,
            "functional_lag_variance": self.functional_lag_variance,
            "random_lags": self.random_lags,
        }
        perturbation_config = {
            "enabled": self.counterfactual_enabled,
            "counterfactual_type": self.counterfactual_type,
            "noise_std": self.counterfactual_variance,
            "apply_probability": self.counterfactual_probability,
        }
        normalization_config = {
            "enabled": self.normalization_enabled,
            "minimum_magnitude": self.normalization_minimum_magnitude,
            "maximum_magnitude": self.normalization_maximum_magnitude,
            "normalization_types": self.normalization_types,
            "apply_probability": self.normalization_apply_probability,
        }
        return LayerOperator(
            seq_len=self.seq_len,
            hidden_dim=size,  # Use hidden_dim as it represents the feature dimension in layers
            device=self.device,
            # TODO: add parameters
            filter_config=filter_config,
            lag_config=lag_config,
            perturbation_config=perturbation_config,
            normalization_config=normalization_config,
        )

    def init_from_yaml(self, yaml_path: str):
        """Initializes the model from a YAML file.

        the format of the yaml file:
        layer i:
            num_nodes: number of nodes in the layer
            next_nodes: number of nodes in the next layer
            activation: activation_function
            edges: [j, k, ...] (indexed according to the index of the node in the next layer)
            weights: [w1, w2, ...] (indexed according to the index of the node in the next layer)
            TODO: initialization values, etc.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        layers = list(config.keys())
        layers.sort()

        network_layers = list()
        for i, layer in enumerate(layers):
            if i == 0:
                in_dim = config[layer]["num_nodes"]
            if self.used_sampler == "ts":
                self.xsampler = TSSampler(
                    self.seq_len,
                    self.num_causes,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                    mixed_names=self.sampling_mixed_names,
                    sigmoid_rate=self.sigmoid_mixed_sampling_rate,
                )
            elif self.used_sampler == "real":
                self.xsampler = RealTSSampler(
                    self.seq_len,
                    self.num_causes,
                    real_data_loader=self.real_data_loader,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                )
            elif self.used_sampler == "tabular":
                self.xsampler = XSampler(
                    self.seq_len,
                    self.num_causes,
                    pre_stats=self.pre_sample_cause_stats,
                    sampling=self.sampling,
                    device=self.device,
                )
            else:
                raise ValueError(
                    f"Invalid sampler: {self.used_sampler}, options are 'ts', 'real', 'tabular'"
                )

            layer_config = config[layer]
            in_dim = layer_config["num_nodes"]
            out_dim = layer_config["next_nodes"]
            activation = layer_config["activation"]
            self.hidden_dim = out_dim
            layer_sequential = self.generate_layer_modules(
                in_dim, is_output_layer=False
            )
            layer_sequential[0] = get_activations_by_name(activation)()

            network_layers.append(layer_sequential)
        self.layers = nn.Sequential(*network_layers).to(self.device)
        self.weight_matrices = [layer[1].weight for layer in network_layers]
        self.initialize_parameters()

        for k, (layer, layer_key) in enumerate(zip(network_layers, layers)):
            layer_config = config[layer_key]
            edge_set = set([tuple(edge) for edge in layer_config["edges"]])
            weights = layer_config["weights"]
            in_dim = layer_config["num_nodes"]
            out_dim = layer_config["next_nodes"]
            # zero out all non-edges
            for i in range(out_dim):
                for j in range(in_dim):
                    if (j, i) not in edge_set:
                        # set the value of this weight to 0 to prevent an edge
                        layer[1].weight.data[i, j] = 0

            if len(weights) > 0:
                layer[1].weight = nn.Parameter(
                    torch.tensor(weights, device=self.device)
                )

    def generate_layer_modules(self, in_dim=0, is_output_layer=False):
        """Generates a layer module with activation, linear transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim
        activation = self.mlp_activations()
        if in_dim == 0:
            in_dim = self.hidden_dim
        linear_layer = nn.Linear(in_dim, out_dim)

        # normalize values and multiply by a constant
        normalization_layer = nn.BatchNorm1d(out_dim).to(self.device)

        # Calculate noise_std regardless of the noise type
        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std

        if self.noise_type == "ts":
            noise_layer = TSNoise(
                seq_len=self.seq_len,
                sampling=self.ts_noise_sampling,
                device=self.device,
                std=self.noise_std,  # Pass noise_std directly to TSNoise
            )
        elif self.noise_type == "mixed":
            noise_layer = MixedTSNoise(
                seq_len=self.seq_len,
                sampling=self.ts_noise_sampling,
                device=self.device,
                ts_std=self.noise_std * self.mixed_noise_ratio,
                gaussian_std=self.noise_std * (1 - self.mixed_noise_ratio),
            )
        else:
            noise_layer = GaussianNoise(noise_std)
            # noise_layer = GaussianNoise(0.0)

        return nn.Sequential(
            activation, linear_layer, normalization_layer, noise_layer
        )

    def initialize_parameters(self):
        """Initializes parameters using block-wise dropout or normal initialization."""
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if self.block_wise_dropout and param.dim() == 2:
                self.initialize_with_block_dropout(param, i)
            else:
                self.initialize_normally(param, i)

    def initialize_with_block_dropout(self, param, index):
        """Initializes parameters using block-wise dropout."""
        nn.init.zeros_(param)
        n_blocks = random.randint(1, math.ceil(math.sqrt(min(param.shape))))
        block_size = [dim // n_blocks for dim in param.shape]
        self.block_size = block_size
        keep_prob = (n_blocks * block_size[0] * block_size[1]) / param.numel()
        for block in range(n_blocks):
            block_slice = tuple(
                slice(dim * block, dim * (block + 1)) for dim in block_size
            )
            nn.init.normal_(
                param[block_slice],
                std=self.init_std
                / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1),
            )

    def initialize_with_stl_dropout(self, param, index):
        """Initializes parameters using stl-block-wise dropout.
        This should only be applied to the first layer's weights.
        afterwards, blocks can be applied normally
        creates recombinations of the STL features, each group is a different combination of S, T and L
        """
        nn.init.zeros_(param)
        n_groups = random.randint(1, math.ceil(math.sqrt(param.shape[0])))
        group_size = [dim // n_groups for dim in param.shape]
        self.group_size = group_size
        keep_prob = (n_groups * group_size[0] * group_size[1]) / param.numel()
        size_thus_far = 0
        for group in range(n_groups):
            # sample random trios of STL features
            indices = torch.randint(
                0,
                int(np.floor(param.shape[1] / 3)),
                size=(3,),
                device=param.device,
            )
            indices[1] += int(torch.floor(param.shape[1] / 3))
            indices[2] += int(
                torch.floor(param.shape[1] / 3)
            ) * 2 + random.randint(
                0, param.shape[1] - (torch.floor(param.shape[1] / 3) * 2)
            )

            target_indices = torch.from_numpy(
                np.arange(group_size[group]) + size_thus_far
            ).to(param.device)
            size_thus_far += group_size[group]
            nn.init.normal_(
                param[target_indices, indices],
                std=self.init_std
                / (keep_prob**0.5 if self.scale_init_std_by_dropout else 1),
            )

    def initialize_normally(self, param, index):
        """Initializes parameters using normal distribution."""
        if param.dim() == 2:  # Applies only to weights, not biases
            dropout_prob = (
                self.mlp_dropout_prob if index > 0 else 0
            )  # No dropout for the first layer's weights
            dropout_prob = min(dropout_prob, 0.99)
            std = self.init_std / (
                (1 - dropout_prob) ** 0.5
                if self.scale_init_std_by_dropout
                else 1
            )
            nn.init.normal_(param, std=std)
            param.data *= torch.bernoulli(
                torch.full_like(param, 1 - dropout_prob)
            )

    def forward(
        self,
        return_indices=False,
        exclude_inputs="allow",
        use_input_as_target=False,
        indices_X_y=None,
        counterfactual_indices=None,
        assigned_inputs=None,
    ):
        """Generates synthetic data by sampling input features and applying MLP transformations."""
        if self.use_layer_operator:
            # need to pad by convolutional filter size
            self.xsampler.seq_len = self.seq_len + sum(
                self.layer_operators[i].used_kernel_size
                for i in range(len(self.layer_operators))
            )
        if assigned_inputs is None:
            causes = self.xsampler.sample()  # (seq_len, num_causes)
        else:
            causes = assigned_inputs

        if not isinstance(causes, torch.Tensor):
            causes = torch.stack(causes, dim=1)  # type: ignore
            assert causes.shape[0] == self.seq_len, (
                f"causes.shape[0]: {causes.shape[0]}, self.seq_len: {self.seq_len}"
            )

        if indices_X_y is not None and counterfactual_indices is not None:
            assert np.all([c in indices_X_y for c in counterfactual_indices]), (
                "Counterfactual indices be contained in indices_X_y"
            )

        # Generate outputs through MLP layers
        outputs: List[Any] = [causes]
        # minl =f"max in input: {outputs[-1].abs().max()} "
        start_idx_at = len(causes)
        for i, layer in enumerate(self.layers):
            layer_output = layer(outputs[-1])
            if self.use_layer_operator:
                layer_output = self.layer_operators[i].apply_all_operations(
                    layer_output,
                    counterfactual_indices=counterfactual_indices,
                    start_idx_at=start_idx_at,
                    noise_std=self.noise_std,
                )
            # TODO: counterfactual perturbation deprecated, used in layer operator instead
            elif counterfactual_indices is not None:
                # apply counterfactual perturbation to the layer output
                layer_output = apply_counterfactual_perturbation(
                    self.counterfactual_perturbation_type,
                    layer_output,
                    counterfactual_indices,
                    start_idx_at,
                    self.noise_std,
                )
            outputs.append(layer_output)
            # minl += f"max in layer {i}: {outputs[-1].abs().max()}, "
        if self.use_layer_operator:
            # padding to accomodate for lags should be shifted back now
            outputs = [o[-self.seq_len :] for o in outputs]
        # logger.info(f"causes: {causes.min()}, {causes.max()}")
        # flat_outputs = torch.cat(outputs, dim=-1)
        # logger.info(f"flat_outputs: {flat_outputs.min()}, {flat_outputs.max()}")
        # # logger.info(f"{flat_outputs[flat_outputs.abs() > 50]} {(flat_outputs.abs() > 50).nonzero()}")

        # # count the number of flat values greater than 10
        # flat_outputs_abs = (flat_outputs.abs() > 15).sum() / flat_outputs.numel()
        # logger.info(f"flat_outputs_abs: {flat_outputs_abs}")
        #     error
        # No longer remove the first two layers, exclude_inputs can be "skip_first"
        # if len(outputs) == 3 and exclude_inputs == "exclude":
        #     outputs = outputs[1:]
        # elif len(outputs) > 3 and exclude_inputs == "exclude":
        #     outputs = outputs[
        #         2:
        #     ]  # Start from 2 because the first layer is only linear without activation

        # flat_outputs = torch.cat(outputs, dim=-1)

        # Handle outputs based on causality
        if self.return_single_output and len(outputs) == 2:
            # Unit test case
            X = outputs[0]
            assert isinstance(X, torch.Tensor)
            X = X[
                :, : self.max_features
            ]  # use the inputs, but they will get pushed out
            y = outputs[-1]
            assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
            # save a plot of y or X:
            # import numpy as np
            # import matplotlib.pyplot as plt
            # plt.plot(X[:,0].cpu().numpy())
            # plt.savefig("temp_figs/X_shape" + str(np.random.randint(100)) + ".png")
            # plt.close()
            indices_X = torch.arange(len(X))
            indices_y = torch.tensor([-1])
        else:
            X, y, indices_X, indices_y = self.handle_outputs(
                causes,
                outputs,
                exclude_inputs=exclude_inputs,
                indices_X_y=indices_X_y,
            )

        # Check for NaNs and handle them by setting to default values
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X[:] = 0.0
            y[:] = -100.0

        if use_input_as_target:
            assert exclude_inputs == "ensure", (
                "use_input_as_target requires exclude_inputs to be 'ensure'"
            )
            # extract the first num_causes features as target
            cause_idx_to_use = torch.randperm(causes.shape[1])[
                : self.num_outputs
            ]
            cause_target = causes[:, cause_idx_to_use]
            # replace the cause_target with the first num_outputs features of y in X
            X[:, cause_idx_to_use] = y
            y = cause_target

        if self.num_outputs == 1:
            y = y.squeeze(-1)

        if return_indices:
            return X, y, indices_X, indices_y
        else:
            return X, y

    def handle_outputs(
        self, causes, outputs, indices_X_y=None, exclude_inputs="allow"
    ):
        """
        Handles outputs based on whether causal or not.

        If causal, sample inputs and target from the graph.
        If not causal, directly use causes as inputs and last output as target.

        Parameters
        ----------
        causes : torch.Tensor
            Causes of shape (seq_len, num_causes)

        outputs : list of torch.Tensor
            List of output tensors from MLP layers

        exclude_inputs : str, default="allow"
            If "exclude", exclude the inputs from the output features (only applicable for causal models)
            If "ensure", forcibly replace the first num_input features with the inputs
            If "allow", allow the inputs to be included in the output features

        Returns
        -------
        X : torch.Tensor
            Input features (seq_len, num_features)

        y : torch.Tensor
            Target (seq_len, num_outputs)
        """
        if self.in_clique:
            # In clique mode, if first layer is chosen all features and target will be mutually disconnected
            exclude_inputs = "exclude"

        if indices_X_y is not None:
            indices_X, indices_y = indices_X_y
            outputs_flat = torch.cat(outputs, dim=-1)
            # exclude inputs by skipping the first k
            # Select input features and targets directly from outputs
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            if self.is_causal:
                outputs_flat = torch.cat(outputs, dim=-1)
                if exclude_inputs == "exclude" or exclude_inputs == "ensure":
                    skip_idx = causes.shape[1]
                elif exclude_inputs == "skip_first":
                    skip_idx = causes.shape[1] + outputs[1].shape[1]
                else:
                    skip_idx = 0
                # if (exclude_inputs == "exclude" or exclude_inputs == "ensure") and len(self.layers) <= 1:
                #     skip_values = causes.shape[-1]
                # else:
                #     skip_values = 0

                if self.in_clique:
                    # When in_clique=True, features and targets are sampled as a block, ensuring that
                    # selected variables may share dense dependencies.
                    start = random.randint(
                        0,
                        outputs_flat.shape[-1]
                        - self.num_outputs
                        - self.num_features
                        - skip_idx,
                    )
                    random_perm = (
                        start
                        + torch.randperm(
                            self.num_outputs + self.num_features,
                            device=self.device,
                        )
                        + skip_idx
                    )
                else:
                    # Otherwise, features and targets are randomly and independently selected
                    random_perm = (
                        torch.randperm(
                            outputs_flat.shape[-1] - 1 - skip_idx,
                            device=self.device,
                        )
                        + skip_idx
                    )

                indices_X = random_perm[
                    self.num_outputs : self.num_outputs + self.num_features
                ]
                if self.y_is_effect:
                    # If targets are effects, take last output dims
                    indices_y = torch.tensor(list(range(-self.num_outputs, 0)))
                else:
                    # Otherwise, take from the beginning of the permuted list
                    indices_y = random_perm[: self.num_outputs]

                if self.sort_features:
                    indices_X, _ = torch.sort(indices_X)

                if exclude_inputs == "ensure":
                    # forcibly replace the first num_input features with the inputs
                    num_req_indices = indices_X.shape[0]
                    input_indices = torch.arange(causes.shape[-1])
                    indices_X = torch.cat([input_indices, indices_X])
                    # sort indices_X
                    # indices_X, _ = torch.sort(indices_X)
                    # remove repeats
                    indices_X = torch.unique(indices_X)
                    indices_X = indices_X[:num_req_indices]

                # indices_X, _ = torch.sort(indices_X) # TODO: debugging line

                X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]

            else:
                # In non-causal mode, use original causes and last layer output
                X = causes
                y = outputs[-1]
                indices_X = torch.arange(self.num_features)
                indices_y = torch.tensor(list(range(-self.num_outputs, 0)))

        # Check if targets are reachable and regenerate if needed (only for num_outputs=1)
        # TODO: Handle cases where num_outputs > 1
        if self.num_outputs == 1 and self.try_regenerate_unreachable_target:
            target_node_index = int(indices_y[0].item())
            is_reachable, layer_idx, local_idx = self.is_target_is_reachable(
                target_node_index
            )
            if not is_reachable:
                # logger.info(f"Target at node index {target_node_index} is not reachable, regenerating...")
                y += self.regenerate_unreachable_target(
                    target_node_index, outputs, layer_idx, local_idx
                )

        return X, y, indices_X, indices_y

    def calculate_node_cumsum(self):
        """Calculate cumulative sum of nodes for each layer in the MLP.

        Returns
        -------
        list
            List containing cumulative sum of nodes up to each layer.
            For example, if there are 3 causes and 2 layers each with 25 nodes,
            returns [3, 28, 53].
        """
        node_counts = []

        # Start with input layer (causes)
        node_counts.append(self.num_causes)

        if self.num_layers == 0:
            # Only one layer: causes -> outputs
            node_counts.append(self.num_causes + self.num_outputs)
        elif self.num_layers == 1:
            # Two layers: causes -> hidden_dim -> outputs
            node_counts.append(self.num_causes + self.hidden_dim)
            node_counts.append(
                self.num_causes + self.hidden_dim + self.num_outputs
            )
        else:
            # Multiple layers: causes -> hidden_dim -> ... -> hidden_dim -> (optional) outputs
            # First layer: causes -> hidden_dim
            node_counts.append(self.num_causes + self.hidden_dim)

            # Middle layers: hidden_dim -> hidden_dim
            for _ in range(self.num_layers - 2):
                node_counts.append(node_counts[-1] + self.hidden_dim)

            # Last layer depends on whether it's causal or not
            if not self.is_causal:
                # Non-causal: add output layer
                node_counts.append(node_counts[-1] + self.num_outputs)
            else:
                # Causal: last layer is hidden_dim -> hidden_dim
                node_counts.append(node_counts[-1] + self.hidden_dim)

        return node_counts

    def get_local_index_for_node(self, node_index: int) -> tuple[int, int]:
        """Get the layer index and local node index within that layer for a given node.

        Parameters
        ----------
        node_index : int
            The node index (0-indexed, reading from first node in first layer,
            then continuing layer by layer).

        Returns
        -------
        tuple[int, int]
            A tuple of (layer_index, local_node_index) where:
            - layer_index: The layer index (0-indexed) that the node belongs to.
            - local_node_index: The index of the node within that layer (0-indexed).
            Returns (-1, -1) if the node index is out of bounds.

        Examples
        --------
        For a model with 3 causes and 2 layers with 25 nodes each:
        - node_cumsum = [3, 28, 53]
        - get_local_index_for_node(0) -> (0, 0) (first node in first layer)
        - get_local_index_for_node(2) -> (0, 2) (last node in first layer)
        - get_local_index_for_node(3) -> (1, 0) (first node in second layer)
        - get_local_index_for_node(27) -> (1, 24) (last node in second layer)
        - get_local_index_for_node(28) -> (2, 0) (first node in third layer)
        """
        # Wrap around if node_index is negative
        if node_index < 0:
            node_index = self.node_cumsum[-1] + node_index

        # If node_index is still negative or larger than the total number of nodes, invalid node index
        if node_index < 0 or node_index >= self.node_cumsum[-1]:
            return (-1, -1)

        # Find the layer index using bisect_right
        layer_idx = bisect.bisect_right(self.node_cumsum, node_index)

        # If bisect_right returns 0, the node is in the first layer
        # If bisect_right returns len(node_cumsum), the node is out of bounds
        if layer_idx == 0:
            # Node is in the first layer
            local_node_idx = node_index

        elif layer_idx > len(self.node_cumsum):
            # Node is out of bounds
            return (-1, -1)

        else:
            # bisect_right gives us the layer index directly
            # For layer > 0, subtract the cumulative count of previous layers
            local_node_idx = node_index - self.node_cumsum[layer_idx - 1]

        return (layer_idx, local_node_idx)

    def is_target_is_reachable(self, node_index: int) -> tuple[bool, int, int]:
        """Check if the target is reachable from the previous layer.

        Parameters
        ----------
        node_index : int
            The node index (0-indexed, reading from first node in first layer,
            then continuing layer by layer).

        Returns
        -------
        tuple[bool, int, int]
            A tuple of (is_reachable, layer_idx, local_node_idx) where:
            - is_reachable: True if the target is reachable from the previous layer
            - layer_idx: The layer index (0-indexed) that the node belongs to
            - local_node_idx: The index of the node within that layer (0-indexed)
        """
        layer_idx, local_idx = self.get_local_index_for_node(node_index)
        if layer_idx == -1:
            raise ValueError(f"Node index {node_index} is out of bounds")
        if layer_idx == 0:
            return True, layer_idx, local_idx

        # Check if less than 10% of the inputs have values > 1e-6
        weights = abs(self.layers[layer_idx - 1][1].weight[local_idx, :])
        significant_connections_ratio = torch.mean(
            (weights > 1e-6).float()
        ).item()
        is_reachable = significant_connections_ratio >= 0.1

        return is_reachable, layer_idx, local_idx

    def regenerate_unreachable_target(
        self, node_index: int, outputs: list, layer_idx: int, local_idx: int
    ) -> torch.Tensor:
        """Regenerate an unreachable target by randomly summing 30% of nodes from the previous layer.

        Parameters
        ----------
        node_index : int
            The node index of the unreachable target
        outputs : list
            List of output tensors from MLP layers
        layer_idx : int
            The layer index of the unreachable target
        local_idx : int
            The local node index of the unreachable target

        Returns
        -------
        torch.Tensor
            The regenerated target values
        """
        if layer_idx == 0:
            raise ValueError("Cannot regenerate target from input layer")

        # Get the previous layer's outputs
        prev_layer_outputs = outputs[
            layer_idx - 1
        ]  # Shape: (seq_len, hidden_dim)

        # Calculate dropout probability (70% dropout = 30% keep)
        keep_prob = 0.3

        # Generate weights for all connections from previous layer
        num_prev_nodes = prev_layer_outputs.shape[1]
        weights = torch.randn(num_prev_nodes, device=self.device)
        weights = weights / torch.norm(weights)  # Normalize weights

        # Create Bernoulli mask to randomly set some weights to zero
        mask = torch.bernoulli(torch.full_like(weights, keep_prob))
        weights = weights * mask

        # logger.info(f"Weights: {weights}") # TODO: comment out

        # Fill the weights into the original weight matrix
        with torch.no_grad():
            if isinstance(self.layers[layer_idx - 1], nn.Sequential):
                self.layers[layer_idx - 1][1].weight[local_idx, :] = weights
            else:
                self.layers[layer_idx - 1].weight[local_idx, :] = weights

        # Weighted sum of all nodes (masked weights will be zero)
        regenerated_target = torch.matmul(
            prev_layer_outputs, weights
        )  # Shape: (seq_len,)

        return regenerated_target.unsqueeze(
            -1
        )  # Add output dimension to match expected shape
