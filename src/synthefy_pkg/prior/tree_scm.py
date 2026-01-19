from __future__ import annotations

import random
from typing import Any, List, Optional

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler
from synthefy_pkg.prior.input_sampling.mixed_sampler import MixedSampler
from synthefy_pkg.prior.input_sampling.real_ts_sampling import RealTSSampler
from synthefy_pkg.prior.input_sampling.stl_sampling import STLSampler
from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler
from synthefy_pkg.prior.ts_noise import MixedTSNoise, TSNoise
from synthefy_pkg.prior.utils import (
    GaussianNoise,
    apply_counterfactual_perturbation,
)


class TreeLayer(nn.Module):
    """A layer that transforms input features using a tree-based model.

    This layer fits a specified tree-based regression model (Decision Tree,
    Extra Trees, Random Forest, or XGBoost) to the input features using
    randomly generated target values. It then uses the trained model
    to predict the outputs.

    Parameters
    ----------
    tree_model : str
        The type of tree-based model to use. Options are "decision_tree",
        "extra_trees", "random_forest", "xgboost".

    max_depth : int
        The maximum depth allowed for the individual trees in the model.

    n_estimators : int
        The number of trees in the ensemble.

    out_dim : int
        The desired output dimension for the transformed features. This determines
        the number of target variables (`y_fake`) generated for fitting the
        multi-output regressor.

    device : str or torch.device
        The device ('cpu' or 'cuda') on which to place the output tensor.
    """

    def __init__(
        self,
        tree_model: str,
        max_depth: int,
        n_estimators: int,
        out_dim: int,
        device: str,
    ):
        super(TreeLayer, self).__init__()
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.out_dim = out_dim
        self.device = device

        if tree_model == "decision_tree":
            self.model = MultiOutputRegressor(
                DecisionTreeRegressor(max_depth=max_depth, splitter="random"),
                n_jobs=-1,
            )
        elif tree_model == "extra_trees":
            self.model = MultiOutputRegressor(
                ExtraTreesRegressor(
                    n_estimators=n_estimators, max_depth=max_depth
                ),
                n_jobs=-1,
            )
        elif tree_model == "random_forest":
            self.model = MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth
                ),
                n_jobs=-1,
            )
        elif tree_model == "xgboost":
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",
                multi_strategy="multi_output_tree",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")

    def forward(self, X):
        """Applies the fitted tree-based transformation to the input features.

        Fits the internal tree model using X and random targets, then predicts
        and returns the outputs.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Transformed features tensor of shape (n_samples, out_dim).
        """
        X = X.nan_to_num(0.0).cpu()
        y_fake = np.random.randn(X.shape[0], self.out_dim)
        self.model.fit(X, y_fake)
        y = self.model.predict(X)
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        if self.out_dim == 1:
            y = y.view(-1, 1)

        return y


class TreeSCM(nn.Module):
    """A Tree-based Structural Causal Model for generating synthetic datasets.
    Similar to MLP-based SCM but uses tree-based models (like Random Forests or XGBoost)
    for potentially non-linear feature transformations instead of linear layers.

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
          intermediate outputs of the tree transformations applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the tree layers becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final tree layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate tree outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate tree outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=5
        Number of tree transformation layers.

    hidden_dim : int, default=10
        Output dimension size for intermediate tree transformations.

    tree_model : str, default="xgboost"
        Type of tree model to use ("decision_tree", "extra_trees", "random_forest", "xgboost").
        XGBoost is favored for performance as it supports multi-output regression natively.

    max_depth_lambda : float, default=0.5
        Lambda parameter for sampling the max_depth for tree models from an exponential distribution.

    n_estimators_lambda : float, default=0.5
        Lambda parameter for sampling the number of estimators (trees) per layer from an exponential distribution.

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
        The base standard deviation for the Gaussian noise added after each tree
        layer's transformation.

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.
        If `True`, the noise standard deviation for each output dimension of a layer
        is sampled from a normal distribution centered at 0 with `noise_std`.
        If `False`, a fixed `noise_std` is used for all dimensions.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    sampling_mixed_names : list[str], default=[]
        List of names for mixed sampling.

    sigmoid_mixed_sampling_rate : int, default=-1
        Rate for sigmoid mixed sampling.

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
        num_layers: int = 5,
        hidden_dim: int = 10,
        tree_model: str = "xgboost",
        max_depth_lambda: float = 0.5,
        n_estimators_lambda: float = 0.5,
        sampling: str = "normal",
        used_sampler: str = "ts",
        ts_noise_sampling: str = "normal",
        pre_sample_cause_stats: bool = False,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        noise_type: str = "gaussian",
        mixed_noise_ratio: float = 0.5,
        device: str = "cpu",
        real_data_loader: Optional[DataLoader] = None,
        sampling_mixed_names: list[str] = [],
        sigmoid_mixed_sampling_rate: int = -1,
        **kwargs,
    ):
        super(TreeSCM, self).__init__()
        # Tree models can be slow so we use less layers, smaller hidden dim, and non-causal mode
        is_causal = False
        num_layers = np.random.randint(1, 3)
        hidden_dim = np.random.randint(3, 10)

        # Data Generation Settings
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.is_causal = is_causal
        self.num_causes = num_causes
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tree_model = tree_model
        self.tree_depth_lambda = max_depth_lambda
        self.tree_n_estimators_lambda = n_estimators_lambda
        self.sampling = sampling
        self.ts_noise_sampling = ts_noise_sampling
        self.pre_sample_cause_stats = pre_sample_cause_stats
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device
        self.noise_type = noise_type
        self.mixed_noise_ratio = mixed_noise_ratio
        self.real_data_loader = real_data_loader
        self.used_sampler = used_sampler
        self.sampling_mixed_names = sampling_mixed_names
        self.sigmoid_mixed_sampling_rate = sigmoid_mixed_sampling_rate

        if self.is_causal:
            # Ensure enough intermediate variables for sampling X and y
            self.hidden_dim = max(
                self.hidden_dim, self.num_outputs + 2 * self.num_features
            )
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
        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(
            np.random.exponential(1 / self.tree_n_estimators_lambda)
        )
        layers = list()
        layers.append(
            TreeLayer(
                tree_model=self.tree_model,
                max_depth=min(max_depth, 4),
                n_estimators=min(n_estimators, 4),
                out_dim=self.hidden_dim,
                device=self.device,
            )
        )
        for _ in range(self.num_layers - 1):
            layers.append(self.generate_layer_modules())
        if not self.is_causal:
            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, tree-based transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim

        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(
            np.random.exponential(1 / self.tree_n_estimators_lambda)
        )
        tree_layer = TreeLayer(
            tree_model=self.tree_model,
            max_depth=min(max_depth, 4),
            n_estimators=min(n_estimators, 4),
            out_dim=out_dim,
            device=self.device,
        )

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std

        if self.noise_type == "gaussian":
            noise_layer = GaussianNoise(noise_std)

        elif self.noise_type == "ts":
            if isinstance(noise_std, torch.Tensor):
                try:
                    if len(noise_std.squeeze()) > 1:
                        noise_std = noise_std.squeeze()[0].item()
                    else:
                        noise_std = noise_std.squeeze().item()
                except TypeError:
                    noise_std = noise_std.item()
            noise_layer = TSNoise(
                seq_len=self.seq_len,
                sampling=self.ts_noise_sampling,
                device=self.device,
                std=self.noise_std,
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
            raise ValueError(f"Invalid noise type: {self.noise_type}")

        return nn.Sequential(tree_layer, noise_layer)

    def forward(
        self,
        return_indices=False,
        exclude_inputs="allow",
        use_input_as_target=False,
        indices_X_y=None,
        counterfactual_indices=None,
        assigned_inputs=None,
    ):
        """Generates synthetic data by sampling input features and applying tree-based transformations."""
        if assigned_inputs is None:
            causes = self.xsampler.sample()  # (seq_len, num_causes)
        else:
            causes = assigned_inputs

        if not isinstance(causes, torch.Tensor):
            causes = torch.stack(causes, dim=1)  # type: ignore
            assert causes.shape[0] == self.seq_len, (
                f"causes.shape[0]: {causes.shape[0]}, self.seq_len: {self.seq_len}"
            )

        # Generate outputs through MLP layers
        outputs: List[Any] = [causes]
        # minl =f"max in input: {outputs[-1].abs().max()} "
        start_idx_at = len(causes)
        for i, layer in enumerate(self.layers):
            layer_output = layer(outputs[-1])
            if counterfactual_indices is not None:
                # apply counterfactual perturbation to the layer output
                layer_output = apply_counterfactual_perturbation(
                    self.counterfactual_perturbation_type,
                    layer_output,
                    counterfactual_indices,
                    start_idx_at,
                    self.noise_std,
                )
            outputs.append(layer_output)
        # No longer remove the first two layers, exclude_inputs can be "skip_first"

        # Handle outputs based on causality
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
        self, causes, outputs, exclude_inputs="allow", indices_X_y=None
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

        Returns
        -------
        X : torch.Tensor
            Input features (seq_len, num_features)

        y : torch.Tensor
            Target (seq_len, num_outputs)
        """
        if indices_X_y is not None:
            indices_X, indices_y = indices_X_y
            outputs_flat = torch.cat(outputs, dim=-1)
            # exclude inputs by skipping the first k
            # Select input features and targets directly from outputs
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            if exclude_inputs == "exclude" or exclude_inputs == "ensure":
                skip_idx = causes.shape[1]
            elif exclude_inputs == "skip_first":
                skip_idx = causes.shape[1] + outputs[1].shape[1]
            else:
                skip_idx = 0
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
                        self.num_outputs + self.num_features, device=self.device
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
                indices_y = list(range(-self.num_outputs, 0))
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
                # indices_X, _ = torch.sort(indices_X)
                indices_X = torch.unique(indices_X)
                indices_X = indices_X[:num_req_indices]

            # Select input features and targets from outputs
            X = outputs_flat[:, indices_X]
            y = outputs_flat[:, indices_y]
        else:
            # In non-causal mode, use original causes and last layer output
            X = causes
            y = outputs[-1]
            indices_X = torch.arange(self.num_features)
            total_len = len(torch.cat(outputs, dim=-1)) + len(causes)
            indices_y = torch.arange(total_len - self.num_outputs, total_len)

        return X, y, indices_X, indices_y
