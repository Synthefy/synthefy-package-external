"""
This module provides functionality for sampling hyperparameters using various probability distributions
and meta-distributions. It's designed to support flexible hyperparameter optimization and configuration
sampling for machine learning models.

Key Components:
1. Basic Distribution Samplers - Functions that create sampling closures for basic distributions
2. HpSampler - A modular class that handles both basic and meta-distribution sampling
3. HpSamplerList - A container class that manages multiple HpSamplers for batch sampling

Meta-distributions are distributions over distribution parameters, allowing for hierarchical sampling
where the parameters of a distribution are themselves sampled from another distribution.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from loguru import logger


def trunc_norm_sampler(mu, sigma):
    """Creates a sampler for truncated normal distribution with given mean and std."""
    return lambda: stats.truncnorm(
        (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
    ).rvs(1)[0]


def beta_sampler(a, b):
    """Creates a sampler for beta distribution with shape parameters a and b."""
    return lambda: np.random.beta(a, b)


def gamma_sampler(a, b):
    """Creates a sampler for gamma distribution with shape parameter a and scale parameter b."""
    return lambda: np.random.gamma(a, b)


def uniform_sampler(a, b, round=False):
    """Creates a sampler for uniform distribution between a and b."""
    return (
        lambda: int(np.round(np.random.uniform(a, b)))
        if round
        else np.random.uniform(a, b)
    )


def uniform_int_sampler(a, b):
    """Creates a sampler for uniform integer distribution between a and b."""
    return lambda: round(np.random.uniform(a, b))


def pareto_sampler(scale, shape):
    """Creates a sampler for Pareto distribution that favors smaller values.
    Lower shape values create heavier tails and more weight on smaller values."""
    return lambda: stats.pareto.rvs(shape, scale=scale)


def log_normal_sampler(mu, sigma):
    """Creates a sampler for log-normal distribution with given log-mean and log-std."""
    return lambda: stats.lognorm.rvs(sigma, scale=math.exp(mu))


class HpSampler(nn.Module):
    """
    A modular hyperparameter sampler that supports both basic and meta-distributions.

    Meta-distributions include:
    - meta_beta: Beta distribution with sampled parameters
    - meta_gamma: Gamma distribution with sampled parameters
    - meta_trunc_norm: Truncated normal with sampled parameters
    - meta_trunc_norm_log_scaled: Log-scaled truncated normal
    - meta_heavy_tailed: Heavy-tailed distribution (Pareto) with sampled parameters
    - meta_heavy_tailed_log_scaled: Log-scaled heavy-tailed distribution (Pareto)
    - meta_choice: Categorical distribution with sampled probabilities
    - meta_choice_mixed: Mixed categorical with sampled probabilities

    Parameters:
        distribution (str): Name of the distribution to use
        device (str): Device to use for tensor operations
        **kwargs: Distribution-specific parameters such as:
            - min, max: bounds for uniform distributions
            - scale: scaling factor for beta distribution
            - lower_bound: minimum value for truncated distributions
            - choice_values: possible values for categorical distributions
    """

    def __init__(self, distribution, device, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.device = device
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.initialize_distribution()

    def initialize_distribution(self):
        if self.distribution.startswith("meta"):
            self.initialize_meta_distribution()
        elif self.distribution == "uniform":
            self.sampler = uniform_sampler(
                self.min, self.max, getattr(self, "round", False)
            )
        elif self.distribution == "beta":
            self.sampler = beta_sampler(self.a, self.b)
        elif self.distribution == "uniform_int":
            self.sampler = uniform_int_sampler(self.min, self.max)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def initialize_meta_distribution(self):
        if self.distribution == "meta_beta":
            self.sampler = self.setup_meta_beta_sampler()
        elif self.distribution == "meta_gamma":
            self.sampler = self.setup_meta_gamma_sampler()
        elif self.distribution == "meta_trunc_norm":
            self.sampler = self.setup_meta_trunc_norm_sampler()
        elif self.distribution == "meta_trunc_norm_log_scaled":
            self.sampler = self.setup_meta_trunc_norm_log_scaled_sampler()
        elif self.distribution == "meta_heavy_tailed":
            self.sampler = self.setup_meta_heavy_tailed_sampler()
        elif self.distribution == "meta_heavy_tailed_log_scaled":
            self.sampler = self.setup_meta_heavy_tailed_log_scaled_sampler()
        elif self.distribution == "meta_choice":
            self.sampler = self.setup_meta_choice_sampler()
        elif self.distribution == "meta_choice_mixed":
            self.sampler = self.setup_meta_choice_mixed_sampler()
        else:
            raise ValueError(
                f"Unsupported meta distribution: {self.distribution}"
            )

    def ensure_hyperparameter(self, attr_name, distribution, min, max):
        if not hasattr(self, attr_name):
            setattr(
                self,
                attr_name,
                HpSampler(
                    distribution=distribution,
                    device=self.device,
                    min=min,
                    max=max,
                ),
            )

    def setup_meta_beta_sampler(self):
        """Sets up a meta-beta distribution sampler.
        Returns a closure that samples beta distribution parameters and then samples from that beta."""
        # Dynamically define b and k if not explicitly provided
        self.ensure_hyperparameter("b", "uniform", self.min, self.max)
        self.ensure_hyperparameter("k", "uniform", self.min, self.max)

        def sampler():
            b = self.b() if callable(self.b) else self.b
            k = self.k() if callable(self.k) else self.k
            return lambda: self.scale * beta_sampler(b, k)()

        return sampler

    def setup_meta_gamma_sampler(self):
        """Sets up a meta-gamma distribution sampler.
        Returns a closure that samples gamma distribution parameters and then samples from that gamma."""
        # Dynamically define alpha and scale if not explicitly provided
        self.ensure_hyperparameter(
            "alpha", "uniform", 0.0, math.log(self.max_alpha)
        )
        self.ensure_hyperparameter("scale", "uniform", 0.0, self.max_scale)

        def sampler():
            alpha = self.alpha() if callable(self.alpha) else self.alpha
            scale = self.scale() if callable(self.scale) else self.scale

            def sub_sampler():
                assert isinstance(scale, float)
                assert isinstance(alpha, float)
                sample = gamma_sampler(
                    math.exp(alpha), scale / math.exp(alpha)
                )()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_sampler(self):
        """Sets up a meta truncated normal distribution sampler.
        Returns a closure that samples normal distribution parameters and then samples from that normal."""
        # Dynamically define mean and std if not explicitly provided
        self.min_std = self.min_std if hasattr(self, "min_std") else 0.01
        self.max_std = self.max_std if hasattr(self, "max_std") else 1.0
        self.ensure_hyperparameter(
            "mean", "uniform", self.min_mean, self.max_mean
        )
        self.ensure_hyperparameter("std", "uniform", self.min_std, self.max_std)

        def sampler():
            mean = self.mean() if callable(self.mean) else self.mean
            std = self.std() if callable(self.std) else self.std

            def sub_sampler():
                sample = trunc_norm_sampler(mean, std)()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_log_scaled_sampler(self):
        """Sets up a log-scaled meta truncated normal distribution sampler.
        Useful for parameters that vary on logarithmic scales."""
        # Dynamically define log_mean and log_std if not explicitly provided
        self.min_std = self.min_std if hasattr(self, "min_std") else 0.01
        self.max_std = self.max_std if hasattr(self, "max_std") else 1.0
        self.ensure_hyperparameter(
            "log_mean",
            "uniform",
            math.log(self.min_mean),
            math.log(self.max_mean),
        )
        self.ensure_hyperparameter(
            "log_std", "uniform", math.log(self.min_std), math.log(self.max_std)
        )

        def sampler():
            log_mean = (
                self.log_mean() if callable(self.log_mean) else self.log_mean
            )
            log_std = self.log_std() if callable(self.log_std) else self.log_std
            assert isinstance(log_mean, float)
            assert isinstance(log_std, float)
            mu = math.exp(log_mean)
            sigma = mu * math.exp(log_std)

            def sub_sampler():
                sample = trunc_norm_sampler(mu, sigma)()
                return (
                    self.lower_bound + round(sample)
                    if self.round
                    else self.lower_bound + sample
                )

            return sub_sampler

        return sampler

    def _plot_pareto_distribution(
        self, scale, shape, sampler_type="heavy_tailed"
    ):
        """Helper method to generate and save Pareto distribution plots."""
        import os

        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import pareto

        sns.set_style("whitegrid")

        # Get the min/max parameters for plotting range
        min_mean = getattr(self, "min_mean", 0.01)
        max_mean = getattr(self, "max_mean", 1.0)
        min_shape = getattr(self, "min_shape", 0.5)
        max_shape = getattr(self, "max_shape", 2.0)

        # Calculate reasonable x range that captures the shape
        # Start from scale (minimum value for Pareto)
        x_min = scale
        # End at a point where the PDF has decayed significantly
        # For Pareto, most of the mass is within the first few multiples of scale
        x_max = (
            scale + scale * 5
        )  # 5x the scale should capture most of the distribution

        # Ensure we don't go beyond reasonable bounds
        x_max = min(x_max, max_mean * 3)

        x = np.linspace(x_min, x_max, 1000)

        plt.figure(figsize=(12, 8))

        # Plot the current distribution
        y_current = pareto.pdf(x, shape, scale=scale)
        plt.plot(
            x,
            y_current,
            "b-",
            linewidth=3,
            label=f"Current: Pareto(shape={shape:.2f}, scale={scale:.2f})",
        )

        # Plot range of possible distributions
        colors = ["r", "g", "orange", "purple"]
        alphas = [0.3, 0.2, 0.15, 0.1]

        # Plot corners of the parameter space
        # corner_scales = [min_mean, max_mean]
        # corner_shapes = [min_shape, max_shape]

        for i, (s, sh) in enumerate(
            [
                (min_mean, min_shape),
                (min_mean, max_shape),
                (max_mean, min_shape),
                (max_mean, max_shape),
            ]
        ):
            if i < len(colors):
                y_corner = pareto.pdf(x, sh, scale=s)
                plt.plot(
                    x,
                    y_corner,
                    color=colors[i],
                    linestyle="--",
                    alpha=alphas[i],
                    linewidth=1,
                    label=f"Corner: Pareto(shape={sh:.2f}, scale={s:.2f})",
                )

        # Calculate and display statistics for current distribution
        mean_val = pareto.mean(shape, scale=scale) if shape > 1 else np.inf
        std_val = pareto.std(shape, scale=scale) if shape > 2 else np.inf

        plt.title(
            f"Pareto Distribution Range ({sampler_type})\n"
            f"Current: Shape={shape:.2f}, Scale={scale:.2f}\n"
            f"Range: Scale=[{min_mean:.2f}, {max_mean:.2f}], Shape=[{min_shape:.2f}, {max_shape:.2f}]",
            fontsize=14,
        )
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)

        # Set reasonable y limits based on the current distribution
        y_max = np.max(y_current) * 1.2  # 20% above the peak
        plt.ylim(0, y_max)

        # Add text box with current statistics and parameter ranges
        stats_text = (
            f"Current Distribution:\nShape: {shape:.2f}\nScale: {scale:.2f}"
        )
        if shape > 1:
            stats_text += f"\nMean: {mean_val:.2f}"
        if shape > 2:
            stats_text += f"\nStd: {std_val:.2f}"

        stats_text += f"\n\nParameter Ranges:\nScale: [{min_mean:.2f}, {max_mean:.2f}]\nShape: [{min_shape:.2f}, {max_shape:.2f}]"

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=10,
            verticalalignment="top",
        )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(f"figures/{sampler_type}", exist_ok=True)
        plt.savefig(
            f"figures/{sampler_type}/pareto_distribution_range_shape_{shape:.2f}_scale_{scale:.2f}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def setup_meta_heavy_tailed_sampler(self):
        """Sets up a meta heavy-tailed distribution sampler.
        Returns a closure that samples parameters for a heavy-tailed (pareto) distribution."""
        # Dynamically define scale and shape if not explicitly provided
        self.min_mean = self.min_mean if hasattr(self, "min_mean") else 0.01
        self.max_mean = self.max_mean if hasattr(self, "max_mean") else 1.0
        self.min_shape = self.min_shape if hasattr(self, "min_shape") else 0.5
        self.max_shape = self.max_shape if hasattr(self, "max_shape") else 2.0
        self.lower_bound = (
            self.lower_bound if hasattr(self, "lower_bound") else 0.0
        )
        self.round = self.round if hasattr(self, "round") else False

        self.ensure_hyperparameter(
            "scale", "uniform", self.min_mean, self.max_mean
        )
        self.ensure_hyperparameter(
            "shape", "uniform", self.min_shape, self.max_shape
        )

        def sampler():
            scale = self.scale() if callable(self.scale) else self.scale
            shape = self.shape() if callable(self.shape) else self.shape
            assert isinstance(scale, float)
            assert isinstance(shape, float)

            def sub_sampler():
                sample = pareto_sampler(scale, shape)()
                return (
                    int(
                        self.lower_bound
                        + np.clip(round(sample), 0, self.max_mean)
                    )
                    if self.round
                    else self.lower_bound + np.clip(sample, 0, self.max_mean)
                )

            # self._plot_pareto_distribution(scale, shape, "heavy_tailed")

            return sub_sampler

        return sampler

    def setup_meta_heavy_tailed_log_scaled_sampler(self):
        """Sets up a log-scaled meta heavy-tailed distribution sampler.
        Useful for parameters that vary on logarithmic scales and have heavy tails.
        Uses a Pareto distribution which favors smaller values with heavy tails."""
        # Dynamically define log_mean, log_std, and shape if not explicitly provided
        self.min_mean = self.min_mean if hasattr(self, "min_mean") else 0.01
        self.max_mean = self.max_mean if hasattr(self, "max_mean") else 1.0
        self.min_shape = self.min_shape if hasattr(self, "min_shape") else 0.5
        self.max_shape = self.max_shape if hasattr(self, "max_shape") else 2.0
        self.lower_bound = (
            self.lower_bound if hasattr(self, "lower_bound") else 0.0
        )
        self.round = self.round if hasattr(self, "round") else False

        self.ensure_hyperparameter(
            "log_mean",
            "uniform",
            math.log(self.min_mean),
            math.log(self.max_mean),
        )
        self.ensure_hyperparameter(
            "shape", "uniform", self.min_shape, self.max_shape
        )

        def sampler():
            log_mean = (
                self.log_mean() if callable(self.log_mean) else self.log_mean
            )
            shape = self.shape() if callable(self.shape) else self.shape
            assert isinstance(log_mean, float)
            assert isinstance(shape, float)

            # Use Pareto distribution for heavy-tailed behavior favoring smaller values
            # Scale parameter determines the minimum value
            scale = math.exp(log_mean)
            # Shape parameter controls the heaviness of the tail (lower = heavier tail)

            def sub_sampler():
                sample = pareto_sampler(scale, shape)()
                return (
                    int(
                        self.lower_bound
                        + np.clip(round(sample), 0, self.max_mean)
                    )
                    if self.round
                    else self.lower_bound + np.clip(sample, 0, self.max_mean)
                )

            # self._plot_pareto_distribution(scale, shape, "heavy_tailed_log_scaled")

            return sub_sampler

        return sampler

    def setup_meta_choice_sampler(self):
        """Sets up a meta-categorical distribution sampler.
        Returns a closure that samples probabilities and then samples categorical values."""
        # Ensure that choice weights are defined or dynamically created
        for i in range(1, len(self.choice_values)):
            self.ensure_hyperparameter(
                f"choice_{i}_weight", distribution="uniform", min=-3.0, max=5.0
            )

        def sampler():
            weights = [1.0]
            for i in range(1, len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                attr = attr() if callable(attr) else attr
                assert isinstance(attr, float)
                weights.append(attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)
            choice_idx = torch.multinomial(weights, 1).item()
            return self.choice_values[choice_idx]

        return sampler

    def setup_meta_choice_mixed_sampler(self):
        """Sets up a mixed meta-categorical distribution sampler.
        Similar to meta_choice but with different probability scaling."""
        # Similar to meta_choice but may include different logic for mixed scenarios
        for i in range(1, len(self.choice_values)):
            self.ensure_hyperparameter(
                f"choice_{i}_weight", distribution="uniform", min=-5.0, max=6.0
            )

        def sampler():
            weights = [1.0]
            for i in range(1, len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                attr = attr() if callable(attr) else attr
                assert isinstance(attr, float)
                weights.append(attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)

            def sub_sampler():
                choice_idx = torch.multinomial(weights, 1).item()
                return self.choice_values[choice_idx]()

            return lambda: sub_sampler

        return sampler

    def forward(self):
        return self.sampler()


class HpSamplerList(nn.Module):
    """
    A container for multiple hyperparameter samplers that handles batch sampling.

    Parameters:
        hyperparameters (dict): Dictionary mapping parameter names to their sampling configurations
        device (str): Device to use for tensor operations

    Example:
        hp_config = {
            'learning_rate': {
                'distribution': 'meta_trunc_norm_log_scaled',
                'min_mean': 1e-4,
                'max_mean': 1e-1
            },
            'num_layers': {
                'distribution': 'uniform_int',
                'min': 2,
                'max': 10
            }
        }
        sampler = HpSamplerList(hp_config, device='cuda')
        params = sampler.sample()  # Returns dict with sampled values
    """

    def __init__(self, hyperparameters, device):
        super().__init__()
        self.device = device
        self.hyperparameters = nn.ModuleDict(
            {
                name: HpSampler(device=device, **params)
                for name, params in hyperparameters.items()
                if params
            }
        )

    def sample(self):
        hps = {name: hp() for name, hp in self.hyperparameters.items()}
        return hps
