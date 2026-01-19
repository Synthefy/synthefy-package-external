from __future__ import annotations

import random

import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X, return_components=False):
        noise = torch.normal(torch.zeros_like(X), self.std)
        if return_components:
            return X, noise
        else:
            return X + noise


class TorchArmaProcess:
    """PyTorch implementation of ARMA process generation that can run on GPU."""

    def __init__(self, ar_coeffs, ma_coeffs, device=None):
        """
        Initialize an ARMA process with specified coefficients.

        Args:
            ar_coeffs: AR coefficients [1, -φ₁, -φ₂, ..., -φₚ]
            ma_coeffs: MA coefficients [1, θ₁, θ₂, ..., θᵧ]
            device: PyTorch device
        """
        self.device = device

        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(ar_coeffs, np.ndarray):
            ar_coeffs = torch.tensor(
                ar_coeffs, dtype=torch.float32, device=device
            )
        if isinstance(ma_coeffs, np.ndarray):
            ma_coeffs = torch.tensor(
                ma_coeffs, dtype=torch.float32, device=device
            )

        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        self.ar_order = len(ar_coeffs) - 1
        self.ma_order = len(ma_coeffs) - 1

        # Check stationarity using PyTorch
        self.is_stationary = self._check_stationarity()

    def _check_stationarity(self):
        """Check if the AR process is stationary by examining polynomial roots."""
        if self.ar_order == 0:
            return True

        # Get AR polynomial coefficients (excluding the lead 1)
        ar_poly = self.ar_coeffs[1:]

        # For stationarity, we need to check if roots of 1 - φ₁z - φ₂z² - ... are outside unit circle
        # This is equivalent to checking if roots of z^p - φ₁z^(p-1) - ... - φₚ are inside unit circle
        # We reverse the coefficients and prepend a 1 to create this polynomial
        rev_poly = torch.cat([torch.ones(1, device=self.device), ar_poly])

        # Use NumPy's roots function since PyTorch doesn't have a direct equivalent
        # We'll need to move back to CPU for this operation
        roots = self._find_polynomial_roots(rev_poly)

        # Check if all roots have magnitude less than 1
        return torch.all(
            torch.abs(roots) < 0.999
        )  # Use 0.999 for numerical stability

    def _find_polynomial_roots(self, coeffs):
        """
        Find roots of a polynomial using companion matrix method.

        Args:
            coeffs: Polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]
                   where the polynomial is a_n * x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0

        Returns:
            Roots of the polynomial as a complex tensor
        """
        # Normalize by leading coefficient
        if coeffs[0] != 1.0:
            coeffs = coeffs / coeffs[0]

        n = len(coeffs) - 1  # Degree of polynomial

        if n == 0:
            return torch.tensor([], device=self.device)
        elif n == 1:
            return -coeffs[1].unsqueeze(0) / coeffs[0]

        # Create companion matrix
        A = torch.zeros(n, n, dtype=torch.complex64, device=self.device)

        # Set subdiagonal to 1
        idx = torch.arange(1, n, device=self.device)
        A[idx, idx - 1] = 1.0

        # Set last row to negated coefficients (excluding leading coefficient)
        A[-1, :] = -coeffs[1:] / coeffs[0]

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(A)

        return eigenvalues

    def generate(self, n_samples, sigma=1.0, burn_in=100, initial_values=None):
        """
        Generate samples from the ARMA process using vectorized operations.

        Args:
            n_samples: Number of samples to generate
            sigma: Standard deviation of the innovation process
            burn_in: Number of initial samples to discard
            initial_values: Initial values for the process

        Returns:
            PyTorch tensor of generated samples
        """
        total_samples = n_samples + burn_in
        max_lag = max(self.ar_order, self.ma_order)

        # Generate white noise
        innovation = torch.randn(total_samples, device=self.device) * sigma

        # Initialize the series with zeros
        series = torch.zeros(total_samples, device=self.device)

        # Set initial values if provided
        if initial_values is not None:
            if isinstance(initial_values, np.ndarray):
                initial_values = torch.tensor(
                    initial_values, device=self.device
                )
            series[:max_lag] = initial_values[:max_lag]

        # Process the entire series in one pass
        for t in range(max_lag, total_samples):
            # Add the innovation term
            series[t] = innovation[t]

            # Add the AR component using vectorized operation
            if self.ar_order > 0:
                # Extract the relevant slice of past values
                past_values = series[t - self.ar_order : t].flip(0)
                series[t] -= torch.sum(self.ar_coeffs[1:] * past_values)

            # Add the MA component using vectorized operation
            if self.ma_order > 0:
                # Extract the relevant slice of past innovations
                past_innovations = innovation[t - self.ma_order : t].flip(0)
                series[t] += torch.sum(self.ma_coeffs[1:] * past_innovations)

        # Discard burn-in samples
        return series[burn_in:]

    @classmethod
    def create_stationary_process(cls, ar_order, ma_order, device=None):
        """
        Create a stationary ARMA process with random coefficients.

        Args:
            ar_order: Order of the AR component
            ma_order: Order of the MA component
            device: PyTorch device

        Returns:
            A stationary TorchArmaProcess instance
        """
        # Generate stationary AR coefficients using the sum < 1 heuristic
        ar_coeffs = torch.ones(ar_order + 1, device=device)

        if ar_order > 0:
            # Generate random coefficients with sum less than 0.9
            raw_coeffs = (
                torch.rand(ar_order, device=device) * 0.3 + 0.1
            )  # Between 0.1 and 0.4
            scaling = 0.8 / raw_coeffs.sum() if raw_coeffs.sum() > 0 else 1.0
            ar_coeffs[1:] = (
                -raw_coeffs * scaling
            )  # Negative for standard ARMA form

        # Generate MA coefficients
        ma_coeffs = torch.ones(ma_order + 1, device=device)

        if ma_order > 0:
            ma_coeffs[1:] = (
                torch.rand(ma_order, device=device) - 0.5
            ) * 0.8  # Between -0.4 and 0.4

        return cls(ar_coeffs, ma_coeffs, device)


def sample_arima_series(self):
    """Generate ARIMA series using a PyTorch-based implementation."""

    ar_len = np.random.randint(1, min(5, self.warmup_samples - 1))
    ma_len = np.random.randint(1, min(5, self.warmup_samples - 1))
    d = np.random.randint(0, 2)  # Integration order (0 or 1)

    # Create a stationary ARMA process using our PyTorch implementation
    arma_process = TorchArmaProcess.create_stationary_process(
        ar_order=ar_len, ma_order=ma_len, device=self.device
    )

    # Generate the base ARMA series
    series = arma_process.generate(
        n_samples=self.seq_len + self.warmup_samples, sigma=1.0, burn_in=50
    )

    # Handle integration (differencing) if d > 0
    if d > 0:
        # For d=1, we do cumulative sum to integrate the series
        series = torch.cumsum(series, dim=0)

        # For higher orders (if you add them later)
        # Apply cumsum multiple times

    # Return only the needed samples (after warmup)
    return series[self.warmup_samples :]


def generate_arima_series(
    arma_process: ArmaProcess,
    d: int = 1,
    n_samples: int = 500,
    sigma: float = 1.0,
    drift: float = 0.0,
    initial_value: float = 0.0,
    arma_mode: bool = False,
) -> np.ndarray:
    """
    Generate synthetic time series data using an ARIMA model.

    Args:
        ar_params (np.ndarray): AR parameters including the 1 for 0-lag coefficient.
                               Example: AR(2) -> [1, -phi_1, -phi_2]
        ma_params (np.ndarray): MA parameters including the 1 for 0-lag coefficient.
                               Example: MA(1) -> [1, theta_1]
        d (int): Integration order. Default is 1.
        n_samples (int): Number of data points to generate. Default is 500.
        sigma (float): Standard deviation of the white noise. Default is 1.0.
        drift (float): Linear drift term. Default is 0.0.
        initial_value (float): Initial value for the series. Default is 0.0.

    Returns:
        np.ndarray: Generated ARIMA time series of length n_samples.

    Raises:
        ValueError: If the AR process is not stationary.
    """
    # Check for stationarity
    if not arma_process.isstationary:
        raise ValueError("AR parameters define a non-stationary process")

    # Generate the ARMA(p,q) process
    W_t = arma_process.generate_sample(nsample=n_samples, scale=sigma)

    if arma_mode:
        return (W_t - np.mean(W_t) + 1e-6) / (np.std(W_t) + 1e-6)

    # Integrate the ARMA process 'd' times
    Y_t = W_t.copy()

    if d > 0:
        for _ in range(d):
            Y_t = np.cumsum(Y_t)
            Y_t = Y_t + initial_value

    # Add drift if specified
    if drift != 0:
        Y_t = Y_t + np.arange(n_samples) * drift

    # Mean-variance normalization of the time series
    mean = np.mean(Y_t)
    std = np.std(Y_t)

    # Avoid division by zero
    Y_t = (Y_t - mean + 1e-6) / (std + 1e-6)

    return Y_t


def apply_counterfactual_perturbation(
    counterfactual_perturbation_type,
    layer_output,
    counterfactual_indices,
    start_idx_at,
    noise_std=0.01,
    perturbation_config=None,
):
    """
    Apply counterfactual perturbations to layer output.

    Args:
        counterfactual_perturbation_type: Type of perturbation to apply
        layer_output: The layer output tensor to perturb
        counterfactual_indices: Indices where to apply the perturbation
        start_idx_at: Starting index offset
        noise_std: Standard deviation for noise perturbations
        perturbation_config: Configuration dictionary for advanced perturbations
    """
    # Get the target tensor slice
    target_slice = layer_output[counterfactual_indices + start_idx_at]

    if counterfactual_perturbation_type == "random_normal":
        layer_output[counterfactual_indices + start_idx_at] = torch.randn_like(
            target_slice
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "random_uniform":
        layer_output[counterfactual_indices + start_idx_at] = torch.rand_like(
            target_slice
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "zero":
        layer_output[counterfactual_indices + start_idx_at] = 0.0
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "additive_norm":
        layer_output[counterfactual_indices + start_idx_at] = (
            torch.randn_like(target_slice) * noise_std + target_slice
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "additive_uniform":
        # Apply Gaussian noise perturbation
        if perturbation_config is None:
            perturbation_config = {"noise_std": noise_std}
        noise_std = perturbation_config.get("noise_std", noise_std)
        noise = torch.rand_like(target_slice) * noise_std - noise_std / 2
        layer_output[counterfactual_indices + start_idx_at] = (
            target_slice + noise
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "scaling":
        # Apply scaling perturbation
        if perturbation_config is None:
            perturbation_config = {"scaling_range": (0.5, 2.0)}
        min_scale, max_scale = perturbation_config.get(
            "scaling_range", (0.5, 2.0)
        )
        scale_factor = random.uniform(min_scale, max_scale)
        layer_output[counterfactual_indices + start_idx_at] = (
            target_slice * scale_factor
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "shifting":
        # Apply shifting perturbation
        if perturbation_config is None:
            perturbation_config = {"shifting_range": (-0.5, 0.5)}
        min_shift, max_shift = perturbation_config.get(
            "shifting_range", (-0.5, 0.5)
        )
        shift_amount = random.uniform(min_shift, max_shift)
        layer_output[counterfactual_indices + start_idx_at] = (
            target_slice + shift_amount
        )
        start_idx_at += len(layer_output)
    elif counterfactual_perturbation_type == "masking":
        # Apply masking perturbation (set some values to zero)
        if perturbation_config is None:
            perturbation_config = {"masking_prob": 0.3}
        mask_prob = perturbation_config.get("masking_prob", 0.3)
        mask = torch.rand_like(target_slice) > mask_prob
        layer_output[counterfactual_indices + start_idx_at] = (
            target_slice * mask
        )
        start_idx_at += len(layer_output)
    else:
        raise ValueError(
            f"Invalid counterfactual perturbation type: {counterfactual_perturbation_type}"
        )
