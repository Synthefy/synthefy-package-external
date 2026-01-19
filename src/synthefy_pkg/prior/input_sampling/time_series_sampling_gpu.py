import random

import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess

from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler
from synthefy_pkg.prior.utils import TorchArmaProcess, generate_arima_series


class TSSampler(XSampler):
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:
    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate

    num_features : int
        Number of features to generate

    pre_stats : bool
        Whether to pre-generate statistics for the input features

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform')

    device : str, default='cpu'
        Device to store tensors on
    """

    def __init__(
        self,
        seq_len,
        num_features,
        pre_stats=False,
        sampling="mixed",
        device="cpu",
    ):
        super().__init__(seq_len, num_features, pre_stats, sampling, device)
        self.warmup_samples = 10
        self.max_freqs = 10
        if pre_stats:
            self._pre_stats()

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        """
        if self.sampling in [
            "fourier",
            "wiener",
            "arima",
            "mixed_series",
            "mixed_both",
        ]:
            samplers = {
                "fourier": self.sample_fourier_series,
                "wiener": self.sample_wiener_process,
                "arima": self.sample_arima_series,
                "mixed_series": self.sample_mixed_series,
                "mixed_both": self.sample_mixed_both,
            }
            X = samplers[self.sampling]()
            return X.cpu().numpy() if return_numpy else X
        else:
            return super().sample(return_numpy)

    def sample_arima_series(self):
        """
        Generate ARIMA series with guaranteed stationarity using PyTorch throughout.
        """

        ar_len = torch.randint(
            1, min(5, self.warmup_samples - 1), (1,), device=self.device
        ).item()
        ma_len = torch.randint(
            1, min(5, self.warmup_samples - 1), (1,), device=self.device
        ).item()
        d = torch.randint(
            0, 2, (1,), device=self.device
        ).item()  # Integration order (0 or 1)

        # Generate stationary AR process by selecting coefficients with sum < 1
        ar_coeffs = torch.zeros((int(ar_len + 1),), device=self.device)
        ar_coeffs[0] = 1.0  # Constant term

        # Generate random coefficients with controlled magnitude
        if ar_len > 0:
            # Generate coefficients such that sum of their absolute values < 0.9
            raw_coeffs = (
                torch.rand((int(ar_len),), device=self.device) * 0.8 + 0.1
            )  # Between 0.1 and 0.9
            scaling = 0.8 / raw_coeffs.sum()  # Ensure sum is < 0.9
            ar_coeffs[1:] = -raw_coeffs * scaling  # Note negative sign

        # Generate invertible MA process with coefficients of small magnitude
        ma_coeffs = torch.zeros((int(ma_len + 1),), device=self.device)
        ma_coeffs[0] = 1.0  # Constant term

        if ma_len > 0:
            ma_coeffs[1:] = (
                torch.rand((int(ma_len),), device=self.device) - 0.5
            ) * 1.0  # Between -0.5 and 0.5

        # Create ARMA process from coefficients using our PyTorch implementation
        arma_generator = TorchArmaProcess(
            ar_coeffs, ma_coeffs, device=self.device
        )

        # Double-check stationarity
        if not arma_generator.is_stationary:
            # Fall back to simple AR(1)
            ar_coeffs = torch.tensor([1.0, -0.5], device=self.device)
            ma_coeffs = torch.tensor([1.0], device=self.device)
            arma_generator = TorchArmaProcess(
                ar_coeffs, ma_coeffs, device=self.device
            )

        # Generate the base ARMA series
        series = arma_generator.generate(
            n_samples=self.seq_len + self.warmup_samples,
            sigma=1.0,
            burn_in=0,  # No burn-in since we'll trim anyway
        )

        # Handle integration (differencing) if d > 0
        if d > 0:
            # For d=1, we do cumulative sum to integrate the series
            series = torch.cumsum(series, dim=0)

        # Normalize the series
        series = (series - torch.mean(series)) / (torch.std(series) + 1e-6)

        # Return only the needed samples (after warmup)
        return series[self.warmup_samples :]

    def _fourier_transformation(
        self, t, period, function="sin", amplitude=1.0, phase=0.0
    ):
        if function == "sin":
            return amplitude * np.sin(2 * np.pi * t / period + phase)
        else:
            return amplitude * np.cos(2 * np.pi * t / period + phase)

    def sample_fourier_series(self):
        # Keep everything in PyTorch tensors
        time = torch.arange(0, self.seq_len, device=self.device)

        # Determine how many sinusoids to combine (between 2 and 5)
        num_sinusoids = np.random.randint(2, 6)

        # Vectorized parameter generation for all sinusoids
        amplitudes = torch.tensor(
            np.random.uniform(0.1, 1.0, num_sinusoids), device=self.device
        )
        periods = torch.tensor(
            np.random.randint(
                max(1, self.seq_len // 20), self.seq_len, num_sinusoids
            ),
            device=self.device,
        )
        phases = torch.tensor(
            np.random.uniform(0, 2 * np.pi, num_sinusoids), device=self.device
        )

        # Create time grid for all sinusoids: shape [seq_len, num_sinusoids]
        time_grid = time.unsqueeze(1).expand(-1, num_sinusoids)

        # Generate random offsets and add to time grid
        offsets = torch.tensor(
            np.random.randint(0, self.seq_len, num_sinusoids),
            device=self.device,
        )
        time_grid = time_grid + offsets

        # Generate functions (sin or cos)
        function_choices = np.random.choice(["sin", "cos"], num_sinusoids)

        # Initialize combined series
        combined_series = torch.zeros(self.seq_len, device=self.device)

        # Compute sine components (vectorized)
        sin_mask = torch.tensor(
            [choice == "sin" for choice in function_choices], device=self.device
        )
        if sin_mask.any():
            sin_indices = torch.where(sin_mask)[0]
            sin_times = time_grid[:, sin_indices]
            sin_periods = periods[sin_indices]
            sin_phases = phases[sin_indices]
            sin_amplitudes = amplitudes[sin_indices]

            # Compute all sine components at once
            sin_components = sin_amplitudes * torch.sin(
                2 * torch.pi * sin_times / sin_periods + sin_phases
            )
            combined_series += torch.sum(sin_components, dim=1)

        # Compute cosine components (vectorized)
        cos_mask = torch.tensor(
            [choice == "cos" for choice in function_choices], device=self.device
        )
        if cos_mask.any():
            cos_indices = torch.where(cos_mask)[0]
            cos_times = time_grid[:, cos_indices]
            cos_periods = periods[cos_indices]
            cos_phases = phases[cos_indices]
            cos_amplitudes = amplitudes[cos_indices]

            # Compute all cosine components at once
            cos_components = cos_amplitudes * torch.cos(
                2 * torch.pi * cos_times / cos_periods + cos_phases
            )
            combined_series += torch.sum(cos_components, dim=1)

        return combined_series

    def sample_fourier_series_frequency_domain(self):
        """Generate time series by sampling in frequency domain and converting to time domain.

        This creates more realistic signals by directly manipulating the frequency spectrum
        and then converting to time domain using inverse FFT using PyTorch throughout.
        """
        # Ensure even sequence length for FFT
        n = self.seq_len

        # Create frequency spectrum (only positive frequencies)
        # Number of frequencies is n//2 + 1 for real signals (including DC and Nyquist)
        num_freqs = torch.randint(
            1, min(n // 2 + 1, self.max_freqs), (1,), device=self.device
        ).item()

        # Sample spectral characteristics
        # 1. Generate random amplitudes with decay (higher frequencies have lower amplitude)
        # This creates a 1/f^beta (pink noise) characteristic, beta controls spectral slope
        beta = torch.FloatTensor(1).uniform_(0.5, 2.0).item()

        # Create frequency array (equivalent to np.fft.rfftfreq)
        freqs = torch.arange(1, n // 2 + 1, device=self.device).float() / n

        # Generate amplitudes with 1/f^beta decay
        amplitudes = torch.FloatTensor(len(freqs)).uniform_(0.1, 1.0).to(
            self.device
        ) * (freqs**-beta)

        # 2. Create random phases (uniform distribution)
        phases = (
            torch.FloatTensor(len(freqs))
            .uniform_(0, 2 * torch.pi)
            .to(self.device)
        )

        # 3. Construct complex spectrum
        spectrum = torch.zeros(
            (int(num_freqs),), dtype=torch.complex64, device=self.device
        )

        # Skip DC component (index 0)
        if len(freqs) > 0 and num_freqs > 1:
            complex_values = amplitudes[: num_freqs - 1] * torch.exp(
                1j * phases[: num_freqs - 1]
            )
            spectrum[1:] = complex_values

        # 4. Add random peaks (resonances) in the spectrum
        num_peaks = torch.randint(1, 5, (1,), device=self.device).item()

        if num_freqs > 2 and num_peaks > 0:
            # Generate random peak positions
            positions_pool = torch.arange(1, num_freqs - 1, device=self.device)

            # Randomly select peak positions
            if len(positions_pool) >= num_peaks:
                peak_indices = torch.randperm(
                    len(positions_pool), device=self.device
                )[:num_peaks]
                peak_positions = positions_pool[peak_indices]

                # Generate random peak amplitudes
                peak_amplitudes = (
                    torch.FloatTensor(num_peaks)
                    .uniform_(2.0, 5.0)
                    .to(self.device)
                )

                # Apply peak amplifications
                spectrum[peak_positions] *= peak_amplitudes

        # 5. Convert to time domain using inverse FFT
        time_series = torch.fft.irfft(spectrum, n=n)

        # 6. Normalize to have zero mean and unit variance
        mean = torch.mean(time_series)
        std = torch.std(time_series) + 1e-8
        time_series = (time_series - mean) / std

        return time_series

    def sample_wiener_process(self):
        """
        Generate a complete Wiener process (Brownian motion) time series using PyTorch.

        Can create:
        - Standard Brownian motion (mu=0, sigma=1)
        - Brownian motion with drift (mu≠0)
        - Scaled Brownian motion (sigma≠1)

        Returns:
        --------
        torch.Tensor
            Wiener process time series of length self.seq_len
        """
        # Randomly sample parameters directly with PyTorch
        mu = torch.FloatTensor(1).uniform_(-0.5, 0.5).item()  # Drift term
        sigma = torch.FloatTensor(1).uniform_(0.5, 2.0).item()  # Volatility
        dt = 1.0  # Time step size
        offset = torch.FloatTensor(1).uniform_(-1.0, 1.0).item()
        drift_variance = (
            torch.FloatTensor(1).uniform_(0.0, 1.0).item()
        )  # Additional constant drift

        # Generate all increments at once (vectorized)
        # For Wiener process, increments are normally distributed with mean=mu*dt and std=sigma*sqrt(dt)
        increments = torch.normal(
            mean=mu * dt,
            std=float(sigma * torch.sqrt(torch.tensor(dt))),
            size=(self.seq_len,),
            device=self.device,
        )

        # Add constant drift
        increments = increments + drift_variance

        # Set first value to offset (initial condition for Brownian motion)
        increments[0] = offset

        # Compute cumulative sum to get the process values
        # This is equivalent to adding each increment to the previous value
        series = torch.cumsum(increments, dim=0)

        return series

    def sample_mixed_series(self):
        """Generate features with mixed distributions."""
        X = []
        for n in range(self.num_features):
            method = np.random.choice(["fourier", "wiener"])
            if method == "fourier":
                x = self.sample_fourier_series_frequency_domain()
            elif method == "wiener":
                x = self.sample_wiener_process()
            else:
                x = self.sample_arima_series()
            X.append(x)
        return torch.stack(X, -1).float()

    def sample_mixed_both(self):
        """Generate features with mixed distributions."""
        X = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "wiener",
                    "arima",
                    "normal",
                    "multinomial",
                    "zipf",
                    "uniform",
                ]
            )
            if method == "fourier":
                x = self.sample_fourier_series()
            elif method == "wiener":
                x = self.sample_wiener_process()
            elif method == "arima":
                x = self.sample_arima_series()
            elif method == "normal":
                x = self.sample_normal(n)
            elif method == "multinomial":
                x = self.sample_multinomial()
            elif method == "zipf":
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1).double()
