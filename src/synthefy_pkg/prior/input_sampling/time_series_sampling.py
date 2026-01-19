import random

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from statsmodels.tsa.arima_process import ArmaProcess

from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler
from synthefy_pkg.prior.utils import generate_arima_series

NOISE_STD = 0.00


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
        sampling="mixed_simple",
        device="cpu",
        mixed_names=[],
        sigmoid_rate=-1,
        effective_seq_len=-1,
    ):
        super().__init__(seq_len, num_features, pre_stats, sampling, device)
        self.warmup_samples = 10
        self.max_freqs = 10
        self.mixed_names = mixed_names
        # takes in a linearly increasing sigmoid rate
        self.sigmoid_rate = np.exp(-sigmoid_rate)
        self.effective_seq_len = seq_len
        if effective_seq_len > 0:
            self.effective_seq_len = effective_seq_len
        # will sample more from the first in the beginning, and then gradually samples until uniform
        if len(mixed_names) > 0:
            # if self.sampling == "mixed_names":
            # logger.info(f"Sampling mixed names: {mixed_names}")
            self.sigmoid_weights = (
                1 - expit(np.arange(len(mixed_names)) * self.sigmoid_rate)
            ) * 10
            self.sigmoid_weights = (
                self.sigmoid_weights / self.sigmoid_weights.sum()
            )
        else:
            # Default to uniform weights if mixed_names is empty
            self.sigmoid_weights = np.array([])
        if pre_stats:
            self._pre_stats()

    def sample(self, return_numpy=False, return_signal_types=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape (seq_len, num_features)
        signal_types : list
            List of signal types for each feature
        """
        if self.sampling in [
            "fourier",
            "fourier_frequency_domain",
            "fourier_both",
            "wiener",
            "arima",
            "ornstein_uhlenbeck",
            "box",
            "wavelet",
            "wedge",
            "impulse",
            "box_periodic",
            "wavelet_periodic",
            "wedge_periodic",
            "impulse_periodic",
            "piecewise_linear_periodic",
            "step",
            "linear_trend",
            "piecewise_splines",
            "piecewise_spline_periodic",
            "mixed_series",
            "mixed_subseries",
            "mixed_both",
            "mixed_subset",
            "mixed_periodic",
            "mixed_simple",
            "mixed_all",
            "mixed_names",
        ]:
            samplers = {
                "fourier": self.sample_multiple(
                    self.sample_fourier_series, self.num_features
                ),
                "fourier_frequency_domain": self.sample_multiple(
                    self.sample_fourier_series_frequency_domain,
                    self.num_features,
                ),
                "fourier_both": self.sample_fourier_both,
                "wiener": self.sample_multiple(
                    self.sample_wiener_process, self.num_features
                ),
                "arima": self.sample_multiple(
                    self.sample_arima_series, self.num_features
                ),
                "box": self.sample_multiple(
                    self.sample_box_function, self.num_features
                ),
                "wavelet": self.sample_multiple(
                    self.sample_wavelet_function, self.num_features
                ),
                "ornstein_uhlenbeck": self.sample_multiple(
                    self.sample_ornstein_uhlenbeck, self.num_features
                ),
                "wedge": self.sample_multiple(
                    self.sample_wedge_function, self.num_features
                ),
                "impulse": self.sample_multiple(
                    self.sample_impulse_function, self.num_features
                ),
                "box_periodic": self.sample_multiple(
                    self.sample_box_function, self.num_features, periodic=True
                ),
                "wavelet_periodic": self.sample_multiple(
                    self.sample_wavelet_function,
                    self.num_features,
                    periodic=True,
                ),
                "wedge_periodic": self.sample_multiple(
                    self.sample_wedge_function, self.num_features, periodic=True
                ),
                "impulse_periodic": self.sample_multiple(
                    self.sample_impulse_function,
                    self.num_features,
                    periodic=True,
                ),
                "step": self.sample_multiple(
                    self.sample_step_function, self.num_features
                ),
                "linear_trend": self.sample_multiple(
                    self.sample_linear_trend, self.num_features
                ),
                "piecewise_splines": self.sample_multiple(
                    self.sample_piecewise_splines, self.num_features
                ),
                "piecewise_spline_periodic": self.sample_multiple(
                    self.sample_piecewise_spline_periodic, self.num_features
                ),
                "piecewise_linear_periodic": self.sample_multiple(
                    self.sample_piecewise_linear_periodic, self.num_features
                ),
                "mixed_simple": self.sample_mixed_simple,
                "mixed_all": self.sample_mixed_all,
                "mixed_subset": self.sample_mixed_subset,
                "mixed_series": self.sample_mixed_series,
                "mixed_subseries": self.sample_mixed_subseries,
                "mixed_periodic": self.sample_mixed_periodic,
                "mixed_both": self.sample_mixed_both,
                "mixed_names": self.sample_mixed_names,
            }
            X, signal_types = samplers[self.sampling]()
            if return_signal_types:
                return (
                    (X.cpu().numpy(), signal_types)
                    if return_numpy
                    else (X, signal_types)
                )
            else:
                return X.cpu().numpy() if return_numpy else X
        else:
            return super().sample(return_numpy, return_signal_types)

    def sample_multiple(
        self, underlying_signal_fn, num_samples, periodic=False
    ):
        """
        Sample multiple signals from the underlying signal function.
        """

        def sample_fn():
            res = [
                underlying_signal_fn(periodic=periodic)
                for _ in range(num_samples)
            ]
            ress = [r[0] for r in res]
            ress = [self.normalize(r) for r in ress]
            signal_types = [r[1] for r in res]
            return torch.stack(ress, -1).float(), signal_types

        return sample_fn

    def sample_periodic(self, underlying_signal_fn):
        """
        Generate a periodic signal with random parameters.

        The signal will be periodic with a random period and phase.
        The signal will be a sine wave with a random amplitude and phase.
        """

        def periodic_signal_fn():
            return underlying_signal_fn(self, is_periodic=True)

        return periodic_signal_fn

    def sample_arima_series(self, periodic=False):
        """
        Generate ARIMA series with guaranteed stationarity using the utility function,
        converting to PyTorch only at the end.
        """

        # Convert parameters to numpy
        ar_len = np.random.randint(1, min(5, self.warmup_samples - 1))
        ma_len = np.random.randint(1, min(5, self.warmup_samples - 1))
        d = np.random.randint(0, 2)  # Integration order (0 or 1)

        # Generate AR and MA coefficients
        ar_coeffs = np.zeros(ar_len + 1)
        ar_coeffs[0] = 1.0

        ma_coeffs = np.zeros(ma_len + 1)
        ma_coeffs[0] = 1.0

        if ar_len > 0:
            # Generate coefficients such that sum of their absolute values < 0.9
            raw_coeffs = (
                np.random.rand(ar_len) * 0.8 + 0.1
            )  # Between 0.1 and 0.9
            scaling = 0.8 / raw_coeffs.sum()  # Ensure sum is < 0.9
            ar_coeffs[1:] = -raw_coeffs * scaling  # Note negative sign

        if ma_len > 0:
            ma_coeffs[1:] = (
                np.random.rand(ma_len) - 0.5
            ) * 1.0  # Between -0.5 and 0.5

        # Create ARMA process using statsmodels
        arma_process = ArmaProcess(ar_coeffs, ma_coeffs)
        # Check if process is stationary
        if not arma_process.isstationary:
            raise ValueError("Process is not stationary")

        # Generate the ARIMA series using the utility function over effective_seq_len
        n_samples = self.effective_seq_len + self.warmup_samples
        series = generate_arima_series(
            arma_process=arma_process,
            d=d,
            n_samples=n_samples,
            sigma=0.5,
            drift=0.0,
            initial_value=0.0,
        )

        # Extract only the needed samples (after warmup)
        series = series[self.warmup_samples :]

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            series = np.tile(series, num_repeats)
            # Trim to exact sequence length
            series = series[: self.seq_len]

        # name signal type based on parameters actually sampled
        signal_type = f"arima_{ar_len}_{ma_len}_{d}"

        # Convert to torch and return
        return torch.tensor(series, device=self.device), signal_type

    def _fourier_transformation(
        self, t, period, function="sin", amplitude=1.0, phase=0.0
    ):
        if function == "sin":
            return amplitude * np.sin(2 * np.pi * t / period + phase)
        else:
            return amplitude * np.cos(2 * np.pi * t / period + phase)

    def sample_fourier_series(self, periodic=False):
        time = torch.arange(0, self.seq_len).numpy()
        # Generate and add multiple sinusoids with different parameters
        # Determine how many sinusoids to combine (between 2 and 5)
        num_sinusoids = np.random.randint(2, 6)

        # Vectorized parameter generation for all sinusoids
        amplitudes = np.random.uniform(0.1, 1.0, num_sinusoids)
        periods = np.random.randint(
            max(1, self.effective_seq_len // 20),
            self.effective_seq_len,
            num_sinusoids,
        )
        phases = np.random.uniform(0, 2 * np.pi, num_sinusoids)

        # Create time grid for all sinusoids: shape [seq_len, num_sinusoids]
        time_grid = np.expand_dims(time, axis=1) @ np.ones((1, num_sinusoids))

        # Generate random offsets and add to time grid
        offsets = np.random.randint(0, self.effective_seq_len, num_sinusoids)
        time_grid = time_grid + offsets

        # Generate functions (sin or cos)
        function_choices = np.random.choice(["sin", "cos"], num_sinusoids)

        # Initialize combined series
        combined_series = np.zeros(self.seq_len)

        # Compute sine components (vectorized)
        sin_mask = function_choices == "sin"
        if np.any(sin_mask):
            sin_indices = np.where(sin_mask)[0]
            sin_times = time_grid[:, sin_indices]
            sin_periods = periods[sin_indices]
            sin_phases = phases[sin_indices]
            sin_amplitudes = amplitudes[sin_indices]

            # Compute all sine components at once
            sin_components = sin_amplitudes * np.sin(
                2 * np.pi * sin_times / sin_periods + sin_phases
            )
            combined_series += np.sum(sin_components, axis=1)

        # Compute cosine components (vectorized)
        cos_mask = function_choices == "cos"
        if np.any(cos_mask):
            cos_indices = np.where(cos_mask)[0]
            cos_times = time_grid[:, cos_indices]
            cos_periods = periods[cos_indices]
            cos_phases = phases[cos_indices]
            cos_amplitudes = amplitudes[cos_indices]

            # Compute all cosine components at once
            cos_components = cos_amplitudes * np.cos(
                2 * np.pi * cos_times / cos_periods + cos_phases
            )
            combined_series += np.sum(cos_components, axis=1)

        # name signal type based on parameters actually sampled
        signal_type = f"fourier_{num_sinusoids}"

        # 7. add noise
        combined_series = combined_series + np.random.normal(
            0, 0.03, self.seq_len
        )

        # Convert back to torch tensor if needed
        return torch.tensor(combined_series, device=self.device), signal_type

    def sample_fourier_series_frequency_domain(self, periodic=False):
        """Generate time series by sampling in frequency domain and converting to time domain.

        This creates more realistic signals by directly manipulating the frequency spectrum
        and then converting to time domain using inverse FFT.
        """
        # Ensure even sequence length for FFT
        n = self.effective_seq_len

        # Create frequency spectrum (only positive frequencies)
        # Number of frequencies is n//2 + 1 for real signals (including DC and Nyquist)
        num_freqs = np.random.randint(1, min(n // 2 + 1, self.max_freqs))

        # Sample spectral characteristics
        # 1. Generate random amplitudes with decay (higher frequencies have lower amplitude)
        # This creates a 1/f^beta (pink noise) characteristic, beta controls spectral slope
        beta = np.random.uniform(
            0.5, 2.0
        )  # Spectral slope (1 = pink noise, 2 = brown noise)
        freqs = np.fft.rfftfreq(n)[1:]  # Exclude DC component
        amplitudes = np.random.uniform(0.1, 1.0, len(freqs)) * (freqs**-beta)

        # 2. Create random phases (uniform distribution)
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))

        # 3. Construct complex spectrum - ensure size matches
        spectrum = np.zeros(
            n // 2 + 1, dtype=complex
        )  # Full spectrum size for rfft
        spectrum[1 : len(amplitudes) + 1] = amplitudes * np.exp(
            1j * phases
        )  # Skip DC component (index 0)

        # 4. Add random peaks (resonances) in the spectrum
        num_peaks = np.random.randint(1, 5)
        peak_positions = np.random.choice(
            np.arange(1, len(spectrum) - 1), size=num_peaks, replace=False
        )
        peak_amplitudes = np.random.uniform(2.0, 5.0, num_peaks)
        spectrum[peak_positions] *= peak_amplitudes

        # 5. Convert to time domain using inverse FFT
        time_series = np.fft.irfft(spectrum, n=n)

        # # 6. Normalize to have zero mean and unit variance
        # time_series = (time_series - np.mean(time_series)) / (
        #     np.std(time_series) + 1e-8
        # )
        # 7. Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            time_series = np.tile(time_series, num_repeats)
            # Trim to exact sequence length
            time_series = time_series[: self.seq_len]

        # 8. add noise
        time_series = time_series + np.random.normal(0, NOISE_STD, self.seq_len)

        # name signal type based on parameters actually sampled
        signal_type = f"fourier_frequency_domain_{num_freqs}"

        return torch.tensor(time_series, device=self.device), signal_type

    def sample_wiener_process(self, periodic=False):
        """
        Generate a complete Wiener process (Brownian motion) time series.

        Can create:
        - Standard Brownian motion (mu=0, sigma=1)
        - Brownian motion with drift (mu≠0)
        - Scaled Brownian motion (sigma≠1)

        Returns:
        --------
        torch.Tensor
            Wiener process time series of length self.seq_len
        """
        # Randomly sample parameters
        mu = np.random.uniform(-0.5, 0.5)  # Drift term
        sigma = np.random.uniform(0.5, 2.0)  # Volatility
        dt = 1.0  # Time step size
        offset = np.random.uniform(-1.0, 1.0)
        drift_variance = np.random.uniform(
            0.0, 1.0
        )  # Additional constant drift

        # Simplify the signal by setting mu and drift_variance to 0 half the time
        if np.random.uniform(0, 1) < 0.5:
            mu = 0.0
            drift_variance = 0.0

        # Generate all increments at once (vectorized) over effective_seq_len
        # For Wiener process, increments are normally distributed with mean=mu*dt and std=sigma*sqrt(dt)
        increments = np.random.normal(
            mu * dt, sigma * np.sqrt(dt), size=self.effective_seq_len
        )

        # Add constant drift
        increments = increments + drift_variance

        # Set first value to 0 (standard initial condition for Brownian motion)
        increments[0] = offset

        # Compute cumulative sum to get the process values
        # This is equivalent to adding each increment to the previous value
        series = np.cumsum(increments)

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            series = np.tile(series, num_repeats)
            # Trim to exact sequence length
            series = series[: self.seq_len]

        # name signal type based on parameters actually sampled
        # perform rounding to 2 decimal places
        mu = round(mu, 2)
        sigma = round(sigma, 2)
        drift_variance = round(drift_variance, 2)
        offset = round(offset, 2)
        signal_type = f"wiener_process_{mu}_{sigma}_{drift_variance}_{offset}"

        return torch.tensor(series, device=self.device), signal_type

    def sample_box_function(self, periodic=False, num_boxes=None):
        """Generate a box function (rectangular pulse) with random width and location.

        The function will be 1 within the box intervals and 0 elsewhere.
        The width and location of the box(es) are randomly chosen.

        Parameters
        ----------
        periodic : bool, default=False
            If True, generates multiple boxes with equal spacing
        num_boxes : int, optional
            Number of boxes to generate. If periodic=True and num_boxes=None,
            a random number of boxes between 2 and 5 will be chosen.

        Returns
        -------
        torch.Tensor
            Box function of length self.seq_len
        """
        # Generate random width between 10% and 40% of sequence length
        min_width = max(2, int(self.effective_seq_len * 0.1))
        max_width = int(self.effective_seq_len * 0.4)
        width = np.random.randint(min_width, max_width)

        # Create the box function
        box = np.zeros(self.seq_len)

        if periodic:
            # Determine number of boxes if not specified
            if num_boxes is None:
                num_boxes = np.random.randint(2, 6)

            # Calculate period to fit all boxes
            period = self.effective_seq_len // num_boxes

            total_num_boxes = self.seq_len // period

            actual_width = min(
                width, period - int(np.ceil(max(period / 2, 1)))
            )  # Leave at least 1 sample gap
            # Generate boxes at regular intervals
            for i in range(total_num_boxes):
                start_pos = i * period
                # Ensure the box fits within its period
                box[start_pos : start_pos + actual_width] = 1.0

            signal_type = f"box_function_periodic_{period}_width_{actual_width}"
        else:
            # Generate random start position for single box
            max_start = self.effective_seq_len - width
            start_pos = np.random.randint(0, max_start + 1)
            box[start_pos : start_pos + width] = 1.0

            # Tile the effective_seq_len signal to fill the full seq_len

            if self.seq_len > self.effective_seq_len:
                # Calculate how many times to repeat the effective_seq_len signal
                num_repeats = int(
                    np.ceil(self.seq_len / self.effective_seq_len)
                )
                box = np.tile(box, num_repeats)
                # Trim to exact sequence length
                box = box[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = f"box_function_{width}_start_{start_pos}"

        # Add small random noise
        box = box + np.random.normal(0, NOISE_STD, self.seq_len)

        return torch.tensor(box, device=self.device), signal_type

    def sample_wavelet_function(
        self, periodic=False, num_wavelets=None, wavelet_type="mexican_hat"
    ):
        """Generate a wavelet function with random parameters.

        Creates either a single wavelet or periodic wavelets with equal spacing.
        Supports different types of wavelets: Mexican hat, Morlet, and Gaussian.

        Parameters
        ----------
        periodic : bool, default=False
            If True, generates multiple wavelets with equal spacing
        num_wavelets : int, optional
            Number of wavelets to generate. If periodic=True and num_wavelets=None,
            a random number of wavelets between 2 and 5 will be chosen.
        wavelet_type : str, default="mexican_hat"
            Type of wavelet to generate. Options: "mexican_hat", "morlet", "gaussian"

        Returns
        -------
        torch.Tensor
            Wavelet function of length self.seq_len
        """
        # Generate random width between 10% and 40% of sequence length
        min_width = max(2, int(self.effective_seq_len * 0.1))
        max_width = int(self.effective_seq_len * 0.4)
        width = np.random.randint(min_width, max_width)

        # Create time points
        t = np.linspace(-3, 3, width)  # Standard range for wavelets

        # Generate base wavelet
        if wavelet_type == "mexican_hat":
            # Mexican hat wavelet: (1 - t^2) * exp(-t^2/2)
            base_wavelet = (1 - t**2) * np.exp(-(t**2) / 2)
        elif wavelet_type == "morlet":
            # Morlet wavelet: exp(-t^2/2) * cos(5t)
            base_wavelet = np.exp(-(t**2) / 2) * np.cos(5 * t)
        else:  # gaussian
            # Gaussian wavelet: exp(-t^2/2)
            base_wavelet = np.exp(-(t**2) / 2)

        # Normalize the base wavelet
        base_wavelet = base_wavelet / np.max(np.abs(base_wavelet))

        # Create the full sequence
        x = np.zeros(self.seq_len)

        if periodic:
            # Determine number of wavelets if not specified
            if num_wavelets is None:
                num_wavelets = np.random.randint(4, 8)

            # Calculate period to fit all wavelets
            period = self.effective_seq_len // num_wavelets

            # Use tile operation to place wavelets at regular intervals
            # Create a period template with zeros and the wavelet at the beginning
            actual_width = min(width, period - 2)  # Leave at least 1 sample gap
            period_template = np.zeros(period)
            period_template[:actual_width] = base_wavelet[:actual_width]

            # Tile the period template to fill the sequence
            x = np.tile(period_template, int(np.ceil(self.seq_len / period)))
            x = x[: self.seq_len]  # Trim to exact sequence length

            # name signal type based on parameters actually sampled
            signal_type = f"wavelet_function_periodic_{period}_width_{width}"
        else:
            # Generate random start position for single wavelet within effective_seq_len
            max_start = self.effective_seq_len - width
            start_pos = np.random.randint(0, max_start + 1)

            # Create the wavelet over effective_seq_len
            x = np.zeros(self.effective_seq_len)
            x[start_pos : start_pos + width] = base_wavelet

            # Tile the effective_seq_len signal to fill the full seq_len
            if self.seq_len > self.effective_seq_len:
                # Calculate how many times to repeat the effective_seq_len signal
                num_repeats = int(
                    np.ceil(self.seq_len / self.effective_seq_len)
                )
                x = np.tile(x, num_repeats)
                # Trim to exact sequence length
                x = x[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = f"wavelet_function_{width}_start_{start_pos}"

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Add small random noise
        noise = np.random.normal(0, NOISE_STD, self.seq_len)
        x = x + noise

        return torch.tensor(x, device=self.device), signal_type

    def sample_ornstein_uhlenbeck(self, periodic=False):
        """Generate an Ornstein-Uhlenbeck process.

        The Ornstein-Uhlenbeck process is a stochastic process that describes
        the velocity of a Brownian particle under friction. It is a mean-reverting
        process that follows the stochastic differential equation:

        dX_t = theta * (mu - X_t)dt + sigma * dW_t

        where:
        - theta is the rate of mean reversion
        - mu is the long-term mean
        - sigma is the volatility
        - W_t is a Wiener process

        Returns
        -------
        torch.Tensor
            Ornstein-Uhlenbeck process of length self.seq_len
        """
        # Randomly sample parameters
        theta = np.random.uniform(0.1, 0.5)  # Mean reversion rate
        mu = np.random.uniform(-1.0, 1.0)  # Long-term mean
        sigma = np.random.uniform(0.1, 0.5)  # Volatility
        dt = 1.0  # Time step size

        # Generate random increments for the Wiener process
        dW = np.random.normal(0, np.sqrt(dt), self.seq_len)

        # Initialize the process
        x = np.zeros(self.seq_len)
        x[0] = np.random.normal(mu, sigma)  # Initial value

        # Vectorized computation of the process
        # Using the exact solution of the OU process:
        # X_t = μ + (X_0 - μ)e^(-θt) + σ∫_0^t e^(-θ(t-s))dW_s
        t = np.arange(self.seq_len) * dt

        # Compute the deterministic part (mean reversion)
        deterministic = mu + (x[0] - mu) * np.exp(-theta * t)

        # Compute the stochastic part (integral of the Wiener process)
        # Using the fact that the integral can be approximated by a sum
        # of independent normal random variables
        stochastic = sigma * np.sqrt((1 - np.exp(-2 * theta * t)) / (2 * theta))
        x = deterministic + stochastic * dW

        # Normalize to have zero mean and unit variance
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # name signal type based on parameters actually sampled
        # round to 2 decimal places
        theta = round(theta, 2)
        mu = round(mu, 2)
        sigma = round(sigma, 2)
        signal_type = f"ornstein_uhlenbeck_{theta}_{mu}_{sigma}"

        return torch.tensor(x, device=self.device), signal_type

    def sample_wedge_function(self, periodic=False, num_periods=None):
        """Generate a wedge (triangle) function with random parameters.

        Creates a triangular wave pattern that can be either increasing,
        decreasing, or symmetric around the middle. Can also generate
        periodic wedges with equal spacing.

        Parameters
        ----------
        periodic : bool, default=False
            If True, generates multiple wedges with equal spacing
        num_periods : int, optional
            Number of periods to generate. If periodic=True and num_periods=None,
            randomly chooses between 2 and 5 periods.

        Returns
        -------
        torch.Tensor
            Wedge function of length self.seq_len
        """
        # Randomly choose wedge type
        wedge_type = np.random.choice(["increasing", "decreasing", "symmetric"])

        if periodic:
            # Determine number of periods if not specified
            if num_periods is None:
                num_periods = np.random.randint(5, 10)

            # Calculate period length
            period_length = int(np.ceil(self.effective_seq_len / num_periods))

            # Generate time points for one period
            t = np.linspace(0, 1, period_length)

            if wedge_type == "increasing":
                # Linear increase from 0 to 1
                one_period = t
            elif wedge_type == "decreasing":
                # Linear decrease from 1 to 0
                one_period = 1 - t
            else:  # symmetric
                # Triangle wave centered at middle
                one_period = 1 - 2 * np.abs(t - 0.5)

            # Repeat the period
            x = np.tile(
                one_period,
                int(num_periods * np.ceil(self.seq_len / period_length)),
            )

            # Trim to exact sequence length
            x = x[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = (
                f"wedge_function_periodic_{num_periods}_type_{wedge_type}"
            )
        else:
            # Generate time points over effective_seq_len
            t = np.linspace(0, 1, self.effective_seq_len)

            if wedge_type == "increasing":
                # Linear increase from 0 to 1
                x = t
            elif wedge_type == "decreasing":
                # Linear decrease from 1 to 0
                x = 1 - t
            else:  # symmetric
                # Triangle wave centered at middle
                x = 1 - 2 * np.abs(t - 0.5)

            # Tile the effective_seq_len signal to fill the full seq_len
            if self.seq_len > self.effective_seq_len:
                # Calculate how many times to repeat the effective_seq_len signal
                num_repeats = int(
                    np.ceil(self.seq_len / self.effective_seq_len)
                )
                x = np.tile(x, num_repeats)
                # Trim to exact sequence length
                x = x[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = f"wedge_function_type_{wedge_type}"

        # print("x", x)
        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Add some random noise
        noise = np.random.normal(0, NOISE_STD, self.seq_len)
        x = x + noise

        return torch.tensor(x, device=self.device), signal_type

    def sample_impulse_function(self, periodic=False, num_periods=None):
        """Generate a function with impulse responses.

        Creates a series of impulse responses (exponential decays) that can be
        either randomly placed or periodically spaced.

        Parameters
        ----------
        periodic : bool, default=False
            If True, generates impulses at regular intervals
        num_periods : int, optional
            Number of periods to generate. If periodic=True and num_periods=None,
            randomly chooses between 2 and 5 periods.

        Returns
        -------
        torch.Tensor
            Impulse function of length self.seq_len
        """
        if periodic:
            # Determine number of periods if not specified
            if num_periods is None:
                num_periods = np.random.randint(5, 20)

            # Calculate period length
            period_length = int(np.ceil(self.effective_seq_len / num_periods))

            # Generate one period of the impulse response
            t = np.arange(period_length)
            decay = np.random.uniform(0.1, 0.5)
            amplitude = np.random.uniform(0.5, 2.0)
            one_period = amplitude * np.exp(-decay * t)

            # Repeat the period
            x = np.tile(
                one_period,
                int(num_periods * np.ceil(self.seq_len / period_length)),
            )

            # Trim to exact sequence length
            x = x[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = f"impulse_function_periodic_{num_periods}"
        else:
            # Randomly choose number of impulses
            num_impulses = np.random.randint(10, 20)

            # Initialize output over effective_seq_len
            x = np.zeros(self.effective_seq_len)

            # Generate impulses
            for _ in range(num_impulses):
                # Random position within effective_seq_len
                pos = np.random.randint(0, self.effective_seq_len)
                # Random decay rate
                decay = np.random.uniform(0.1, 0.5)
                # Random amplitude
                amplitude = np.random.uniform(1.0, 3.0)

                # Generate exponential decay
                t = np.arange(self.effective_seq_len - pos)
                impulse = amplitude * np.exp(-decay * t)
                x[pos:] += impulse

            # Tile the effective_seq_len signal to fill the full seq_len
            if self.seq_len > self.effective_seq_len:
                # Calculate how many times to repeat the effective_seq_len signal
                num_repeats = int(
                    np.ceil(self.seq_len / self.effective_seq_len)
                )
                x = np.tile(x, num_repeats)
                # Trim to exact sequence length
                x = x[: self.seq_len]

            # name signal type based on parameters actually sampled
            signal_type = f"impulse_function_type_{num_impulses}"

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Add small random noise
        noise = np.random.normal(0, NOISE_STD, self.seq_len)
        x = x + noise

        return torch.tensor(x, device=self.device), signal_type

    def sample_step_function(self, num_steps=None, periodic=False):
        """Generate a step function with random steps.

        Creates a function that changes value at random positions,
        simulating a series of step changes.

        Parameters
        ----------
        num_steps : int, optional
            Number of steps to generate. If None, randomly chooses between 2 and 5.

        Returns
        -------
        torch.Tensor
            Step function of length self.seq_len
        """
        if num_steps is None:
            num_steps = np.random.randint(7, 15)

        # Generate step positions over effective_seq_len
        step_positions = np.sort(
            np.random.choice(
                np.arange(1, self.effective_seq_len - 1),
                size=num_steps - 1,
                replace=False,
            )
        )
        step_positions = np.concatenate(
            [[0], step_positions, [self.effective_seq_len]]
        )

        # Generate random step values
        step_values = np.random.normal(0, 1, num_steps)

        # Create step function over effective_seq_len
        x = np.zeros(self.effective_seq_len)
        for i in range(num_steps):
            x[step_positions[i] : step_positions[i + 1]] = step_values[i]

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            x = np.tile(x, num_repeats)
            # Trim to exact sequence length
            x = x[: self.seq_len]

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Add small noise
        noise = np.random.normal(0, NOISE_STD, self.seq_len)
        x = x + noise

        # name signal type based on parameters actually sampled
        signal_type = f"step_function_num_steps_{num_steps}"

        return torch.tensor(x, device=self.device), signal_type

    def sample_linear_trend(self, add_noise=True, periodic=False):
        """Generate a linear trend with optional noise.

        Creates a linear trend that can be either increasing or decreasing,
        with optional random noise added.

        Parameters
        ----------
        add_noise : bool, default=True
            Whether to add random noise to the trend.

        Returns
        -------
        torch.Tensor
            Linear trend of length self.seq_len
        """
        # Generate time points over effective_seq_len
        t = np.linspace(0, 1, self.effective_seq_len)

        # Random slope
        slope = np.random.uniform(-2.0, 2.0)

        # Generate trend
        x = slope * t

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            x = np.tile(x, num_repeats)
            # Trim to exact sequence length
            x = x[: self.seq_len]

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        if add_noise:
            # Add random noise
            noise = np.random.normal(0, NOISE_STD, self.seq_len)
            x = x + noise

        # name signal type based on parameters actually sampled
        signal_type = f"linear_trend_slope_{slope}"

        return torch.tensor(x, device=self.device), signal_type

    def sample_piecewise_linear_periodic(
        self,
        num_periods=None,
        segments_per_period=None,
        add_noise=True,
    ):
        """Generate periodic piecewise linear trends by repeating linear patterns.

        Creates a periodic signal by generating a single piecewise linear pattern
        and then repeating it at regular intervals. This creates repeating patterns
        of linear segments that are useful for modeling periodic phenomena with
        different growth rates in each phase.

        Parameters
        ----------
        num_periods : int, optional
            Number of periods to generate. If None, randomly chooses between 2 and 6.
        segments_per_period : int, optional
            Number of linear segments per period. If None, randomly chooses between 2 and 5.
        add_noise : bool, default=True
            Whether to add random noise to the signal.

        Returns
        -------
        torch.Tensor
            Periodic piecewise linear function of length self.seq_len
        str
            Signal type description
        """
        if num_periods is None:
            num_periods = np.random.randint(5, 20)

        if segments_per_period is None:
            segments_per_period = np.random.randint(2, 5)

        # Calculate period length
        period_length = int(np.ceil(self.effective_seq_len / num_periods))

        # Temporarily modify seq_len to generate one period
        original_seq_len, original_effective_seq_len = (
            self.seq_len,
            self.effective_seq_len,
        )
        self.seq_len, self.effective_seq_len = period_length, period_length

        # Generate one period using a simplified piecewise linear method
        one_period = self.sample_piecewise_linear_period(
            segments_per_period, add_noise=False
        )

        # Restore original seq_len
        self.seq_len, self.effective_seq_len = (
            original_seq_len,
            original_effective_seq_len,
        )

        # Convert to numpy for tiling
        one_period = one_period.cpu().numpy()

        # Use tile operation to repeat the period
        x = np.tile(one_period, int(np.ceil(self.seq_len / period_length)))
        x = x[: self.seq_len]  # Trim to exact sequence length

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        if add_noise:
            # Add random noise
            noise = np.random.normal(0, NOISE_STD, self.seq_len)
            x = x + noise

        # name signal type based on parameters actually sampled
        signal_type = f"piecewise_linear_periodic_{num_periods}_{period_length}_s{segments_per_period}"

        return torch.tensor(x, device=self.device), signal_type

    def sample_piecewise_linear_period(self, num_segments=None, add_noise=True):
        """Helper method to generate a single piecewise linear period.

        This is used internally by sample_piecewise_linear_periodic to generate
        one period that can then be tiled.
        """
        if num_segments is None:
            num_segments = np.random.randint(4, 8)

        # Calculate segment length over effective_seq_len
        segment_length = self.effective_seq_len // num_segments

        # Generate one period over effective_seq_len
        x_period = np.zeros(self.effective_seq_len)

        # Generate random slopes for each segment
        slopes = np.random.uniform(-3.0, 3.0, num_segments)

        # Build the piecewise linear function for one period
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, self.effective_seq_len)

            # Time points for this segment (normalized to [0, 1])
            t_segment = np.linspace(0, 1, end_idx - start_idx)

            # Generate linear trend for this segment
            segment_values = slopes[i] * t_segment

            # Add offset to ensure continuity (optional)
            if i > 0:
                # Make segments continuous by adjusting offset
                prev_end_value = x_period[start_idx - 1] if start_idx > 0 else 0
                segment_values = segment_values + prev_end_value

            x_period[start_idx:end_idx] = segment_values

        # Normalize
        x_period = (x_period - np.mean(x_period)) / (np.std(x_period) + 1e-8)

        if add_noise:
            # Add random noise
            noise = np.random.normal(0, NOISE_STD, self.effective_seq_len)
            x_period = x_period + noise

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            x = np.tile(x_period, num_repeats)
            # Trim to exact sequence length
            x = x[: self.seq_len]

        return torch.tensor(x_period, device=self.device)

    def sample_piecewise_splines(
        self, num_segments=None, continuous=True, degree=3, periodic=False
    ):
        """Generate piecewise splines with optional continuity constraints.

        Creates a series of spline segments that can be either continuous
        or discontinuous at their boundaries. Each segment is a polynomial
        of specified degree.

        Parameters
        ----------
        num_segments : int, optional
            Number of spline segments to generate. If None, randomly chooses
            between 2 and 5 segments.
        continuous : bool, default=True
            If True, ensures continuity at segment boundaries
        degree : int, default=3
            Degree of the polynomial splines (1=linear, 2=quadratic, 3=cubic)

        Returns
        -------
        torch.Tensor
            Piecewise spline function of length self.seq_len
        """
        from scipy.interpolate import BSpline, make_interp_spline

        if num_segments is None:
            num_segments = np.random.randint(4, 8)

        # Calculate segment length over effective_seq_len
        segment_length = self.effective_seq_len // num_segments

        # Generate segment boundaries over effective_seq_len
        boundaries = np.arange(0, self.effective_seq_len + 1, segment_length)
        if boundaries[-1] != self.effective_seq_len:
            boundaries[-1] = self.effective_seq_len

        # Initialize output over effective_seq_len
        x = np.zeros(self.effective_seq_len)

        # Generate control points for each segment
        if continuous:
            # For continuous splines, we need to ensure continuity at boundaries
            # Start with random control points
            control_points = np.random.normal(0, 1, num_segments + 1)

            # Adjust control points to ensure continuity
            for i in range(1, num_segments):
                # Make the end point of previous segment match start point of current segment
                control_points[i] = (
                    control_points[i - 1] + control_points[i + 1]
                ) / 2
        else:
            # For discontinuous splines, each segment can have independent control points
            control_points = np.random.normal(0, 1, num_segments + 1)

        # Generate spline for each segment
        for i in range(num_segments):
            # Get segment boundaries
            start, end = boundaries[i], boundaries[i + 1]
            segment_len = end - start

            # Generate time points for this segment
            t = np.linspace(0, 1, segment_len)

            if degree == 1:
                # Linear interpolation
                x[start:end] = (
                    control_points[i] * (1 - t) + control_points[i + 1] * t
                )
            else:
                # Higher degree spline - use simpler approach to avoid BSpline issues
                # Create a polynomial of the specified degree
                coeffs = np.random.normal(0, 1, degree + 1)

                # Evaluate polynomial at time points
                polynomial = np.polyval(coeffs, t)

                # Ensure continuity by adjusting the polynomial
                if continuous and i > 0:
                    # Adjust to match the end value of previous segment
                    offset = x[start - 1] - polynomial[0]
                    polynomial += offset

                x[start:end] = polynomial

        # Tile the effective_seq_len signal to fill the full seq_len
        if self.seq_len > self.effective_seq_len:
            # Calculate how many times to repeat the effective_seq_len signal
            num_repeats = int(np.ceil(self.seq_len / self.effective_seq_len))
            x = np.tile(x, num_repeats)
            # Trim to exact sequence length
            x = x[: self.seq_len]

        # Normalize
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        # Add small random noise
        noise = np.random.normal(0, NOISE_STD, self.seq_len)
        x = x + noise

        # name signal type based on parameters actually sampled
        signal_type = f"piecewise_splines_num_segments_{num_segments}_continuous_{continuous}_degree_{degree}"

        return torch.tensor(x, device=self.device), signal_type

    def sample_piecewise_spline_periodic(
        self,
        num_periods=None,
        segments_per_period=None,
        continuous=True,
        degree=3,
    ):
        """Generate periodic piecewise splines by repeating spline patterns.

        Creates a periodic signal by generating a single spline pattern and then
        repeating it at regular intervals. This creates smooth, repeating patterns
        that are useful for modeling periodic phenomena.

        Parameters
        ----------
        num_periods : int, optional
            Number of periods to generate. If None, randomly chooses between 2 and 6.
        segments_per_period : int, optional
            Number of spline segments per period. If None, randomly chooses between 2 and 4.
        continuous : bool, default=True
            If True, ensures continuity at segment boundaries within each period
        degree : int, default=3
            Degree of the polynomial splines (1=linear, 2=quadratic, 3=cubic)

        Returns
        -------
        torch.Tensor
            Periodic piecewise spline function of length self.seq_len
        """
        if num_periods is None:
            num_periods = np.random.randint(4, 20)

        if segments_per_period is None:
            segments_per_period = np.random.randint(2, 5)

        # Calculate period length
        period_length = int(np.ceil(self.effective_seq_len / num_periods))

        # Temporarily modify seq_len to generate one period
        original_seq_len, original_effective_seq_len = (
            self.seq_len,
            self.effective_seq_len,
        )
        self.seq_len, self.effective_seq_len = period_length, period_length

        # Generate one period using the existing piecewise splines method
        one_period, _ = self.sample_piecewise_splines(
            num_segments=segments_per_period,
            continuous=continuous,
            degree=degree,
        )

        # Restore original seq_len
        self.seq_len, self.effective_seq_len = (
            original_seq_len,
            original_effective_seq_len,
        )

        # Convert to numpy for tiling
        one_period = one_period.cpu().numpy()

        # Use tile operation to repeat the period
        x = np.tile(one_period, int(np.ceil(self.seq_len / period_length)))
        x = x[: self.seq_len]  # Trim to exact sequence length

        # name signal type based on parameters actually sampled
        signal_type = f"piecewise_spline_periodic_{num_periods}_{period_length}_s{segments_per_period}_c{continuous}_d{degree}"

        return torch.tensor(x, device=self.device), signal_type

    def normalize(self, x):
        return (x - torch.mean(x)) / (torch.std(x) + 1e-8)

    def sample_mixed_simple(self):
        """Generate features with mixed distributions."""
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(["fourier", "wiener", "arima"])
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            else:
                x, signal_type = self.sample_arima_series()
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_series(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "wiener",
                    "arima",
                    "ornstein_uhlenbeck",
                    "impulse_periodic",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "ornstein_uhlenbeck":
                x, signal_type = self.sample_ornstein_uhlenbeck()
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_subseries(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "fourier_frequency_domain",
                    "arima",
                    "wavelet",
                    "wiener",
                    "impulse",
                    "step",
                    "linear_trend",
                    "piecewise_splines_periodic",
                    "piecewise_linear_periodic",
                    "wavelet_periodic",
                    "box_periodic",
                    "wedge_periodic",
                    "impulse_periodic",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "wavelet":
                x, signal_type = self.sample_wavelet_function()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "impulse":
                x, signal_type = self.sample_impulse_function()
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            elif method == "step":
                x, signal_type = self.sample_step_function()
            elif method == "linear_trend":
                x, signal_type = self.sample_linear_trend()
            elif method == "piecewise_splines_periodic":
                x, signal_type = self.sample_piecewise_splines(periodic=True)
            elif method == "piecewise_linear_periodic":
                x, signal_type = self.sample_piecewise_linear_periodic()
            elif method == "wavelet_periodic":
                x, signal_type = self.sample_wavelet_function(periodic=True)
            elif method == "box_periodic":
                x, signal_type = self.sample_box_function(periodic=True)
            elif method == "wedge_periodic":
                x, signal_type = self.sample_wedge_function(periodic=True)
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            else:
                raise ValueError(f"Unsupported method: {method}")
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_fourier_both(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "fourier_frequency_domain",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_all(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "fourier_frequency_domain",
                    "wiener",
                    "arima",
                    "ornstein_uhlenbeck",
                    "box",
                    "wavelet",
                    "wedge",
                    "impulse",
                    "step",
                    "linear_trend",
                    "piecewise_splines",
                    "piecewise_linear_periodic",
                    "box_periodic",
                    "wavelet_periodic",
                    "wedge_periodic",
                    "impulse_periodic",
                    "piecewise_spline_periodic",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "ornstein_uhlenbeck":
                x, signal_type = self.sample_ornstein_uhlenbeck()
            elif method == "box":
                x, signal_type = self.sample_box_function()
            elif method == "box_periodic":
                x, signal_type = self.sample_box_function(periodic=True)
            elif method == "wavelet":
                x, signal_type = self.sample_wavelet_function()
            elif method == "wavelet_periodic":
                x, signal_type = self.sample_wavelet_function(periodic=True)
            elif method == "wedge":
                x, signal_type = self.sample_wedge_function()
            elif method == "wedge_periodic":
                x, signal_type = self.sample_wedge_function(periodic=True)
            elif method == "impulse":
                x, signal_type = self.sample_impulse_function()
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            elif method == "step":
                x, signal_type = self.sample_step_function()
            elif method == "linear_trend":
                x, signal_type = self.sample_linear_trend()
            elif method == "piecewise_splines":
                x, signal_type = self.sample_piecewise_splines()
            elif method == "piecewise_linear_periodic":
                x, signal_type = self.sample_piecewise_linear_periodic()
            elif method == "piecewise_spline_periodic":
                x, signal_type = self.sample_piecewise_spline_periodic()
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_names(self):
        X = []
        signal_types = []

        # Check if mixed_names is empty and fallback to uniform
        if len(self.mixed_names) == 0:
            logger.error("mixed_names is empty, falling back to uniform")
            # Use the parent class's sample_uniform method
            x_full, signal_type = self.sample_uniform()
            x = x_full[:, 0]  # Take only the first feature
            signal_type = "uniform"  # Return single signal type
            return x, signal_type

        for n in range(self.num_features):
            method = np.random.choice(
                self.mixed_names,
                p=self.sigmoid_weights,
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "box":
                x, signal_type = self.sample_box_function()
            elif method == "box_periodic":
                x, signal_type = self.sample_box_function(periodic=True)
            elif method == "wavelet":
                x, signal_type = self.sample_wavelet_function()
            elif method == "wavelet_periodic":
                x, signal_type = self.sample_wavelet_function(periodic=True)
            elif method == "wedge":
                x, signal_type = self.sample_wedge_function()
            elif method == "wedge_periodic":
                x, signal_type = self.sample_wedge_function(periodic=True)
            elif method == "impulse":
                x, signal_type = self.sample_impulse_function()
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            elif method == "step":
                x, signal_type = self.sample_step_function()
            elif method == "linear_trend":
                x, signal_type = self.sample_linear_trend()
            elif method == "piecewise_splines":
                x, signal_type = self.sample_piecewise_splines()
            elif method == "piecewise_linear_periodic":
                x, signal_type = self.sample_piecewise_linear_periodic()
            elif method == "normal":
                x, _ = self.sample_normal(n)
                signal_type = "normal"
            elif method == "uniform":
                # Extract only the first feature from the uniform sampler to maintain consistency
                x_full, signal_type = self.sample_uniform()
                x = x_full[:, 0]  # Take only the first feature
                signal_type = "uniform"  # Return single signal type
            else:
                # Handle unsupported methods by defaulting to uniform sampling
                logger.warning(
                    f"Unsupported method '{method}' in mixed_names, defaulting to uniform"
                )
                x_full, signal_type = self.sample_uniform()
                x = x_full[:, 0]  # Take only the first feature
                signal_type = "uniform"  # Return single signal type
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        try:
            X = torch.stack(X, -1).float()
        except:
            # logger.error(f"Error stacking X: {X}")
            for x, signal_type in zip(X, signal_types):
                logger.error(f"x.shape: {x.shape}, signal_type: {signal_type}")
            raise
        return X, signal_types

    def sample_mixed_subset(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "fourier_frequency_domain",
                    "wiener",
                    "arima",
                    "box",
                    "wavelet",
                    "wedge",
                    "impulse",
                    "step",
                    "linear_trend",
                    "piecewise_splines",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "box":
                x, signal_type = self.sample_box_function()
            elif method == "wavelet":
                x, signal_type = self.sample_wavelet_function()
            elif method == "wedge":
                x, signal_type = self.sample_wedge_function()
            elif method == "impulse":
                x, signal_type = self.sample_impulse_function()
            elif method == "step":
                x, signal_type = self.sample_step_function()
            elif method == "linear_trend":
                x, signal_type = self.sample_linear_trend()
            elif method == "piecewise_splines":
                x, signal_type = self.sample_piecewise_splines()
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_periodic(self):
        X = []
        signal_types = []
        for n in range(self.num_features):
            method = np.random.choice(
                [
                    "fourier",
                    "fourier_frequency_domain",
                    "arima"
                    # "wavelet",
                    # "wedge",
                    # "impulse",
                    "box_periodic",
                    "wavelet_periodic",
                    "wedge_periodic",
                    "impulse_periodic",
                    "piecewise_spline_periodic",
                ]
            )
            if method == "fourier":
                x, signal_type = self.sample_fourier_series()
            elif method == "fourier_frequency_domain":
                x, signal_type = self.sample_fourier_series_frequency_domain()
            elif method == "wavelet":
                x, signal_type = self.sample_wavelet_function(periodic=True)
            elif method == "wedge":
                x, signal_type = self.sample_wedge_function(periodic=True)
            elif method == "impulse":
                x, signal_type = self.sample_impulse_function(periodic=True)
            elif method == "box_periodic":
                x, signal_type = self.sample_box_function(periodic=True)
            elif method == "wavelet_periodic":
                x, signal_type = self.sample_wavelet_function(periodic=True)
            elif method == "wedge_periodic":
                x, signal_type = self.sample_wedge_function(periodic=True)
            elif method == "impulse_periodic":
                x, signal_type = self.sample_impulse_function(periodic=True)
            elif method == "piecewise_spline_periodic":
                x, signal_type = self.sample_piecewise_spline_periodic()
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types

    def sample_mixed_both(self):
        """Generate features with mixed distributions."""
        X = []
        signal_types = []
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
                x, signal_type = self.sample_fourier_series()
            elif method == "wiener":
                x, signal_type = self.sample_wiener_process()
            elif method == "arima":
                x, signal_type = self.sample_arima_series()
            elif method == "normal":
                x, _ = self.sample_normal(n)
                signal_type = "normal"
            elif method == "multinomial":
                x, signal_type = self.sample_multinomial()
            elif method == "zipf":
                x, signal_type = self.sample_zipf()
            else:  # default to uniform
                # Use the parent class method but extract only one feature for consistency
                x_full, signal_type = self.sample_uniform()
                x = x_full[:, 0]  # Take only the first feature
                signal_type = "uniform"  # Return single signal type
            x = self.normalize(x)
            X.append(x)
            signal_types.append(signal_type)
        return torch.stack(X, -1).float(), signal_types
