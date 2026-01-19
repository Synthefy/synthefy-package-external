import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess
from torch import nn

from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler
from synthefy_pkg.prior.utils import GaussianNoise


class TSNoise(nn.Module):
    """Time series based noise layer that uses TSSampler to generate structured noise.

    This layer adds time series-based noise to the input tensor, where the noise
    is generated using various time series patterns like Fourier series, Wiener process,
    ARIMA, etc.

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate for noise

    sampling : str, default='mixed_simple'
        The type of time series to generate. Options include:
        - 'fourier': Fourier series
        - 'wiener': Wiener process
        - 'arima': ARIMA process
        - 'mixed_simple': Mix of basic time series
        - 'mixed_all': Mix of all available time series types
        - 'mixed_subset': Mix of a subset of time series types
        - 'mixed_periodic': Mix of periodic time series
        - 'mixed_both': Mix of time series and basic distributions

    device : str, default='cpu'
        Device to store tensors on

    std : float, default=0.01
        Standard deviation multiplier for the generated noise.
    """

    def __init__(
        self, seq_len, sampling="mixed_simple", device="cpu", std=0.01
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sampling = sampling
        self.device = device
        self.std = std

        self.ts_sampler = TSSampler(
            seq_len=seq_len,
            num_features=1,  # We only need one feature for noise
            sampling=sampling,
            device=device,
        )

    def forward(self, X, return_components=False):
        """Add time series based noise to the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of any shape

        Returns
        -------
        torch.Tensor
            Input tensor with added time series noise
        """
        # Generate noise using TSSampler
        noise, _ = self.ts_sampler.sample(return_signal_types=True)

        # Reshape noise to match input dimensions
        if len(X.shape) > 1:
            noise = noise.expand(*X.shape)

        if return_components:
            return X, noise * self.std
        else:
            return X + noise * self.std


class MixedTSNoise(nn.Module):
    def __init__(
        self,
        seq_len,
        sampling="mixed_simple",
        device="cpu",
        ts_std=0.01,
        gaussian_std=0.01,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sampling = sampling
        self.device = device
        self.ts_std = ts_std
        self.gaussian_std = gaussian_std
        self.ts_noise = TSNoise(seq_len, sampling, device, ts_std)
        self.gaussian_noise = GaussianNoise(
            gaussian_std
        )  # TODO: the only time independent noise is Gaussian at the moment, but we could vary this

    def forward(self, X, return_components=False):
        """Add time series based noise to the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of any shape

        Returns
        -------
        torch.Tensor
            Input tensor with added time series noise
        """
        X, ts_noise = self.ts_noise(X, return_components=True)
        X, gaussian_noise = self.gaussian_noise(X, return_components=True)
        if return_components:
            return X, ts_noise, gaussian_noise
        else:
            return X + ts_noise + gaussian_noise
