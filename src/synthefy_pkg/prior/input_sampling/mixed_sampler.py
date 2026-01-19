from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler
from synthefy_pkg.prior.input_sampling.real_ts_sampling import RealTSSampler
from synthefy_pkg.prior.input_sampling.stl_sampling import STLSampler
from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler


class MixedSampler(XSampler):
    """Comprehensive input sampler that combines all available sampling methods.

    This sampler can generate features using:
    - Synthetic signals (Fourier, ARIMA, Wiener, etc.)
    - Real time series data
    - STL decomposition components (trend, seasonal, residual)

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate
    num_features : int
        Number of features to generate
    real_data_loader : Optional[DataLoader], default=None
        DataLoader for real time series data
    pre_stats : bool, default=False
        Whether to pre-generate statistics for the input features
    sampling : str, default='mixed_all'
        Feature sampling strategy
    device : str, default='cpu'
        Device to store tensors on
    effective_seq_len : int, default=-1
        Effective sequence length for synthetic signals
    sampling_forms : List[str], default=None
        List of sampling forms to use. Options:
        - 'synthetic': Use TSSampler methods
        - 'real': Use RealTSSampler methods
        - 'stl': Use STLSampler methods
        - 'mixed': Combine all forms
    synthetic_methods : List[str], default=None
        List of specific synthetic methods to use. If None, uses all available.
    real_ratio : float, default=0.3
        Ratio of real data features when using mixed sampling
    stl_ratio : float, default=0.3
        Ratio of STL decomposition features when using mixed sampling
    synthetic_ratio : float, default=0.4
        Ratio of synthetic features when using mixed sampling
    use_random_ratios : bool, default=True
        Whether to use random ratios for each sampling call instead of fixed ratios
    ratio_variation : float, default=0.2
        Range of variation for random ratios (±ratio_variation around base ratios)
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        real_data_loader: Optional[DataLoader] = None,
        pre_stats: bool = False,
        sampling: str = "mixed_all",
        device: str = "cpu",
        effective_seq_len: int = -1,
        sampling_forms: Optional[List[str]] = None,
        real_ratio: float = 0.3,
        stl_ratio: float = 0.3,
        synthetic_ratio: float = 0.4,
        use_random_ratios: bool = True,
        ratio_variation: float = 0.2,
    ):
        super().__init__(seq_len, num_features, pre_stats, sampling, device)

        # Initialize sub-samplers
        self.ts_sampler = TSSampler(
            seq_len=seq_len,
            num_features=num_features,
            pre_stats=pre_stats,
            sampling=sampling,
            device=device,
            effective_seq_len=effective_seq_len,
        )

        self.real_ts_sampler = None
        self.stl_sampler = None

        if real_data_loader is not None:
            self.real_ts_sampler = RealTSSampler(
                seq_len=seq_len,
                num_features=num_features,
                real_data_loader=real_data_loader,
                pre_stats=pre_stats,
                sampling=sampling,
                device=device,
                effective_seq_len=effective_seq_len,
            )

            self.stl_sampler = STLSampler(
                seq_len=seq_len,
                num_features=num_features,
                pre_stats=pre_stats,
                sampling=sampling,
                device=device,
                effective_seq_len=effective_seq_len,
            )

        # Sampling configuration
        self.sampling_forms = (
            sampling_forms
            if sampling_forms is not None
            else ["synthetic", "real", "stl"]
        )
        self.real_ratio = real_ratio
        self.stl_ratio = stl_ratio
        self.synthetic_ratio = synthetic_ratio

        # Store the base ratios for random selection ranges
        self.base_real_ratio = real_ratio
        self.base_stl_ratio = stl_ratio
        self.base_synthetic_ratio = synthetic_ratio
        self.use_random_ratios = use_random_ratios
        self.ratio_variation = ratio_variation

        # Validate base ratios sum to 1.0
        total_ratio = real_ratio + stl_ratio + synthetic_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Base ratios must sum to 1.0, got {total_ratio}")

        # Ensure we have the required samplers for requested forms
        if "real" in self.sampling_forms and self.real_ts_sampler is None:
            raise ValueError(
                "real_data_loader required when using 'real' sampling form"
            )
        if "stl" in self.sampling_forms and self.stl_sampler is None:
            raise ValueError(
                "real_data_loader required when using 'stl' sampling form"
            )

    def _generate_random_ratios(self) -> Tuple[float, float, float]:
        """Generate random ratios for each component within a range around the base ratios.

        Returns
        -------
        Tuple[float, float, float]
            Random ratios for (real, stl, synthetic) that sum to 1.0
        """
        # Generate random ratios within the variation range
        real_ratio = np.random.uniform(
            max(0, self.base_real_ratio - self.ratio_variation),
            min(1, self.base_real_ratio + self.ratio_variation),
        )
        stl_ratio = np.random.uniform(
            max(0, self.base_stl_ratio - self.ratio_variation),
            min(1, self.base_stl_ratio + self.ratio_variation),
        )
        synthetic_ratio = np.random.uniform(
            max(0, self.base_synthetic_ratio - self.ratio_variation),
            min(1, self.base_synthetic_ratio + self.ratio_variation),
        )

        # Normalize to sum to 1.0
        total = real_ratio + stl_ratio + synthetic_ratio
        real_ratio /= total
        stl_ratio /= total
        synthetic_ratio /= total

        return real_ratio, stl_ratio, synthetic_ratio

    def _get_ratios(self) -> Tuple[float, float, float]:
        """Get ratios for sampling, either random or fixed based on configuration.

        Returns
        -------
        Tuple[float, float, float]
            Ratios for (real, stl, synthetic) that sum to 1.0
        """
        if self.use_random_ratios:
            return self._generate_random_ratios()
        else:
            return (
                self.base_real_ratio,
                self.base_stl_ratio,
                self.base_synthetic_ratio,
            )

    def sample(
        self, return_numpy: bool = False, return_signal_types: bool = False
    ):
        """Generate features using the specified sampling strategy."""

        # Get ratios for this sampling call (random or fixed)
        real_ratio, stl_ratio, synthetic_ratio = self._get_ratios()

        # Determine the number of features for each form using the ratios
        n_real = int(self.num_features * real_ratio)
        n_stl = int(self.num_features * stl_ratio)
        n_synthetic = self.num_features - n_real - n_stl

        # Initialize variables
        synthetic_samples = None
        synthetic_signal_types = None
        real_samples = None
        real_signal_types = None
        real_batch = None
        stl_samples = None
        stl_signal_types = None

        # then sample the features for each form
        if n_synthetic > 0:
            self.ts_sampler.num_features = n_synthetic
            synthetic_result = self.ts_sampler.sample(
                return_numpy, return_signal_types
            )
            if return_signal_types:
                synthetic_samples, synthetic_signal_types = synthetic_result
            else:
                synthetic_samples = synthetic_result

        if self.real_ts_sampler is not None and n_real > 0:
            self.real_ts_sampler.num_features = n_real
            real_sample_list = self.real_ts_sampler.sample(
                return_numpy, return_signal_types, return_precomputed_batch=True
            )
            real_samples = real_sample_list[0]
            real_signal_types = (
                real_sample_list[1] if return_signal_types else None
            )
            real_batch = (
                real_sample_list[2]
                if return_signal_types
                else real_sample_list[1]
            )

        if self.stl_sampler is not None and n_stl > 0:
            self.stl_sampler.num_features = n_stl
            stl_samples = self.stl_sampler.sample(
                return_numpy, return_signal_types, precomputed_batch=real_batch
            )
            stl_signal_types = None
            if return_signal_types:
                stl_signal_types = stl_samples[1]
                stl_samples = stl_samples[0]
            else:
                stl_samples = stl_samples[0]

        # Collect samples and signal types, only if they exist
        samples_to_concat = []
        signal_types_to_concat = []

        if synthetic_samples is not None:
            samples_to_concat.append(synthetic_samples)
            if return_signal_types and synthetic_signal_types is not None:
                signal_types_to_concat.append(synthetic_signal_types)

        if real_samples is not None:
            samples_to_concat.append(real_samples)
            if return_signal_types and real_signal_types is not None:
                signal_types_to_concat.append(real_signal_types)

        if stl_samples is not None:
            samples_to_concat.append(stl_samples)
            if return_signal_types and stl_signal_types is not None:
                signal_types_to_concat.append(stl_signal_types)

        if return_signal_types:
            # Filter out None values from signal types
            valid_signal_types = [
                st for st in signal_types_to_concat if st is not None
            ]
            # add an assertion to make sure that all the values are tensors
            assert all(
                isinstance(sample, torch.Tensor) for sample in samples_to_concat
            ), "All samples must be tensors"
            return torch.cat(samples_to_concat, dim=0), sum(
                valid_signal_types, []
            )  # type: ignore
        else:
            return torch.cat(samples_to_concat, dim=0)  # type: ignore
