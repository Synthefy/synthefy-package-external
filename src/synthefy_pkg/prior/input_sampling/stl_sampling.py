from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from synthefy_pkg.preprocessing.data_summary_utils.stl_decomposition import (
    perform_stl_decomposition,
)
from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler


class STLSampler(XSampler):
    """Input sampler for using real time series features for prior datasets.
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
        real_data_loader: Optional[DataLoader] = None,
        pre_stats=False,
        sampling="mixed",
        device="cpu",
        effective_seq_len=-1,
        sampling_forms=["synthetic", "real", "stl"],
    ):
        super().__init__(seq_len, num_features, pre_stats, sampling, device)
        self.real_data_loader = real_data_loader
        if self.real_data_loader is None:
            raise ValueError("real_data_loader must be provided")

        # Add missing attributes inherited from TSSampler
        self.warmup_samples = 10
        self.max_freqs = 10
        self.effective_seq_len = seq_len
        if effective_seq_len > 0:
            self.effective_seq_len = effective_seq_len
        # extract the continuous_start_idx and continuous_end_idx from the real_data_loader
        try:
            self.continuous_start_idx = (
                self.real_data_loader.dataset.continuous_start_idx  # type: ignore
            )
            self.continuous_end_idx = (
                self.real_data_loader.dataset.continuous_end_idx  # type: ignore
            )
        except AttributeError:
            # Fallback values if attributes don't exist
            self.continuous_start_idx = 0
            self.continuous_end_idx = self.num_features

        self.sampling_forms = sampling_forms

    def sample(
        self,
        return_numpy=False,
        return_signal_types=False,
        precomputed_batch=None,
    ):
        # Ensure real_data_loader is not None
        assert self.real_data_loader is not None, (
            "real_data_loader must be provided"
        )

        # Get a batch from the dataloader (it's already a DataLoader)
        if precomputed_batch is None:
            batch = next(iter(self.real_data_loader))
        else:
            batch = precomputed_batch
        window = batch["timeseries"].to(self.device).float()
        timeseries = window[..., 9011:9795]

        updated_trend = []
        updated_seasonal = []
        updated_residual = []
        for j in range(timeseries.shape[0]):
            for k in range(timeseries.shape[1]):
                sample = timeseries[j, k].cpu().numpy()
                valid_indices = np.where(~np.isnan(sample))[0]
                valid_values = sample[valid_indices]
                interpolation_indices = np.linspace(
                    0, self.seq_len - 1, len(valid_indices)
                )
                if len(valid_indices) == 0:
                    interpolated_values = np.zeros(self.seq_len)
                else:
                    interpolated_values = np.interp(
                        np.arange(self.seq_len),
                        interpolation_indices,
                        valid_values,
                    )

                stl_decomp = perform_stl_decomposition(
                    torch.from_numpy(interpolated_values)
                    .to(self.device)
                    .float(),
                )
                updated_trend.append(
                    torch.from_numpy(stl_decomp["trend"])
                    .to(self.device)
                    .float()
                )
                updated_seasonal.append(
                    torch.from_numpy(stl_decomp["seasonal"])
                    .to(self.device)
                    .float()
                )
                updated_residual.append(
                    torch.from_numpy(stl_decomp["resid"])
                    .to(self.device)
                    .float()
                )

        updated_trend = torch.stack(updated_trend)
        updated_seasonal = torch.stack(updated_seasonal)
        updated_residual = torch.stack(updated_residual)

        assert not torch.any(updated_trend.isnan()), (
            "nan found in updated_trend"
        )
        assert not torch.any(updated_seasonal.isnan()), (
            "nan found in updated_seasonal"
        )
        assert not torch.any(updated_residual.isnan()), (
            "nan found in updated_residual"
        )
        # randomly pick self.num_features features from len(updated_timeseries)

        chosen_features = np.random.choice(
            len(updated_trend),
            np.ceil(self.num_features / 3).astype(int),
            replace=False,
        )
        chosen_features = np.atleast_1d(chosen_features)
        chosen_timeseries = []
        for feature in chosen_features:
            chosen_timeseries.append(updated_trend[feature])
        for feature in chosen_features:
            chosen_timeseries.append(updated_seasonal[feature])
        for feature in chosen_features:
            chosen_timeseries.append(updated_residual[feature])
        chosen_timeseries = torch.stack(chosen_timeseries, dim=-1)[
            ..., : self.num_features
        ]
        result = chosen_timeseries.to(self.device).float()

        if return_numpy:
            result = result.cpu().numpy()
        else:
            result = result
        if return_signal_types:
            return result, chosen_features
        return result
