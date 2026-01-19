from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from synthefy_pkg.prior.input_sampling.basic_sampling import XSampler


class RealTSSampler(XSampler):
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

    def sample(
        self,
        return_numpy=False,
        return_signal_types=False,
        return_precomputed_batch=False,
    ):
        # Ensure real_data_loader is not None
        assert self.real_data_loader is not None, (
            "real_data_loader must be provided"
        )

        # Get a batch from the dataloader (it's already a DataLoader)
        batch = next(iter(self.real_data_loader))
        window = batch["timeseries"].to(self.device).float()
        timeseries = window[..., 9011:9795]

        updated_timeseries_list = []
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
                updated_timeseries_list.append(
                    torch.from_numpy(interpolated_values)
                    .to(self.device)
                    .float()
                )

        updated_timeseries = torch.stack(updated_timeseries_list)
        assert not torch.any(updated_timeseries.isnan()), (
            "nan found in updated_timeseries"
        )

        # randomly pick self.num_features features from len(updated_timeseries)
        chosen_features = np.random.choice(
            len(updated_timeseries), self.num_features, replace=False
        )
        chosen_features = np.atleast_1d(chosen_features)
        chosen_timeseries = []
        for feature in chosen_features:
            chosen_timeseries.append(updated_timeseries[feature])
        chosen_timeseries = np.stack(chosen_timeseries, axis=-1)
        result = torch.from_numpy(chosen_timeseries).to(self.device).float()

        if return_numpy:
            result = result.cpu().numpy()
        else:
            result = result
        return_result = result
        if return_signal_types:
            return_result = [result, chosen_features]
            if return_precomputed_batch:
                return_result = [result, chosen_features, batch]
        else:
            if return_precomputed_batch:
                return_result = [result, batch]
        return return_result
