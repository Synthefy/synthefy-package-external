from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nested import nested_tensor

from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.prior.dataset import PriorDataset, add_synthetic_timestamps
from synthefy_pkg.prior.mlp_scm import MLPSCM
from synthefy_pkg.prior.observation import Obs
from synthefy_pkg.prior.tree_scm import TreeSCM
from synthefy_pkg.utils.time_and_freq_utils import add_synthetic_time_features

config_path = "configs/tabicl/prior/prior_config.yaml"


def generate_scm_object(config_path: Union[str, TabICLPriorConfig], n_features):
    if isinstance(config_path, str):
        config = TabICLPriorConfig.from_yaml(yaml_path=config_path)
    else:
        config = config_path
    config.n_features = n_features

    single_dataset = PriorDataset(config=config)
    # Generate data to get the DAG
    result = single_dataset.prior.get_batch(
        batch_size=1,
        probing=True,
    )
    assert isinstance(result, list)
    X, y, d, scm_object, params, indices_X, indices_y = result[0]

    indices_X_y = (indices_X, indices_y)
    return single_dataset, scm_object, params, indices_X_y


def collect_inputs(
    scm_object: MLPSCM,
    num_inputs_to_collect: int,
):
    actual_sequence_length, scm_object.xsampler.seq_len = (
        scm_object.xsampler.seq_len,
        num_inputs_to_collect,
    )
    fixed_inputs, sample_types = scm_object.xsampler.sample(
        return_signal_types=True
    )
    print("sample_types", sample_types, fixed_inputs.shape)
    # create a plot of the inputs just to check
    import matplotlib.pyplot as plt

    plt.plot(fixed_inputs[:10000, 0])
    plt.savefig("fixed_inputs.png")
    plt.close()
    scm_object.xsampler.seq_len = actual_sequence_length
    return fixed_inputs, sample_types


def get_single_dataset(
    config: TabICLPriorConfig,
    batch_size: int,
    dataset_object: PriorDataset,
    scm_object: MLPSCM,
    params: Dict[str, Any],
    indices_X_y: Tuple[Tensor, Tensor],
    fixed_inputs: Tensor,
):
    X_list, y_list, d_list = [], [], []
    input_start_idxs = np.random.randint(
        0, fixed_inputs.shape[0] - scm_object.seq_len, size=(batch_size,)
    )
    for i in range(batch_size):
        # input_start_idx = np.random.randint(0, fixed_inputs.shape[0] - scm_object.seq_len)
        # input_start_idx = 0
        assert isinstance(fixed_inputs, Tensor)
        X, y, d, _, _ = get_single_dataset_from_scm(
            config,
            scm_object,
            params,
            indices_X_y,
            assigned_inputs=fixed_inputs[
                input_start_idxs[i] : input_start_idxs[i] + scm_object.seq_len
            ],
        )
        X_list.append(X)
        y_list.append(y)
        d_list.append(d)

    X, y, d, seq_lens, train_sizes = post_dataset_modification(
        config,
        dataset_object,
        params,
        X_list,
        y_list,
        d_list,
        input_start_idxs,
        use_freqs=["minutely"] * batch_size,
    )

    return X, y, d, seq_lens, train_sizes


def get_single_dataset_from_scm(
    config: TabICLPriorConfig,
    prior_object: Union[MLPSCM, TreeSCM],
    params: Dict[str, Any],
    indices_X_y: Tuple[Tensor, Tensor],
    assigned_inputs: Tensor,
    input_start_idx: int = 0,
):
    # Always get indices since we might need them
    X, y, indices_X, indices_y = prior_object(
        return_indices=True,
        exclude_inputs=config.exclude_inputs,
        indices_X_y=indices_X_y,
        assigned_inputs=assigned_inputs,
    )

    X, y = Obs(params)(X, y)

    assert "original_seq_len" in params, "params must contain original_seq_len"
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == params["max_features"], (
        f"X.shape[1]: {X.shape[1]}, params['max_features']: {params['max_features']}"
    )
    assert X.shape[0] == params["original_seq_len"]

    # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
    d = torch.tensor(
        [params["num_features"]],
        device=config.prior_device,
        dtype=torch.long,
    )

    # Only keep valid datasets with sufficient features and balanced classes

    return X, y, d, indices_X, indices_y


def post_dataset_modification(
    config: TabICLPriorConfig,
    dataset_object: PriorDataset,
    params: Dict[str, Any],
    X_list: List[Tensor],
    y_list: List[Tensor],
    d_list: List[Tensor],
    input_start_idxs: Optional[Union[List[int], np.ndarray]] = None,
    use_freqs: Optional[Union[List[str], np.ndarray]] = None,
):
    # Combine Results
    if config.seq_len_per_gp:
        # Use nested tensors for variable sequence lengths
        X = nested_tensor(
            [x.to(config.prior_device) for x in X_list],
            device=config.prior_device,
        )
        y = nested_tensor(
            [y.to(config.prior_device) for y in y_list],
            device=config.prior_device,
        )
    else:
        # Stack into regular tensors for fixed sequence length
        X = torch.stack(X_list).to(config.prior_device)  # (B, T, H)
        y = torch.stack(y_list).to(config.prior_device)  # (B, T)

    if (
        len(config.add_synthetic_timestamps) > 0
        and params["dataset_has_timestamp"]
    ):
        if config.add_time_stamps_as_features:
            X, d_list, _, _, frequencies, start_times = (
                add_synthetic_time_features(
                    X,
                    d_list,
                    config.add_synthetic_timestamps,
                    config.max_features,
                    series_flags=torch.ones(
                        X.shape[0], dtype=torch.bool, device=X.device
                    ),
                    frequencies=use_freqs,
                    start_times=[i * 60 for i in input_start_idxs]
                    if input_start_idxs is not None
                    else None,
                )
            )
        else:
            X, d_list, frequencies, start_times, num_added_features = (
                add_synthetic_timestamps(
                    X,
                    d_list,
                    config.add_synthetic_timestamps,
                    config.max_seq_len,
                    config.max_features,
                    frequencies=use_freqs,
                    start_times=[i * 60 for i in input_start_idxs]
                    if input_start_idxs is not None
                    else None,
                )
            )
        # y = self.add_synthetic_timestamps(y)

    # Metadata (always regular tensors)
    d = torch.stack(d_list).to(
        config.prior_device
    )  # Actual number of features after filtering out constant ones
    # seq_lens = torch.tensor(
    #     [params["seq_len"] for params in param_list],
    #     device=self.config.prior_device,
    #     dtype=torch.long,
    # )
    batch_size = len(X_list)
    seq_lens = torch.tensor(
        [params["original_seq_len"] for _ in range(batch_size)],
        device=config.prior_device,
        dtype=torch.long,
    )
    train_sizes = torch.tensor(
        [params["train_size"] for _ in range(batch_size)],
        device=config.prior_device,
        dtype=torch.long,
    )
    return X, y, d, seq_lens, train_sizes
