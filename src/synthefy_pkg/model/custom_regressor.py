import os
import pickle
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from synthefy_pkg.model.architectures.resnet1d import resnet1d50, resnet1d101
from synthefy_pkg.utils.tstr_utils import extract_col_indices

COMPILE = False


def get_regression_model(model_name: str) -> Callable:
    if model_name == "resnet1d50":
        return resnet1d50
    elif model_name == "resnet1d101":
        return resnet1d101
    else:
        raise ValueError("Unknown model name")


class TimeSeriesRegressor(torch.nn.Module):
    def __init__(
        self,
        regressor_config: DictConfig,
        dataset_config: DictConfig,
        device: str,
    ):
        super(TimeSeriesRegressor, self).__init__()
        self.dataset_config = dataset_config
        self.regressor_config = regressor_config
        self.device = device

        regression_model = get_regression_model(
            self.regressor_config.model_name
        )

        self.index_of_interest = self.dataset_config["regression_index"]
        self.continuous_input_indices = self.dataset_config.get(
            "continuous_input_channels", None
        )
        self.discrete_input_indices = self.dataset_config.get(
            "discrete_input_channels", None
        )
        if self.continuous_input_indices is None:
            self.continuous_input_indices = torch.tensor(
                extract_col_indices(
                    dataset_config,
                    self.dataset_config.get(
                        "regression_continuous_input_cols", []
                    ),
                    "continuous",
                )
            )
        if self.discrete_input_indices is None:
            self.discrete_input_indices = torch.tensor(
                extract_col_indices(
                    dataset_config,
                    self.dataset_config.get(
                        "regression_original_discrete_input_cols", []
                    ),
                    "discrete",
                )
            )

        # don't allow the target index to be in the continuous or discrete input indices
        if self.index_of_interest in self.continuous_input_indices:
            raise ValueError(
                f"Target index (index_of_interest: {self.index_of_interest}) is in continuous input indices: {self.continuous_input_indices}"
            )

        if self.dataset_config.use_condition_input:
            self.regressor = regression_model(
                num_classes=1,
                input_channels=self.dataset_config.num_channels
                + len(self.discrete_input_indices)
                + len(self.continuous_input_indices),
            )
        else:
            self.regressor = regression_model(
                num_classes=1,
                input_channels=self.dataset_config.num_channels,
            )

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        timeseries = input_dict["timeseries"]
        return self.regressor(timeseries)

    def prepare_training_input(
        self, train_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Assert no overlap between index_of_interest and continuous input indices
        assert self.index_of_interest not in self.continuous_input_indices, (
            f"Target index (index_of_interest: {self.index_of_interest}) cannot be in continuous input indices: {self.continuous_input_indices}"
        )
        timeseries_full = train_batch["timeseries_full"].float().to(self.device)
        assert timeseries_full.shape[1] == self.dataset_config.num_channels, (
            "The number of input features is not correct"
        )
        assert (
            timeseries_full.shape[2] == self.dataset_config.time_series_length
        ), "The length of the time series is not correct"

        actual_horizon = self.dataset_config.time_series_length
        required_horizon = self.dataset_config.required_time_series_length
        assert required_horizon <= actual_horizon, (
            "required_horizon > actual_horizon"
        )

        input_dict = {}
        # input_dict["timeseries"] = timeseries_full

        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )
        continuous_label_embedding = continuous_label_embedding.permute(
            0, 2, 1
        )  # B x C x H
        # create input_continuous_label_embedding with every channel except the channel of interest
        input_continuous_label_embedding = continuous_label_embedding[
            :, self.continuous_input_indices
        ]  # pick only the continuous input indices

        # target = continuous_label_embedding[:, self.index_of_interest].unsqueeze(1)

        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )  #
        if len(discrete_label_embedding.shape) != 3:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(
                1
            ).repeat(1, actual_horizon, 1)

        discrete_label_embedding = discrete_label_embedding.permute(
            0, 2, 1
        )  # B x C x H
        input_discrete_label_embedding = discrete_label_embedding[
            :, self.discrete_input_indices
        ]  # pick only the discrete input indices

        if self.dataset_config.use_condition_input:
            inp_ = torch.cat(
                (
                    timeseries_full,
                    input_continuous_label_embedding,
                    input_discrete_label_embedding,
                ),
                dim=1,
            )
        else:
            inp_ = timeseries_full

        target_channel = continuous_label_embedding[:, self.index_of_interest]
        assert (
            target_channel.shape[1] == self.dataset_config.time_series_length
        ), "The target channel should have only one channel"
        target = torch.max(target_channel, dim=1)[0].unsqueeze(1)

        input_dict["timeseries"] = inp_
        input_dict["target"] = target

        return input_dict

    def get_output(self, timeseries):
        return self.regressor(timeseries)
