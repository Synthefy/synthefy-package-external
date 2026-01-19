from typing import Dict, Tuple

import numpy as np
import torch

from synthefy_pkg.model.cltsp.cltsp_utils import (
    ConditionEncoder,
    TimeSeriesEncoder,
)


class CLTSP_v3(torch.nn.Module):
    def __init__(
        self,
        cltsp_config,
    ):
        super(CLTSP_v3, self).__init__()
        self.cltsp_config = cltsp_config
        self.dataset_config = cltsp_config.dataset_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.timeseries_encoder = TimeSeriesEncoder(
            cltsp_config=self.cltsp_config.encoder_config,
            dataset_config=self.dataset_config,
            device=self.device,
        )
        self.condition_encoder = ConditionEncoder(
            cltsp_config=self.cltsp_config.encoder_config,
            dataset_config=self.dataset_config,
            device=self.device,
        )

    def forward(self, input) -> Tuple[torch.Tensor, torch.Tensor]:
        timeseries_input = input["timeseries_input"]
        discrete_condition_input = input["discrete_condition_input"]
        continuous_condition_input = input["continuous_condition_input"]

        # print("within forward function of CLTSP_v1")
        # print(
        #     timeseries_input.shape,
        #     discrete_condition_input.shape,
        #     continuous_condition_input.shape,
        # )

        timeseries_embedding = self.timeseries_encoder(timeseries_input)
        condition_embedding = self.condition_encoder(
            discrete_condition_input, continuous_condition_input
        )

        return timeseries_embedding, condition_embedding

    def prepare_training_input(self, train_batch: Dict[str, torch.Tensor]):
        # timeseries
        timeseries_full = train_batch["timeseries_full"].float().to(self.device)
        timeseries_full = timeseries_full.permute(0, 2, 1)
        assert timeseries_full.shape[2] == self.dataset_config.num_channels, (
            "The number of input features is not correct"
        )

        actual_horizon = self.dataset_config.time_series_length
        required_horizon = self.dataset_config.required_time_series_length
        assert required_horizon < actual_horizon, (
            "required_horizon must be less than actual_horizon - num_positive_samples"
        )

        # discrete label embedding
        discrete_label_embedding = (
            train_batch["discrete_label_embedding"].float().to(self.device)
        )
        if len(discrete_label_embedding.shape) == 2:
            # we will enter here only if we have constant discrete labels
            # so we repeat the discrete label embedding along the time dimension, that is the first dimension
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(
                1, actual_horizon, 1
            )
            assert torch.all(
                discrete_label_embedding[:, 0, :]
                == discrete_label_embedding[:, 1, :]
            ), "discrete label embedding is not constant"

        assert discrete_label_embedding.shape[1] == actual_horizon, (
            "Wrong shape"
        )
        assert (
            discrete_label_embedding.shape[2]
            == self.dataset_config.num_discrete_conditions
        ), "The number of discrete labels is not correct"

        # continuous label embedding
        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )
        if continuous_label_embedding.shape[-1] == 0:
            label_embedding = discrete_label_embedding
        else:
            assert continuous_label_embedding.shape[1] == actual_horizon, (
                "Wrong shape"
            )
            assert (
                continuous_label_embedding.shape[2]
                == self.dataset_config.num_continuous_labels
            ), "The number of continuous labels is not correct"
            label_embedding = torch.cat(
                (discrete_label_embedding, continuous_label_embedding), dim=-1
            )
        _, labels = torch.unique(label_embedding, dim=0, return_inverse=True)
        labels = labels.detach()
        # print(torch.min(labels), torch.max(labels))

        # obtaining random patches
        timeseries_list = []
        discrete_label_embedding_list = []
        continuous_label_embedding_list = []
        num_positive_samples = (
            self.cltsp_config.encoder_config.num_positive_samples
        )
        random_indices = np.random.randint(
            0, actual_horizon - required_horizon, size=num_positive_samples
        )
        for index in random_indices:
            timeseries_list.append(
                timeseries_full[:, index : index + required_horizon]
            )
            discrete_label_embedding_list.append(
                discrete_label_embedding[:, index : index + required_horizon]
            )
            continuous_label_embedding_list.append(
                continuous_label_embedding[:, index : index + required_horizon]
            )
        timeseries_tensor = torch.cat(timeseries_list, dim=0)
        discrete_label_embeddings_tensor = torch.cat(
            discrete_label_embedding_list, dim=0
        )
        continuous_label_embeddings_tensor = torch.cat(
            continuous_label_embedding_list, dim=0
        )
        bs = timeseries_full.shape[0]
        assert torch.all(
            timeseries_tensor[:bs]
            == timeseries_full[
                :, random_indices[0] : random_indices[0] + required_horizon
            ]
        )

        input = {
            "timeseries_input": timeseries_tensor,
            "discrete_condition_input": discrete_label_embeddings_tensor,
            "continuous_condition_input": continuous_label_embeddings_tensor,
            "labels": labels,
        }
        return input

    def get_timeseries_embedding(self, timeseries):
        return self.timeseries_encoder(timeseries)

    def get_condition_embedding(self, discrete_condition, continuous_condition):
        return self.condition_encoder(discrete_condition, continuous_condition)
