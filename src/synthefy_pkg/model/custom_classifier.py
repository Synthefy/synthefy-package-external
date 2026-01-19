from typing import Dict

import numpy as np
import torch

from synthefy_pkg.model.architectures.resnet1d import resnet1d50, resnet1d101
from synthefy_pkg.utils.tstr_utils import extract_col_indices

COMPILE = False


def get_classification_model(model_name):
    """
    Retrieve the classification model based on the model name.

    Args:
        model_name (str): The name of the model to retrieve. Options are 'resnet1d50' or 'resnet1d101'.

    Returns:
        torch.nn.Module: The corresponding ResNet1D model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == "resnet1d50":
        return resnet1d50
    elif model_name == "resnet1d101":
        return resnet1d101
    else:
        raise ValueError("Unknown model name")


class TimeSeriesClassifier(torch.nn.Module):
    """
    A time series classifier using a ResNet1D architecture.

    Attributes:
        dataset_config (DictConfig): Configuration for the dataset.
        classifier_config (DictConfig): Configuration for the classifier.
        device (str): The device to run the model on ('cpu' or 'cuda').
        indices (torch.Tensor): Indices for classification from discrete windows array.
        classifier (torch.nn.Module): The classification model.
    """

    def __init__(self, classifier_config, dataset_config, device):
        """
        Initialize the TimeSeriesClassifier.

        Args:
            classifier_config (DictConfig): Configuration for the classifier.
            dataset_config (DictConfig): Configuration for the dataset.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        super(TimeSeriesClassifier, self).__init__()
        self.dataset_config = dataset_config
        self.classifier_config = classifier_config
        self.device = (
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        self.indices = torch.tensor(self.dataset_config.classification_indices)
        self.continuous_input_indices = extract_col_indices(
            dataset_config,
            self.dataset_config.get("classification_continuous_input_cols", []),
            "continuous",
        )
        self.discrete_input_indices = extract_col_indices(
            dataset_config,
            self.dataset_config.get(
                "classification_original_discrete_input_cols", []
            ),
            "discrete",
        )
        classification_model = get_classification_model(
            self.classifier_config.model_name
        )

        if self.dataset_config.classification_use_condition_input:
            self.classifier = classification_model(
                num_classes=self.indices.shape[0],
                input_channels=self.dataset_config.num_channels
                + len(self.discrete_input_indices)
                + len(self.continuous_input_indices),
            )
        else:
            self.classifier = classification_model(
                num_classes=self.indices.shape[0],
                input_channels=self.dataset_config.num_channels,
            )

        # Force all parameters to float32
        for param in self.classifier.parameters():
            param.data = param.data.to(torch.float32)

        # Move model to device and ensure float32
        self.classifier = self.classifier.to(self.device).to(torch.float32)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Args:
            input_dict (Dict[str, torch.Tensor]): A dictionary containing the input tensor with key 'timeseries'.

        Returns:
            torch.tensor: The output of the classifier.
        """
        # Keep everything on the same device (GPU)
        timeseries = input_dict["timeseries"].to(self.device).to(torch.float32)

        # Perform forward pass
        output = self.classifier(timeseries)

        return output

    def prepare_training_input(self, train_batch):
        """
        Prepare the input data for training.

        Args:
            train_batch (dict): A batch of training data.

        Returns:
            dict: A dictionary containing the prepared input data and labels.
        """
        # Assert no overlap between classification indices and discrete input indices
        assert not any(
            idx in self.discrete_input_indices for idx in self.indices
        ), "Classification indices and discrete input indices must not overlap"
        timeseries_full = (
            train_batch["timeseries_full"]
            .to(dtype=torch.float32)
            .to(self.device)
        )
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
        continuous_label_embedding = (
            train_batch["continuous_label_embedding"].float().to(self.device)
        )
        continuous_label_embedding = continuous_label_embedding.permute(
            0, 2, 1
        )  # B x C x W
        input_continuous_label_embedding = continuous_label_embedding[
            :, self.continuous_input_indices
        ]  # pick only the continuous input indices
        if self.dataset_config.use_reduced_horizon:
            random_index = np.random.randint(
                0, actual_horizon - required_horizon
            )
            timeseries_full = timeseries_full[
                :, :, random_index : random_index + required_horizon
            ]
        else:
            timeseries_full = timeseries_full

        discrete_label_embedding = (
            train_batch["discrete_label_embedding"]
            .to(dtype=torch.float32)
            .to(self.device)
        )
        # We can handle 2D or 3D with Ellipsis `...` in indices
        input_discrete_label_embedding = discrete_label_embedding[
            ..., self.discrete_input_indices
        ]  # pick only the discrete input indices
        if len(discrete_label_embedding.shape) == 2:
            window_size = timeseries_full.shape[2]
            input_discrete_label_embedding = (
                input_discrete_label_embedding.unsqueeze(-1).expand(
                    -1, -1, window_size
                )
            )  # B x C x W
            classification_label = discrete_label_embedding[:, self.indices]
        else:
            input_discrete_label_embedding = (
                input_discrete_label_embedding.permute(0, 2, 1)
            )  # B x C x W

            # Check if there are multiple labels in the discrete label embedding
            if self.dataset_config.multi_label:
                classification_logits = discrete_label_embedding[
                    :, :, self.indices
                ]
                assert classification_logits.shape[1] == actual_horizon
                assert classification_logits.shape[2] == self.indices.shape[0]
                classification_logits = torch.sum(classification_logits, dim=1)
                assert classification_logits.shape[1] == self.indices.shape[0]
                classification_label = torch.zeros_like(classification_logits)
                classification_label[classification_logits > 0] = 1

            else:
                classification_label = discrete_label_embedding[
                    :, 0, self.indices
                ]

        if self.dataset_config.classification_use_condition_input:
            input_dict["timeseries"] = torch.cat(
                (
                    timeseries_full,
                    input_continuous_label_embedding,
                    input_discrete_label_embedding,
                ),
                dim=1,
            )
        else:
            input_dict["timeseries"] = timeseries_full

        if not self.dataset_config.multi_label:
            assert (
                torch.sum(classification_label) == classification_label.shape[0]
            ), "there are multiple labels in the target, which is wrong"
            classification_label = torch.argmax(classification_label, dim=1)

        input_dict["labels"] = classification_label
        return input_dict

    def get_logits(self, timeseries):
        """
        Get the logits from the classifier.

        Args:
            timeseries (torch.tensor): The input timeseries data.

        Returns:
            torch.tensor: The logits from the classifier.
        """
        return self.classifier(timeseries)
