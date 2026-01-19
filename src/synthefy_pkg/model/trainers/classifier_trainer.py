import os
from typing import Dict

import lightning as L
import torch
from omegaconf import DictConfig

from synthefy_pkg.model.custom_classifier import TimeSeriesClassifier
from synthefy_pkg.postprocessing.utils import plot_learning_curve

COMPILE = False


def load_classifier_model(config: DictConfig) -> torch.nn.Module:
    classifier_config = DictConfig(config.classifier)
    dataset_config = DictConfig(config.dataset)
    device = config.device

    classifier_model = TimeSeriesClassifier(
        classifier_config=classifier_config,
        dataset_config=dataset_config,
        device=device,
    )
    return classifier_model


class ClassifierTrainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier_model = load_classifier_model(config=config)
        self.dataset_config = DictConfig(config.dataset)
        if self.dataset_config.multi_label:
            self.classification_loss_criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.classification_loss_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, torch.Tensor]):
        input = self.classifier_model.prepare_training_input(batch)
        logits = self.classifier_model(input)
        return input, logits

    def calculate_loss(self, input, logits):
        labels = input["labels"]
        loss = self.classification_loss_criterion(logits, labels)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.classifier_model.parameters(),
            lr=float(self.config.training.learning_rate),
        )

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(dtype=torch.float32).to(self.config.device)
        input, logits = self.forward(batch)
        loss = self.calculate_loss(input, logits)
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(dtype=torch.float32).to(self.config.device)
        input, logits = self.forward(batch)
        loss = self.calculate_loss(input, logits)

        # Log to Lightning
        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.current_epoch > 1:
            plot_learning_curve(
                input_logs_dir=os.path.dirname(str(self.trainer.log_dir)),
                output_fig_path=os.path.join(
                    str(self.trainer.log_dir), "learning_curve.png"
                ),
            )

        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for key, value in batch.items():
            batch[key] = value.to(dtype=torch.float32).to(self.config.device)
        input, logits = self.forward(batch)
        loss = self.calculate_loss(input, logits)

        # Calculate predictions
        preds = (
            torch.sigmoid(logits)
            if self.dataset_config.multi_label
            else torch.softmax(logits, dim=1)
        )

        # Log the loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "preds": preds.detach().cpu(),
            "labels": input["labels"].detach().cpu(),
        }

    def on_test_epoch_start(self) -> None:
        # Initialize lists to store predictions and labels
        self.test_step_outputs = []

    def on_test_batch_end(self, outputs, batch, batch_idx) -> None:
        # Store the outputs
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        # Aggregate all predictions and labels
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        # Store predictions and labels as instance attributes
        self.test_predictions = all_preds
        self.test_labels = all_labels
