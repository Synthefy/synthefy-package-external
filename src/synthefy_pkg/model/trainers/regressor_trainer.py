from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
from omegaconf import DictConfig

from synthefy_pkg.model.custom_regressor import TimeSeriesRegressor


def load_regression_model(config: DictConfig) -> torch.nn.Module:
    regressor_config = DictConfig(config.regressor)
    dataset_config = DictConfig(config.dataset)
    device = config.device

    regression_model = TimeSeriesRegressor(
        regressor_config=regressor_config,
        dataset_config=dataset_config,
        device=device,
    )
    return regression_model


class RegressorTrainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.regressor_model = load_regression_model(config=config)
        self.dataset_config = DictConfig(config.dataset)
        self.criterion = torch.nn.MSELoss()

    def forward(self, batch: Dict[str, torch.Tensor]):
        inp = self.regressor_model.prepare_training_input(batch)
        logits = self.regressor_model(inp)
        return inp, logits

    def calculate_loss(self, inp, logits):
        labels = inp["target"]
        loss = self.criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.regressor_model.parameters(),
            lr=float(self.config.training.learning_rate),
        )

    def training_step(self, batch: Dict[str, torch.Tensor]):
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        inp, logits = self.forward(batch)
        loss = self.calculate_loss(inp, logits)
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor]):
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        inp, logits = self.forward(batch)
        loss = self.calculate_loss(inp, logits)
        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        inp, logits = self.forward(batch)
        loss = self.calculate_loss(inp, logits)

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "preds": logits.detach().cpu(),
            "labels": inp["target"].detach().cpu(),
        }

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def on_test_batch_end(self, outputs, batch, batch_idx) -> None:
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        self.test_predictions = all_preds
        self.test_labels = all_labels
