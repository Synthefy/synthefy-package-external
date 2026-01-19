import torch

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_trainer import (
    TimeSeriesDecoderForecastingTrainer,
)
from synthefy_pkg.model.utils.lr_scheduler import get_lr_scheduler

COMPILE = False


class MetadataPretrainForecastingTrainer(TimeSeriesDecoderForecastingTrainer):
    """
    exactly the same as TimeSeriesDecoderForecastingTrainer, but only returns the metadata encoder as an optimizer
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        assert hasattr(self.decoder_model, "metadata_encoder")

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(
            self.decoder_model.metadata_encoder.parameters(),
            lr=self.training_config.learning_rate,
        )

        lr_scheduler = get_lr_scheduler(
            optimizer=adam_optimizer,
            lr_scheduler_config=self.training_config.lr_scheduler_config,
        )

        if lr_scheduler is None:
            return {
                "optimizer": adam_optimizer,
            }

        return {
            "optimizer": adam_optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
