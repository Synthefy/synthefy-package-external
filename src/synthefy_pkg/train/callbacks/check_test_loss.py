import lightning as L
import torch
import torch.nn.functional as F
from loguru import logger


class CheckTestLossCallback(L.Callback):
    def __init__(self, test_dataloader, check_test_every_n_epoch):
        super().__init__()
        self.test_dataloader = test_dataloader
        self.check_test_every_n_epoch = check_test_every_n_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip testing if not on the right epoch interval
        if trainer.current_epoch % self.check_test_every_n_epoch != 0:
            return

        device = pl_module.device
        pl_module.eval()

        # Track which models need to be set to eval mode
        models_to_check = ["decoder_model", "denoiser_model"]
        for model_name in models_to_check:
            if hasattr(pl_module, model_name):
                getattr(pl_module, model_name).eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                # logger.info(
                #     f"test batch {i}",
                #     hasattr(self.test_dataloader.dataset, "__len__"),
                # )
                if not hasattr(self.test_dataloader.dataset, "__len__") and (
                    (
                        hasattr(self.test_dataloader.dataset, "length")
                        and i > self.test_dataloader.dataset.length
                    )
                    or (
                        not hasattr(self.test_dataloader.dataset, "length")
                        and i > 1000
                    )
                ):
                    break
                # Move batch data to correct device
                for key, value in batch.items():
                    # for FMs, skip the scalars which are stored as a list
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                batch["epoch"] = trainer.current_epoch

                # Get model predictions
                input, prediction = pl_module.forward(batch)

                # Determine correct target based on model type
                if hasattr(pl_module, "foundation_model"):
                    pred = prediction["prediction"]
                    target_mask = prediction["target_mask"]
                    loss_sum = pl_module.calculate_loss(
                        input, pred, target_mask
                    )
                    total_loss += loss_sum.item()
                    total_samples += pred.numel()
                    continue
                elif hasattr(pl_module, "decoder_model"):
                    target = input["forecast"]
                elif hasattr(pl_module, "denoiser_model"):
                    target = input["noise"]
                else:
                    continue

                # Calculate MSE loss
                loss_sum = F.mse_loss(target, prediction, reduction="sum")
                total_loss += loss_sum.item()
                total_samples += target.numel()

        # Calculate and log mean MSE
        mean_mse = total_loss / total_samples if total_samples > 0 else 0.0

        # Log metrics
        pl_module.log(
            "test_loss",
            mean_mse,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Restore training mode
        pl_module.train()
        for model_name in models_to_check:
            if hasattr(pl_module, model_name):
                getattr(pl_module, model_name).train()
