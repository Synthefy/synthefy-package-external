import os

import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.rank_zero import rank_zero_only
from loguru import logger
from tqdm import tqdm


class GlobalStepCheckpointCallback(L.Callback):
    def __init__(
        self, dirpath, save_checkpoint_every_n_steps, run_ar_val_every_n_steps, num_val_batches, dataset_generator
    ):
        super().__init__()
        self.dirpath = dirpath
        self.save_checkpoint_every_n_steps = save_checkpoint_every_n_steps
        self.run_ar_val_every_n_steps = run_ar_val_every_n_steps
        self.counter = 0
        self.num_val_batches = num_val_batches
        self.dataset_generator = dataset_generator

    def _get_max_validation_batches(self, val_dataloader) -> int:
        """
        Determine the maximum number of validation batches to process.
        Returns a safe upper limit for validation iterations.
        """
        # First, try to get the length from the dataloader itself
        if hasattr(val_dataloader, "__len__"):
            return len(val_dataloader)

        # If dataloader doesn't have length, check the dataset
        if hasattr(val_dataloader, "dataset"):
            dataset = val_dataloader.dataset

            # Try standard dataset length
            if hasattr(dataset, "__len__"):
                return len(dataset) // getattr(val_dataloader, "batch_size", 1)

            # Try custom length attribute
            if hasattr(dataset, "length"):
                return dataset.length // getattr(val_dataloader, "batch_size", 1)

        # Fallback to a reasonable default
        return 1000

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            trainer.global_step % self.run_ar_val_every_n_steps == 0
        ):  # Use instance variable
            logger.info(
                f"Plots should be saved at step {trainer.global_step}"
            )
            # try:
            # Manual validation calculation
            pl_module.eval()  # Set model to evaluation mode
            val_loss = 0.0
            num_batches = 0

            # Get validation dataloader
            val_dataloader = self.dataset_generator.val_dataloader()

            max_batches = self._get_max_validation_batches(val_dataloader)
            pl_module.on_validation_start()

            # Manual validation loop
            with torch.no_grad():
                for i, val_batch in tqdm(
                    enumerate(val_dataloader),
                    total=max_batches,
                    desc=f"Running val for checkpoint (step {trainer.global_step})",
                    leave=False
                ):
                    # Ensures break in lengthless dataloaders
                    if i >= max_batches:
                        break

                    # Move data to device
                    if hasattr(val_batch, 'items') and hasattr(val_batch, '__setitem__'):
                        for key, value in val_batch.items():
                            if isinstance(value, torch.Tensor):
                                val_batch[key] = value.to(pl_module.device)
                        val_batch["epoch"] = trainer.current_epoch

                    # Forward pass
                    val_loss_batch = pl_module.validation_step(
                        val_batch, i, compute_ar_loss=True
                    )
                    if isinstance(val_loss_batch, dict):
                        if "val_loss" in val_loss_batch:
                            val_loss += (
                                val_loss_batch["val_loss"].item()
                                if isinstance(
                                    val_loss_batch["val_loss"],
                                    torch.Tensor,
                                )
                                else val_loss_batch["val_loss"]
                            )
                    elif isinstance(val_loss_batch, torch.Tensor):
                        val_loss += val_loss_batch.item()
                    elif (
                        isinstance(val_loss_batch, (int, float))
                        and val_loss_batch is not None
                    ):
                        val_loss += val_loss_batch

                    num_batches += 1

                    if num_batches > self.num_val_batches:
                        break

            # Calculate average validation loss
            avg_val_loss = (
                val_loss / num_batches if num_batches > 0 else float("inf")
            )
            pl_module.train()  # Set model back to training mode

            # Save checkpoint with validation loss in filename
            checkpoint_path = os.path.join(
                self.dirpath,
                f"checkpoint_step_{trainer.global_step}_{pl_module.config.save_key}_{avg_val_loss:.4f}.ckpt",
            )
            if trainer.global_step % self.save_checkpoint_every_n_steps == 0:
                trainer.save_checkpoint(checkpoint_path)
                logger.info(
                    f"Manually saved checkpoint to {checkpoint_path} with {pl_module.config.save_key}: {avg_val_loss:.4f}"
                )
            # except Exception as e:
            #     logger.error(f"Failed to save checkpoint: {str(e)}")
