import os
from typing import Any, Dict, Literal, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from tqdm import tqdm

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.load_models import load_cltsp_model
from synthefy_pkg.postprocessing.utils import plot_learning_curve
from synthefy_pkg.utils.basic_utils import get_cltsp_config


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, config):
        super(SupConLoss, self).__init__()
        self.temperature = config.supervised_contrastive_learning.temperature
        self.contrast_mode = (
            config.supervised_contrastive_learning.contrast_mode
        )
        self.base_temperature = (
            config.supervised_contrastive_learning.base_temperature
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            assert mask is not None
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = (
            torch.exp(logits) * logits_mask + 1e-8
        )  # for stability, else the loss goes to NaN
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class CLTSPTrainer(L.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.cltsp_config = get_cltsp_config(config=config)
        self.cltsp_model = load_cltsp_model(config=config)

        self.sup_con_loss = SupConLoss(config=config.encoder_config)

        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        input = self.cltsp_model.prepare_training_input(batch)
        # for key, value in input.items():
        #     print(key, value.shape)
        # print("----------------------------------")
        timeseries_emb, condition_emb = self.cltsp_model(input)
        # print(timeseries_emb.shape, condition_emb.shape)
        return input, timeseries_emb, condition_emb

    def calculate_mod_clip_loss(
        self, timeseries_emb, condition_emb
    ) -> torch.Tensor:
        batch_size = timeseries_emb.shape[0]
        logits = (
            condition_emb @ timeseries_emb.T
        ) / self.config.training_config.temperature
        target = torch.arange(batch_size).to(self.config.device)
        texts_loss = torch.nn.functional.cross_entropy(logits, target)
        timeseries_loss = torch.nn.functional.cross_entropy(logits.T, target)
        loss = (timeseries_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def calculate_loss(
        self, input, timeseries_emb, condition_emb
    ) -> Tuple[
        torch.Tensor,
        Union[torch.Tensor, int],
        Union[torch.Tensor, int],
    ]:
        # print(timeseries_emb.shape, condition_emb.shape)
        clip_loss = self.calculate_mod_clip_loss(timeseries_emb, condition_emb)

        num_samples_in_batch = int(
            timeseries_emb.shape[0] // self.cltsp_config.num_positive_samples  # type: ignore
        )
        assert num_samples_in_batch == input["labels"].shape[0]

        if self.config.training_config.use_timeseries_scl_loss:
            timeseries_emb_split = timeseries_emb.reshape(
                self.cltsp_config.num_positive_samples,  # type: ignore
                num_samples_in_batch,
                -1,
            )  # (num_positive_samples, batch_size, embedding_dim)
            # print(timeseries_emb_split[0], timeseries_emb)
            # print(
            #     timeseries_emb_split[0].shape, timeseries_emb[:num_samples_in_batch].shape
            # )
            assert torch.all(
                timeseries_emb[:num_samples_in_batch] == timeseries_emb_split[0]
            )
            timeseries_emb_split_reshaped = torch.einsum(
                "ijk->jik", timeseries_emb_split
            )  # (batch_size, num_positive_samples, embedding_dim)
            assert torch.all(
                timeseries_emb[:num_samples_in_batch]
                == timeseries_emb_split_reshaped[:, 0, :]
            )
            supervised_contrastive_learning_loss_for_timeseries = (
                self.sup_con_loss(
                    timeseries_emb_split_reshaped, input["labels"]
                )
            )
        else:
            supervised_contrastive_learning_loss_for_timeseries = 0

        if self.config.training_config.use_condition_scl_loss:
            condition_emb_split = condition_emb.reshape(
                self.cltsp_config.num_positive_samples,  # type: ignore
                num_samples_in_batch,
                -1,
            )  # (num_positive_samples, batch_size, embedding_dim)
            assert torch.all(
                condition_emb[:num_samples_in_batch] == condition_emb_split[0]
            )
            condition_emb_split_reshaped = torch.einsum(
                "ijk->jik", condition_emb_split
            )  # (batch_size, num_positive_samples, embedding_dim)
            assert torch.all(
                condition_emb[:num_samples_in_batch]
                == condition_emb_split_reshaped[:, 0, :]
            )
            supervised_contrastive_learning_loss_for_condition = (
                self.sup_con_loss(condition_emb_split_reshaped, input["labels"])
            )
        else:
            supervised_contrastive_learning_loss_for_condition = 0

        return (
            clip_loss,
            supervised_contrastive_learning_loss_for_timeseries,
            supervised_contrastive_learning_loss_for_condition,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.cltsp_model.parameters(),
            lr=self.config.training_config.learning_rate,
        )

    def _process_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move batch to device and process it."""
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        return batch

    def _calculate_total_loss(
        self,
        clip_loss: torch.Tensor,
        scl_loss_ts: torch.Tensor,
        scl_loss_cn: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the total loss based on enabled loss components.
        Return torch.tensor
        """

        clip_factor = 1 if self.config.training_config.use_clip_loss else 0
        condition_scl_factor = (
            1 if self.config.training_config.use_condition_scl_loss else 0
        )
        timeseries_scl_factor = (
            1 if self.config.training_config.use_timeseries_scl_loss else 0
        )

        return (
            (timeseries_scl_factor * scl_loss_ts)
            + (clip_factor * clip_loss)
            + (condition_scl_factor * scl_loss_cn)
        )

    def _log_losses(
        self,
        prefix: str,
        total_loss: torch.Tensor | int,
        clip_loss: torch.Tensor | int,
        scl_loss_ts: torch.Tensor | int,
        scl_loss_cn: torch.Tensor | int,
    ) -> None:
        """Log all relevant losses with the given prefix."""
        self.log(
            f"{prefix}_loss",
            total_loss,
            sync_dist=True,
            on_step=True if prefix == "train" else False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.config.training_config.use_clip_loss:
            self.log(
                f"{prefix}_clip_loss",
                clip_loss,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        if self.config.training_config.use_timeseries_scl_loss:
            self.log(
                f"{prefix}_scl_loss_ts",
                scl_loss_ts,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        if self.config.training_config.use_condition_scl_loss:
            self.log(
                f"{prefix}_scl_loss_cn",
                scl_loss_cn,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        """Shared step logic for train/val/test."""
        batch = self._process_batch(batch)
        input, timeseries_emb, condition_emb = self.forward(batch)

        clip_loss, scl_loss_ts, scl_loss_cn = self.calculate_loss(
            input, timeseries_emb, condition_emb
        )

        total_loss = self._calculate_total_loss(
            clip_loss,
            scl_loss_ts,  # type: ignore
            scl_loss_cn,  # type: ignore
        )
        self._log_losses(
            prefix, total_loss, clip_loss, scl_loss_ts, scl_loss_cn
        )

        return total_loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._shared_step(batch, "val")
        if self.current_epoch > 1:
            log_dir = self.trainer.log_dir
            if log_dir is not None:
                plot_learning_curve(
                    input_logs_dir=os.path.dirname(log_dir),
                    output_fig_path=os.path.join(log_dir, "learning_curve.png"),
                    run_name=self.config.run_name,
                    dataset_name=self.config.dataset_name,
                )
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

    def predict(self, dataloader) -> Dict[str, Any]:
        """Run prediction on the dataloader and return embeddings and losses.

        Args:
            dataloader: DataLoader containing the data to predict on

        Returns:
            Dictionary containing:
            - timeseries_embeddings: Embeddings of the time series as numpy array
            - condition_embeddings: Embeddings of the conditions as numpy array
            - contrastive_losses: Contrastive losses between pairs as numpy array
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        timeseries_embeddings = []
        condition_embeddings = []
        contrastive_losses = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Generating embeddings", unit="batch"
            ):
                # Move batch to device
                batch = self._process_batch(batch)

                # Get embeddings
                input_dict, timeseries_emb, condition_emb = self.forward(batch)

                # Calculate loss
                clip_loss, _, _ = self.calculate_loss(
                    input_dict, timeseries_emb, condition_emb
                )
                # Store results
                timeseries_embeddings.append(timeseries_emb.cpu().numpy())
                condition_embeddings.append(condition_emb.cpu().numpy())
                # Reshape the scalar loss to a 1D array
                contrastive_losses.append(clip_loss.cpu().numpy().reshape(1))

        final_predictions = {
            "timeseries_embeddings": np.concatenate(
                timeseries_embeddings, axis=0
            ),
            "condition_embeddings": np.concatenate(
                condition_embeddings, axis=0
            ),
            "contrastive_losses": np.array(contrastive_losses).flatten(),
        }

        return final_predictions
