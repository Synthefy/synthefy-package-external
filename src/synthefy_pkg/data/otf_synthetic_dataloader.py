import copy

import torch
from einops import rearrange, repeat
from loguru import logger
from torch.utils.data import DataLoader

from synthefy_pkg.configs.execution_configurations import (
    Configuration,
    DatasetConfig,
)
from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.fm_evals.dataloading.synthetic_medium_lag_dataloader import (
    SyntheticMediumLagDataloader,
)
from synthefy_pkg.prior.dataset import PriorDataset
from synthefy_pkg.utils.basic_utils import seed_everything


class OTFSyntheticDataloader(SynthesisBaseDataModule):
    def __init__(self, config: Configuration):
        logger.info("Initializing On the Fly Synthetic Dataloader")
        assert config.dataset_config is not None, (
            "dataset_config is required for OTFSyntheticDataloader"
        )
        assert config.foundation_model_config is not None, (
            "foundation_model_config is required for OTFSyntheticDataloader"
        )

        self.batch_size = config.dataset_config.batch_size
        self.num_workers = config.dataset_config.num_workers
        self.context_length = config.foundation_model_config.context_length
        self.num_correlates = config.dataset_config.num_correlates
        self.use_window_counts = config.dataset_config.use_window_counts
        self.shuffle = True
        self.config = config
        self.prior_config = config.prior_config
        self.dataset_length = config.prior_config.dataset_length

        # override certain prior config values
        # self.prior_config.max_seq_len = self.context_length
        # self.prior_config.min_features = self.num_correlates - 5
        # self.prior_config.max_features = self.num_correlates - 1

        # self.prior_config.min_features = config.dataset_config.num_channels
        # self.prior_config.max_features = config.dataset_config.num_channels
        # self.prior_config.batch_size = self.batch_size
        # self.prior_config.device = self.config.device
        # # self.prior_config.prior_device = self.dataset_config.device # TODO: does putting both on the save device work?
        # self.prior_config.max_features = self.num_correlates - 1
        # self.prior_config.max_seq_len = config.dataset_config.time_series_length

        # TODO: train val test all have the same dataset config
        #      it's possible we might want to pass different configs for each
        self.dataset_length = self.config.prior_config.dataset_length
        dataset = PriorDataset(
            config=self.prior_config,
            real_data_config=self.config.dataset_config
            if self.config.dataset_config.mixed_real_synthetic_sampling
            else None,
            fm_config=self.config.foundation_model_config,
        )
        logger.info("Loading train dataset")
        self.train_dataset = dataset
        valtest_prior_config = copy.deepcopy(config.prior_config)
        valtest_prior_config.dataset_length = (
            max(config.prior_config.dataset_length // 8, 32)
        )
        valtest_prior_config.check_for_curriculum_config = False
        valtest_prior_config.run_id = valtest_prior_config.run_id + "_valtest"

        # Will be overridden if run_val_with_eval_bench is True
        self.val_dataset = PriorDataset(
            config=valtest_prior_config,
            real_data_config=self.config.dataset_config
            if self.config.dataset_config.mixed_real_synthetic_sampling
            else None,
            fm_config=self.config.foundation_model_config,
        )
        self.test_dataset = PriorDataset(
            config=valtest_prior_config,
            real_data_config=self.config.dataset_config
            if self.config.dataset_config.mixed_real_synthetic_sampling
            else None,
            fm_config=self.config.foundation_model_config,
        )

        super().__init__(
            config, self.train_dataset, self.val_dataset, self.test_dataset
        )

    def worker_init_fn(self, worker_id):
        """Initialize each worker with a unique seed based on the global seed."""
        # Set a different seed for each worker and device combination
        seed_everything(self.config.seed, worker_id=worker_id, seed_by_device=True)

    def collate_fn(self, batch):
        """
        Reshapes a flat list of examples to
        [batch, num_correlates, window_size] - works even for the very last
        (potentially incomplete) batch on each GPU.
        """
        vals = {
            "X": torch.stack([b[0][0] for b in batch]),
            "y": torch.stack([b[1] for b in batch]).squeeze(1),
            "d": torch.stack([b[2] for b in batch]).squeeze(-1),
            "seq_lens": torch.stack([b[3] for b in batch]).squeeze(-1),
            "train_sizes": torch.stack([b[4] for b in batch]).squeeze(-1),
            "series_flags": torch.stack([b[5] for b in batch]).squeeze(-1),
        }
        return vals

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=4,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        if self.config.dataset_config.run_val_with_eval_bench:
            return SyntheticMediumLagDataloader(
                self.config.dataset_config.eval_bench_data_path,
                num_target_rows=self.config.dataset_config.forecast_length,
                max_length=self.config.dataset_config.num_eval_bench_batches,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=4,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self):
        if self.config.dataset_config.run_val_with_eval_bench:
            return SyntheticMediumLagDataloader(
                self.config.dataset_config.eval_bench_data_path,
                num_target_rows=self.config.dataset_config.forecast_length,
                max_length=self.config.dataset_config.num_eval_bench_batches,
            )
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=4,
            worker_init_fn=self.worker_init_fn,
        )

    def update_config(self, config):
        # self.prior_config = config
        # self.config.prior_config = config
        self.train_dataset.update_config(config)
