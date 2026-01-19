import argparse
import copy

import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from synthefy_pkg.configs.execution_configurations import (
    Configuration,
)
from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.data.data_gen_base import SynthesisBaseDataModule
from synthefy_pkg.prior.genload import LoadPriorDataset


class LoadSavedSyntheticDataloader(SynthesisBaseDataModule):
    def __init__(self, config: Configuration):
        logger.info("Initializing Saved Synthetic dataloader")
        assert config.dataset_config is not None, (
            "dataset_config is required for SavedSyntheticDataloader"
        )
        assert config.foundation_model_config is not None, (
            "foundation_model_config is required for SavedSyntheticDataloader"
        )
        assert config.prior_config is not None, (
            "prior_config is required for SavedSyntheticDataloader"
        )
        assert config.prior_config.prior_dir is not None, (
            "prior_dir is required in prior_config for SavedSyntheticDataloader"
        )

        self.batch_size = config.dataset_config.batch_size
        self.num_workers = 0  # Always disable multiprocessing to prevent CUDA IPC and other multiprocessing issues
        if config.dataset_config.num_workers > 0:
            logger.warning(
                f"Multiprocessing disabled for LoadSavedSyntheticDataloader (requested {config.dataset_config.num_workers} workers, using 0 instead)"
            )
        self.context_length = config.foundation_model_config.context_length
        self.num_correlates = config.dataset_config.num_correlates
        self.use_window_counts = config.dataset_config.use_window_counts
        self.shuffle = (
            False  # No need to shuffle as we're loading pre-generated data
        )
        self.config = config
        self.prior_config = config.prior_config

        # IMPORTANT: Load data on CPU in worker processes to prevent CUDA IPC memory leaks
        # Data will be moved to GPU in the main process (collate_fn) for training
        cpu_prior_config = copy.deepcopy(self.prior_config)
        cpu_prior_config.prior_device = (
            "cpu"  # Force CPU loading in worker processes
        )

        # Create LoadPriorDataset instances for train, val, and test
        # Each dataset will load from its respective split directory

        # Train dataset configuration
        train_config = copy.deepcopy(cpu_prior_config)
        train_config.batch_size = self.batch_size
        train_config.load_prior_start = (
            0  # Start from beginning for train split
        )
        logger.info(
            f"Loading train dataset from 'train' split on CPU (will transfer to {self.prior_config.prior_device} in main process)"
        )
        self.train_dataset = LoadPriorDataset(
            config=train_config, split="train"
        )

        # Validation dataset configuration
        val_config = copy.deepcopy(cpu_prior_config)
        val_config.batch_size = (
            self.batch_size
        )  # Use same batch size as train for consistency
        val_config.load_prior_start = 0  # Start from beginning for val split
        logger.info(
            f"Loading validation dataset from 'val' split on CPU (will transfer to {self.prior_config.prior_device} in main process)"
        )
        self.val_dataset = LoadPriorDataset(config=val_config, split="val")

        # Test dataset configuration
        test_config = copy.deepcopy(cpu_prior_config)
        test_config.batch_size = (
            self.batch_size
        )  # Use same batch size as train for consistency
        test_config.load_prior_start = 0  # Start from beginning for test split
        logger.info(
            f"Loading test dataset from 'test' split on CPU (will transfer to {self.prior_config.prior_device} in main process)"
        )
        self.test_dataset = LoadPriorDataset(config=test_config, split="test")

        # Store the target device for GPU training
        self.target_device = self.prior_config.prior_device

        super().__init__(
            config, self.train_dataset, self.val_dataset, self.test_dataset
        )

    def collate_fn(self, batch):
        """
        Collate function for the dataloader.
        Each batch item from LoadPriorDataset already contains multiple examples.
        This function also handles moving data from CPU to the target device if needed.
        """
        # Each item in batch is already a tuple of (X, y, d, seq_lens, train_sizes)
        # We just need to combine them
        if len(batch) == 1:
            # If batch size is 1, just return the first item
            result = {
                "X": batch[0][0],
                "y": batch[0][1],
                "d": batch[0][2],
                "seq_lens": batch[0][3],
                "train_sizes": batch[0][4],
                "series_flags": batch[0][5],
            }
        else:
            # Code should never be reached as dataset batch size is 1
            # If batch size > 1, combine the batches
            result = {
                "X": torch.cat([b[0] for b in batch]),
                "y": torch.cat([b[1] for b in batch]),
                "d": torch.cat([b[2] for b in batch]),
                "seq_lens": torch.cat([b[3] for b in batch]),
                "train_sizes": torch.cat([b[4] for b in batch]),
                "series_flags": torch.cat([b[5] for b in batch]),
            }

        # Verify all tensors are on CPU (as expected from worker processes)
        # This confirms our CUDA IPC prevention is working correctly
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    logger.warning(
                        f"Tensor '{key}' arrived on {value.device}, expected CPU. This may indicate CUDA IPC issues."
                    )

        # Move tensors to GPU in main process (prevents CUDA IPC memory leaks from worker processes)
        # This ensures GPU acceleration for training while avoiding worker process CUDA issues
        if self.target_device != "cpu":
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    try:
                        result[key] = value.to(
                            self.target_device, non_blocking=True
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to move tensor '{key}' to {self.target_device}: {e}"
                        )
                        raise

        return result

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # We use batch_size=1 because LoadPriorDataset already returns batches
            num_workers=self.num_workers,
            persistent_workers=False,  # Always False to prevent CUDA IPC issues
            shuffle=False,  # No need to shuffle as we're loading pre-generated data
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=2
            if self.num_workers > 0
            else None,  # Reduced prefetch to minimize memory usage
            pin_memory=False,  # Disabled because we handle CPU->GPU transfer in collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=False,  # Always False to prevent CUDA IPC issues
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=2
            if self.num_workers > 0
            else None,  # Reduced prefetch to minimize memory usage
            pin_memory=False,  # Disabled because we handle CPU->GPU transfer in collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=False,  # Always False to prevent CUDA IPC issues
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            prefetch_factor=2
            if self.num_workers > 0
            else None,  # Reduced prefetch to minimize memory usage
            pin_memory=False,  # Disabled because we handle CPU->GPU transfer in collate_fn
        )
