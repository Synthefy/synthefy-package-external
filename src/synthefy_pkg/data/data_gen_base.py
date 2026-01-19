from abc import abstractmethod

import lightning as L
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.configs.relational_config import SubDatasetConfig

COMPILE = False


class DataGenBase:
    @abstractmethod
    def __init__(self) -> None:
        self.batch_size = 0

    @abstractmethod
    def __iter__(self):
        while False:
            yield None

    def get_iterator(self):
        return self.__iter__()

    def get_batch(self, batch_size):
        raise NotImplementedError("Implement get_batch method")


class SynthesisBaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: Configuration | SubDatasetConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        forecasting: bool = False,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.forecasting = forecasting

        self.kwargs = (
            {
                "num_workers": config.num_workers,
                "pin_memory": True,
                "multiprocessing_context": "fork",
            }
            if config.device == "cuda"
            else {"num_workers": config.num_workers}
        )

        self.config = config
        if (
            self.forecasting
        ):  # TODO: Can we eliminate this? I think data_gen_base still uses it.
            self.dataset_config = self.config.dataset_config

    def train_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.train_dataset,
                batch_size=self.dataset_config.batch_size,
                **self.kwargs,  # pyright: ignore
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                **self.kwargs,  # pyright: ignore
            )

    def val_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.val_dataset,
                batch_size=self.dataset_config.batch_size,
                **self.kwargs,  # pyright: ignore
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                **self.kwargs,  # pyright: ignore
            )

    def test_dataloader(self):
        if self.forecasting:
            return DataLoader(
                self.test_dataset,
                batch_size=self.dataset_config.batch_size,
                **self.kwargs,  # pyright: ignore
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                **self.kwargs,  # pyright: ignore
            )
