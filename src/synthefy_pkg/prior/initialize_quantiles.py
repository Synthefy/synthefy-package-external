from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthefy_pkg.configs.tabicl_config import TabICLPriorConfig
from synthefy_pkg.prior.dataset import PriorDataset


def create_dataloader(config: Dict[str, Any]) -> DataLoader:
    """
    Create a dataloader from the config.
    """
    ticl_config = TabICLPriorConfig()
    for key, value in config.items():
        setattr(ticl_config, key, value)

    dataset = PriorDataset(ticl_config)

    return DataLoader(
        dataset,
        batch_size=None,  # No additional batching since PriorDataset handles batching internally
        shuffle=False,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True if ticl_config.prior_device == "cpu" else False,
        pin_memory_device=ticl_config.device
        if ticl_config.prior_device == "cpu"
        else "",
    )


def initialize_quantiles(
    config: Dict[str, Any], use_learned_quantiles: bool = False
) -> List[float]:
    """
    Initialize the quantiles by creating a dataloader, loading batches
    then computing
    """
    if use_learned_quantiles:
        # create a dataloader
        dataloader = create_dataloader(config)
        dataloader_iter = iter(dataloader)

        y_quantiles = list()
        print("Pretraining quantiles...")
        for _ in tqdm(range(1000), desc="Pretraining quantiles"):
            batch = next(dataloader_iter)
            _, micro_y, _, _, _ = batch
            y_quantiles.append(micro_y.flatten())
        y_quantiles = torch.cat(y_quantiles, dim=0)
        return y_quantiles.quantile(torch.linspace(0, 1, 5000)).tolist()
    else:
        # return preset quantiles within a range
        return torch.linspace(-80, 80, 5000).tolist()
