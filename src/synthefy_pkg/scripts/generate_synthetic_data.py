import os

import numpy as np
import torch
from loguru import logger

from synthefy_pkg.utils.basic_utils import ENDC, OKBLUE

COMPILE=True

def generate_synthetic_dataset(
    dataloader,
    synthesizer,
    synthesis_function,
    save_dir,
    dataset_config,
    in_gan_space=False,
    scaler=None,
    train_or_val_or_test: str = "",
):
    if train_or_val_or_test not in ["train", "val", "test"]:
        raise ValueError(
            "train_or_val_or_test should be one of 'train', 'val', or 'test'"
        )
    horizon = dataset_config.time_series_length
    num_channels = dataset_config.num_channels
    num_discrete_conditions = dataset_config.num_discrete_conditions
    num_continuous_labels = dataset_config.num_continuous_labels

    timeseries_str_to_add = "timeseries.npy"
    discrete_conditions_str_to_add = "discrete_conditions.npy"
    continuous_conditions_str_to_add = "continuous_conditions.npy"
    combined_data_str_to_add = "combined_data.pkl"

    def get_file_locations(data_type):
        timeseries_loc = os.path.join(save_dir, f"{data_type}_{timeseries_str_to_add}")
        discrete_conditions_loc = os.path.join(
            save_dir, f"{data_type}_{discrete_conditions_str_to_add}"
        )
        continuous_conditions_loc = os.path.join(
            save_dir, f"{data_type}_{continuous_conditions_str_to_add}"
        )
        combined_data_loc = os.path.join(
            save_dir, f"{data_type}_{combined_data_str_to_add}"
        )

        return (
            timeseries_loc,
            discrete_conditions_loc,
            continuous_conditions_loc,
            combined_data_loc,
        )

    (
        timeseries_loc,
        discrete_conditions_loc,
        continuous_conditions_loc,
        combined_data_loc,
    ) = get_file_locations(train_or_val_or_test)

    if os.path.exists(timeseries_loc):
        logger.info(
            f"The synthetic dataset already exists. Skipping generation. `{timeseries_loc}`"
        )
        return None
    else:
        logger.info("Let's start the data generation process")
        logger.info(f"The synthetic dataset will be stored in: {save_dir}")
        logger.info(f"The synthetic timeseries will be stored in: {timeseries_loc}")
        logger.info(
            f"The synthetic discrete conditions will be stored in: {discrete_conditions_loc}"
        )
        logger.info(
            f"The synthetic continuous conditions will be stored in: {continuous_conditions_loc}"
        )

    logger.info(OKBLUE + "Generating synthetic samples" + ENDC)
    timeseries = []
    discrete_conditions = []
    continuous_conditions = []
    num_test_samples = 0

    original_timeseries = []

    combined_data = {
        "original_timeseries": [],
        "synthetic_timeseries": [],
        "discrete_conditions": [],
        "continuous_conditions": [],
    }

    for batch_idx, batch in enumerate(dataloader):
        for key, value in batch.items():
            batch[key] = value.to(synthesizer.config.device)
        dataset_dict = synthesis_function(
            batch=batch,
            synthesizer=synthesizer,
        )

        if in_gan_space:
            # print(OKBLUE + "Converting the samples from gan space to normal space" + ENDC)
            dataset_dict = scaler.convert_from_gan_to_normal(
                dataset_dict=dataset_dict,
            )

        timeseries.append(dataset_dict["timeseries"])
        discrete_conditions.append(dataset_dict["discrete_conditions"])
        continuous_conditions.append(dataset_dict["continuous_conditions"])
        original_timeseries.append(batch["timeseries_full"].cpu().numpy())

        num_test_samples += dataset_dict["timeseries"].shape[0]

        logger.info(OKBLUE + "Generated %d samples" % (num_test_samples) + ENDC)
        if num_test_samples > 5:
            break
        timeseries_to_save = np.concatenate(timeseries, axis=0)
        assert timeseries_to_save.shape[-1] == horizon
        assert timeseries_to_save.shape[-2] == num_channels

        discrete_conditions_to_save = np.concatenate(discrete_conditions, axis=0)
        assert discrete_conditions_to_save.shape[-1] == num_discrete_conditions
        if len(discrete_conditions_to_save.shape) > 2:
            assert discrete_conditions_to_save.shape[-2] == horizon

        continuous_conditions_to_save = np.concatenate(continuous_conditions, axis=0)
        assert continuous_conditions_to_save.shape[-1] == num_continuous_labels
        if len(continuous_conditions_to_save.shape) > 2:
            assert continuous_conditions_to_save.shape[-2] == horizon

        original_timeseries_to_save = np.concatenate(original_timeseries, axis=0)
        assert original_timeseries_to_save.shape[-1] == horizon
        assert original_timeseries_to_save.shape[-2] == num_channels

        np.save(timeseries_loc, timeseries_to_save)
        np.save(discrete_conditions_loc, discrete_conditions_to_save)
        np.save(continuous_conditions_loc, continuous_conditions_to_save)

        combined_data["synthetic_timeseries"] = timeseries_to_save
        combined_data["discrete_conditions"] = discrete_conditions_to_save
        combined_data["continuous_conditions"] = continuous_conditions_to_save

        combined_data["original_timeseries"] = original_timeseries_to_save

        # save the combined data as pickle file
        torch.save(combined_data, combined_data_loc, pickle_protocol=4)
