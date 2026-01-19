import os
from copy import deepcopy
from typing import Any, Dict

import yaml
from frozendict import frozendict
from loguru import logger
from mergedeep import Strategy, merge

DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL_CHANNELS = 256
DEFAULT_FORECAST_LENGTH_MULTIPLIER = 0.5

BASE_DATASET_CONFIG = frozendict(
    {
        "dataset_name": "",
        "num_channels": 0,  # this will be overwritten with values from preprocessing
        "forecast_length": 0,  # this will be overwritten with values from preprocessing
        "time_series_length": 0,  # this will be overwritten with values from preprocessing
        "required_time_series_length": 0,  # this will be overwritten with values from preprocessing
        "num_discrete_conditions": 0,  # this will be overwritten with values from preprocessing
        "num_continuous_labels": 0,  # this will be overwritten with values from preprocessing
        "num_discrete_labels": 1,
        "discrete_condition_embedding_dim": 64,
        "latent_dim": 64,
        "batch_size": 16,
        "use_metadata": True,
        "use_timestamp": False,  # For forecasting v2 model
    }
)

BASE_METADATA_ENCODER_CONFIG = frozendict(
    {
        "channels": DEFAULT_MODEL_CHANNELS,
        "n_heads": 8,
        "num_encoder_layers": 2,
    }
)


BASE_EXECUTION_CONFIG = frozendict(
    {
        "save_path": "training_logs",
        "run_name": "{task}_{dataset_name}",  # added with format in generate_default_train_config
        "generation_save_path": "generation_logs",
        "experiment_name": "Time_Series_Diffusion_Training",
    }
)

BASE_SYNTHESIS_TRAINING_CONFIG = frozendict(
    {
        "max_epochs": 500,
        "learning_rate": 1e-4,
        "n_plots": 4,
        "auto_lr_find": True,
        "check_val_every_n_epoch": 5,
        "check_test_every_n_epoch": 5,
        "log_every_n_steps": 1,
        "num_devices": 1,
        "strategy": "auto",  # TODO - should i make this ddp_find_unused_parameters_true always?
    }
)

BASE_FORECAST_TRAINING_CONFIG = frozendict(
    {
        "max_epochs": 30,
        "learning_rate": 1e-4,
        "n_plots": 4,
        "auto_lr_find": True,
        "check_val_every_n_epoch": 1,
        "check_test_every_n_epoch": 1,
        "log_every_n_steps": 1,
        "num_devices": 1,
        "strategy": "auto",  # TODO - should i make this ddp_find_unused_parameters_true always?
    }
)

BASE_SYNTHESIS_DENOISER_CONFIG = frozendict(
    {
        "denoiser_name": "patched_diffusion_transformer",
        "positional_embedding_dim": 128,
        "channel_embedding_dim": 16,
        "channels": DEFAULT_MODEL_CHANNELS,
        "n_heads": 8,
        "n_layers": 12,
        "dropout_pos_enc": 0.2,
        "use_metadata": True,
        "use_periodic_projection": True,
        "patch_len": 32,
        "stride": 32,
        "metadata_embedding_dim": 32,
    }
)

BASE_FORECASTING_DENOISER_CONFIG = frozendict(
    {
        "denoiser_name": "synthefy_forecasting_model_v2a",
        "d_model": DEFAULT_MODEL_CHANNELS,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.1,
        "e_layers": 6,
        "d_layers": 1,
        "d_ff": 512,
        "n_heads": 8,
        "activation": "gelu",
        "factor": 3,
        "output_attention": True,
        "use_metadata": True,
    }
)

BASE_TRAIN_SYNTHESIS_CONFIG = frozendict(
    {
        "device": DEFAULT_DEVICE,
        "task": "synthesis",
        "dataset_config": BASE_DATASET_CONFIG,
        "denoiser_config": BASE_SYNTHESIS_DENOISER_CONFIG,
        "metadata_encoder_config": BASE_METADATA_ENCODER_CONFIG,
        "execution_config": BASE_EXECUTION_CONFIG,
        "training_config": BASE_SYNTHESIS_TRAINING_CONFIG,
    }
)

BASE_TRAIN_FORECASTING_CONFIG = frozendict(
    {
        "device": DEFAULT_DEVICE,
        "task": "forecast",
        "dataset_config": BASE_DATASET_CONFIG,
        "denoiser_config": BASE_FORECASTING_DENOISER_CONFIG,
        "metadata_encoder_config": BASE_METADATA_ENCODER_CONFIG,
        "execution_config": BASE_EXECUTION_CONFIG,
        "training_config": BASE_FORECAST_TRAINING_CONFIG,
    }
)


def generate_default_train_config(
    task: str,
    dataset_name: str,
    time_series_length: int,
    num_channels: int,
    num_discrete_conditions: int,
    num_continuous_labels: int,
    num_timestamp_labels: int,
    save_to_examples_dir: bool = False,
) -> Dict[str, Any]:
    """
    Get the default training configuration for a given task and dataset.

    Args:
        task: The type of task to configure (e.g., "synthesis" or "forecast")
    """
    if task not in ["synthesis", "forecast"]:
        raise ValueError(f"Invalid task: {task}")
    # Create a mutable copy of the frozen base config
    base_config = deepcopy(
        dict(
            BASE_TRAIN_SYNTHESIS_CONFIG
            if task == "synthesis"
            else BASE_TRAIN_FORECASTING_CONFIG
        )
    )
    # make each dict in the base config mutable
    for key, value in base_config.items():
        if isinstance(value, frozendict):
            base_config[key] = deepcopy(dict(value))

    # Create a mutable copy of the dataset config
    base_config["dataset_config"] = dict(base_config["dataset_config"])

    # fill the dataset config requirements
    dataset_updates = {
        "dataset_name": dataset_name,
        "time_series_length": time_series_length,
        "forecast_length": int(
            time_series_length * DEFAULT_FORECAST_LENGTH_MULTIPLIER
        ),
        "required_time_series_length": time_series_length,
        "num_channels": num_channels,
        "num_discrete_conditions": num_discrete_conditions,
        "num_continuous_labels": num_continuous_labels,
        "num_timestamp_labels": num_timestamp_labels,
    }
    base_config["dataset_config"].update(dataset_updates)

    # denoiser config updates
    if task == "synthesis":
        if time_series_length < 32:
            base_config["denoiser_config"]["patch_len"] = min(
                8, time_series_length
            )
            base_config["denoiser_config"]["stride"] = min(
                8, time_series_length
            )
        else:
            base_config["denoiser_config"]["patch_len"] = 32
            base_config["denoiser_config"]["stride"] = 32
    # add the formatted run name
    base_config["execution_config"]["run_name"] = base_config[
        "execution_config"
    ]["run_name"].format(task=task, dataset_name=dataset_name)

    # TODO - Add base TSTR config
    # Note - not doing this now since it requires comments in the yaml file, and it not trivial to do this I think.

    # save to examples dir
    if save_to_examples_dir:
        # TODO - make the configs only end in forecast not forecasting
        suffix = "synthesis" if task == "synthesis" else "forecasting"
        try:
            package_base = os.getenv("SYNTHEFY_PACKAGE_BASE")
            if not package_base:
                raise ValueError(
                    "SYNTHEFY_PACKAGE_BASE is not set to save training configs - not saving to examples dir"
                )

            train_file_save_path = os.path.join(
                package_base,
                "examples",
                "configs",
                f"{task}_configs",
                f"config_{dataset_name}_{suffix}.yaml",
            )
            if os.path.exists(train_file_save_path):
                logger.warning(
                    f"Config for {dataset_name}-{task} training already exists at {train_file_save_path}, not saving again."
                )

                # TODO - figure out how to not overwrite comments in the yaml file, then we can uncomment the below.
                """
                existing_config = yaml.load(open(os.path.join(package_base, "examples", "configs", f"{task}_configs", f"config_{dataset_name}_{suffix}.yaml")), yaml.SafeLoader)
                base_config = merge(destination=base_config, existing_config, strategy=Strategy.REPLACE) # overwrite base with existing
                # Ensure these specific keys are always set to our values - these are required.
                for key in dataset_updates:
                    base_config["dataset_config"][key] = dataset_updates[key]
                """
            else:
                # TODO - this will no longer be conditional once we figure out how to not overwrite comments in the yaml file.
                with open(train_file_save_path, "w") as f:
                    yaml.dump(base_config, f, sort_keys=False)
                logger.info(
                    f"Saved training config to examples dir: {train_file_save_path}"
                )
        except Exception as e:
            logger.warning(f"Error saving training config to examples dir: {e}")

    return base_config
