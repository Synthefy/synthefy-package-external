import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from filelock import FileLock
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from synthefy_pkg.configs.dataset_configs import (
    DatasetConfig,
)
from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.load_models import load_diffusion_transformer
from synthefy_pkg.model.trainers.diffusion_model import (
    TimeSeriesDiffusionModelTrainer,
)
from synthefy_pkg.model.trainers.metadata_only_forecasting_trainer import (
    MetadataPretrainForecastingTrainer,
)
from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer import (
    TimeSeriesDecoderForecastingFoundationTrainer,
)
from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_trainer import (
    TimeSeriesDecoderForecastingTrainer,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKBLUE
# from synthefy_pkg.utils.constrained_synthesis_utils import (
#     get_equality_constraints,
#     project_all_samples_to_equality_constraints,
# )

COMPILE = True


def check_generation_dir_exists(save_dir):
    if not os.path.exists(save_dir):
        return False
    # Check if save dir is empty or not
    if os.path.exists(save_dir) and os.listdir(save_dir):
        logger.info(
            OKBLUE
            + f"The synthetic dataset already exists. Skipping generation. `{save_dir}`"
            + ENDC
        )
        return True
    else:
        logger.info(OKBLUE + "Let's start the data generation process" + ENDC)
        logger.info(
            OKBLUE
            + "The synthetic dataset will be stored in: "
            + str(save_dir)
            + ENDC
        )
        return False


def add_missing_timeseries_columns(
    df: pd.DataFrame, preprocess_config_path: str
) -> pd.DataFrame:
    """Add missing timeseries columns with zero values to ensure preprocessing compatibility.
    Args:
        df (pd.DataFrame): Input DataFrame
        preprocess_config_path (str): Path to preprocessing configuration file
    Returns:
        pd.DataFrame: DataFrame with added missing timeseries columns
    """
    with open(preprocess_config_path, "r") as f:
        preprocess_config = yaml.safe_load(f)
    timeseries_cols = preprocess_config.get("timeseries", {}).get("cols", [])
    for col in timeseries_cols:
        if col not in df.columns:
            logger.warning(f"Adding missing timeseries column: {col} with 0s")
            df[col] = np.zeros(len(df))
    return df


def generate_synthetic_dataset_for_foundation_forecasting(
    dataset_config: DatasetConfig,
    save_dir,
    synthesizer,
    synthesis_function,
    dataloader=None,
    batch=None,
    batch_idx=None,
    train_or_val_or_test: str = "",
    task: str = "synthesis",
    gpu_id: Optional[int] = None,
):
    """Generate synthetic data either for a full dataloader or a single batch.
    Note that constraints are stored in the dataset config.

    Args:
        dataset_config: Configuration for the dataset
        save_dir: Directory to save results
        synthesizer: Model used for synthesis
        synthesis_function: Function that performs the synthesis
        dataloader: Optional dataloader to process multiple batches
        batch: Optional single batch to process
        batch_idx: Required if processing single batch
        train_or_val_or_test: Dataset split identifier
        task: Type of synthesis task
        gpu_id: Optional GPU ID for single batch processing
    """
    # Check if dataloader is empty
    if dataloader is not None and dataloader.dataset is None:
        logger.warning(
            f"Dataloader is empty for {train_or_val_or_test} set. Skipping generation."
        )
        return
    dataset_str = f"{train_or_val_or_test}_dataset"
    if "probabilistic_forecast" in task:
        dataset_str = "probabilistic_" + dataset_str

    save_dir = os.path.join(save_dir, dataset_str)
    os.makedirs(save_dir, exist_ok=True)

    # For full dataloader processing, check if already generated
    if dataloader is not None and check_generation_dir_exists(save_dir):
        return None

    # Create filename for H5 file
    prefix = "probabilistic_" if "probabilistic_forecast" in task else ""
    h5_filename = f"{prefix}{train_or_val_or_test}_combined_data.h5"
    h5_file_path = os.path.join(save_dir, h5_filename)

    # Create a lock file associated with the H5 file
    lock_path = h5_file_path + ".lock"
    lock = FileLock(lock_path)

    # Log generation info
    if dataloader is not None:
        # the on the fly dataloader cannot have a __len__ method
        if dataset_config.dataloader_name != "OTFSyntheticDataloader":
            total_samples = len(dataloader)
        else:
            total_samples = dataset_config.num_datasets
        logger.info(
            OKBLUE + f"Will generate {total_samples} synthetic samples" + ENDC
        )
        logger.info(OKBLUE + "Generating synthetic samples" + ENDC)
        batches = enumerate(tqdm(dataloader))
    else:
        logger.info(
            f"Generating synthetic data for batch {batch_idx} on GPU {gpu_id}"
        )
        if batch is None or batch_idx is None:
            raise ValueError(
                "batch/batch_idx cannot be None when dataloader is None"
            )
        batches = [(batch_idx, batch)]

    kwargs = {}
    if dataset_config.use_constraints:
        kwargs["dataset_config"] = dataset_config

    # Ensure the H5 file exists by creating it if needed
    with lock:
        if not os.path.exists(h5_file_path):
            with h5py.File(h5_file_path, "w") as f:
                pass

    for current_batch_idx, current_batch in batches:
        # Move batch to device
        for key, value in current_batch.items():
            if isinstance(value, torch.Tensor):
                current_batch[key] = value.to(synthesizer.config.device)
        # Generate synthetic data
        dataset_dict = synthesis_function(
            batch=current_batch,
            synthesizer=synthesizer,
            **kwargs,
        )
        # Get data from current batch
        predicted_forecast = [
            dataset_dict["predicted_forecast"].detach().cpu().numpy()
        ]
        true_forecast = [dataset_dict["true_forecast"].detach().cpu().numpy()]
        forecast_mask = [dataset_dict["forecast_mask"].detach().cpu().numpy()]
        history = [dataset_dict["history"].detach().cpu().numpy()]
        dataset_ids = [dataset_dict["dataset_ids"].detach().cpu().numpy()]

        # Concatenate all the data
        predicted_forecast = np.concatenate(predicted_forecast, axis=0)
        true_forecast = np.concatenate(true_forecast, axis=0)
        forecast_mask = np.concatenate(forecast_mask, axis=0)
        history = np.concatenate(history, axis=0)
        dataset_ids = np.concatenate(dataset_ids, axis=0)

        # Use file locking when writing to the H5 file
        with lock:
            with h5py.File(h5_file_path, "a") as f:
                # Check if datasets already exist
                if "predicted_forecast" not in f:
                    # Create datasets with maxshape for resizing
                    f.create_dataset(
                        "predicted_forecast",
                        data=predicted_forecast,
                        maxshape=(None,) + predicted_forecast.shape[1:],
                    )
                    f.create_dataset(
                        "true_forecast",
                        data=true_forecast,
                        maxshape=(None,) + true_forecast.shape[1:],
                    )
                    f.create_dataset(
                        "forecast_mask",
                        data=forecast_mask,
                        maxshape=(None,) + forecast_mask.shape[1:],
                    )
                    f.create_dataset(
                        "history",
                        data=history,
                        maxshape=(None,) + history.shape[1:],
                    )
                    f.create_dataset(
                        "dataset_ids",
                        data=dataset_ids,
                        maxshape=(None,) + dataset_ids.shape[1:],
                    )
                else:
                    # Resize datasets and append new data
                    for name, data in [
                        ("predicted_forecast", predicted_forecast),
                        ("true_forecast", true_forecast),
                        ("forecast_mask", forecast_mask),
                        ("history", history),
                        ("dataset_ids", dataset_ids),
                    ]:
                        dataset = cast(h5py.Dataset, f[name])
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[current_size:new_size] = data

    logger.info(f"All results stored in H5 file: {h5_file_path}")


def generate_synthetic_dataset(
    dataset_config,
    save_dir,
    synthesizer,
    synthesis_function,
    dataloader=None,
    batch=None,
    batch_idx=None,
    train_or_val_or_test: str = "",
    task: str = "synthesis",
    gpu_id: Optional[int] = None,
):
    """Generate synthetic data either for a full dataloader or a single batch.
    Note that constraints are stored in the dataset config.

    Args:
        dataset_config: Configuration for the dataset
        save_dir: Directory to save results
        synthesizer: Model used for synthesis
        synthesis_function: Function that performs the synthesis
        dataloader: Optional dataloader to process multiple batches
        batch: Optional single batch to process
        batch_idx: Required if processing single batch
        train_or_val_or_test: Dataset split identifier
        task: Type of synthesis task
        gpu_id: Optional GPU ID for single batch processing
    """
    # Check if dataloader is empty
    if dataloader is not None and dataloader.dataset is None:
        logger.warning(
            f"Dataloader is empty for {train_or_val_or_test} set. Skipping generation."
        )
        return

    horizon = dataset_config.time_series_length
    num_channels = dataset_config.num_channels
    num_discrete_conditions = dataset_config.num_discrete_conditions
    num_continuous_labels = dataset_config.num_continuous_labels
    dataset_str = f"{train_or_val_or_test}_dataset"
    if "probabilistic_forecast" in task:
        dataset_str = "probabilistic_" + dataset_str

    save_dir = os.path.join(save_dir, dataset_str)
    os.makedirs(save_dir, exist_ok=True)

    # For full dataloader processing, check if already generated
    if dataloader is not None and check_generation_dir_exists(save_dir):
        return None

    # Create filename for H5 file
    prefix = "probabilistic_" if "probabilistic_forecast" in task else ""
    h5_filename = f"{prefix}{train_or_val_or_test}_combined_data.h5"
    h5_file_path = os.path.join(save_dir, h5_filename)

    # Create a lock file associated with the H5 file
    lock_path = h5_file_path + ".lock"
    lock = FileLock(lock_path)

    # Log generation info
    if dataloader is not None:
        total_samples = len(dataloader.dataset)
        logger.info(
            OKBLUE + f"Will generate {total_samples} synthetic samples" + ENDC
        )
        logger.info(OKBLUE + "Generating synthetic samples" + ENDC)
        batches = enumerate(tqdm(dataloader))
    else:
        logger.info(
            f"Generating synthetic data for batch {batch_idx} on GPU {gpu_id}"
        )
        if batch is None or batch_idx is None:
            raise ValueError(
                "batch/batch_idx cannot be None when dataloader is None"
            )
        batches = [(batch_idx, batch)]

    kwargs = {}
    if dataset_config.use_constraints:
        kwargs["dataset_config"] = dataset_config

    # Ensure the H5 file exists by creating it if needed
    with lock:
        if not os.path.exists(h5_file_path):
            with h5py.File(h5_file_path, "w") as f:
                pass

    for current_batch_idx, current_batch in batches:
        # Move batch to device
        for key, value in current_batch.items():
            if isinstance(value, torch.Tensor):
                current_batch[key] = value.to(synthesizer.config.device)
        # Generate synthetic data
        dataset_dict = synthesis_function(
            batch=current_batch,
            synthesizer=synthesizer,
            **kwargs,
        )

        # Get data from current batch
        timeseries = [dataset_dict["timeseries"]]
        discrete_conditions = [dataset_dict["discrete_conditions"]]
        continuous_conditions = [dataset_dict["continuous_conditions"]]
        original_timeseries = (
            [current_batch["timeseries_full"].cpu().numpy()]
            if "timeseries_full" in current_batch
            else [current_batch["timeseries"].cpu().numpy()]
        )

        # Concatenate all the data
        timeseries = np.concatenate(timeseries, axis=0)
        discrete_conditions = np.concatenate(discrete_conditions, axis=0)
        continuous_conditions = np.concatenate(continuous_conditions, axis=0)
        original_timeseries = np.concatenate(original_timeseries, axis=0)

        # Use file locking when writing to the H5 file
        with lock:
            with h5py.File(h5_file_path, "a") as f:
                # Check if datasets already exist
                if "synthetic_timeseries" not in f:
                    # Create datasets with maxshape for resizing
                    f.create_dataset(
                        "synthetic_timeseries",
                        data=timeseries,
                        maxshape=(None,) + timeseries.shape[1:],
                    )
                    f.create_dataset(
                        "discrete_conditions",
                        data=discrete_conditions,
                        maxshape=(None,) + discrete_conditions.shape[1:],
                    )
                    f.create_dataset(
                        "continuous_conditions",
                        data=continuous_conditions,
                        maxshape=(None,) + continuous_conditions.shape[1:],
                    )
                    f.create_dataset(
                        "original_timeseries",
                        data=original_timeseries,
                        maxshape=(None,) + original_timeseries.shape[1:],
                    )
                else:
                    # Resize datasets and append new data
                    for name, data in [
                        ("synthetic_timeseries", timeseries),
                        ("discrete_conditions", discrete_conditions),
                        ("continuous_conditions", continuous_conditions),
                        ("original_timeseries", original_timeseries),
                    ]:
                        dataset = cast(h5py.Dataset, f[name])
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[current_size:new_size] = data

    # Validate shapes in the final H5 file
    with lock:
        with h5py.File(h5_file_path, "r") as f:
            timeseries_to_save = cast(h5py.Dataset, f["synthetic_timeseries"])
            discrete_conditions_to_save = cast(
                h5py.Dataset, f["discrete_conditions"]
            )
            continuous_conditions_to_save = cast(
                h5py.Dataset, f["continuous_conditions"]
            )

            # Validate shapes
            assert timeseries_to_save.shape[-1] == horizon
            assert timeseries_to_save.shape[-2] == num_channels
            assert (
                discrete_conditions_to_save.shape[-1] == num_discrete_conditions
            )
            assert (
                continuous_conditions_to_save.shape[-1] == num_continuous_labels
            )
            if dataset_config.dataloader_name != "FoundationModelDataLoader":
                if len(discrete_conditions_to_save.shape) > 2:
                    assert discrete_conditions_to_save.shape[-2] == horizon
                if len(continuous_conditions_to_save.shape) > 2:
                    assert continuous_conditions_to_save.shape[-2] == horizon

    logger.info(f"All results stored in H5 file: {h5_file_path}")


# TODO - remove this, its unused. Instead using get_synthesis_via_diffusion in diffusion_model.py
# TODO - the synthesis_via_X functions are very similar - refactor to use a common function
def synthesis_via_diffusion(batch, synthesizer, similarity_guidance_dict=None):
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]
    sample = input_["sample"]
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            logger.debug(f"Diffusion step {t} of {T}")
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            epsilon_theta = synthesizer(synthesis_input)
            x = (
                x
                - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
            ) / torch.sqrt(Alpha[t])
            noise = torch.randn_like(x).to(device)
            if t > 0:
                x = x + Sigma[t] * noise

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def get_sample_est_from_noisy_sample(
    noisy_sample, noise_est, current_alpha_bar
):
    sample_est = (1 / (current_alpha_bar**0.5 + 1e-8)) * (
        noisy_sample - noise_est * (1 - current_alpha_bar) ** 0.5
    )
    return sample_est


# TODO - the synthesis_via_X functions are very similar - refactor to use a common function
def projected_synthesis_via_diffusion(
    batch, synthesizer, control_param: float = 0.0
):
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]

    sample = input_["sample"]
    x = torch.randn_like(sample).to(device)

    B = x.shape[0]

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            epsilon_theta = synthesizer(synthesis_input)

            # get the sample estimate from the noisy sample and the noise estimate
            x0_est = get_sample_est_from_noisy_sample(
                x, epsilon_theta, Alpha_bar[t]
            )

            if t > 0:
                noise = torch.randn_like(x).to(device)
                x = (
                    (Alpha_bar[t - 1] ** 0.5) * x0_est
                    + (1.0 - Alpha_bar[t - 1] - control_param) ** 0.5
                    * epsilon_theta
                    + noise * (control_param**0.5)
                )
            else:
                x = x0_est

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


# TODO - the synthesis_via_X functions are very similar - refactor to use a common function
# def synthesis_via_projected_diffusion(
#     batch,
#     synthesizer,
#     dataset_config: DictConfig,
# ) -> Dict[str, np.ndarray]:
#     """Synthesize data using projected diffusion with equality constraints.

#     This function performs synthesis using a diffusion model while projecting samples to satisfy
#     equality constraints. The constraints are extracted either from the input batch or from
#     predetermined during preprocessing stage min/max values stored in dataset files.

#     The function supports various constraint types including:
#     - min/max: Minimum and maximum values per channel
#     - argmax/argmin: Index of maximum/minimum values
#     - mean: Mean value per channel
#     - mean change: Mean of differences between consecutive values
#     - autocorrelation: Autocorrelation at specified lags

#     Args:
#         batch: Input batch containing data and condition embeddings of shape (batch, channels, horizon)
#         synthesizer: Diffusion model used for synthesis
#         dataset_config (DictConfig): Configuration containing:
#             - dataset_name (str): Name of dataset for loading predetermined constraints from preprocessing stage.
#             - constraints: List of constraints to enforce
#                     (e.g. ["min", "max", "mean", "argmax", "argmin", "mean_change", "autocorr_1", "max and argmax"])
#             - projection_during_synthesis: Method to use for projection during synthesis: "strict" or "clipping"
#                 - "clipping": requires "min" or/and "max" constraints
#                 - "strict": can handle all constraints
#             - extract_equality_constraints_from_windows (bool): Whether to extract constraints from batch
#             - predetermined_constraint_values: Dictionary containing predetermined min/max or other constraint values for constraints
#                 e.g. {"min": 0.0, "max": 1.0}
#             - selectively_denoise: Whether to use selective denoising approach

#     Returns:
#         dict: Dictionary containing:
#             - timeseries: The synthesized time series data (numpy array)
#             - discrete_conditions: Discrete condition embeddings used (numpy array)
#             - continuous_conditions: Continuous condition embeddings used (numpy array)
#     """
#     T, Alpha, Alpha_bar, Sigma = (
#         synthesizer.diffusion_hyperparameters["T"],
#         synthesizer.diffusion_hyperparameters["Alpha"],
#         synthesizer.diffusion_hyperparameters["Alpha_bar"],
#         synthesizer.diffusion_hyperparameters["Sigma"],
#     )
#     device = synthesizer.device
#     Alpha = Alpha.to(device)
#     Alpha_bar = Alpha_bar.to(device)
#     Sigma = Sigma.to(device)

#     input_ = synthesizer.prepare_training_input(batch)
#     discrete_cond_input = input_["discrete_cond_input"]
#     continuous_cond_input = input_["continuous_cond_input"]

#     sample = input_["sample"]
#     equality_constraints = get_equality_constraints(dataset_config, sample)
#     B, K, H = sample.shape
#     x = torch.randn_like(sample).to(device)
#     warm_start_samples = sample.detach().cpu().numpy()

#     if dataset_config.selectively_denoise:
#         T = 20
#         Alpha_bar_current = Alpha_bar[T - 1]
#         x = (Alpha_bar_current**0.5) * sample + (
#             1 - Alpha_bar_current
#         ) ** 0.5 * torch.randn_like(sample).to(device)
#         logger.info("Using the selective denoising approach")
#         warm_start_samples = x.detach().cpu().numpy()

#     differences_list = []
#     with torch.no_grad():
#         for t in range(T - 1, -1, -1):
#             # sleep(0.001)  # TODO @sai - do we need this?
#             # logger.info(t)
#             diffusion_steps = torch.LongTensor(
#                 [
#                     t,
#                 ]
#                 * B
#             ).to(device)

#             # get the denoiser input
#             synthesis_input = {
#                 "noisy_sample": x,
#                 "discrete_cond_input": discrete_cond_input,
#                 "continuous_cond_input": continuous_cond_input,
#                 "diffusion_step": diffusion_steps,
#             }

#             # get the noise estimate
#             epsilon_theta = synthesizer(synthesis_input)

#             # get the clean sample estimate
#             # logger.info(Alpha_bar[t-1], Alpha_bar[t], t)
#             x0_est = get_sample_est_from_noisy_sample(
#                 x, epsilon_theta, Alpha_bar[t]
#             )
#             x0_est_numpy = x0_est.detach().cpu().numpy()

#             # perform penalty-based projection
#             if dataset_config.gamma_choice == "lin":
#                 penalty_coefficient = Alpha_bar[t].item() * 1e5
#             elif dataset_config.gamma_choice == "quad":
#                 penalty_coefficient = Alpha_bar[t].item() ** 2 * 1e5
#             else:
#                 penalty_coefficient = np.clip(
#                     np.exp(1 / (1 - Alpha_bar[t - 1].item())), 0.1, 1e5
#                 )

#             projected_x0_est = project_all_samples_to_equality_constraints(
#                 x0_est_numpy,
#                 equality_constraints,
#                 penalty_coefficient=penalty_coefficient,
#                 warm_start_samples=warm_start_samples,
#                 projection_method=dataset_config.projection_during_synthesis,
#                 dataset_name=dataset_config.dataset_name,
#             )
#             warm_start_samples = projected_x0_est
#             projected_x0_est = torch.tensor(projected_x0_est).to(device)

#             control_param = ((1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])) * (
#                 1 - Alpha_bar[t] / Alpha_bar[t - 1]
#             )  # DDPM
#             if t > 0:
#                 noise = torch.randn_like(x).to(device)
#                 x = (
#                     (Alpha_bar[t - 1] ** 0.5) * projected_x0_est
#                     + (1.0 - Alpha_bar[t - 1] - control_param) ** 0.5
#                     * epsilon_theta
#                     + noise * (control_param**0.5)
#                 )
#                 Alpha_bar_prev = Alpha_bar[t - 1]
#                 true_noisy_sample = (Alpha_bar_prev**0.5) * sample
#                 diff = true_noisy_sample - x
#                 samplewise_diff = torch.mean(torch.square(diff), dim=(1, 2))
#                 differences_list.append({t: samplewise_diff})

#             else:
#                 x = projected_x0_est

#     synthesized_timeseries = synthesizer.prepare_output(x)

#     dataset_dict = {
#         "timeseries": synthesized_timeseries,
#         "equality_constraints": equality_constraints,
#         "discrete_conditions": batch["discrete_label_embedding"]
#         .detach()
#         .cpu()
#         .numpy(),
#         "continuous_conditions": batch["continuous_label_embedding"]
#         .detach()
#         .cpu()
#         .numpy(),
#         "differences": differences_list,
#     }

#     return dataset_dict


def synthesis_via_gan(batch, synthesizer, similarity_guidance_dict=None):
    input_ = synthesizer.prepare_training_input(batch)
    synthesized = synthesizer.generator(
        x=input_["noise_for_generator"],
        y=input_["discrete_cond_input"],
        z=input_["continuous_cond_input"],
    )

    synthesized_timeseries = synthesizer.prepare_output(synthesized)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def forecast_via_diffusion(batch, synthesizer, forecast_length: int = 32):
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]
    sample = input_["sample"]
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device) 

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            # logger.debug(f"Diffusion step {t} of {T}")
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            epsilon_theta = synthesizer(synthesis_input)
            x0_est = get_sample_est_from_noisy_sample(x, epsilon_theta, Alpha_bar[t])
            x0_est[:, :, :-forecast_length] = sample[
                :, :, :-forecast_length
            ]  # assigin the observed part of the time series

            control_param = ((1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])) * (
                1 - Alpha_bar[t] / Alpha_bar[t - 1]
            )  # DDPM
            if t > 0:
                noise = torch.randn_like(x).to(device)
                x = (
                    (Alpha_bar[t - 1] ** 0.5) * x0_est
                    + (1.0 - Alpha_bar[t - 1] - control_param) ** 0.5 * epsilon_theta
                    + noise * (control_param**0.5)
                )
            else:
                x = x0_est

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def forecast_via_decoders(batch, synthesizer):
    # This needs work... I don't want to freeze/unfreeze layers because of prepare_training_input()
    batch["epoch"] = (
        0  # hardcoded to epoch 0 since we're not training, only sampling
    )
    decoder_input = synthesizer.prepare_training_input(batch, log_dir=None)
    forecast = synthesizer(decoder_input)

    # if these components are missing, replace with dummy values
    if "discrete_label_embedding" in batch:
        discrete_conditions = (
            batch["discrete_label_embedding"].detach().cpu().numpy()
        )
    else:
        discrete_conditions = np.zeros(
            (batch["timeseries"].shape[0], batch["timeseries"].shape[1], 0)
        )  # TODO: if it is necessary and not in the batch, this will produce an error downstream
    if "continuous_label_embedding" in batch:
        continuous_conditions = (
            batch["continuous_label_embedding"].detach().cpu().numpy()
        )
    elif (
        "metadata" in batch
    ):  # replace continuous labels with metadata in Foundation model context
        continuous_conditions = batch["metadata"].detach().cpu().numpy()
    else:
        continuous_conditions = np.zeros(
            (batch["timeseries"].shape[0], batch["timeseries"].shape[1], 0)
        )  # TODO: if it is necessary and not in the batch, this will produce an error downstream

    # append history and forecast to form timeseries
    history = decoder_input["history"]
    timeseries = torch.cat([history, forecast], dim=1)
    timeseries = timeseries.permute(0, 2, 1).cpu().detach().numpy()

    dataset_dict = {
        "timeseries": timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def probabilistic_forecast_via_decoders(batch, synthesizer, num_forecasts=100):
    batch["epoch"] = 0
    decoder_input = synthesizer.prepare_training_input(batch, log_dir=None)

    # Set model to training mode to enable dropout
    synthesizer.train()

    # TODO: Parallelize this
    forecasts = []
    for idx in range(num_forecasts):
        forecast = synthesizer(decoder_input)
        forecasts.append(forecast)
    forecasts = torch.stack(forecasts)
    forecasts = forecasts.permute(1, 0, 2, 3)

    # Set model back to evaluation mode
    synthesizer.eval()

    discrete_conditions = (
        batch["discrete_label_embedding"].detach().cpu().numpy()
    )
    continuous_conditions = (
        batch["continuous_label_embedding"].detach().cpu().numpy()
    )

    # append history and forecast to form timeseries
    # (batch_size, num_channels, history_len) > (batch_size, num_forecasts, num_channels, history_len)
    history = (
        decoder_input["history"].unsqueeze(1).repeat(1, num_forecasts, 1, 1)
    )
    # (batch_size, num_forecasts, num_channels, history_len) + (batch_size, num_forecasts, num_channels, forecast_len) > (batch_size, num_forecasts, num_channels, history_len + forecast_len)
    timeseries = torch.cat([history, forecasts], dim=2)
    # (batch_size, num_forecasts, num_channels, history_len + forecast_len) > (batch_size, num_forecasts, history_len + forecast_len, num_channels) - TODO: double check this.
    timeseries = timeseries.permute(0, 1, 3, 2).cpu().detach().numpy()

    dataset_dict = {
        "timeseries": timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def autoregressive_forecast_via_foundation_model_v2(batch, synthesizer):
    # This function is basically no longer used, but keeping imports just in case
    from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v2 import (
        forecast_next_step,
        get_hidden_states,
        get_target_embeddings,
    )

    decoder_input = synthesizer.prepare_training_input(
        batch, log_dir=None
    )  # SynthefyForecastingModelV1.prepare_training_input()

    history_length = (
        synthesizer.dataset_config.time_series_length
        - synthesizer.dataset_config.forecast_length
    )

    decoder_input_subset = {
        "timestamps": decoder_input["timestamps"][..., :history_length],
        "timestamps_mask": decoder_input["timestamps_mask"][
            ..., :history_length
        ],
        "descriptions": decoder_input["descriptions"],
        "descriptions_mask": decoder_input["descriptions_mask"],
        "continuous": decoder_input["continuous"][..., :history_length],
        "continuous_mask": decoder_input["continuous_mask"][
            ..., :history_length
        ],
    }
    decoder_target_subset = {
        "timestamps": decoder_input["timestamps"][..., history_length:],
        "timestamps_mask": decoder_input["timestamps_mask"][
            ..., history_length:
        ],
        "descriptions": decoder_input["descriptions"],
        "descriptions_mask": decoder_input["descriptions_mask"],
        "continuous": decoder_input["continuous"][..., history_length:],
        "continuous_mask": decoder_input["continuous_mask"][
            ..., history_length:
        ],
    }

    target_embeddings = get_target_embeddings(
        synthesizer, decoder_target_subset
    )

    batch_size = decoder_input["timestamps"].shape[0]
    num_correlates = decoder_input["timestamps"].shape[1]

    forecast_length = synthesizer.dataset_config.forecast_length

    forecast_list = []

    for fidx in tqdm(
        range(forecast_length), total=forecast_length, desc="Forecasting"
    ):
        hidden_states = get_hidden_states(synthesizer, decoder_input_subset)
        hidden_states = hidden_states.reshape(
            batch_size, num_correlates, -1, synthesizer.decoder_model_dims
        )
        assert hidden_states.shape[-2] == history_length + fidx
        required_hidden_states = hidden_states[:, :, -1]
        required_target_embeddings = target_embeddings[:, :, fidx]

        next_step_forecast = forecast_next_step(
            synthesizer, required_hidden_states, required_target_embeddings
        )

        # update the decoder_input_subset with the next step forecast
        decoder_input_subset["continuous"] = torch.cat(
            [
                decoder_input_subset["continuous"],
                next_step_forecast.unsqueeze(-1),
            ],
            dim=-1,
        )
        decoder_input_subset["timestamps"] = decoder_input["timestamps"][
            ..., : history_length + fidx + 1
        ]
        decoder_input_subset["continuous_mask"] = torch.cat(
            [
                decoder_input_subset["continuous_mask"],
                torch.zeros_like(
                    decoder_input_subset["continuous_mask"][..., :1]
                ).bool(),
            ],
            dim=-1,
        )
        decoder_input_subset["timestamps_mask"] = torch.cat(
            [
                decoder_input_subset["timestamps_mask"],
                torch.zeros_like(
                    decoder_input_subset["timestamps_mask"][..., :1]
                ).bool(),
            ],
            dim=-1,
        )
        forecast_list.append(next_step_forecast)

    forecast = decoder_target_subset["continuous"].squeeze(-2)
    predicted_forecast = torch.cat(forecast_list, dim=-1)

    dataset_dict = {
        "predicted_forecast": predicted_forecast,
        "true_forecast": forecast,
        "forecast_mask": decoder_target_subset["continuous_mask"],
        "history": decoder_input_subset["continuous"],
    }

    return dataset_dict


def load_synthesis_model(
    configuration: Configuration,
    model_checkpoint_path: Optional[str],
) -> Tuple[TimeSeriesDiffusionModelTrainer, int, int]:
    # Load weights to resume training
    diffusion_model = load_diffusion_transformer(configuration)

    if model_checkpoint_path is None:
        logger.warning(OKBLUE + "No model checkpoint provided" + ENDC)
        model_trainer = TimeSeriesDiffusionModelTrainer(
            configuration, diffusion_model
        )
        start_epoch = 0
        global_step = 0

    else:
        logger.warning(
            OKBLUE
            + f"Loading model from checkpoint {model_checkpoint_path}"
            + ENDC
        )

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {model_checkpoint_path} does not exist."
            )

        try:
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=torch.device(configuration.training_config.device),
            )
            start_epoch = int(checkpoint["epoch"] + 1)
            global_step = int(checkpoint["global_step"])
            logger.info(
                f"Resuming from epoch: {start_epoch}, global_step: {global_step}"
            )
            del checkpoint
            model_trainer = (
                TimeSeriesDiffusionModelTrainer.load_from_checkpoint(
                    model_checkpoint_path,
                    config=configuration,
                    diffusion_model=diffusion_model,
                )
            )
            logger.info(
                "Model loaded from checkpoint: {}".format(model_checkpoint_path)
            )
            return model_trainer, start_epoch, global_step
        except Exception as e:
            logger.exception(
                f"Error loading checkpoint from {model_checkpoint_path}: {e}"
            )
            raise e

    return model_trainer, start_epoch, global_step


def load_forecast_model(
    configuration: Configuration,
    model_checkpoint_path: Optional[str],
) -> Tuple[
    Union[
        TimeSeriesDecoderForecastingTrainer,
        MetadataPretrainForecastingTrainer,
        TimeSeriesDecoderForecastingFoundationTrainer,
    ],
    int,
    int,
]:
    if configuration.trainer == "metadata_encoder":
        trainer_class = MetadataPretrainForecastingTrainer
    elif configuration.trainer == "foundation_model":
        trainer_class = TimeSeriesDecoderForecastingFoundationTrainer
    elif configuration.trainer == "forecasting_model":
        trainer_class = TimeSeriesDecoderForecastingTrainer
    elif configuration.trainer == "default_trainer":
        logger.warning(
            "Default trainer for forecast model loading is TimeSeriesDecoderForecastingTrainer."
        )
        trainer_class = TimeSeriesDecoderForecastingTrainer
    elif configuration.trainer == "synthesis_model":
        raise ValueError(
            "Synthesis trainer cannot be used for forecasting task."
        )
    else:
        raise ValueError(
            f"Invalid trainer for load_forecast_model: {configuration.trainer}"
        )

    if not model_checkpoint_path:
        logger.warning(OKBLUE + "No model checkpoint provided" + ENDC)
        model_trainer = trainer_class(configuration)
        start_epoch = 0
        global_step = 0
    else:
        # Load weights to resume training
        logger.warning(
            OKBLUE
            + f"Loading model from checkpoint {model_checkpoint_path}"
            + ENDC
        )

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file {model_checkpoint_path} does not exist."
            )

        try:
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=torch.device(configuration.device),
            )
            start_epoch = int(checkpoint["epoch"] + 1)
            global_step = int(checkpoint["global_step"])
            logger.info(
                f"Resuming from epoch: {start_epoch}, global_step: {global_step}"
            )

            # Check if we need to handle the architecture differences
            if hasattr(configuration, "training_config") and getattr(
                configuration.training_config, "strict_load", True
            ):
                # Regular loading
                model_trainer = trainer_class.load_from_checkpoint(
                    model_checkpoint_path,
                    config=configuration,
                    scaler=None,
                    strict=True,
                )
                logger.info(
                    "Model loaded from checkpoint: {}".format(
                        model_checkpoint_path
                    )
                )
                return model_trainer, start_epoch, global_step
            else:
                # Create model first
                model_trainer = trainer_class(config=configuration)

                # Filter out the postprocessor layers that don't match
                filtered_state_dict = {
                    k: v
                    for k, v in checkpoint["state_dict"].items()
                    if not (
                        k.startswith("decoder_model.postprocessor")
                        or k.startswith("decoder_model.distribution")
                    )
                }

                # Load partial state dict
                model_trainer.load_state_dict(filtered_state_dict, strict=False)
                logger.info(
                    "Loaded checkpoint partially - skipped postprocessor layers due to architecture change"
                )

                return model_trainer, start_epoch, global_step

        except Exception as e:
            logger.exception(
                f"Error loading checkpoint from {model_checkpoint_path}: {e}"
            )
            raise e

    return model_trainer, start_epoch, global_step
