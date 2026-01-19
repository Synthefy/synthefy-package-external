import os

import h5py
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from synthefy_pkg.model.trainers.diffusion_model import (
    get_synthesis_via_diffusion as synthesis_via_diffusion,
)
from synthefy_pkg.utils.basic_utils import (
    ENDC,
    OKBLUE,
    OKYELLOW,
    get_dataloader_purpose,
    get_dataset_config,
    import_from,
    seed_everything,
)
from synthefy_pkg.utils.synthesis_utils import (
    forecast_via_decoders,
    forecast_via_diffusion,
    synthesis_via_gan,
)

synthesis_function_dict = {
    "synthesis_via_diffusion": synthesis_via_diffusion,
    "synthesis_via_gan": synthesis_via_gan,
    "forecast_via_diffusion": forecast_via_diffusion,
    "forecast_via_decoders": forecast_via_decoders,
}

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))

COMPILE = False


def load_pretrained_model(model_type, model_checkpoint_path, config):
    print(OKBLUE + "loading model from checkpoint" + ENDC)
    model = model_type.load_from_checkpoint(
        model_checkpoint_path, config=config, scaler=None, strict=True
    )
    print(OKBLUE + "model loaded from checkpoint" + ENDC)
    return model


def generate_synthetic_dataset(
    dataloader,
    synthesizer,
    synthesis_function,
    save_dir,
    dataset_config,
    in_gan_space=False,
    scaler=None,
    train=False,
    val=False,
    test=False,
    task="synthesis",
):
    combined_data_str_to_add = "combined_data.h5"

    def append_to_hdf5(
        synthetic_timeseries,
        discrete_conditions,
        continuous_conditions,
        original_timeseries,
    ):
        with h5py.File(combined_data_loc, "a") as f:
            # Create or append to synthetic_timeseries dataset
            if "synthetic_timeseries" not in f:
                f.create_dataset(
                    "synthetic_timeseries",
                    data=synthetic_timeseries,
                    maxshape=(None, *synthetic_timeseries.shape[1:]),
                )
            else:
                f["synthetic_timeseries"].resize(  # type: ignore
                    (
                        f["synthetic_timeseries"].shape[0]  # type: ignore
                        + synthetic_timeseries.shape[0]
                    ),
                    axis=0,
                )
                f["synthetic_timeseries"][-synthetic_timeseries.shape[0] :] = (  # type: ignore
                    synthetic_timeseries
                )

            # Create or append to discrete_conditions dataset
            if "discrete_conditions" not in f:
                f.create_dataset(
                    "discrete_conditions",
                    data=discrete_conditions,
                    maxshape=(None, *discrete_conditions.shape[1:]),
                )
            else:
                f["discrete_conditions"].resize(  # type: ignore
                    (
                        f["discrete_conditions"].shape[0]  # type: ignore
                        + discrete_conditions.shape[0]
                    ),
                    axis=0,
                )
                f["discrete_conditions"][-discrete_conditions.shape[0] :] = (  # type: ignore
                    discrete_conditions
                )

            # Create or append to continuous_conditions dataset
            if "continuous_conditions" not in f:
                f.create_dataset(
                    "continuous_conditions",
                    data=continuous_conditions,
                    maxshape=(None, *continuous_conditions.shape[1:]),
                )
            else:
                f["continuous_conditions"].resize(  # type: ignore
                    (
                        f["continuous_conditions"].shape[0]  # type: ignore
                        + continuous_conditions.shape[0]
                    ),
                    axis=0,
                )
                f["continuous_conditions"][  # type: ignore
                    -continuous_conditions.shape[0] :
                ] = continuous_conditions

            # Create or append to original_timeseries dataset
            if "original_timeseries" not in f:
                f.create_dataset(
                    "original_timeseries",
                    data=original_timeseries,
                    maxshape=(None, *original_timeseries.shape[1:]),
                )
            else:
                f["original_timeseries"].resize(  # type: ignore
                    (
                        f["original_timeseries"].shape[0]  # type: ignore
                        + original_timeseries.shape[0]
                    ),
                    axis=0,
                )
                f["original_timeseries"][-original_timeseries.shape[0] :] = (  # type: ignore
                    original_timeseries
                )

        # logger.debug("Data appended successfully to HDF5 file.")

    if train:
        combined_data_loc = (
            os.path.join(save_dir, "train_" + f"{combined_data_str_to_add}")
            if "probabilistic" not in task
            else os.path.join(
                save_dir, "probabilistic_train_" + f"{combined_data_str_to_add}"
            )
        )

    if val:
        combined_data_loc = (
            os.path.join(save_dir, "val_" + f"{combined_data_str_to_add}")
            if "probabilistic" not in task
            else os.path.join(
                save_dir, "probabilistic_val_" + f"{combined_data_str_to_add}"
            )
        )

    if test:
        combined_data_loc = (
            os.path.join(save_dir, "test_" + f"{combined_data_str_to_add}")
            if "probabilistic" not in task
            else os.path.join(
                save_dir, "probabilistic_test_" + f"{combined_data_str_to_add}"
            )
        )

    if os.path.exists(combined_data_loc):
        print(
            OKBLUE
            + f"The synthetic dataset already exists. Skipping data generation. {combined_data_loc}"
            + ENDC
        )
        return None

    else:
        print(OKBLUE + "Let's start the data generation process" + ENDC)
        print(
            OKBLUE
            + "The synthetic dataset will be stored in: "
            + str(save_dir)
            + ENDC
        )
        print(
            OKBLUE
            + "The combined synthetic data will be stored in: "
            + str(combined_data_loc)
            + ENDC
        )

    print(OKBLUE + "Generating synthetic samples" + ENDC)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        for key, value in batch.items():
            if key in [
                "window_indices",
                "shard_indices",
                "shard_local_indices",
            ]:
                continue
            batch[key] = value.to(synthesizer.config.device)

        if "timeseries_full" not in batch:
            # There are 2 ways to do this, which one we prefer?
            # batch["timeseries_full"] = batch["timeseries"][:, :, batch["timeseries"].shape[2]-dataset_config.time_series_length-1:-1]
            batch["timeseries_full"] = batch["timeseries"][
                :,
                :,
                dataset_config.continuous_start_idx : dataset_config.dataset_idx_start_idx,
            ]
            assert (
                batch["timeseries_full"].shape[2]
                == dataset_config.time_series_length
            )

        dataset_dict = synthesis_function(
            batch=batch,
            synthesizer=synthesizer,
        )

        if in_gan_space:
            # print(OKBLUE + "Converting the samples from gan space to normal space" + ENDC)
            if scaler is None:
                raise ValueError(
                    "Scaler is None. Please provide a scaler to convert the samples from gan space to normal space."
                )
            dataset_dict = scaler.convert_from_gan_to_normal(
                dataset_dict=dataset_dict,
            )

        # Append to hdf5 file
        append_to_hdf5(
            dataset_dict["timeseries"],
            dataset_dict["discrete_conditions"],
            dataset_dict["continuous_conditions"],
            batch["timeseries_full"].cpu().numpy(),
        )


@hydra.main(config_path="../../../examples/configs/", version_base="1.1")
def main(config: DictConfig):
    assert load_dotenv(
        os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env")
    )
    seed_everything(config.seed)
    pl_dataloader = import_from(
        f"synthefy_pkg.data.{config.dataloader_file}",
        f"{config.dataloader_model}",
    )(config)  # typically ForecastingDataLoader
    get_dataloader_purpose(pl_dataloader)

    # define model type
    synthesizer_wrapper_type = import_from(
        f"synthefy_pkg.model.{config.synthesizer_wrapper_model_file}_trainer",
        config.synthesizer_wrapper_model_name,  # typically TimeSeriesDecoderForecastingTrainer
    )

    # load model with pretrained weights
    synthesizer_wrapper = load_pretrained_model(
        synthesizer_wrapper_type,
        config.synthesizer_wrapper_checkpoint_path,
        config,
    )

    if config.should_compile_torch:
        synthesizer_wrapper = torch.compile(
            synthesizer_wrapper
        )  # compiles the model and *step (training/validation/prediction)
        # torch._dynamo.config.log_level = logging.ERROR

    L.seed_everything(config.seed)

    torch.set_float32_matmul_precision("high")
    synthesizer_wrapper.eval()
    for parameter in synthesizer_wrapper.parameters():
        parameter.requires_grad = False

    # assign config.denosier_checkpoint_path to denoiser_model.log_dir excluding the last two folders in the path
    save_dir = "/" + os.path.join(
        *config.synthesizer_wrapper_checkpoint_path.split("/")[:-2]
    )
    save_dir = os.path.join(save_dir, "vanilla_sampling")
    print(
        OKYELLOW
        + "All the results will be stored in this directory: "
        + str(save_dir)
        + ENDC
    )
    os.makedirs(save_dir, exist_ok=True)

    # select the synthesizer and the synthesis function
    if "diffusion" in config.synthesizer_wrapper_model_file:
        synthesizer = synthesizer_wrapper.denoiser_model
        synthesis_function_str = "_via_diffusion"
    elif "gan" in config.synthesizer_wrapper_model_file:
        synthesizer = synthesizer_wrapper.synthesizer
        synthesis_function_str = "_via_gan"
    elif "decoder" in config.synthesizer_wrapper_model_file:
        synthesizer = synthesizer_wrapper.decoder_model
        synthesis_function_str = "_via_decoders"
    else:
        raise NotImplementedError

    # modify the synthesis function
    if config.task == "forecast":
        synthesis_function_str = "forecast" + synthesis_function_str
    elif config.task == "synthesis":
        synthesis_function_str = "synthesis" + synthesis_function_str

    synthesis_function = synthesis_function_dict[synthesis_function_str]

    print(
        OKYELLOW
        + "The synthesis function is: "
        + str(synthesis_function_str)
        + ENDC
    )

    dataset_config = get_dataset_config(config)

    in_gan_space = True if config.experiment == "gan" else False

    scaler = None

    print(
        OKYELLOW
        + "Let us first generate the synthetic dataset for the test conditions"
        + ENDC
    )
    test_dataloader = pl_dataloader.test_dataloader()
    generate_synthetic_dataset(
        dataloader=test_dataloader,
        synthesizer=synthesizer,
        synthesis_function=synthesis_function,
        save_dir=save_dir,
        dataset_config=dataset_config,
        in_gan_space=in_gan_space,
        scaler=scaler,
        test=True,
        task=config.task,
    )


if __name__ == "__main__":
    main()
