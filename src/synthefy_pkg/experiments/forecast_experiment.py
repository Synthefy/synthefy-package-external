import os
from typing import Any, Callable, Dict, List, Optional, Union, cast

import h5py
import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger

from synthefy_pkg.data.foundation_dataloader import FoundationModelDataLoader
from synthefy_pkg.data.general_pl_dataloader import ForecastingDataLoader
from synthefy_pkg.data.sharded_dataloader import ShardedDataloaderV1
from synthefy_pkg.data.v3_sharded_dataloader import V3ShardedDataloader
from synthefy_pkg.experiments.experiment import Experiment
from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_trainer import (
    TimeSeriesDecoderForecastingTrainer,
)
from synthefy_pkg.train.model_train import ModelTrain
from synthefy_pkg.utils.basic_utils import seed_everything
from synthefy_pkg.utils.sagemaker_utils import (
    get_sagemaker_save_dir,
    is_running_in_sagemaker,
)
from synthefy_pkg.utils.synthesis_utils import (
    autoregressive_forecast_via_foundation_model_v2,
    forecast_via_decoders,
    generate_synthetic_dataset,
    load_forecast_model,
    probabilistic_forecast_via_decoders,
)

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert SYNTHEFY_PACKAGE_BASE is not None
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
assert SYNTHEFY_DATASETS_BASE is not None

COMPILE = True


class ForecastExperiment(Experiment):
    def __init__(
        self, config_source: Union[str, Dict[str, Any]] = "config.yaml"
    ):
        super().__init__(config_source)
        self.metadata_encoder_run_folder = "meta_pretrain"

    def _setup(self):
        seed_everything(self.configuration.seed)
        # Note: This class supports only SynthefyForecastingModelV1 and TimeSeriesDecoderForecastingTrainer
        # Other models are not supported yet.

    def _setup_train(
        self,
        model_checkpoint_path: Optional[str] = None,
        metadata_pretrain_epochs: int = 0,
    ):
        if (
            self.configuration.dataset_config.dataloader_name
            == "FoundationModelDataLoader"
        ):
            self.data_loader = FoundationModelDataLoader(self.configuration)
            raise NotImplementedError(
                "FoundationModelDataLoader and forecasting inference are not supported"
            )
        elif (
            self.configuration.dataset_config.dataloader_name
            == "ShardedDataloaderV1"
        ):
            self.data_loader = ShardedDataloaderV1(self.configuration)
        elif (
            self.configuration.dataset_config.dataloader_name
            == "V3ShardedDataloader"
        ):
            self.data_loader = V3ShardedDataloader(self.configuration)
        else:
            self.data_loader = ForecastingDataLoader(self.configuration)

        if hasattr(self.configuration, "metadata_encoder_config"):
            setattr(
                self.configuration.metadata_encoder_config,
                "metadata_pretrain_epochs",
                metadata_pretrain_epochs,
            )
        self.model_trainer, start_epoch, global_step = load_forecast_model(
            self.configuration, model_checkpoint_path
        )
        torch.set_float32_matmul_precision("high")

        self.training_runner = ModelTrain(
            config=self.configuration,
            dataset_generator=self.data_loader,
            model_trainer=self.model_trainer,
            start_epoch=start_epoch,
            global_step=global_step,
        )
        # This isn't very proper because the model tries to write its own plots to log_dir
        self.model_trainer.log_dir = self.training_runner.log_dir

        self._setup_sagemaker_checkpoint_callback()

    def _set_run_name_for_metadata_encoder_training(self):
        if hasattr(self.configuration, "run_name"):
            self.configuration.run_name = os.path.join(
                self.configuration.run_name, self.metadata_encoder_run_folder
            )

    def _reset_run_name(self):
        if hasattr(self.configuration, "run_name"):
            assert (
                self.metadata_encoder_run_folder in self.configuration.run_name
            )
            self.configuration.run_name = os.path.dirname(
                self.configuration.run_name
            )

    def train_metadata_encoder(
        self, model_checkpoint_path: Optional[str] = None
    ):
        self.configuration.training_config.max_epochs, train_epochs = (
            self.configuration.metadata_encoder_config.metadata_pretrain_epochs,
            self.configuration.training_config.max_epochs,
        )
        # make sure the run name is updated to save these graphs in a separate folder
        # metadata encoder training logged into a subfolder
        self._set_run_name_for_metadata_encoder_training()
        self._setup_train(
            model_checkpoint_path=model_checkpoint_path,
            metadata_pretrain_epochs=self.configuration.metadata_encoder_config.metadata_pretrain_epochs,
        )
        self.training_runner.train()

        # revert the number of epochs to run as the metadata number + train number (so loading happens properly)
        self.configuration.training_config.max_epochs = (
            self.configuration.metadata_encoder_config.metadata_pretrain_epochs
            + train_epochs
        )
        # main training logged in original log directory
        self._reset_run_name()

        return os.path.join(self.training_runner.save_path, "best_model.ckpt")

    def train(self, model_checkpoint_path: Optional[str] = None):
        if (
            hasattr(self, "configuration")
            and hasattr(self.configuration, "metadata_encoder_config")
            and getattr(
                self.configuration.metadata_encoder_config,
                "metadata_pretrain_epochs",
                0,
            )
            > 0
        ):
            # modifies checkpoint path to the metadata encoder checkpoint path

            model_checkpoint_path = self.train_metadata_encoder(
                model_checkpoint_path=model_checkpoint_path
            )

        self._setup_train(
            model_checkpoint_path=model_checkpoint_path,
            metadata_pretrain_epochs=0,
        )
        self.training_runner.train()

    def _setup_inference(
        self, model_checkpoint_path: str
    ) -> Union[TimeSeriesDecoderForecastingTrainer, Callable]:
        # Load model with pretrained weights
        model, _, _ = load_forecast_model(
            self.configuration, model_checkpoint_path
        )

        if self.configuration.should_compile_torch:
            # compiles the model and *step (training/validation/prediction)
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Torch compile failed: {str(e)}")

        L.seed_everything(self.configuration.seed)

        torch.set_float32_matmul_precision("high")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        if (
            hasattr(self.configuration, "denoiser_config")
            and self.configuration.denoiser_config is not None
            and self.configuration.denoiser_config.use_probabilistic_forecast
        ):
            self.synthesis_function = probabilistic_forecast_via_decoders
        else:
            # FoundationModelDataLoader and ShardedDataloaderV1 are for Foundation Model
            # ForecastingDataLoader is for trainable models.
            if (
                self.configuration.dataset_config.dataloader_name
                == "FoundationModelDataLoader"
            ):
                self.data_loader = FoundationModelDataLoader(self.configuration)
                raise NotImplementedError(
                    "FoundationModelDataLoader and forecasting inference are not supported"
                )
            elif (
                self.configuration.dataset_config.dataloader_name
                == "ShardedDataloaderV1"
            ):
                self.data_loader = ShardedDataloaderV1(self.configuration)
                self.synthesis_function = model.decoder_model.synthesis_function
            elif (
                self.configuration.dataset_config.dataloader_name
                == "V3ShardedDataloader"
            ):
                self.data_loader = V3ShardedDataloader(self.configuration)
                self.synthesis_function = model.decoder_model.synthesis_function
            else:
                self.data_loader = ForecastingDataLoader(self.configuration)
                self.synthesis_function = forecast_via_decoders

        return model

    def forecast_one_window(self, model_checkpoint_path: str, batch: dict):
        model = self._setup_inference(
            model_checkpoint_path=model_checkpoint_path
        )

        for key, value in batch.items():
            batch[key] = torch.tensor(value, dtype=torch.float32).to(
                model.decoder_model.config.device
            )

        batch_size = batch["timeseries_full"].shape[0]
        channels = batch["timeseries_full"].shape[1]

        dataset_dict = self.synthesis_function(
            batch=batch,
            synthesizer=model.decoder_model,
        )

        if not dataset_dict["timeseries"].shape[0] == batch_size:
            raise ValueError(
                f"Expected batch size {batch_size}, got {dataset_dict['timeseries'].shape[0]}"
            )
        if not dataset_dict["timeseries"].shape[1] == channels:
            raise ValueError(
                f"Expected {channels} channels, got {dataset_dict['timeseries'].shape[1]}"
            )

        # Note: timeseries is the concatenation of the history and the forecast.
        # Note: drop the batch dimension
        return dataset_dict["timeseries"][0]

    def forecast_n_windows(self):
        raise NotImplementedError

    def generate_synthetic_data(
        self,
        model_checkpoint_path: str,
        splits: List[str] = ["test"],
        output_dir: Optional[str] = None,
    ) -> bool:
        model = self._setup_inference(
            model_checkpoint_path=model_checkpoint_path
        )
        if (
            self.configuration.dataset_config.dataloader_name
            == "FoundationModelDataLoader"
        ):
            self.data_loader = FoundationModelDataLoader(self.configuration)
        else:
            self.data_loader = ForecastingDataLoader(self.configuration)

        # Get base save directory
        if output_dir is None:
            if is_running_in_sagemaker():
                save_dir = get_sagemaker_save_dir("")
                logger.info(
                    f"Running in SageMaker environment, saving to: {save_dir}"
                )
            else:
                if SYNTHEFY_DATASETS_BASE is None:
                    raise ValueError(
                        "SYNTHEFY_DATASETS_BASE environment variable is not set"
                    )
                save_dir = self.configuration.get_save_dir(
                    SYNTHEFY_DATASETS_BASE
                )

        else:
            save_dir = output_dir

        os.makedirs(save_dir, exist_ok=True)

        for split in splits:
            logger.info(
                f"Generating the synthetic dataset for the {split} conditions"
            )
            dataloader = getattr(self.data_loader, f"{split}_dataloader")()

            task = (
                "probabilistic_forecast"
                if hasattr(self.configuration, "denoiser_config")
                and self.configuration.denoiser_config is not None
                and self.configuration.denoiser_config.use_probabilistic_forecast
                else "forecast"
            )

            # skipping probabilistic forecast for any split other than the test
            # this is too expensive to run for all splits and we only need some quantitative metrics on the test set
            if task == "probabilistic_forecast" and split != "test":
                logger.warning(
                    f"Skipping probabilistic forecast for the {split} split"
                )
                continue

            generate_synthetic_dataset(
                dataloader=dataloader,
                synthesizer=model.decoder_model,  # SynthefyForecastingModelV{1,2,2a}
                synthesis_function=self.synthesis_function,  # {forecast_via_decoders, probabilistic_forecast_via_decoders, TabPFNBaseline.synthesis_function}
                save_dir=save_dir,
                dataset_config=self.configuration.dataset_config,
                train_or_val_or_test=split,
                task=task,
            )

            dataset_str = f"{split}_dataset"
            if "probabilistic_forecast" in task:
                dataset_str = "probabilistic_" + dataset_str

            save_dir = os.path.join(save_dir, dataset_str)
            prefix = (
                "probabilistic_" if "probabilistic_forecast" in task else ""
            )
            h5_filename = f"{prefix}{split}_combined_data.h5"
            h5_file_path = os.path.join(save_dir, h5_filename)

            if (
                self.configuration.dataset_config.dataloader_name
                == "FoundationModelDataLoader"
            ):
                with h5py.File(h5_file_path, "r") as f:
                    timeseries = cast(h5py.Dataset, f["synthetic_timeseries"])
                    original_timeseries = cast(
                        h5py.Dataset, f["original_timeseries"]
                    )
                    logger.info(
                        f"MSE: {np.mean((np.array(timeseries) - np.array(original_timeseries)) ** 2)}"
                    )
                    logger.info(
                        f"MAE: {np.mean(np.abs(np.array(timeseries) - np.array(original_timeseries)))}"
                    )

        logger.info(f"All the results stored at: {save_dir}")

        # Always return True for forecast experiment since no multi-GPU generation
        return True
