import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.data.general_pl_dataloader import (
    SynthesisModelv1DataLoader,
    SynthesisModelv1DatasetFromMetadata,
)
from synthefy_pkg.experiments.experiment import Experiment
from synthefy_pkg.model.load_models import load_cltsp_model_from_checkpoint
from synthefy_pkg.model.trainers.cltsp_trainer import CLTSPTrainer
from synthefy_pkg.preprocessing.preprocess import (
    LAG_COL_FORMAT,
    DataPreprocessor,
)
from synthefy_pkg.train.model_train import ModelTrain
from synthefy_pkg.utils.basic_utils import (
    ENDC,
    OKBLUE,
    OKYELLOW,
    get_num_devices,
    seed_everything,
)
from synthefy_pkg.utils.synthesis_utils import (
    generate_synthetic_dataset,
    load_synthesis_model,
)

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

COMPILE = True


MAX_STATUS_PERCENTAGE_FOR_CELERY = 95


class CLTSPExperiment(Experiment):
    def __init__(
        self, config_source: Union[str, Dict[str, Any]] = "config.yaml"
    ):
        super().__init__(config_source)

    def _setup(self):
        seed_everything(self.configuration.seed)
        # TODO - add MLFLOW
        # mlf_logger = MLFlowLogger(
        #     experiment_name=config.experiment_name,
        #     run_name=config.run_name,
        #     tracking_uri="file://" + config.mlflow_folder,
        # )

    def _setup_train(self, model_checkpoint_path: Optional[str] = None):
        self.data_loader = SynthesisModelv1DataLoader(
            self.configuration.dataset_config
        )

        torch.set_float32_matmul_precision("high")
        self.model_trainer = CLTSPTrainer(
            config=self.configuration,
        )

        self.training_runner = ModelTrain(
            config=self.configuration,
            dataset_generator=self.data_loader,
            model_trainer=self.model_trainer,
            start_epoch=0,
            global_step=0,
            use_test_loss_callback=False,
        )

        self._setup_sagemaker_checkpoint_callback()

    def _setup_inference(self, model_checkpoint_path: str):
        """
        Inputs:
        - model_checkpoint_path: path to the model checkpoint
        Outputs:
        - model: the loaded model
        - checkpoint_dict: the checkpoint dictionary
        """
        logger.info(
            f"Setting seed for inference to {self.configuration.inference_seed}"
        )
        seed_everything(self.configuration.inference_seed)

        logger.info(OKBLUE + "loading model from checkpoint" + ENDC)
        model = load_cltsp_model_from_checkpoint(
            self.configuration, model_checkpoint_path
        )

        if self.configuration.should_compile_torch:
            # compiles the model and *step (training/validation/prediction)
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")
        L.seed_everything(self.configuration.inference_seed)

        torch.set_float32_matmul_precision("high")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def train(self, model_checkpoint_path: Optional[str] = None):
        self._setup_train(model_checkpoint_path=model_checkpoint_path)
        self.training_runner.train()

    def generate_synthetic_data(self):
        pass

    def predict(
        self,
        model_checkpoint_path: str,
        h5_file_path: Optional[str] = None,
        synthetic_or_original: Optional[str] = "synthetic",
        df: Optional[pd.DataFrame] = None,
        preprocess_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gets the embeddings and losses from the trained model.
        Can either provide an h5 file

        Inputs:
        - model_checkpoint_path: path to the model checkpoint
        - h5_file_path: path to the h5 file (optional if df is provided)
        - synthetic_or_original: "synthetic" or "original" (only used with h5_file_path)
        - df: pandas dataframe of the data to predict on (optional if h5_file_path is provided)
        - preprocess_config_path: path to the preprocess config (optional if h5_file_path is provided)

        Outputs:
        - predictions: Dict[str, Any] predictions from the model
        """

        if (
            not hasattr(self.configuration, "encoder_config")
            or self.configuration.encoder_config is None
        ):
            raise ValueError(
                "encoder_config must be configured before prediction"
            )

        if not hasattr(
            self.configuration.encoder_config, "num_positive_samples"
        ):
            self.configuration.encoder_config.num_positive_samples = 1
        num_positive_samples_save = (
            self.configuration.encoder_config.num_positive_samples
        )
        # self.configuration.dataset_config.batch_size = 1
        self.configuration.encoder_config.num_positive_samples = 1
        if synthetic_or_original not in ("synthetic", "original"):
            raise ValueError(
                "synthetic_or_original must be either 'synthetic' or 'original'"
            )

        # Setup model trainer for inference
        model_trainer = self._setup_inference(model_checkpoint_path)

        # TODO - can i get these from dataloader with df metadata?
        timeseries = None
        discrete_conditions = None
        continuous_conditions = None
        # Load data either from provided arrays or from h5 file
        if df is None:
            if h5_file_path is None:
                raise ValueError(
                    "Either provide all data arrays or a valid h5_file_path"
                )
            with h5py.File(h5_file_path, "r") as f:
                timeseries = np.array(f[f"{synthetic_or_original}_timeseries"])
                discrete_conditions = np.array(f["discrete_conditions"])
                continuous_conditions = np.array(f["continuous_conditions"])

            # setup the dataloader
            pl_dataloader = SynthesisModelv1DataLoader(
                config=self.configuration.dataset_config,
                timeseries_dataset=timeseries,
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            )
        else:
            pl_dataloader = SynthesisModelv1DataLoader(
                config=self.configuration.dataset_config,
                metadata_for_synthesis=df,
                preprocess_config_path=preprocess_config_path,
            )

        # Get the test dataloader which is actually iterable
        test_dataloader = pl_dataloader.test_dataloader()

        # Get predictions using the actual PyTorch DataLoader
        predictions = model_trainer.predict(test_dataloader)

        # Create a more detailed results dictionary
        results = {
            "timeseries_embeddings": predictions["timeseries_embeddings"],
            "condition_embeddings": predictions["condition_embeddings"],
            "contrastive_losses": predictions["contrastive_losses"],
            "timeseries": timeseries,
            "discrete_conditions": discrete_conditions,
            "continuous_conditions": continuous_conditions,
        }

        self.configuration.encoder_config.num_positive_samples = (
            num_positive_samples_save
        )
        return results
