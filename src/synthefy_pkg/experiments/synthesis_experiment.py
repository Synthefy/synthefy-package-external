import gc
import os
import pickle
import signal
import time
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from synthefy_pkg.data.general_pl_dataloader import (
    SynthesisModelv1DataLoader,
    SynthesisModelv1DatasetFromMetadata,
)
from synthefy_pkg.experiments.experiment import Experiment
from synthefy_pkg.model.diffusion.diffusion_transformer import (
    DiffusionTransformer,
)
from synthefy_pkg.model.trainers.diffusion_model import (
    get_synthesis_via_diffusion,
)
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
    load_pickle_from_path,
    seed_everything,
)
from synthefy_pkg.utils.sagemaker_utils import (
    get_sagemaker_save_dir,
    is_running_in_sagemaker,
)
from synthefy_pkg.utils.scaling_utils import (
    load_continuous_scalers,
    load_discrete_encoders,
    load_timeseries_scalers,
    transform_using_scaler,
)
from synthefy_pkg.utils.synthesis_utils import (
    check_generation_dir_exists,
    forecast_via_diffusion,
    generate_synthetic_dataset,
    load_synthesis_model,
    projected_synthesis_via_diffusion,
    synthesis_via_projected_diffusion,
)

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

COMPILE = True


MAX_STATUS_PERCENTAGE_FOR_CELERY = 95
  

class SynthesisExperiment(Experiment):
    def __init__(
        self, config_source: Union[str, Dict[str, Any]] = "config.yaml", synthesis_task: str = "synthesis", forecast_length: int = 32,
    ):
        super().__init__(config_source)
        self.model = None
        self.synthesizer = None
        self.synthesis_task = synthesis_task
        self.forecast_length = forecast_length

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

        self.model_trainer, start_epoch, global_step = load_synthesis_model(
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
        self._setup_sagemaker_checkpoint_callback()

    def _setup_inference(self, model_checkpoint_path: str):
        """
        Inputs:
        - model_checkpoint_path: path to the model checkpoint
        Outputs:
        - None (models are stored as class attributes)
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU memory before setting up inference: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
            )
        logger.info(
            f"Setting seed for inference to {self.configuration.inference_seed}"
        )
        seed_everything(self.configuration.inference_seed)

        logger.info(OKBLUE + "loading model from checkpoint" + ENDC)
        model, _, _ = load_synthesis_model(
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

        synthesizer = model.denoiser_model

        self.model = model
        self.synthesizer = synthesizer

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU memory after setting up inference: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
            )
        if not hasattr(self, "synthesizer"):
            self.synthesizer = None

    def train(self, model_checkpoint_path: Optional[str] = None):
        self._setup_train(model_checkpoint_path=model_checkpoint_path)
        self.training_runner.train()

    def _preprocess_window_data(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        window_size: int,
        ts_history: np.ndarray,
        preprocess_config: dict,
        preprocess_config_path: str,
        saved_scalers: dict,
        encoders: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to preprocess window data for synthesis.

        Args:
            df: Input DataFrame containing metadata
            start_idx: Start index of current window
            end_idx: End index of current window
            window_size: Size of each window
            ts_history: Historical time series data
            preprocess_config: Preprocessing configuration dictionary
            preprocess_config_path: Path to preprocessing config file
            saved_scalers: Dictionary of saved scalers
            encoders: Dictionary of saved encoders

        Returns:
            tuple: (continuous_conditions_batch, discrete_conditions_batch, original_discrete_windows)
        """
        # Update lag data columns with historical values
        for idx, col in enumerate(preprocess_config["timeseries"]["cols"]):
            lag_data_col = LAG_COL_FORMAT.format(
                col=col, window_size=window_size
            )
            df[lag_data_col] = 0.0
            if start_idx > 0:
                df.iloc[
                    start_idx:end_idx, df.columns.get_indexer([lag_data_col])[0]
                ] = ts_history[0, idx, :]

        # Initialize preprocessor
        preprocessor = DataPreprocessor(preprocess_config_path)
        preprocessor.group_labels_cols = (
            []
            if not preprocessor.use_label_col_as_discrete_metadata
            else preprocessor.group_labels_cols
        )
        # timeseries data is not needed for synthesis (and its not guaranteed to be present in the df)
        preprocessor.timeseries_cols = []
        preprocessor.timeseries_scalers_info = {}
        preprocessor.add_lag_data = False

        # Update continuous columns
        lag_cols = [
            LAG_COL_FORMAT.format(col=col, window_size=window_size)
            for col in preprocess_config["timeseries"]["cols"]
        ]
        preprocessor.continuous_cols = (
            preprocess_config["continuous"]["cols"] + lag_cols
        )

        # Process data
        preprocessor.process_data(
            df.iloc[start_idx:end_idx],
            saved_scalers=saved_scalers,
            saved_encoders=encoders,
            save_files_on=False,
        )

        return (
            preprocessor.windows_data_dict["continuous"]["windows"],
            preprocessor.windows_data_dict["discrete"]["windows"],
            preprocessor.windows_data_dict["original_discrete"]["windows"],
        )

    def generate_long_term_synthetic_data(
        self,
        model_checkpoint_path: str,
        preprocess_config_path: str,
        metadata_for_synthesis: Dict[str, Any],
        num_windows_to_generate: int = -1,
        control_param: float = 0.0,
    ) -> np.ndarray:
        """
        Inputs:
        - model_checkpoint_path: path to the model checkpoint
        - preprocess_config_path: path to the preprocess config
        - metadata_for_synthesis: metadata for synthesis -> JSON formatted dictionary for 1 subject/metadata
                                    -> Note - this should be from something like df.to_json()
        - num_windows_to_generate: number of windows to synthesize. -1 for all windows
        - control_param: control parameter for the diffusion model

        Outputs:
        - generated_time_series: (1, num_channels, num_windows_to_generate * window_size)

        Loads the model and generates the long term synthetic data.
        Dataset must be trained on the lag timeseries data
        """
        # Validate metadata
        if not metadata_for_synthesis:
            raise ValueError("metadata_for_synthesis cannot be empty")

        # Load preprocessing config to check required columns
        try:
            with open(preprocess_config_path) as f:
                preprocess_config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load preprocessing config: {e}")

        self._setup_inference(model_checkpoint_path)

        if self.synthesizer is None:
            raise RuntimeError(
                "Synthesizer not loaded. Call _setup_inference() first."
            )

        num_channels = self.configuration.dataset_config.num_channels
        window_size = self.configuration.dataset_config.time_series_length
        # note - the stride = window size since we require training on the lag timeseries data, so no overlap
        stride = self.configuration.dataset_config.time_series_length

        # Convert metadata to DataFrame and validate
        try:
            df = pd.DataFrame(metadata_for_synthesis)
            if num_windows_to_generate == -1:
                num_windows_to_generate = len(df) // window_size
            df = df.iloc[0 : num_windows_to_generate * window_size]

            if len(df) < window_size:
                raise ValueError(
                    f"Metadata length ({len(df)}) is insufficient for requested number of windows"
                    f" ({num_windows_to_generate}) with window size {window_size}"
                )
            for col in preprocess_config.get("group_labels", {}).get(
                "cols", []
            ):
                if df[col].nunique() > 1:
                    raise ValueError(
                        f"Grouping column {col} has multiple unique values: {df[col].unique()}"
                    )

        except Exception as e:
            raise ValueError(f"Failed to convert metadata to DataFrame: {e}")

        # Check for required columns
        required_cols = []
        for category in ["discrete", "continuous"]:
            cols = preprocess_config.get(category, {}).get("cols", [])
            required_cols.extend(cols)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing necessary metadata columns: {missing_cols}"
            )

        # get the scalers + encoders for preprocessing
        dataset_name = self.configuration.dataset_config.dataset_name
        saved_scalers = {
            "timeseries": load_timeseries_scalers(dataset_name),
            "continuous": load_continuous_scalers(dataset_name),
        }
        encoders = load_discrete_encoders(dataset_name)

        num_windows_max = min(num_windows_to_generate, len(df) // window_size)
        if num_windows_max != num_windows_to_generate:
            logger.warning(
                f"Cannot generate {num_windows_to_generate} windows, only {num_windows_max} windows can be generated"
            )

        # we will append the synthetic data to generated_time_series
        generated_time_series = np.zeros(
            (1, num_channels, num_windows_max * window_size)
        )  # (batch_size=1, num_channels, num_windows_max * window_size)
        # we will also keep track of the history for the current time stamp
        # note - the history is always delayed by the window size, so we initialize it with zeros, window_size long
        history = np.zeros(
            (1, num_channels, window_size + num_windows_max * window_size)
        )  # (batch_size=1, num_channels, window_size + num_windows_max * window_size)

        logger.info(f"Starting generation of {num_windows_max} windows")
        total_len = len(df)
        for tidx in tqdm(
            range(0, total_len, stride),
            desc="Iterative Synthesis",
            total=int(total_len // stride),
        ):
            start_idx = tidx
            end_idx = start_idx + window_size

            end_flag = False
            if end_idx >= total_len:
                end_idx = total_len
                start_idx = end_idx - window_size
                end_flag = True

            # Process data
            (
                continuous_conditions_batch,
                discrete_conditions_batch,
                original_discrete_windows,
            ) = self._preprocess_window_data(
                df=df,
                start_idx=start_idx,
                end_idx=end_idx,
                window_size=window_size,
                ts_history=history[:, :, start_idx:end_idx],
                preprocess_config=preprocess_config,
                preprocess_config_path=preprocess_config_path,
                saved_scalers=saved_scalers,
                encoders=encoders,
            )

            dummy_timeseries_full = torch.zeros(
                1, num_channels, window_size
            ).to(
                self.synthesizer.config.device
            )  # (batch_size=1, num_channels, window_size)

            # now, let's synthesize the time series
            synthesized_dict = projected_synthesis_via_diffusion(
                batch={
                    "continuous_label_embedding": torch.tensor(
                        continuous_conditions_batch
                    ).to(self.synthesizer.config.device),
                    "discrete_label_embedding": torch.tensor(
                        discrete_conditions_batch
                    ).to(self.synthesizer.config.device),
                    "timeseries_full": dummy_timeseries_full,
                },
                synthesizer=self.synthesizer,
            )
            # unscale the data before putting it into history - and since we need unscaled for preprocessing
            synthesized_dict["timeseries"] = transform_using_scaler(
                windows=synthesized_dict["timeseries"],
                timeseries_or_continuous="timeseries",
                original_discrete_windows=original_discrete_windows,
                dataset_name=dataset_name,
                inverse_transform=True,
                transpose_timeseries=True,
            )

            # update the generated time series by direct assignment
            generated_time_series[:, :, start_idx:end_idx] = synthesized_dict[
                "timeseries"
            ]
            history[:, :, window_size + start_idx : window_size + end_idx] = (
                synthesized_dict["timeseries"]
            )

            # if the end flag is true, break the loop
            if end_flag:
                break

        return generated_time_series

    # TODO - support constrained synthesis for API (update this function)
    def generate_one_synthetic_window(
        self,
        model_checkpoint_path: str,
        continuous_conditions: np.ndarray,
        discrete_conditions: np.ndarray,
        timeseries: Optional[np.ndarray] = None,
    ):
        self._setup_inference(model_checkpoint_path)
        if self.synthesizer is None:
            raise RuntimeError(
                "Synthesizer not loaded. Call _setup_inference() first."
            )


        # Select syn()thesis function based on task type
        if self.synthesis_task == "forecast":
            synthesis_function = partial(forecast_via_diffusion, forecast_length=self.forecast_length)
        else:
            synthesis_function = (
                synthesis_via_projected_diffusion
                if self.configuration.dataset_config.use_constraints
                else get_synthesis_via_diffusion
            )

        ### above is same as generate_synthetic_data, but for one window only. ### --> TODO make a helper function

        # convert to torch tensors needed?
        continuous_conditions_tensor = torch.tensor(
            continuous_conditions, dtype=torch.float32
        )
        discrete_conditions_tensor = torch.tensor(
            discrete_conditions, dtype=torch.float32
        )

        continuous_conditions_tensor = continuous_conditions_tensor.to(
            self.synthesizer.config.device
        )
        discrete_conditions_tensor = discrete_conditions_tensor.to(
            self.synthesizer.config.device
        )
        kwargs = {}
        if self.configuration.dataset_config.use_constraints:
            kwargs["dataset_config"] = self.configuration.dataset_config

        # Use provided timeseries for forecast task, otherwise zeros
        if timeseries is not None:
            timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32)
        else:
            timeseries_tensor = torch.zeros(
                1,
                self.configuration.dataset_config.num_channels,
                self.configuration.dataset_config.time_series_length,
            )
        timeseries_tensor = timeseries_tensor.to(self.synthesizer.config.device)

        dataset_dict = synthesis_function(
            batch={
                "continuous_label_embedding": continuous_conditions_tensor,
                "discrete_label_embedding": discrete_conditions_tensor,
                "timeseries_full": timeseries_tensor,
            },
            synthesizer=self.synthesizer,
            **kwargs,
        )
        return dataset_dict["timeseries"]  # this is the synthesized data

    def generate_synthetic_data(
        self,
        model_checkpoint_path: str,
        metadata_for_synthesis: Optional[pd.DataFrame] = None,
        preprocess_config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        splits: List[str] = ["test"],
        task_id: Optional[str] = None,
        output_filename_prefix: Optional[str] = None,
    ) -> bool:
        """
        Generate synthetic data for the given splits.
        inputs:
        - model_checkpoint_path: path to the model checkpoint
        - metadata_for_synthesis: metadata for synthesis -> Dataframe of metadata to synthesize for
        - preprocess_config_path: path to the preprocess config
        - output_dir: directory to save the generated data
        - splits: list of splits to generate data for
        - task_id: optional task ID for progress tracking (fastapi/celery/redis)
        - output_filename_prefix: optional prefix for the output filename - this will overwrite the split name
        outputs:
        - is_main_process: flag indicating if this is the main process so postprocessing can be done on the main process (examples/generate_synthetic_data.py)
        """
        trainer = None
        if metadata_for_synthesis is None:
            self.data_loader = SynthesisModelv1DataLoader(
                self.configuration.dataset_config
            )
        else:
            self.data_loader = SynthesisModelv1DataLoader(
                self.configuration.dataset_config,
                metadata_for_synthesis=metadata_for_synthesis,
                preprocess_config_path=preprocess_config_path,
            )

        self._setup_inference(model_checkpoint_path)

        # Get base save directory - handle SageMaker paths
        if output_dir is None:
            if is_running_in_sagemaker():
                save_dir = get_sagemaker_save_dir("")
                logger.info(
                    f"Running in SageMaker environment, saving to: {save_dir}"
                )
            else:
                save_dir = self.configuration.get_save_dir(
                    str(os.getenv("SYNTHEFY_DATASETS_BASE")),
                )
        else:
            save_dir = output_dir

        os.makedirs(save_dir, exist_ok=True)

        if (
            self.configuration.dataset_config.use_constraints
            and len(self.configuration.dataset_config.constraints) > 0
        ):
            logger.info(
                f"Using constraints: {self.configuration.dataset_config.constraints} - This will be slow. Remove them with use_constraints=False from the synthesis config if you want to turn them off."
            )
        elif (
            self.configuration.dataset_config.use_constraints
            and len(self.configuration.dataset_config.constraints) == 0
        ):
            raise ValueError("No constraints provided")

        if self.synthesis_task == "forecast":
            synthesis_function = partial(forecast_via_diffusion, forecast_length=self.forecast_length)
        else:
            synthesis_function = (
                synthesis_via_projected_diffusion
                if self.configuration.dataset_config.use_constraints
                else get_synthesis_via_diffusion
            )

        # Get base save directory - handle SageMaker paths
        if output_dir is None:
            if is_running_in_sagemaker():
                save_dir = get_sagemaker_save_dir("")
                logger.info(
                    f"Running in SageMaker environment, saving to: {save_dir}"
                )
            else:
                save_dir = self.configuration.get_save_dir(
                    str(os.getenv("SYNTHEFY_DATASETS_BASE")),
                )
        else:
            save_dir = output_dir

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Only strategy kwarg if num_devices > 1
        num_devices = get_num_devices(
            self.configuration.training_config.num_devices
        )
        strategy = "ddp" if num_devices > 1 else "auto"

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=num_devices,
            logger=False,
            enable_progress_bar=True,
            strategy=strategy,
        )
        for split in splits:
            logger.info(
                f"Generating synthetic dataset for the {split} conditions."
            )
            dataloader = getattr(self.data_loader, f"{split}_dataloader")()
            train_or_val_or_test = (
                split
                if split != "test" or metadata_for_synthesis is None
                else f"custom_{timestamp}"
            )
            if output_filename_prefix is not None:
                train_or_val_or_test = f"{output_filename_prefix}"

            if check_generation_dir_exists(
                os.path.join(save_dir, f"{train_or_val_or_test}_dataset")
            ):
                return True

            inference_module = SynthesisInferenceModule(
                self.synthesizer,
                synthesis_function,
                self.configuration.dataset_config,
                train_or_val_or_test,
                save_dir,
                inference_seed=self.configuration.inference_seed,
                task_id=task_id,
                total_batches=len(dataloader),
            )
            trainer.predict(inference_module, dataloaders=dataloader)

        # Ensure all processes have finished
        if (
            hasattr(trainer.strategy, "barrier")
            and trainer.strategy.barrier is not None
        ):
            try:
                logger.info(
                    "Synchronizing processes at barrier (interruptible)"
                )
                self.interruptible_barrier(trainer.strategy)
            except Exception as e:
                logger.warning(
                    f"Failed to synchronize at barrier: {e}, continuing cleanup"
                )

        is_main_process = True
        if hasattr(trainer.strategy, "local_rank"):
            is_main_process = getattr(trainer.strategy, "local_rank", 0) == 0
        elif hasattr(trainer.strategy, "global_rank"):
            is_main_process = getattr(trainer.strategy, "global_rank", 0) == 0

        return is_main_process

    def interruptible_barrier(self, strategy, max_wait_seconds=30):
        """A barrier implementation that can be interrupted with CTRL+C"""

        if not hasattr(strategy, "barrier") or strategy.barrier is None:
            return

        # Set up signal handler
        interrupted = False
        original_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler(sig, frame):
            nonlocal interrupted
            interrupted = True
            logger.warning("CTRL+C detected, will exit after timeout")
            # Don't call original handler yet - we want to try cleanup first

        try:
            signal.signal(signal.SIGINT, sigint_handler)

            # Try barrier with polling for interruption
            start_time = time.time()
            barrier_complete = False

            while not barrier_complete and not interrupted:
                try:
                    # Try with very short timeout
                    if (
                        hasattr(strategy.barrier, "__call__")
                        and "timeout" in strategy.barrier.__code__.co_varnames
                    ):
                        strategy.barrier(timeout=0.5)
                        barrier_complete = True
                    else:
                        # No timeout support, just try once
                        strategy.barrier()
                        barrier_complete = True
                except Exception as e:
                    if (
                        interrupted
                        or time.time() - start_time > max_wait_seconds
                    ):
                        logger.warning(f"Barrier timed out or interrupted: {e}")
                        break
                    time.sleep(0.1)  # Small sleep to prevent CPU spinning

        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, original_handler)

            # If we were interrupted, trigger the original handler now
            if interrupted:
                if callable(original_handler):
                    original_handler(signal.SIGINT, None)
                else:
                    # If it's not callable, just raise the signal
                    signal.raise_signal(signal.SIGINT)

    def cleanup(self):
        """Clean up model resources for this specific experiment instance."""
        try:
            # Only cleanup if we have loaded models
            if hasattr(self, "model") and self.model is not None:
                # Move model to CPU to free GPU memory
                if hasattr(self.model, "to"):
                    self.model.to("cpu")
                del self.model
                self.model = None

            if hasattr(self, "synthesizer") and self.synthesizer is not None:
                del self.synthesizer
                self.synthesizer = None

            # Force garbage collection for this instance only
            import gc

            gc.collect()

            logger.info(
                "SynthesisExperiment cleanup completed (model-specific)"
            )
        except Exception as e:
            logger.error(f"Error during SynthesisExperiment cleanup: {e}")


class SynthesisInferenceModule(L.LightningModule):
    """
    A simple LightningModule wrapper for our synthesizer.
    It simply calls the synthesis function (e.g.,
    get_synthesis_via_diffusion or synthesis_via_projected_diffusion)
    using the provided synthesizer.
    """

    def __init__(
        self,
        synthesizer,
        synthesis_function,
        dataset_config,
        train_or_val_or_test,
        save_dir,
        inference_seed,
        task_id: Optional[str] = None,
        total_batches: Optional[int] = None,
    ):
        super().__init__()
        self.synthesizer = synthesizer
        self.synthesis_function = synthesis_function
        self.dataset_config = dataset_config
        self.train_or_val_or_test = train_or_val_or_test
        self.save_dir = save_dir
        self.inference_seed = inference_seed
        self.task_id = task_id
        self.total_batches = total_batches

        seed_everything(seed=self.inference_seed)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            from celery import current_task
        except ImportError:
            current_task = None

        gpu_id = getattr(self, "local_rank", 0)

        # Only update if we're the main GPU
        if gpu_id == 0 and self.total_batches is not None:
            # Calculate progress percentage

            progress_percentage = min(
                np.round((batch_idx) / self.total_batches * 100, 2),
                MAX_STATUS_PERCENTAGE_FOR_CELERY,
            )

            # Update Celery task status with just the percentage
            if current_task and self.task_id:
                current_task.update_state(  # pyright: ignore
                    state="PROGRESS",
                    meta={
                        "progress_percentage": progress_percentage,
                    },
                )

        return generate_synthetic_dataset(
            dataset_config=self.dataset_config,
            save_dir=self.save_dir,
            synthesizer=self.synthesizer,
            synthesis_function=self.synthesis_function,
            batch=batch,
            batch_idx=batch_idx,
            train_or_val_or_test=self.train_or_val_or_test,
            gpu_id=gpu_id,
        )
