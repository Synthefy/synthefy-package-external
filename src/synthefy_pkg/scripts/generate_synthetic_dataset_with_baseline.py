import os
from typing import Any, List, Tuple, Union

import yaml
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.foundation_dataloader import FoundationModelDataLoader
from synthefy_pkg.data.general_pl_dataloader import ForecastingDataLoader
from synthefy_pkg.data.sharded_dataloader import ShardedDataloaderV1
from synthefy_pkg.data.v3_sharded_dataloader import V3ShardedDataloader
from synthefy_pkg.model.baselines.forecast_via_diffusion_baseline import (
    ForecastViaDiffusionBaseline,
    MetadataStrategy,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v1 import (
    SynthefyForecastingModelV1,
)
from synthefy_pkg.model.load_models import load_timeseries_decoder
from synthefy_pkg.scripts.generate_synthetic_dataset import (
    generate_synthetic_dataset,
)
from synthefy_pkg.utils.basic_utils import ENDC, OKYELLOW, seed_everything
from synthefy_pkg.utils.docker_utils import check_huggingface_access

COMPILE = True
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")


def load_config(config_or_config_path: Union[DictConfig, str]) -> DictConfig:
    if isinstance(config_or_config_path, DictConfig):
        return config_or_config_path
    elif isinstance(config_or_config_path, str):
        with open(config_or_config_path, "r") as file:
            config = yaml.safe_load(file)
        config = DictConfig(config)
        return config
    else:
        raise ValueError(
            "Input must be either a DictConfig object or a string path to a config file."
        )


def get_baseline_model(
    baseline: str,
    configuration: Configuration,
    decoder_model: Any,
    dictconfig: Union[DictConfig, None] = None,
    use_probabilistic_forecast: bool = False,
):
    """
    Return a baseline model based on the baseline name.
    args:
        baseline: str, the name of the baseline model
        configuration: Configuration, the configuration object
        decoder_model: SynthefyForecastingModelV1, the decoder model
        dictconfig: Union[DictConfig, None], the dictionary configuration object; must be non-None for the forecast_via_diffusion baseline; unused otherwise.
    returns:
        A baseline model
    """
    extra_kwargs = {}
    if (
        configuration.dataset_config.dataloader_name == "ShardedDataloaderV1"
        or configuration.dataset_config.dataloader_name == "V3ShardedDataloader"
    ):
        extra_kwargs["fmv2_prepare_fn"] = decoder_model.prepare_training_input
        extra_kwargs["num_channels"] = (
            configuration.dataset_config.num_correlates
        )

    if baseline == "chronos":
        check_huggingface_access()
        from synthefy_pkg.model.baselines.chronos_baseline import (
            ChronosBaseline,
        )

        return ChronosBaseline(
            device=configuration.device,
            pred_len=configuration.dataset_config.forecast_length,
            prepare_fn=decoder_model.prepare_training_input,
            use_probabilistic_forecast=use_probabilistic_forecast,
        )
    elif baseline == "timesfm":
        check_huggingface_access()
        from synthefy_pkg.model.baselines.timesfm_baseline import (
            TimesfmBaseline,
        )

        return TimesfmBaseline(
            device=configuration.device,
            batch_size=configuration.dataset_config.batch_size,
            pred_len=configuration.dataset_config.forecast_length,
            prepare_fn=decoder_model.prepare_training_input,
            use_timesfm2=False,
        )
    elif baseline == "timesfm2":
        check_huggingface_access()
        from synthefy_pkg.model.baselines.timesfm_baseline import (
            TimesfmBaseline,
        )

        return TimesfmBaseline(
            device=configuration.device,
            batch_size=configuration.dataset_config.batch_size,
            pred_len=configuration.dataset_config.forecast_length,
            prepare_fn=decoder_model.prepare_training_input,
            use_timesfm2=True,
        )
    elif baseline == "prophet":
        from synthefy_pkg.model.baselines.prophet_baseline import (
            ProphetBaseline,
        )

        kwargs = {
            "seq_len": configuration.dataset_config.time_series_length,
            "pred_len": configuration.dataset_config.forecast_length,
            "batch_size": configuration.dataset_config.batch_size,
            "num_channels": configuration.dataset_config.num_channels,
            "use_probabilistic_forecast": use_probabilistic_forecast,
        }
        kwargs.update(extra_kwargs)

        return ProphetBaseline(**kwargs)
    elif baseline == "stl":
        from synthefy_pkg.model.baselines.stl_baseline import STLBaseline

        kwargs = {
            "seq_len": configuration.dataset_config.time_series_length,
            "pred_len": configuration.dataset_config.forecast_length,
            "batch_size": configuration.dataset_config.batch_size,
            "num_channels": configuration.dataset_config.num_channels,
        }
        kwargs.update(extra_kwargs)

        return STLBaseline(**kwargs)  # type: ignore
    elif baseline == "tabpfn":
        from synthefy_pkg.model.baselines.tabpfn_baseline import TabPFNBaseline

        return TabPFNBaseline(
            device=configuration.device,
            batch_size=configuration.dataset_config.batch_size,
            pred_len=configuration.dataset_config.forecast_length,
            prepare_fn=decoder_model.prepare_training_input,
            use_metadata=False,
            metadata_strategy=MetadataStrategy.NAIVE,
        )
    elif baseline == "forecast_via_diffusion":
        assert dictconfig is not None
        # TODO: Can remove usage of dictconfig and use the configuration object instead.
        # TODO: don't hardcode the model_checkpoint_path
        model_checkpoint_path = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            "models",
            f"{configuration.dataset_config.dataset_name}_synthesis.ckpt",
        )
        return ForecastViaDiffusionBaseline(
            config=dictconfig,
            model_checkpoint_path=model_checkpoint_path,
            use_probabilistic_forecast=use_probabilistic_forecast,
            require_time_invariant_metadata=True,
            # TODO: Set this based on a config parameter, not hardcoded here...
            # metadata_strategy=MetadataStrategy.NAIVE,
            # metadata_strategy=MetadataStrategy.DELETE,
            # metadata_strategy=MetadataStrategy.REPEAT,
            metadata_strategy=MetadataStrategy.REPEAT_WINDOW,
        )
    else:
        raise ValueError(f"Baseline {baseline} not implemented")


def run(
    config: DictConfig,
    baseline: str = "chronos",
    splits: Tuple[str] | List[str] = ("test",),
):
    configuration = Configuration(config)

    seed_everything(configuration.seed)
    DEVICE = configuration.device
    logger.info(f"Using device: {DEVICE}")

    if (
        configuration.dataset_config.dataloader_name
        == "FoundationModelDataLoader"
    ):
        pl_dataloader = FoundationModelDataLoader(configuration)
    elif configuration.dataset_config.dataloader_name == "ShardedDataloaderV1":
        pl_dataloader = ShardedDataloaderV1(configuration)
    elif configuration.dataset_config.dataloader_name == "V3ShardedDataloader":
        pl_dataloader = V3ShardedDataloader(configuration)
    else:
        pl_dataloader = ForecastingDataLoader(configuration)

    save_dir = configuration.get_save_dir(SYNTHEFY_DATASETS_BASE)

    synthesizer = DictConfig({"config": {"device": DEVICE}})

    decoder_model = load_timeseries_decoder(configuration, prefer_sfmv1=True)

    use_probabilistic_forecast = False
    if (
        hasattr(configuration, "denoiser_config")
        and configuration.denoiser_config is not None
    ):
        use_probabilistic_forecast = (
            configuration.denoiser_config.use_probabilistic_forecast
        )

    baseline_model = get_baseline_model(
        baseline,
        configuration,
        decoder_model,
        config,
        use_probabilistic_forecast,
    )

    assert configuration.task in ("forecast", "probabilistic_forecast")

    task = (
        "probabilistic_forecast" if use_probabilistic_forecast else "forecast"
    )

    for split in splits:
        logger.info(
            OKYELLOW
            + f"Generating the synthetic dataset for the {split} conditions"
            + ENDC
        )
        dataloader = getattr(pl_dataloader, f"{split}_dataloader")()
        split_kwargs = {"test": False, "val": False, "train": False}
        split_kwargs[split] = True

        if use_probabilistic_forecast:
            save_dir_split = os.path.join(
                save_dir, f"probabilistic_{split}_dataset"
            )
        else:
            save_dir_split = os.path.join(save_dir, f"{split}_dataset")

        os.makedirs(save_dir_split, exist_ok=True)

        logger.info(
            OKYELLOW
            + "All the results will be stored in this directory: "
            + str(save_dir_split)
            + ENDC
        )

        generate_synthetic_dataset(
            dataloader=dataloader,
            synthesizer=synthesizer,
            synthesis_function=baseline_model.synthesis_function,
            save_dir=save_dir_split,
            dataset_config=configuration.dataset_config,
            in_gan_space=False,
            scaler=None,
            task=task,
            **split_kwargs,
        )
