from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.model.cltsp.cltsp_v3 import CLTSP_v3
from synthefy_pkg.model.diffusion.diffusion_transformer import (
    DiffusionTransformer,
)
from synthefy_pkg.model.diffusion.patched_diffusion_transformer import (
    PatchedDiffusionTransformer,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v1 import (
    SynthefyForecastingModelV1,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v2 import (
    SynthefyForecastingModelV2,
)
from synthefy_pkg.model.forecasting.synthefy_forecasting_model_v2a import (
    SynthefyForecastingModelV2a,
)

COMPILE = False


def load_timeseries_decoder(config: Configuration, prefer_sfmv1: bool = False):
    if hasattr(config, "foundation_model_config"):
        if (
            config.foundation_model_config.model_name
            == "synthefy_foundation_forecasting_model_v1"
        ):
            from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model import (
                SynthefyFoundationForecastingModel,
            )

            model = SynthefyFoundationForecastingModel(config)

            raise ValueError("SynthefyFoundationForecastingModel is deprecated")
        elif (
            config.foundation_model_config.model_name
            == "synthefy_foundation_forecasting_model_v2"
        ):
            from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v2 import (
                SynthefyFoundationForecastingModelV2,
            )

            model = SynthefyFoundationForecastingModelV2(config)

            raise ValueError(
                "SynthefyFoundationForecastingModelV2 is deprecated"
            )
        elif (
            config.foundation_model_config.model_name
            == "synthefy_foundation_forecasting_model_v3"
        ):
            from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v3 import (
                SynthefyFoundationForecastingModelV3,
            )

            model = SynthefyFoundationForecastingModelV3(config)

            raise ValueError(
                "SynthefyFoundationForecastingModelV3 is deprecated"
            )

        elif (
            config.foundation_model_config.model_name
            == "synthefy_foundation_forecasting_model_v3e"
        ):
            from synthefy_pkg.model.foundation_model.synthefy_foundation_forecasting_model_v3e import (
                SynthefyFoundationForecastingModelV3E,
            )

            model = SynthefyFoundationForecastingModelV3E(config)
        elif config.foundation_model_config.model_name in [
            "tabicl",
            "multilayer_tabicl",
            "univariate_tabicl",
        ]:
            from synthefy_pkg.model.architectures.tabicl_wrapper import (
                TabICLModel,
            )

            model = TabICLModel(config)

    elif hasattr(config, "denoiser_config"):
        assert config.denoiser_config is not None, (
            "config must have nonempty denoiser_config"
        )
        if (
            config.denoiser_config.denoiser_name
            == "synthefy_forecasting_model_v1"
        ) or prefer_sfmv1:
            model = SynthefyForecastingModelV1(config)
        elif (
            config.denoiser_config.denoiser_name
            == "synthefy_forecasting_model_v2"
        ):
            model = SynthefyForecastingModelV2(config)
        elif (
            config.denoiser_config.denoiser_name
            == "synthefy_forecasting_model_v2a"
        ):
            model = SynthefyForecastingModelV2a(config)
        else:
            raise ValueError("decoder name not recognized")
    elif hasattr(config, "timesfm_config"):
        from synthefy_pkg.model.baselines.timesfm import TimesFM

        model = TimesFM(config)
    elif hasattr(config, "patchtst_config"):
        from synthefy_pkg.model.architectures.patchtst import PatchTST

        model = PatchTST(config)
    else:
        raise ValueError("config does not have valid model config")
    return model


def load_diffusion_transformer(config: Configuration):
    assert (
        hasattr(config, "denoiser_config")
        and config.denoiser_config is not None
    ), "config must have denoiser_config"
    if hasattr(config, "denoiser_config"):
        if hasattr(config.denoiser_config, "denoiser_name"):
            if (
                config.denoiser_config.denoiser_name
                == "patched_diffusion_transformer"
            ):
                return PatchedDiffusionTransformer(config)
            elif (
                config.denoiser_config.denoiser_name
                == "flexible_patched_diffusion_transformer"
            ):
                from synthefy_pkg.model.diffusion.flexible_patched_diffusion_transformer import (
                    FlexiblePatchedDiffusionTransformer,
                )

                return FlexiblePatchedDiffusionTransformer(config)
            else:
                return DiffusionTransformer(config)
        else:
            return DiffusionTransformer(config)
    else:
        raise ValueError("config does not have denoiser_config")


def load_cltsp_model(config: Union[DictConfig, Configuration]) -> CLTSP_v3:
    if hasattr(config, "encoder_config"):
        if hasattr(config.encoder_config, "encoder_name"):  # type: ignore
            if config.encoder_config.encoder_name == "cltsp_v3":  # type: ignore
                return CLTSP_v3(config)  # type: ignore
            else:
                raise ValueError("encoder name not recognized")
        else:
            raise ValueError("encoder_config does not have encoder_name")
    else:
        raise ValueError("config does not have encoder_config")


def load_cltsp_model_from_checkpoint(
    config: Union[DictConfig, Configuration],
    checkpoint_path: str,
    strict: bool = True,
) -> Any:
    """Load a CLTSP trainer from a checkpoint.

    Args:
        config: Configuration object
        checkpoint_path: Path to the checkpoint file
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        The loaded CLTSPTrainer instance ready for inference
    """
    # avoid circular import
    from synthefy_pkg.model.trainers.cltsp_trainer import CLTSPTrainer

    try:
        # Load trainer from checkpoint
        trainer = CLTSPTrainer.load_from_checkpoint(
            checkpoint_path,
            config=config,
            strict=strict,
            map_location=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

        # Handle torch compilation if needed
        if (
            hasattr(config, "should_compile_torch")
            and config.should_compile_torch
        ):
            try:
                trainer.cltsp_model = torch.compile(trainer.cltsp_model)  # type: ignore
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")

        torch.set_float32_matmul_precision("high")

        return trainer

    except Exception as e:
        logger.exception(
            f"Error loading trainer from checkpoint {checkpoint_path}: {e}"
        )
        raise
