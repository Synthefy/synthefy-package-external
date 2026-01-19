import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from chronos import ChronosPipeline
from einops import rearrange
from torch import Tensor

from synthefy_pkg.model.architectures.tabicl.layers import SkippableLinear
from synthefy_pkg.utils.synthesis_utils import load_forecast_model


class LinearEmbedder(nn.Module):
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        skip_value = kwargs.get("skip_value", -100.0)
        skip_number = kwargs.get("skip_number", 0.0)
        self.embed_dim = embed_dim
        self.linear = SkippableLinear(
            1, embed_dim, skip_value=skip_value, skip_number=skip_number
        )

    def to(self, device: str):
        self.linear = self.linear.to(device)
        return self

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert x.ndim == 3, "Input must be 3D"
        x = x.unsqueeze(-1)
        return self.linear(x), None


class ChronosEmbedder(nn.Module):
    def __init__(self, embed_dim: int, device: str = "cuda"):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                device_map="cpu",
                torch_dtype=torch.bfloat16,
            )
        self.linear = nn.Linear(512, embed_dim)

    def to(self, device: str):
        # Reinitialize the model with the new device
        self.device = device
        return self

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            assert x.ndim == 3, "Input must be 3D"
            b, nc, horizon_len = x.shape
            # Move input to cpu because that is necessary for chronos
            x = x.to("cpu")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = self.model.embed(x.reshape(-1, x.shape[-1]))[0]
            embeddings = embeddings[:, 1:]
            embeddings = embeddings.to(self.device)
        embeddings = embeddings.reshape(b, nc, horizon_len, -1)
        embeddings = self.linear(embeddings)
        return embeddings, None


class SFMV3Embedder(nn.Module):
    def __init__(
        self,
        config: str,
        checkpoint: str,
        embed_dim: int,
        out_dim: int = -1,
        device: str = "cuda",
    ):
        super().__init__()
        raise NotImplementedError(
            "FoundationForecastExperiment has been removed. This class needs to be updated to use an alternative experiment class."
        )
        # from synthefy_pkg.experiments.foundation_forecast_experiment import (
        #     FoundationForecastExperiment,
        # )
        from synthefy_pkg.model.trainers.timeseries_decoder_forecasting_foundation_trainer import (
            TimeSeriesDecoderForecastingFoundationTrainer,
        )

        self.config = config
        self.device = device

        # experiment = FoundationForecastExperiment(self.config)

        # if (
        #     out_dim > 0
        #     and experiment.configuration.foundation_model_config.num_bins
        #     != out_dim
        # ):
        #     experiment.configuration.training_config.strict_load = False
        #     experiment.configuration.foundation_model_config.num_bins = out_dim

        # Load model
        # state_embedder_trainer, _, _ = load_forecast_model(
        #     experiment.configuration, checkpoint
        # )
        raise NotImplementedError(
            "FoundationForecastExperiment has been removed. This class needs to be updated to use an alternative experiment class."
        )
        assert (
            isinstance(
                state_embedder_trainer,
                TimeSeriesDecoderForecastingFoundationTrainer,
            )
            and state_embedder_trainer.decoder_model is not None
        )
        self.state_embedder = state_embedder_trainer
        self.projection_layer = nn.Linear(
            self.state_embedder.decoder_model.decoder_model_dims, embed_dim
        )
        self.layer_norm = nn.LayerNorm(
            self.state_embedder.decoder_model.decoder_model_dims,
            elementwise_affine=True,
        )
        # self.state_embedder.eval()
        # # set requires_grad to False for all parameters
        # for param in self.state_embedder.parameters():
        #     param.requires_grad = False

    def to(self, device: str):
        self.device = device
        self.state_embedder.to(device)
        self.projection_layer = self.projection_layer.to(device)
        return self

    def forward(
        self, src: Tensor, tabicl_target_mask: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        batch_size = src.shape[0]
        time_series_length = src.shape[1]
        num_correlates = src.shape[2]

        decoder_input = self.state_embedder.decoder_model.prepare_training_input_for_tabicl_embedder(
            src, target_mask=tabicl_target_mask
        )
        output_dict = self.state_embedder.decoder_model.forward(decoder_input)

        # apply layer norm to embeddings
        assert isinstance(output_dict, dict), "output_dict must be a dict"
        embeddings = output_dict["embeddings"]
        embeddings_norm = self.layer_norm(embeddings)
        projected_embeddings = self.projection_layer(embeddings_norm)  # type: ignore
        # if torch.isnan(projected_embeddings).sum() > 0:
        #     print("nan in projected_embeddings")
        #     from IPython import embed; embed()
        projected_embeddings = rearrange(
            projected_embeddings,
            "(b nc) t e -> b t nc e",
            b=batch_size,
            nc=num_correlates,
            t=time_series_length,
        )
        logits = rearrange(
            output_dict["logits"],  # type: ignore
            "(b nc) t e -> b t nc e",
            b=batch_size,
            nc=num_correlates,
            t=time_series_length,
        )

        return projected_embeddings, logits


def initialize_embedder(
    embed_dim: int, model_name: str = "linear", device: str = "cuda", **kwargs
):
    if model_name == "chronos-t5-small":
        model = ChronosEmbedder(embed_dim, device=device)
    elif model_name == "linear":
        model = LinearEmbedder(embed_dim, **kwargs)
        model = model.to(device)
    elif model_name == "sfm_v3e":
        config = kwargs.get("config", "")
        output_dim = kwargs.get("output_dim", -1)
        checkpoint = kwargs.get("checkpoint", "")
        model = SFMV3Embedder(
            config, checkpoint, embed_dim, out_dim=output_dim, device=device
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model
