from typing import Union

import torch
from loguru import logger
from torch.optim.lr_scheduler import (
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from transformers import (
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from synthefy_pkg.configs.lr_configs import (
    ConstantLRConfig,
    CosineWarmupLRConfig,
    CosineWithRestartsLRConfig,
    CyclicLRConfig,
    ExponentialLRConfig,
    LinearWarmupLRConfig,
    PolynomialDecayWarmupLRConfig,
    ReduceLROnPlateauConfig,
)
from synthefy_pkg.train.lr import get_cosine_with_restarts


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler_config: Union[
        None,
        ExponentialLRConfig,
        ReduceLROnPlateauConfig,
        CyclicLRConfig,
        ConstantLRConfig,
        LinearWarmupLRConfig,
        CosineWarmupLRConfig,
        CosineWithRestartsLRConfig,
        PolynomialDecayWarmupLRConfig,
    ],
) -> Union[
    None, ExponentialLR, ReduceLROnPlateau, CyclicLR, _LRScheduler, LambdaLR
]:
    if lr_scheduler_config is None:
        return None

    logger.info(f"Using scheduler: {lr_scheduler_config.scheduler_name}")
    if lr_scheduler_config.scheduler_name == "exponential":
        assert isinstance(lr_scheduler_config, ExponentialLRConfig)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_scheduler_config.gamma
        )

    elif lr_scheduler_config.scheduler_name == "reduce_lr_on_plateau":
        assert isinstance(lr_scheduler_config, ReduceLROnPlateauConfig)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_scheduler_config.factor,
            patience=lr_scheduler_config.patience,
        )

    elif lr_scheduler_config.scheduler_name == "cyclic":
        assert isinstance(lr_scheduler_config, CyclicLRConfig)
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_scheduler_config.base_lr,
            max_lr=lr_scheduler_config.max_lr,
            step_size_up=lr_scheduler_config.step_size_up,
            step_size_down=lr_scheduler_config.step_size_down,
            cycle_momentum=False,
        )
    elif lr_scheduler_config.scheduler_name == "constant":
        assert isinstance(lr_scheduler_config, ConstantLRConfig)
        return get_constant_schedule(optimizer=optimizer)

    elif lr_scheduler_config.scheduler_name == "linear_warmup":
        assert isinstance(lr_scheduler_config, LinearWarmupLRConfig)
        # Calculate warmup steps based on proportion if provided
        warmup_steps = lr_scheduler_config.warmup_steps
        if lr_scheduler_config.warmup_proportion >= 0:
            warmup_steps = int(
                lr_scheduler_config.max_steps
                * lr_scheduler_config.warmup_proportion
            )

        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=lr_scheduler_config.max_steps,
        )

    elif lr_scheduler_config.scheduler_name == "cosine_warmup":
        assert isinstance(lr_scheduler_config, CosineWarmupLRConfig)
        # Calculate warmup steps based on proportion if provided
        warmup_steps = lr_scheduler_config.warmup_steps
        if lr_scheduler_config.warmup_proportion >= 0:
            warmup_steps = int(
                lr_scheduler_config.max_steps
                * lr_scheduler_config.warmup_proportion
            )

        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=lr_scheduler_config.max_steps,
        )

    elif lr_scheduler_config.scheduler_name == "cosine_with_restarts":
        assert isinstance(lr_scheduler_config, CosineWithRestartsLRConfig)
        # Calculate warmup steps based on proportion if provided
        warmup_steps = lr_scheduler_config.warmup_steps
        if lr_scheduler_config.warmup_proportion >= 0:
            warmup_steps = int(
                lr_scheduler_config.max_steps
                * lr_scheduler_config.warmup_proportion
            )

        return get_cosine_with_restarts(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=lr_scheduler_config.max_steps,
            num_cycles=lr_scheduler_config.cosine_num_cycles,
            amplitude_decay=lr_scheduler_config.cosine_amplitude_decay,
            lr_end=lr_scheduler_config.cosine_lr_end,
        )

    elif lr_scheduler_config.scheduler_name == "polynomial_decay_warmup":
        assert isinstance(lr_scheduler_config, PolynomialDecayWarmupLRConfig)
        # Calculate warmup steps based on proportion if provided
        warmup_steps = lr_scheduler_config.warmup_steps
        if lr_scheduler_config.warmup_proportion >= 0:
            warmup_steps = int(
                lr_scheduler_config.max_steps
                * lr_scheduler_config.warmup_proportion
            )

        return get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=lr_scheduler_config.max_steps,
            lr_end=lr_scheduler_config.poly_decay_lr_end,
            power=lr_scheduler_config.poly_decay_power,
        )

    else:
        raise ValueError(
            f"Unknown scheduler_name: {lr_scheduler_config.scheduler_name}"
        )
