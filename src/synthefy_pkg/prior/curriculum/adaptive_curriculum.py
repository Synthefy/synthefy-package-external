from typing import Optional

import numpy as np
from loguru import logger

from synthefy_pkg.configs.tabicl_config import (
    AssignmentConfig,
    TabICLCurriculumConfig,
    TabICLPriorConfig,
)
from synthefy_pkg.prior.curriculum.simple_curriculum import Curriculum


class AdaptiveCurriculum(Curriculum):
    def __init__(
        self,
        config: TabICLCurriculumConfig,
        prior_config: TabICLPriorConfig,
        assign_values: Optional[AssignmentConfig] = None,
    ):
        super().__init__(config, prior_config, assign_values)
        self.last_ratio = 0

    def _compute_ratio(self, **kwargs):
        loss = kwargs["loss"]
        step = kwargs["global_step"]
        logger.info(f"loss: {loss}, step: {step}")
        if step < kwargs["delay_count"] or step == 1:
            ratio = 0
        else:
            ratio = max(
                0,
                min(
                    1,
                    (self.config.max_loss - loss)
                    / (self.config.max_loss - self.config.min_loss),
                ),
            )
            logger.info(
                f"ratio_diff: {self.config.max_loss - loss} / {self.config.max_loss - self.config.min_loss}"
            )
            logger.info(
                f"ratio: {ratio}, last ratio: {self.last_ratio}, min_step_size {kwargs['min_step_size']}, ratio check {np.abs(self.last_ratio - ratio) > kwargs['min_step_size']} ratio update {self.last_ratio + kwargs['min_step_size'] * np.sign(ratio - self.last_ratio)}"
            )
            # Smooth the ratio to avoid large jumps
            if np.abs(self.last_ratio - ratio) > kwargs["min_step_size"]:
                ratio = self.last_ratio + kwargs["min_step_size"] * np.sign(
                    ratio - self.last_ratio
                )
            if self.config.floor_ratio:
                # only update the ratio if it is greater than the last ratio
                # ensures ratio is non-decreasing
                if ratio < self.last_ratio:
                    ratio = self.last_ratio

            self.last_ratio = ratio
        return ratio

    def _update(self, **kwargs):
        configs, updated_values = self._assign_curriculum_values(**kwargs)
        return configs, updated_values


class FloorAdaptiveCurriculum(Curriculum):
    def __init__(
        self,
        config: TabICLCurriculumConfig,
        prior_config: TabICLPriorConfig,
        assign_values: Optional[AssignmentConfig] = None,
    ):
        # as long as performance is above a floor, add to the ratio
        super().__init__(config, prior_config, assign_values)
        self.last_ratio = 0

    def _compute_ratio(self, **kwargs):
        loss = kwargs["loss"]
        step = kwargs["global_step"]
        logger.debug(f"loss: {loss}, step: {step}")
        if step < kwargs["delay_count"] or step == 1:
            ratio = 0
        else:
            if float(loss) >= float(self.config.min_loss):
                logger.debug(
                    f"loss {loss} above floor {self.config.min_loss}, ratio unchanged {self.last_ratio}"
                )
                ratio = self.last_ratio
            else:
                logger.info(
                    f"updated ratio{self.last_ratio} to {self.last_ratio + kwargs['min_step_size']} with loss: {loss}, min_loss: {self.config.min_loss} step: {step}"
                )
                # Bound the ratio to be between 0 and 1
                ratio = max(0, min(1, self.last_ratio + kwargs["min_step_size"]))

            self.last_ratio = ratio
        return ratio

    def _update(self, **kwargs):
        configs, updated_values = self._assign_curriculum_values(**kwargs)
        return configs, updated_values
