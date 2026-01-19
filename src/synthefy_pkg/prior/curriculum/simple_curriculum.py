import copy
from typing import Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.tabicl_config import (
    AssignmentConfig,
    TabICLCurriculumConfig,
    TabICLPriorConfig,
)


class Curriculum:
    def __init__(
        self,
        config: TabICLCurriculumConfig,
        prior_config: TabICLPriorConfig,
        assign_values: Optional[AssignmentConfig] = None,
    ):
        self.config = config
        self.prior_config = prior_config
        self.assign_values = (
            assign_values if assign_values is not None else DictConfig({})
        )

    def _compute_value(self, ratio: float, value_min: float, value_max: float):
        """
        computes the value of the curriculum feature using a linear operation on the ratio
        TODO: support other operations
        """
        if isinstance(value_min, list):
            assert isinstance(value_max, list), (
                "value_max must be a list if value_min is a list"
            )
            return [
                vmin + ratio * (vmax - vmin)
                for vmin, vmax in zip(value_min, value_max)
            ]  # computes ratio for each element in the list
        else:  # handles floats
            return value_min + ratio * (value_max - value_min)

    def _compute_ratio(self, **kwargs):
        raise NotImplementedError

    def _assign_curriculum_values(self, **kwargs):
        new_config = copy.deepcopy(self.prior_config)
        updated_values = {}
        new_assignments = copy.deepcopy(self.assign_values)
        configs = [new_config, new_assignments]
        features = [
            self.config.curriculum_features,
            self.config.assignment_features,
        ]
        for config, feature_set in zip(configs, features):
            for key, value in feature_set.items():
                logger.debug(f"key: {key}, value: {value}")
                delay_count = value.get("delay_count", 0)
                min_step_size = value.get("min_step_size", 0.05)
                use_ratio = self._compute_ratio(
                    delay_count=delay_count,
                    min_step_size=min_step_size,
                    **kwargs,
                )
                if isinstance(getattr(config, key), dict):
                    logger.debug(
                        f"key: {key}, distribution: {value['distribution']}"
                    )
                    config_random_dict = getattr(config, key)
                    if config_random_dict["distribution"] == "uniform":
                        config_random_dict["min"] = self._compute_value(
                            use_ratio, value["min_min"], value["max_min"]
                        )
                        config_random_dict["max"] = self._compute_value(
                            use_ratio, value["min_max"], value["max_max"]
                        )
                    elif (
                        config_random_dict["distribution"]
                        == "meta_trunc_norm_log_scaled"
                    ):
                        config_random_dict["max_mean"] = self._compute_value(
                            use_ratio,
                            value["min_max_mean"],
                            value["max_max_mean"],
                        )
                        config_random_dict["min_mean"] = self._compute_value(
                            use_ratio,
                            value["min_min_mean"],
                            value["max_min_mean"],
                        )
                        config_random_dict["lower_bound"] = self._compute_value(
                            use_ratio,
                            value["min_lower_bound"],
                            value["max_lower_bound"],
                        )
                    elif config_random_dict["distribution"] == "meta_beta":
                        config_random_dict["scale"] = self._compute_value(
                            use_ratio, value["min_scale"], value["max_scale"]
                        )
                        config_random_dict["min"] = self._compute_value(
                            use_ratio, value["min_min"], value["max_min"]
                        )
                        config_random_dict["max"] = self._compute_value(
                            use_ratio, value["min_max"], value["max_max"]
                        )
                    elif config_random_dict["distribution"] == "meta_choice":
                        config_random_dict["choice_values"] = (
                            value["min_choice_values"]
                            + value["max_choice_values"][
                                1 : 1
                                + np.round(
                                    self._compute_value(
                                        use_ratio,
                                        0,
                                        len(value["max_choice_values"][1:]),
                                    )
                                )
                            ]
                        )
                    else:
                        raise ValueError(
                            f"Unknown distribution: {config_random_dict['distribution']}"
                        )
                    setattr(config, key, config_random_dict)
                    config_random_dict["ratio"] = use_ratio
                    updated_values[key] = config_random_dict
                else:
                    if isinstance(getattr(config, key), int):
                        value = int(
                            np.round(
                                self._compute_value(
                                    use_ratio, value["min"], value["max"]
                                )
                            )
                        )
                        setattr(
                            config,
                            key,
                            value,
                        )
                        updated_values[key] = {
                            "value": value,
                            "ratio": use_ratio,
                        }
                    else:  # key is a float or list
                        value = self._compute_value(
                            use_ratio, value["min"], value["max"]
                        )
                        setattr(
                            config,
                            key,
                            value,
                        )
                        updated_values[key] = {
                            "value": value,
                            "ratio": use_ratio,
                        }
        return configs, updated_values

    def _update(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self._update(**kwargs)


class LinearCurriculum(Curriculum):
    def __init__(
        self,
        config: TabICLCurriculumConfig,
        prior_config: TabICLPriorConfig,
        assign_values: Optional[AssignmentConfig] = None,
    ):
        super().__init__(config, prior_config, assign_values)

    def _compute_ratio(self, **kwargs):
        if self.config.max_steps > 0:
            steps = kwargs["global_step"]
            ratio = max(
                0,
                min(1, (steps - kwargs["delay_count"]) / self.config.max_steps),
            )
        else:
            epoch = kwargs["epoch"]
            ratio = max(
                0,
                min(
                    1, (epoch - kwargs["delay_count"]) / self.config.max_epochs
                ),
            )
        return ratio

    def _update(self, **kwargs):
        configs, updated_values = self._assign_curriculum_values(**kwargs)
        return configs, updated_values
