import os
from typing import Tuple

COMPILE = False


def get_dataset_paths(
    log_dir, experiment, train=False, val=False, test=False
) -> Tuple[str, str, str]:
    if experiment.lower() == "gan":
        timeseries_str_to_add = "timeseries_for_gan.npy"
        discrete_conditions_str_to_add = "discrete_conditions_for_gan.npy"
        continuous_conditions_str_to_add = "continuous_conditions_for_gan.npy"
    elif experiment.lower() == "metrics":
        timeseries_str_to_add = "timeseries_for_metrics.npy"
        discrete_conditions_str_to_add = "discrete_conditions_for_metrics.npy"
        continuous_conditions_str_to_add = "continuous_conditions_for_metrics.npy"
    else:
        timeseries_str_to_add = "timeseries.npy"
        discrete_conditions_str_to_add = "discrete_conditions.npy"
        continuous_conditions_str_to_add = "continuous_conditions.npy"

    if experiment.lower() == "metrics":
        timeseries_loc = os.path.join(log_dir, timeseries_str_to_add)
        discrete_conditions_loc = os.path.join(log_dir, discrete_conditions_str_to_add)
        continuous_conditions_loc = os.path.join(
            log_dir, continuous_conditions_str_to_add
        )
    else:
        if train:
            timeseries_loc = os.path.join(log_dir, "train_" + timeseries_str_to_add)
            discrete_conditions_loc = os.path.join(
                log_dir, "train_" + discrete_conditions_str_to_add
            )
            continuous_conditions_loc = os.path.join(
                log_dir, "train_" + continuous_conditions_str_to_add
            )
        elif val:
            timeseries_loc = os.path.join(log_dir, "val_" + timeseries_str_to_add)
            discrete_conditions_loc = os.path.join(
                log_dir, "val_" + discrete_conditions_str_to_add
            )
            continuous_conditions_loc = os.path.join(
                log_dir, "val_" + continuous_conditions_str_to_add
            )
        elif test:
            timeseries_loc = os.path.join(log_dir, "test_" + timeseries_str_to_add)
            discrete_conditions_loc = os.path.join(
                log_dir, "test_" + discrete_conditions_str_to_add
            )
            continuous_conditions_loc = os.path.join(
                log_dir, "test_" + continuous_conditions_str_to_add
            )

    return timeseries_loc, discrete_conditions_loc, continuous_conditions_loc
