import os

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.data.general_pl_dataloader import ForecastingDataLoader
from synthefy_pkg.scripts.forecasting_evaluation import metric
from synthefy_pkg.utils.basic_utils import ENDC, OKYELLOW, seed_everything

DEFAULT_SEED = 42
SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")
assert SYNTHEFY_DATASETS_BASE is not None


@hydra.main(config_path="../examples/configs/", version_base="1.1")
def main(config: DictConfig):
    seed_everything(config.get("seed", DEFAULT_SEED))

    configuration = Configuration(config=config)

    pl_dataloader = ForecastingDataLoader(configuration)

    torch.set_float32_matmul_precision("high")

    save_dir = os.path.join(
        str(SYNTHEFY_DATASETS_BASE),
        configuration.generation_save_path,
        configuration.dataset_name,
        configuration.experiment_name,
        configuration.run_name,
        "test_dataset",
    )
    logger.info(
        OKYELLOW
        + "Loading data from: "
        + str(save_dir)
        + ENDC
        + "\nNote: We only analyze test data"
    )
    os.makedirs(save_dir, exist_ok=True)

    test_dataset_obj = pl_dataloader.test_dataloader().dataset

    # Note: we skip checking the conditions for now. TODO: Reintroduce this...

    test_timeseries = test_dataset_obj.timeseries_dataset
    synthetic_test_timeseries = np.load(os.path.join(save_dir, "test_timeseries.npy"))

    forecast_length = configuration.dataset_config.forecast_length
    gt = test_timeseries[:, :, -forecast_length:]
    pred = synthetic_test_timeseries[:, :, -forecast_length:]
    mae, mse, rmse, mape, mspe, rmsle = metric(pred, gt)

    logger.info(
        f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, MSPE: {mspe}, RMSLE: {rmsle}"
    )


if __name__ == "__main__":
    main()

"""
Chronos Workflow:

Sampling:
    `HYDRA_FULL_ERROR=1 python3 -m ipdb src/synthefy_pkg/scripts/generate_synthetic_dataset_with_chronos.py --config-name=config_twamp_one_month_forecasting.yaml`

Compute Stats (this script):
    `HYDRA_FULL_ERROR=1 python3 -m ipdb examples/forecasting_evaluation_chronos.py --config-name=config_twamp_one_month_forecasting.yaml`

Compute Stats (alt):
    `python -m ipdb src/synthefy_pkg/postprocessing/postprocess.py     --config=/home/ubuntu/code/synthefy-package/examples/configs/config_twamp_one_month_forecasting.yaml`
"""
