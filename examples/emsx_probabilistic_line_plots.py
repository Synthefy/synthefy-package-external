import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from swarm_visualizer.utility.general_utils import set_plot_properties

from synthefy_pkg.configs.execution_configurations import Configuration

set_plot_properties()

SYNTHEFY_DATASETS_BASE = os.environ.get("SYNTHEFY_DATASETS_BASE")


def probabilistic_to_deterministic(sampled_data):
    assert len(sampled_data.shape) == 4
    return sampled_data[:, 0, :, :]


def plot_emsx_plots(config: Configuration, save_dir: str, data_dir: str):
    time_series_length = config.dataset_config.time_series_length  # 192
    forecast_length = config.dataset_config.forecast_length  # 96

    test_unscaled_probabilistic_synthesized = np.load(
        os.path.join(
            data_dir,
            "unscaled_data_probabilistic/test_synthesized_timeseries_unscaled.pkl.npy",
        )
    )
    test_unscaled_probabilistic_synthesized_chronos = np.load(
        os.path.join(
            f"{data_dir}_chronos",
            "unscaled_data_probabilistic/test_synthesized_timeseries_unscaled.pkl.npy",
        )
    )
    test_unscaled_probabilistic_synthesized_prophet = np.load(
        os.path.join(
            f"{data_dir}_prophet",
            "unscaled_data_probabilistic/test_synthesized_timeseries_unscaled.pkl.npy",
        )
    )

    test_unscaled_real_probabilistic = np.load(
        os.path.join(
            data_dir,
            "unscaled_data_probabilistic/test_real_timeseries_unscaled.pkl.npy",
        )
    )

    test_unscaled_synthesized_prophet = probabilistic_to_deterministic(
        test_unscaled_probabilistic_synthesized_prophet
    )

    test_unscaled_real = test_unscaled_real_probabilistic

    try:
        assert np.all(
            np.abs(
                test_unscaled_synthesized_prophet[:, :, :forecast_length]
                - test_unscaled_real[:, :, :forecast_length]
            )
            < 1e-2
        ), f"""Synthesized Prophet History is not close to the real timeseries history; Max error in Prophet = {np.max(
                np.abs(
                    test_unscaled_synthesized_prophet[:, :, :96]
                    - test_unscaled_real[:, :, :96]
                )
            )}
        """
    except:
        num_errors = np.count_nonzero(
            np.abs(
                test_unscaled_synthesized_prophet[:, :, :forecast_length]
                - test_unscaled_real[:, :, :forecast_length]
            )
            < 1e-2
        )
        errors = np.abs(
            test_unscaled_synthesized_prophet[:, :, :forecast_length]
            - test_unscaled_real[:, :, :forecast_length]
        )
        print(f"Max error: {np.max(errors)}")
        print(f"Min error: {np.min(errors)}")
        print(f"Mean error: {np.mean(errors)}")
        print(f"Std error: {np.std(errors)}")
        print(f"Number of errors: {num_errors}")
        raise

    # Why do we do manual clipping? I think it's just for the SE plots.
    test_unscaled_probabilistic_synthesized = np.clip(
        test_unscaled_probabilistic_synthesized, 0, 1000000
    )
    test_unscaled_probabilistic_synthesized_chronos = np.clip(
        test_unscaled_probabilistic_synthesized_chronos, 0, 1000000
    )

    test_unscaled_probabilistic_mean = np.mean(
        test_unscaled_probabilistic_synthesized, axis=1
    )
    test_unscaled_probabilistic_std = np.std(
        test_unscaled_probabilistic_synthesized, axis=1
    )

    test_unscaled_probabilistic_mean_chronos = np.mean(
        test_unscaled_probabilistic_synthesized_chronos, axis=1
    )
    test_unscaled_probabilistic_std_chronos = np.std(
        test_unscaled_probabilistic_synthesized_chronos, axis=1
    )

    scaled_test_real = test_unscaled_real.copy()
    scaled_test_synthesized_probabilistic_mean = test_unscaled_probabilistic_mean.copy()

    BATCHES = scaled_test_real.shape[0]
    CHANNELS = scaled_test_real.shape[1]

    for i in range(BATCHES):
        for j in range(CHANNELS):
            min_ = np.min(test_unscaled_real[i, j])
            max_ = np.max(test_unscaled_real[i, j])
            scaled_test_real[i, j] = (scaled_test_real[i, j] - min_) / max(
                max_ - min_, np.finfo(float).eps
            )
            scaled_test_synthesized_probabilistic_mean[i, j] = (
                scaled_test_synthesized_probabilistic_mean[i, j] - min_
            ) / max(max_ - min_, np.finfo(float).eps)

    # Pick an arbitrary channel to be the reference for "best" by MSE
    channel = 1
    error = np.abs(scaled_test_real - scaled_test_synthesized_probabilistic_mean)
    channel_error = error[:, channel, :]
    mse = np.mean(channel_error**2, axis=1)
    top_10_idx = np.argsort(mse)[:100]

    os.makedirs(save_dir, exist_ok=True)

    for i in top_10_idx:
        plt.figure()
        plt.plot(
            test_unscaled_real[i, channel], label="Dataset", linewidth=3, color="blue"
        )
        # plot the mean of the probabilistic synthesized
        plt.plot(
            test_unscaled_probabilistic_mean[i, channel],
            label="Synthefy",
            linewidth=3,
            color="red",
            linestyle="--",
        )
        # create a shaded region for the probabilistic synthesized with +/- 1 std
        plt.fill_between(
            np.arange(forecast_length, time_series_length, 1),
            test_unscaled_probabilistic_mean[i, channel, forecast_length:]
            - test_unscaled_probabilistic_std[i, channel, forecast_length:],
            test_unscaled_probabilistic_mean[i, channel, forecast_length:]
            + test_unscaled_probabilistic_std[i, channel, forecast_length:],
            color="red",
            alpha=0.3,
        )

        plt.axvline(x=forecast_length, color="black", linestyle="--")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.savefig(os.path.join(save_dir, f"se_plot_sfm1_{i}.png"))

        # plot chronos mean and std
        plt.plot(
            test_unscaled_probabilistic_mean_chronos[i, channel],
            label="Chronos",
            linewidth=3,
            color="green",
            linestyle="-.",
        )
        plt.fill_between(
            np.arange(forecast_length, time_series_length, 1),
            test_unscaled_probabilistic_mean_chronos[i, channel, forecast_length:]
            - test_unscaled_probabilistic_std_chronos[i, channel, forecast_length:],
            test_unscaled_probabilistic_mean_chronos[i, channel, forecast_length:]
            + test_unscaled_probabilistic_std_chronos[i, channel, forecast_length:],
            color="green",
            alpha=0.3,
        )
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"se_plot_sfm1_chronos_{i}.png"))
        plt.close()


def main(config_filepath: str):
    config = Configuration(config_filepath=config_filepath)
    assert config.task == "forecast"
    assert config.dataset_config.dataset_name == "se_emsx"
    assert config.denoiser_config.use_probabilistic_forecast

    data_dir = config.get_save_dir(SYNTHEFY_DATASETS_BASE)
    save_dir = os.path.join(data_dir, "emsx_probabilistic_line_plots")
    print(f"Saving SE plots to {save_dir}")
    plot_emsx_plots(config, save_dir, data_dir)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--config",
        type=str,
        default="examples/configs/forecast_configs/config_se_emsx_forecasting.yaml",
    )
    args = argparse.parse_args()
    main(args.config)
