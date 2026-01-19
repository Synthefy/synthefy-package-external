import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from swarm_visualizer.lineplot import plot_overlaid_lineplot
from swarm_visualizer.utility.general_utils import set_plot_properties

from synthefy_pkg.configs.execution_configurations import Configuration

COMPILE = False


def generate_ecg_plots(
    config: Configuration,
    save_path: str,
    actual_timeseries: np.ndarray,
    synthesized_timeseries: np.ndarray,
):
    set_plot_properties()
    fig, axs = plt.subplots(
        nrows=config.training_config.n_plots,
        ncols=config.dataset_config.num_input_features,
        sharey=True,
        sharex=True,
        figsize=(8 * config.dataset_config.num_input_features, 32),
    )

    for plot_idx in range(config.training_config.n_plots):
        for feat_idx in range(config.dataset_config.num_input_features):
            true_plot = actual_timeseries[plot_idx, feat_idx]
            synthesized_timeseries_plot = synthesized_timeseries[
                plot_idx, feat_idx
            ]

            NORMALIZED_TS_DICT = {
                "True": {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": true_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "b",
                },
                "Synthesized": {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": synthesized_timeseries_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "g",
                },
            }

            plot_overlaid_lineplot(
                normalized_ts_dict=NORMALIZED_TS_DICT,
                title_str="Feature {} Plot".format(feat_idx + 1),
                ylabel="Value",
                xlabel="Time",
                # fontsize=30,
                xticks=None,
                ylim=None,
                DEFAULT_ALPHA=1.0,
                legend_present=True,
                DEFAULT_MARKERSIZE=15,
                delete_yticks=False,
                ax=axs[plot_idx, feat_idx],
            )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def generate_electricity_plots(
    config: Configuration,
    save_path: str,
    actual_timeseries: np.ndarray,
    synthesized_timeseries: np.ndarray,
):
    set_plot_properties()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=config.training_config.n_plots,
        sharey=True,
        sharex=True,
        figsize=(32, 8),
    )
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def generate_waveforms_plots(
    config: Configuration,
    save_path: str,
    actual_timeseries: np.ndarray,
    synthesized_timeseries: np.ndarray,
):
    set_plot_properties()
    fig, axs = plt.subplots(
        nrows=config.training_config.n_plots,
        ncols=config.dataset_config.num_input_features,
        sharey=True,
        sharex=True,
        figsize=(8 * config.dataset_config.num_input_features, 32),
    )

    for plot_idx in range(config.training_config.n_plots):
        for feat_idx in range(config.dataset_config.num_input_features):
            true_plot = actual_timeseries[plot_idx, feat_idx]
            synthesized_timeseries_plot = synthesized_timeseries[
                plot_idx, feat_idx
            ]

            NORMALIZED_TS_DICT = {
                "True": {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": true_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "b",
                },
                "Synthesized": {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": synthesized_timeseries_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "g",
                },
            }
            if config.dataset_config.num_input_features == 1:
                req_ax = axs[plot_idx]
            else:
                req_ax = axs[plot_idx, feat_idx]

            plot_overlaid_lineplot(
                normalized_ts_dict=NORMALIZED_TS_DICT,
                title_str="Feature {} Plot".format(feat_idx + 1),
                ylabel="Value",
                xlabel="Time",
                fontsize=30,
                xticks=None,
                ylim=None,
                DEFAULT_ALPHA=1.0,
                legend_present=True,
                DEFAULT_MARKERSIZE=15,
                delete_yticks=False,
                ax=req_ax,
            )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def generate_tsdiffusion_plots(
    config: Configuration,
    save_path: str,
    actual_timeseries: np.ndarray,
    synthesized_timeseries: np.ndarray,
    labels: Optional[np.ndarray] = None,
    task: str = "synthesis",
):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_plot_properties()
    num_input_features = min(actual_timeseries.shape[1], 12)
    num_plots = min(config.training_config.n_plots, actual_timeseries.shape[0])

    fig, axs = plt.subplots(
        nrows=num_plots,
        ncols=num_input_features,
        sharey=True,
        sharex=True,
        figsize=(8 * num_input_features, 8 * num_plots),
    )

    for plot_idx in range(num_plots):
        if num_input_features < actual_timeseries.shape[1]:
            feat_idxs = np.random.choice(
                actual_timeseries.shape[1], num_input_features, replace=False
            )
        else:
            feat_idxs = np.arange(actual_timeseries.shape[1])
        for feat_idx, feat in enumerate(feat_idxs):
            true_plot = actual_timeseries[plot_idx, feat]
            synthesized_timeseries_plot = synthesized_timeseries[plot_idx, feat]

            if task == "synthesis":
                word = "Synthesized"
            elif task == "forecast":
                word = "Forecasted"
            NORMALIZED_TS_DICT = {
                "True": {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": true_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "b",
                },
                word: {
                    "xvec": np.arange(0, true_plot.shape[0], 1),
                    "ts_vector": synthesized_timeseries_plot,
                    "lw": 3.0,
                    "linestyle": "-",
                    "color": "g",
                },
            }
            if num_input_features == 1 and num_plots > 1:
                req_ax = axs[plot_idx]
            elif num_plots == 1 and num_input_features > 1:
                req_ax = axs[feat_idx]
            elif num_plots == 1 and num_input_features == 1:
                req_ax = axs
            else:
                req_ax = axs[plot_idx, feat_idx]

            plot_overlaid_lineplot(
                normalized_ts_dict=NORMALIZED_TS_DICT,
                ylabel="Value",
                xlabel="Time",
                xticks=None,
                ylim=None,
                DEFAULT_ALPHA=1.0,
                legend_present=True,
                DEFAULT_MARKERSIZE=15,
                delete_yticks=False,
                ax=req_ax,
            )

    # After creating your plots, set the x-axis tick visibility for all subplots
    for ax in axs.flat:  # or axs.flatten() or axs.ravel()
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.legend(fontsize=24)

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
