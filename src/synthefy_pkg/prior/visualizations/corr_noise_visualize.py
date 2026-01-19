import argparse
import math  # Import math for ceil and sqrt
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from synthefy_pkg.prior.input_sampling.time_series_sampling import TSSampler
from synthefy_pkg.prior.mlp_scm import MLPSCM
from synthefy_pkg.prior.ts_noise import MixedTSNoise, TSNoise
from synthefy_pkg.prior.utils import GaussianNoise


def get_mlp_scm_outputs(mlp: MLPSCM):
    outputs = [mlp.xsampler.sample()]
    denoised_outputs = [outputs[-1]]
    noise_samples = [outputs[-1]]
    for layer in mlp.layers:
        if isinstance(layer, nn.Sequential):
            val = outputs[-1]
            for sublayer in layer.children():
                if isinstance(sublayer, GaussianNoise) or isinstance(
                    sublayer, TSNoise
                ):
                    val, noise = sublayer(val, return_components=True)
                    outputs.append(val + noise)
                    denoised_outputs.append(val)
                    noise_samples.append(noise)
                elif isinstance(sublayer, MixedTSNoise):
                    val, ts_noise, gaussian_noise = sublayer(
                        val, return_components=True
                    )
                    outputs.append(val + ts_noise + gaussian_noise)
                    denoised_outputs.append(val)
                    noise_samples.append(ts_noise + gaussian_noise)
                else:
                    val = sublayer(val)
        else:
            outputs.append(layer(outputs[-1]))
            denoised_outputs.append(outputs[-1])
            noise_samples.append(outputs[-1])
    X_g, y_g, indices_X, indices_y = mlp.handle_outputs(
        outputs[-1], outputs[2:]
    )
    denoised_X_g, denoised_y_g, _, _ = mlp.handle_outputs(
        denoised_outputs[-1],
        denoised_outputs[2:],
        indices_X_y=(indices_X, indices_y),
    )
    Noise_X_g, Noise_y_g, _, _ = mlp.handle_outputs(
        noise_samples[-1], noise_samples[2:], indices_X_y=(indices_X, indices_y)
    )
    X_g = X_g.cpu().detach().numpy()
    y_g = y_g.cpu().detach().numpy()
    denoised_X_g = denoised_X_g.cpu().detach().numpy()
    denoised_y_g = denoised_y_g.cpu().detach().numpy()
    Noise_X_g = Noise_X_g.cpu().detach().numpy()
    Noise_y_g = Noise_y_g.cpu().detach().numpy()
    return X_g, y_g, denoised_X_g, denoised_y_g, Noise_X_g, Noise_y_g


def visualize_noise_and_data(
    seq_len=1000,
    num_features=10,
    num_outputs=1,
    noise_std=10,
    noise_sampling="linear_trend",
    input_sampling="normal",
    device="cpu",
    save_dir="figures/noise_compare",
):
    os.makedirs(save_dir, exist_ok=True)

    # Calculate grid size for subplots: num_features rows, 2 columns (Gaussian vs TS)
    total_rows = num_features
    total_cols = 2

    # --- Gaussian Noise Layer ---
    gaussian_noise = GaussianNoise(noise_std)
    x = torch.zeros((seq_len, num_features), device=device)
    gaussian_noise_sample = gaussian_noise(x).cpu().numpy()

    # --- TS Noise Layer (using TSSampler as noise) ---
    ts_noise = TSSampler(
        seq_len, num_features, sampling=noise_sampling, device=device
    )
    ts_noise_sample = ts_noise.sample(return_signal_types=True)
    assert isinstance(ts_noise_sample, tuple)
    ts_noise_sample = ts_noise_sample[0].cpu().numpy()

    # --- Plot noise samples as time series (separated) ---
    fig_noise, axes_noise = plt.subplots(
        total_rows, total_cols, figsize=(6 * total_cols, 3 * total_rows)
    )
    # axes_noise is already a 2D array (num_features, 2)
    for i in range(num_features):
        # Plot Gaussian Noise for Feature i
        ax_gauss = axes_noise[i, 0]
        ax_gauss.plot(gaussian_noise_sample[:, i])
        ax_gauss.set_title(f"Feature {i + 1} (Gaussian Noise)")
        ax_gauss.set_xlabel("Sample")
        ax_gauss.set_ylabel("Value")

        # Plot TS Noise for Feature i
        ax_ts = axes_noise[i, 1]
        ax_ts.plot(ts_noise_sample[:, i])
        ax_ts.set_title(f"Feature {i + 1} (TS Noise)")
        ax_ts.set_xlabel("Sample")
        ax_ts.set_ylabel("Value")

    # No unused subplots in this layout

    fig_noise.suptitle("Noise Layer Output Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mlpscm_noise_comparison.png"))
    plt.close(fig_noise)

    # --- MLPSCM with Gaussian Noise ---
    mlp_gauss = MLPSCM(
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
        noise_std=noise_std,
        used_sampler="ts",
        noise_type="gaussian",
        sampling=input_sampling,
        device=device,
    )
    X_g, y_g, denoised_X_g, denoised_y_g, Noise_X_g, Noise_y_g = (
        get_mlp_scm_outputs(mlp_gauss)
    )

    # Create side-by-side visualizations
    def plot_three_components(
        X_orig,
        X_denoised,
        X_noise,
        y_orig,
        y_denoised,
        y_noise,
        title_prefix="",
    ):
        """Plot original, denoised, and noise components as time series."""
        fig, axes = plt.subplots(
            1 + X_orig.shape[1], 3, figsize=(20, 4 * (1 + X_orig.shape[1]))
        )

        # Plot X components as time series (first feature)
        for i in range(X_orig.shape[1]):
            time_steps = range(len(X_orig))
            axes[i, 0].plot(time_steps, X_orig[:, i], color="blue", alpha=0.8)
            axes[i, 0].set_title(f"{title_prefix}Original X (Feature {i + 1})")
            axes[i, 0].set_xlabel("Time Step")
            axes[i, 0].set_ylabel("Value")

            axes[i, 1].plot(
                time_steps, X_denoised[:, i], color="green", alpha=0.8
            )
            axes[i, 1].set_title(f"{title_prefix}Denoised X (Feature {i + 1})")
            axes[i, 1].set_xlabel("Time Step")
            axes[i, 1].set_ylabel("Value")

            axes[i, 2].plot(time_steps, X_noise[:, i], color="red", alpha=0.8)
            axes[i, 2].set_title(f"{title_prefix}Noise X (Feature {i + 1})")
            axes[i, 2].set_xlabel("Time Step")
            axes[i, 2].set_ylabel("Value")

        # Plot y components as time series
        axes[i + 1, 0].plot(
            time_steps, y_orig.flatten(), color="blue", alpha=0.8
        )
        axes[i + 1, 0].set_title(f"{title_prefix}Original y")
        axes[i + 1, 0].set_xlabel("Time Step")
        axes[i + 1, 0].set_ylabel("Value")

        axes[i + 1, 1].plot(
            time_steps, y_denoised.flatten(), color="green", alpha=0.8
        )
        axes[i + 1, 1].set_title(f"{title_prefix}Denoised y")
        axes[i + 1, 1].set_xlabel("Time Step")
        axes[i + 1, 1].set_ylabel("Value")

        axes[i + 1, 2].plot(
            time_steps, y_noise.flatten(), color="red", alpha=0.8
        )
        axes[i + 1, 2].set_title(f"{title_prefix}Noise y")
        axes[i + 1, 2].set_xlabel("Time Step")
        axes[i + 1, 2].set_ylabel("Value")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_dir,
                f"{title_prefix}_mlpscm_noise_feature_signal_comparison.png",
            )
        )
        plt.close(fig)

    # Visualize the three components
    plot_three_components(
        X_g,
        denoised_X_g,
        Noise_X_g,
        y_g,
        denoised_y_g,
        Noise_y_g,
        "Gaussian MLP SCM - ",
    )

    # --- MLPSCM with TS Noise ---
    mlp_ts = MLPSCM(
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
        noise_std=noise_std,
        used_sampler="ts",
        noise_type="ts",
        sampling=input_sampling,
        ts_noise_sampling=noise_sampling,
        device=device,
    )
    X_t, y_t, denoised_X_t, denoised_y_t, Noise_X_t, Noise_y_t = (
        get_mlp_scm_outputs(mlp_ts)
    )

    plot_three_components(
        X_t,
        denoised_X_t,
        Noise_X_t,
        y_t,
        denoised_y_t,
        Noise_y_t,
        "TS MLP SCM - ",
    )

    # --- MLPSCM with Mixed Noise ---

    mlp_mixed = MLPSCM(
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
        noise_std=noise_std,
        used_sampler="ts",
        noise_type="mixed",
        mixed_noise_ratio=1.0,
        sampling=input_sampling,
        ts_noise_sampling=noise_sampling,
        device=device,
    )
    X_m, y_m, denoised_X_m, denoised_y_m, Noise_X_m, Noise_y_m = (
        get_mlp_scm_outputs(mlp_mixed)
    )

    plot_three_components(
        X_m,
        denoised_X_m,
        Noise_X_m,
        y_m,
        denoised_y_m,
        Noise_y_m,
        "Mixed MLP SCM 1.0 mixing - ",
    )

    mlp_mixed = MLPSCM(
        seq_len=seq_len,
        num_features=num_features,
        num_outputs=num_outputs,
        noise_std=noise_std,
        used_sampler="ts",
        noise_type="mixed",
        mixed_noise_ratio=0.0,
        sampling=input_sampling,
        ts_noise_sampling=noise_sampling,
        device=device,
    )
    X_m, y_m, denoised_X_m, denoised_y_m, Noise_X_m, Noise_y_m = (
        get_mlp_scm_outputs(mlp_mixed)
    )

    plot_three_components(
        X_m,
        denoised_X_m,
        Noise_X_m,
        y_m,
        denoised_y_m,
        Noise_y_m,
        "Mixed MLP SCM 0.0 mixing - ",
    )

    # --- Plot data samples as time series (separated) ---
    fig_data, axes_data = plt.subplots(
        total_rows, total_cols, figsize=(6 * total_cols, 3 * total_rows)
    )
    # axes_data is already a 2D array (num_features, 2)
    for i in range(num_features):
        # Plot Gaussian Data for Feature i
        ax_gauss = axes_data[i, 0]
        ax_gauss.plot(X_g[:, i])
        ax_gauss.set_title(f"Feature {i + 1} (Gaussian Data)")
        ax_gauss.set_xlabel("Sample")
        ax_gauss.set_ylabel("Value")

        # Plot TS Data for Feature i
        ax_ts = axes_data[i, 1]
        ax_ts.plot(X_t[:, i])
        ax_ts.set_title(f"Feature {i + 1} (TS Data)")
        ax_ts.set_xlabel("Sample")
        ax_ts.set_ylabel("Value")

    # No unused subplots in this layout

    fig_data.suptitle("MLPSCM Data Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mlpscm_feature_comparison.png"))
    plt.close(fig_data)

    # --- Optionally, plot output y as well ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("MLPSCM Output y (Gaussian Noise)")
    plt.plot(y_g)
    plt.xlabel("Sample")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.title("MLPSCM Output y (TS Noise)")
    plt.plot(y_t)
    plt.xlabel("Sample")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mlpscm_output_comparison.png"))
    plt.close()
    print(f"Saved figures to {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=1)
    parser.add_argument(
        "--noise_sampling", type=str, default="impulse_periodic"
    )
    parser.add_argument("--input_sampling", type=str, default="linear_trend")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_noise_and_data(
        seq_len=args.seq_len,
        num_features=args.num_features,
        num_outputs=args.num_outputs,
        noise_std=args.noise_std,
        noise_sampling=args.noise_sampling,
        input_sampling=args.input_sampling,
    )


if __name__ == "__main__":
    main()

# uv run src/synthefy_pkg/prior/visualizations/corr_noise_visualize.py
