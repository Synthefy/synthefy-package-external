#!/usr/bin/env python3
"""
Compare two PyTorch Lightning metrics.csv logs (baseline vs. new) to detect anomalies.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from synthefy_pkg.utils.basic_utils import compare_system_specs

COMPILE = False


def load_metrics(csv_file: str) -> pd.DataFrame:
    """
    Load the metrics CSV into a pandas DataFrame.

    We assume the CSV has columns: epoch, step, train_loss_step, train_loss_epoch, val_loss, test_loss, etc.

    :param csv_file: Path to the metrics CSV file.
    :return: A pandas DataFrame with columns that may include:
             ['epoch', 'step', 'train_loss_epoch', 'val_loss', 'test_loss', ...]
    """
    df = pd.read_csv(csv_file)
    df = df.fillna(np.nan)
    return df


def align_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by epoch and compute the mean (or last) of each metric within that epoch.
    This ensures that we have exactly one row per epoch for simpler comparisons.

    :param df: The raw metrics DataFrame, which may log multiple steps per epoch.
    :return: A new DataFrame indexed by epoch, with columns representing average metrics per epoch.
    """
    if "epoch" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'epoch' column for alignment."
        )

    grouped = df.groupby("epoch").mean(numeric_only=True).reset_index()
    return grouped


def plot_comparison(
    baseline: pd.DataFrame,
    new: pd.DataFrame,
    metric_cols=("train_loss_epoch", "val_loss", "test_loss"),
    image_path: str = "losses_comparison_new_vs_baseline.png",
):
    """
    Plot all metrics in a single figure comparing baseline vs. new run.

    :param baseline: DataFrame with at least 'epoch' and metric columns.
    :param new: DataFrame with same columns as baseline.
    :param metric_cols: Which metrics to plot (list or tuple of column names).
    :param image_path: Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    valid_metrics = [
        m for m in metric_cols if m in baseline.columns and m in new.columns
    ]

    baseline_style = {"linestyle": "-", "marker": "o", "markersize": 4}
    new_style = {"linestyle": "--", "marker": "s", "markersize": 4}

    for metric in valid_metrics:
        plt.plot(
            baseline["epoch"],
            baseline[metric],
            label=f"Baseline {metric}",
            **baseline_style,
        )
        plt.plot(
            new["epoch"],
            new[metric],
            label=f"New {metric}",
            **new_style,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Comparison of Metrics between Baseline and New Run")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


def compute_statistics(
    baseline: pd.DataFrame, new: pd.DataFrame, metric="val_loss"
):
    """
    Compute and return basic statistics comparing baseline vs. new runs
    for a specified metric (e.g. val_loss).

    :param baseline: Aligned DataFrame with columns ['epoch', metric].
    :param new: Aligned DataFrame with columns ['epoch', metric].
    :param metric: The metric column name to compare (e.g. 'val_loss').
    :return: A dict containing the results.
    """
    results = {
        "metric": metric,
        "baseline_final": np.nan,
        "baseline_best": np.nan,
        "new_final": np.nan,
        "new_best": np.nan,
        "mean_diff": np.nan,
    }

    if metric not in baseline.columns or metric not in new.columns:
        logger.error(
            f"Metric '{metric}' not found in both DataFrames. Skipping stats."
        )
        return results

    bvals = baseline[metric].dropna()
    nvals = new[metric].dropna()

    if bvals.empty or nvals.empty:
        logger.error(f"No valid data to compare for metric '{metric}'.")
        return results

    results["baseline_final"] = bvals.iloc[-1]
    results["new_final"] = nvals.iloc[-1]
    results["baseline_best"] = bvals.min()
    results["new_best"] = nvals.min()

    merged = pd.merge(
        baseline[["epoch", metric]],
        new[["epoch", metric]],
        on="epoch",
        how="inner",
        suffixes=("_b", "_n"),
    )
    merged.dropna(inplace=True)
    if not merged.empty:
        diffs = (merged[f"{metric}_b"] - merged[f"{metric}_n"]).abs()
        results["mean_diff"] = diffs.mean()

    return results


def flag_anomaly(stat_results: dict, threshold=0.05):
    """
    Decide if the new run is "abnormal" relative to baseline based on:
    1) Final validation loss comparison.
    2) Best metric value comparison.
    3) Mean absolute difference across epochs.

    :param stat_results: dict from compute_statistics()
    :param threshold: Threshold percentage (as decimal) for difference to consider "large".
    :return: A bool indicating if run is flagged as abnormal, plus a detailed message.
    """
    metric = stat_results["metric"]
    baseline_best = stat_results["baseline_best"]
    baseline_final = stat_results["baseline_final"]
    new_best = stat_results["new_best"]
    new_final = stat_results["new_final"]
    mean_diff = stat_results["mean_diff"]

    is_abnormal = False
    messages = []

    def compare_values(new, baseline, label):
        percent_diff = abs((new - baseline) / baseline)
        direction = "worse" if new > baseline else "better"
        if percent_diff > threshold:
            return (
                True,
                f"[{metric}] {label} ({new:.4f}) is {percent_diff * 100:.1f}% {direction} "
                f"than baseline ({baseline:.4f}), exceeding {threshold * 100:.1f}% threshold.",
            )
        return (
            False,
            f"[{metric}] {label} within normal bounds (difference {percent_diff * 100:.1f}%).",
        )

    final_abnormal, final_msg = compare_values(
        new_final, baseline_final, "Final value"
    )
    best_abnormal, best_msg = compare_values(
        new_best, baseline_best, "Best value"
    )

    if final_abnormal or best_abnormal:
        is_abnormal = True
    messages.extend([final_msg, best_msg])

    if not np.isnan(mean_diff):
        mean_diff_pct = mean_diff / abs(baseline_best)
        if mean_diff_pct > threshold:
            is_abnormal = True
            messages.append(
                f"[{metric}] Mean absolute difference across epochs ({mean_diff:.4f}) "
                f"is {mean_diff_pct * 100:.1f}% of baseline best, exceeding {threshold * 100:.1f}% threshold."
            )
        else:
            messages.append(
                f"[{metric}] Mean absolute difference ({mean_diff:.4f}) within normal bounds "
                f"({mean_diff_pct * 100:.1f}% of baseline best)."
            )

    if not is_abnormal:
        messages.append(
            f"[{metric}] All metrics within normal bounds (no anomalies detected)."
        )

    return is_abnormal, "\n".join(messages)


def main(args):
    compare_system_specs(args.baseline_csv)

    df_baseline = load_metrics(args.baseline_csv)
    df_new = load_metrics(args.new_csv)

    df_baseline_epoch = align_by_epoch(df_baseline)
    df_new_epoch = align_by_epoch(df_new)

    plot_comparison(
        df_baseline_epoch,
        df_new_epoch,
        metric_cols=args.metrics,
        image_path=args.image_path,
    )

    return_code = 0
    for m in args.metrics:
        stats = compute_statistics(df_baseline_epoch, df_new_epoch, metric=m)
        print(f"\n=== Stats for metric '{m}' ===")
        print(f"Baseline final:  {stats['baseline_final']:.2f}")
        print(f"Baseline best:   {stats['baseline_best']:.2f}")
        print(f"New final:       {stats['new_final']:.2f}")
        print(f"New best:        {stats['new_best']:.2f}")
        print(f"Mean diff:       {stats['mean_diff']:.2f}")

        is_abnormal, msg = flag_anomaly(stats, threshold=args.threshold)
        print(msg)
        if is_abnormal:
            logger.error(
                f"The `{m}` in the new run is flagged as abnormal with the threshold of {args.threshold * 100:.2f}%."
            )
            return_code = 1

    return return_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Lightning metrics CSV logs."
    )
    parser.add_argument(
        "baseline_csv", type=str, help="Path to baseline metrics.csv"
    )
    parser.add_argument("new_csv", type=str, help="Path to new run metrics.csv")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["train_loss_epoch", "val_loss", "test_loss"],
        help="Which metrics to compare and plot. Default: train_loss_epoch val_loss test_loss",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold for difference in best metric to consider abnormal (as decimal, e.g. 0.05 for 5%).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="losses_comparison_new_vs_baseline.png",
        help="Path to save the comparison plot image. Default: losses_comparison_new_vs_baseline.png",
    )
    args = parser.parse_args()
    return_code = main(args)
    exit(return_code)

# Sample command to run this script:
"""
python3 src/synthefy_pkg/scripts/compare_lightning_logs_with_base.py \
    src/synthefy_pkg/scripts/base_results/air_quality_metrics.csv \
    <path_to_new_metrics_csv_file>
"""
