#!/usr/bin/env python3
"""
Script to aggregate multivariate predictability and predictability statistics by domain.
Also creates scatterplots comparing predictability scores against MAPE values from S3.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import boto3
import botocore.exceptions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def load_domain_mapping(domains_file: str) -> Dict[str, str]:
    """Load the domain mapping from domains.json file."""
    with open(domains_file, "r") as f:
        domains_data = json.load(f)

    dataset_to_domain = {}
    for domain_entry in domains_data["datasets"]:
        domain = domain_entry["domain"]
        for dataset_name in domain_entry["dataset_name"]:
            dataset_to_domain[dataset_name] = domain

    return dataset_to_domain


def load_predictability_stats(predictability_file: str) -> Dict[str, float]:
    """Load predictability statistics from JSON file."""
    with open(predictability_file, "r") as f:
        data = json.load(f)

    # Extract values from nested structure
    predictability_values = {}
    for dataset_name, nested_dict in data.items():
        # Get the first (and typically only) value from the nested dict
        for key, value in nested_dict.items():
            predictability_values[dataset_name] = value
            break  # Only take the first value
    return predictability_values


def load_multivariate_stats(multivariate_file: str) -> Dict[str, float]:
    """Load multivariate predictability statistics from JSON file."""
    with open(multivariate_file, "r") as f:
        return json.load(f)


def download_s3_aggregated_data(
    dataset_names: List[str],
    s3_bucket: str = "synthefy-fm-dataset-forecasts",
    s3_prefix: str = "aggregated",
) -> Dict[str, Dict]:
    """Download aggregated JSON files from S3 for given dataset names."""
    s3_client = boto3.client("s3")
    aggregated_data = {}

    print(
        f"Downloading aggregated data from S3 bucket: {s3_bucket}/{s3_prefix}"
    )

    for dataset_name in dataset_names:
        s3_key = f"{s3_prefix}/{dataset_name}_aggregated.json"
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json", delete=False
            ) as temp_file:
                s3_client.download_file(s3_bucket, s3_key, temp_file.name)

                with open(temp_file.name, "r") as f:
                    data = json.load(f)
                    aggregated_data[dataset_name] = data

                os.unlink(temp_file.name)
        except botocore.exceptions.ClientError as e:
            print(f"Warning: Could not download {s3_key}: {e}")
            continue
    return aggregated_data


def extract_mape_values(aggregated_data: Dict[str, Dict]) -> Dict[str, float]:
    """Extract mean MAPE values from aggregated data."""
    data_model_mape_values = dict()
    for dataset_name, data in aggregated_data.items():
        if "model_mape_values" in data:
            data_model_mape_values[dataset_name] = np.clip(
                np.mean(list(data["model_mape_values"].values())), 0.0, 1.0
            )
        else:
            data_model_mape_values[dataset_name] = 0.5
    return data_model_mape_values


def extract_univariate_multivariate_mape_diff(
    aggregated_data: Dict[str, Dict],
) -> Dict[str, float]:
    """Extract MAPE values for univariate and multivariate models and compute mean difference (univariate - multivariate)."""
    mape_differences = dict()

    for dataset_name, data in aggregated_data.items():
        if "model_mape_values" not in data:
            mape_differences[dataset_name] = 0.0
            continue

        model_mape_values = data["model_mape_values"]

        # Extract univariate model MAPE values
        univariate_mape_values = []
        multivariate_mape_values = []

        for model_name, mape_value in model_mape_values.items():
            if "univariate" in model_name.lower():
                univariate_mape_values.append(mape_value)
            elif "multivariate" in model_name.lower():
                multivariate_mape_values.append(mape_value)

        # Compute mean difference (univariate - multivariate)
        if univariate_mape_values and multivariate_mape_values:
            univariate_mean = np.mean(univariate_mape_values)
            multivariate_mean = np.mean(multivariate_mape_values)
            difference = univariate_mean - multivariate_mean
            # Clip to reasonable range
            mape_differences[dataset_name] = np.clip(difference, -0.2, 0.2)
        else:
            mape_differences[dataset_name] = 0.0

    return mape_differences


def create_domain_legend(
    dataset_to_domain: Dict[str, str],
    output_dir: str = "/home/caleb/synthefy-package",
):
    """Create a separate horizontal legend file for domain colors."""

    # Get all unique domains
    all_domains = sorted(set(dataset_to_domain.values()))

    # Create color mapping consistent with the plots
    colors = plt.colormaps["tab20"](np.linspace(0, 1, len(all_domains)))
    domain_color_map = {
        domain: colors[i] for i, domain in enumerate(all_domains)
    }

    # Create legend figure with larger size for two rows
    fig, ax_plot = plt.subplots(figsize=(12, 4))
    ax = cast(Axes, ax_plot)

    # Create legend elements
    legend_elements = []
    for domain in all_domains:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=domain_color_map[domain],
                markersize=16,
                label=domain,
            )
        )

    # Calculate number of columns for two equal rows
    n_domains = len(all_domains)
    ncol = (n_domains + 1) // 2  # Round up to ensure two rows

    # Create legend with two rows
    ax.legend(
        handles=legend_elements,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=14,
    )

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Save legend as PDF
    legend_file_pdf = os.path.join(output_dir, "domain_legend.pdf")
    plt.savefig(
        legend_file_pdf, format="pdf", bbox_inches="tight", pad_inches=0.1
    )

    # Save legend as SVG
    legend_file_svg = os.path.join(output_dir, "domain_legend.svg")
    plt.savefig(
        legend_file_svg, format="svg", bbox_inches="tight", pad_inches=0.1
    )

    plt.close()
    print(f"Domain legend saved to: {legend_file_pdf}")
    print(f"Domain legend saved to: {legend_file_svg}")

    return legend_file_pdf


def create_scatterplot(
    predictability_stats: Dict[str, float],
    multivariate_stats: Dict[str, float],
    mape_values: Dict[str, float],
    mape_diff_values: Dict[str, float],
    dataset_to_domain: Dict[str, str],
    output_dir: str = "/home/caleb/synthefy-package",
):
    """Create separate scatterplots comparing predictability scores against MAPE values with regression lines."""

    all_datasets = (
        set(predictability_stats.keys())
        | set(multivariate_stats.keys())
        | set(mape_values.keys())
        | set(mape_diff_values.keys())
    )
    df = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "domain": dataset_to_domain.get(dataset_name, "Unknown"),
                "predictability": predictability_stats.get(
                    dataset_name, np.nan
                ),
                "multivariate": multivariate_stats.get(dataset_name, np.nan),
                "mape": mape_values.get(dataset_name, np.nan),
                "mape_diff": mape_diff_values.get(dataset_name, np.nan),
            }
            for dataset_name in all_datasets
        ]
    )

    # Clip outliers to 3 standard deviations
    for col in ["predictability", "multivariate", "mape", "mape_diff"]:
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Plot 1: Predictability vs MAPE
    valid_data = df.dropna(subset=["predictability", "mape"])
    if not valid_data.empty:
        fig, ax_plot = plt.subplots(1, 1, figsize=(10, 8))
        ax = cast(Axes, ax_plot)

        # Create scatter plot with diverse colors
        domain_codes = valid_data["domain"].astype("category").cat.codes
        colors = plt.colormaps["tab20"](
            np.linspace(0, 1, len(valid_data["domain"].unique()))
        )
        ax.scatter(
            valid_data["predictability"],
            valid_data["mape"],
            color=[colors[code] for code in domain_codes],
            alpha=0.9,
            s=80,
        )

        # Add regression line
        x_vals = np.asarray(valid_data["predictability"].values)
        y_vals = np.asarray(valid_data["mape"].values)
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel("Univariate Predictability Score", fontsize=24)
        ax.set_ylabel("Mean MAPE", fontsize=24)
        ax.set_title(
            "Univariate Predictability vs MAPE", fontsize=28, fontweight="bold"
        )
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True, alpha=0.3)

        # Calculate correlation and R-squared
        corr = np.corrcoef(valid_data["predictability"], valid_data["mape"])[
            0, 1
        ]
        r_squared = corr**2

        # Add statistics text
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}\nR²: {r_squared:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Add dataset labels
        for _, row in valid_data.iterrows():
            ax.annotate(
                row["dataset_name"],
                (row["predictability"], row["mape"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        output_file1 = os.path.join(output_dir, "predictability_vs_mape.pdf")
        plt.savefig(output_file1, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Predictability vs MAPE plot saved to: {output_file1}")

    # Plot 2: Multivariate vs MAPE Difference (Univariate - Multivariate)
    valid_data = df.dropna(subset=["multivariate", "mape_diff"])
    if not valid_data.empty:
        fig, ax_plot = plt.subplots(1, 1, figsize=(10, 8))
        ax = cast(Axes, ax_plot)

        # Create scatter plot with diverse colors
        domain_codes = valid_data["domain"].astype("category").cat.codes
        colors = plt.colormaps["tab20"](
            np.linspace(0, 1, len(valid_data["domain"].unique()))
        )
        ax.scatter(
            valid_data["multivariate"],
            valid_data["mape_diff"],
            color=[colors[code] for code in domain_codes],
            alpha=0.9,
            s=80,
        )

        # Add regression line
        x_vals = np.asarray(valid_data["multivariate"].values)
        y_vals = np.asarray(valid_data["mape_diff"].values)
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel("Multivariate Predictability Score", fontsize=24)
        ax.set_ylabel(
            "MAPE Difference (Univariate - Multivariate)", fontsize=24
        )
        ax.set_title(
            "Multivariate Predictability vs MAPE Difference",
            fontsize=28,
            fontweight="bold",
        )
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True, alpha=0.3)

        # Calculate correlation and R-squared
        corr = np.corrcoef(valid_data["multivariate"], valid_data["mape_diff"])[
            0, 1
        ]
        r_squared = corr**2

        # Add statistics text
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}\nR²: {r_squared:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Add dataset labels
        for _, row in valid_data.iterrows():
            ax.annotate(
                row["dataset_name"],
                (row["multivariate"], row["mape_diff"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        output_file2 = os.path.join(output_dir, "multivariate_vs_mape_diff.pdf")
        plt.savefig(output_file2, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Multivariate vs MAPE Difference plot saved to: {output_file2}")

    # Plot 3: Predictability vs MAPE (by domain)
    valid_data = df.dropna(subset=["predictability", "mape"])
    if not valid_data.empty:
        fig, ax_plot = plt.subplots(1, 1, figsize=(12, 8))
        ax = cast(Axes, ax_plot)

        # Get all unique domains from the mapping to ensure complete legend
        all_domains = sorted(set(dataset_to_domain.values()))

        # Create color mapping for consistent colors across plots
        colors = plt.colormaps["tab20"](np.linspace(0, 1, len(all_domains)))
        domain_color_map = {
            domain: colors[i] for i, domain in enumerate(all_domains)
        }

        # Create scatter plot by domain
        for domain in all_domains:
            domain_data = valid_data[valid_data["domain"] == domain]
            if not domain_data.empty:
                ax.scatter(
                    domain_data["predictability"],
                    domain_data["mape"],
                    color=domain_color_map[domain],
                    label=domain,
                    alpha=0.9,
                    s=80,
                )
            else:
                # Add empty entry to legend for domains without data
                ax.scatter(
                    [],
                    [],
                    color=domain_color_map[domain],
                    label=domain,
                    alpha=0.9,
                    s=80,
                )

        # Add single regression line across all data
        x_vals = np.asarray(valid_data["predictability"].values)
        y_vals = np.asarray(valid_data["mape"].values)
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel("Predictability Score", fontsize=24)
        ax.set_ylabel("Mean MAPE", fontsize=24)
        ax.set_title("Predictability vs MAPE", fontsize=28, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file3 = os.path.join(
            output_dir, "predictability_vs_mape_by_domain.pdf"
        )
        plt.savefig(output_file3, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"Predictability vs MAPE by domain plot saved to: {output_file3}")

    # Plot 4: Multivariate vs MAPE Difference
    valid_data = df.dropna(subset=["multivariate", "mape_diff"])
    if not valid_data.empty:
        fig, ax_plot = plt.subplots(1, 1, figsize=(12, 8))
        ax = cast(Axes, ax_plot)

        # Get all unique domains from the mapping to ensure complete legend
        all_domains = sorted(set(dataset_to_domain.values()))

        # Create color mapping for consistent colors across plots
        colors = plt.colormaps["tab20"](np.linspace(0, 1, len(all_domains)))
        domain_color_map = {
            domain: colors[i] for i, domain in enumerate(all_domains)
        }

        # Create scatter plot by domain
        for domain in all_domains:
            domain_data = valid_data[valid_data["domain"] == domain]
            if not domain_data.empty:
                ax.scatter(
                    domain_data["multivariate"],
                    domain_data["mape_diff"],
                    color=domain_color_map[domain],
                    label=domain,
                    alpha=0.9,
                    s=80,
                )
            else:
                # Add empty entry to legend for domains without data
                ax.scatter(
                    [],
                    [],
                    color=domain_color_map[domain],
                    label=domain,
                    alpha=0.9,
                    s=80,
                )

        # Add single regression line across all data
        x_vals = np.asarray(valid_data["multivariate"].values)
        y_vals = np.asarray(valid_data["mape_diff"].values)
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel("Multivariate Predictability Score", fontsize=24)
        ax.set_ylabel("MAPE Difference (Uni - Multi)", fontsize=24)
        ax.set_title(
            "Multivariate Predictability vs MAPE Difference",
            fontsize=28,
            fontweight="bold",
        )
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file4 = os.path.join(
            output_dir, "multivariate_vs_mape_diff_by_domain.pdf"
        )
        plt.savefig(output_file4, format="pdf", bbox_inches="tight")
        plt.close()
        print(
            f"Multivariate vs MAPE Difference by domain plot saved to: {output_file4}"
        )

    # Save summary CSV
    summary_file = os.path.join(output_dir, "predictability_mape_summary.csv")
    df.to_csv(summary_file, index=False)
    print(f"Summary data saved to: {summary_file}")

    # Create separate domain legend
    create_domain_legend(dataset_to_domain, output_dir)

    return df


def aggregate_by_domain(
    dataset_to_domain: Dict[str, str],
    predictability_stats: Dict[str, float],
    multivariate_stats: Dict[str, float],
) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate statistics by domain."""
    domain_stats = {
        domain: {"predictability": [], "multivariate": []}
        for domain in set(dataset_to_domain.values())
    }

    for dataset_name, domain in dataset_to_domain.items():
        if dataset_name in predictability_stats:
            domain_stats[domain]["predictability"].append(
                predictability_stats[dataset_name]
            )
        if dataset_name in multivariate_stats:
            domain_stats[domain]["multivariate"].append(
                multivariate_stats[dataset_name]
            )

    return domain_stats


def compute_domain_statistics(
    domain_stats: Dict[str, Dict[str, List[float]]],
) -> pd.DataFrame:
    """Compute mean and standard deviation for each domain."""
    results = []

    # Collect all predictability and multivariate values across all domains
    all_predictability_values = []
    all_multivariate_values = []

    for domain, stats in domain_stats.items():
        row: Dict[str, Any] = {"Domain": domain}

        # Predictability statistics
        if stats["predictability"]:
            row.update(
                {
                    "Predictability_Mean": float(
                        np.mean(stats["predictability"])
                    ),
                    "Predictability_Std": float(
                        np.std(stats["predictability"])
                    ),
                    "Predictability_Count": int(len(stats["predictability"])),
                }
            )
            all_predictability_values.extend(stats["predictability"])
        else:
            row.update(
                {
                    "Predictability_Mean": float("nan"),
                    "Predictability_Std": float("nan"),
                    "Predictability_Count": 0,
                }
            )

        # Multivariate statistics
        if stats["multivariate"]:
            row.update(
                {
                    "Multivariate_Mean": float(np.mean(stats["multivariate"])),
                    "Multivariate_Std": float(np.std(stats["multivariate"])),
                    "Multivariate_Count": int(len(stats["multivariate"])),
                }
            )
            all_multivariate_values.extend(stats["multivariate"])
        else:
            row.update(
                {
                    "Multivariate_Mean": float("nan"),
                    "Multivariate_Std": float("nan"),
                    "Multivariate_Count": 0,
                }
            )

        results.append(row)

    # Add combined statistics row
    combined_row: Dict[str, Any] = {"Domain": "Combined"}

    # Combined predictability statistics
    if all_predictability_values:
        combined_row.update(
            {
                "Predictability_Mean": float(
                    np.mean(all_predictability_values)
                ),
                "Predictability_Std": float(np.std(all_predictability_values)),
                "Predictability_Count": int(len(all_predictability_values)),
            }
        )
    else:
        combined_row.update(
            {
                "Predictability_Mean": float("nan"),
                "Predictability_Std": float("nan"),
                "Predictability_Count": 0,
            }
        )

    # Combined multivariate statistics
    if all_multivariate_values:
        combined_row.update(
            {
                "Multivariate_Mean": float(np.mean(all_multivariate_values)),
                "Multivariate_Std": float(np.std(all_multivariate_values)),
                "Multivariate_Count": int(len(all_multivariate_values)),
            }
        )
    else:
        combined_row.update(
            {
                "Multivariate_Mean": float("nan"),
                "Multivariate_Std": float("nan"),
                "Multivariate_Count": 0,
            }
        )

    results.append(combined_row)

    return pd.DataFrame(results)


def save_domain_table(df: pd.DataFrame, output_file: str):
    """Save domain aggregation results as CSV and print summary statistics."""
    df_sorted = df.sort_values("Domain").reset_index(drop=True)

    # Save the main table as CSV
    df_sorted.to_csv(output_file, index=False)
    print(f"Domain aggregation table saved to: {output_file}")

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Total domains: {len(df_sorted)}")
    print(
        f"Domains with predictability data: {len(df_sorted[df_sorted['Predictability_Count'] > 0])}"
    )
    print(
        f"Domains with multivariate data: {len(df_sorted[df_sorted['Multivariate_Count'] > 0])}"
    )

    # Overall statistics
    all_predictability = []
    all_multivariate = []

    for _, row in df_sorted.iterrows():
        if not pd.isna(row["Predictability_Mean"]):
            all_predictability.extend(
                [row["Predictability_Mean"]] * int(row["Predictability_Count"])
            )
        if not pd.isna(row["Multivariate_Mean"]):
            all_multivariate.extend(
                [row["Multivariate_Mean"]] * int(row["Multivariate_Count"])
            )

    if all_predictability:
        print(
            f"Overall Predictability - Mean: {np.mean(all_predictability):.4f}, Std: {np.std(all_predictability):.4f}"
        )
    if all_multivariate:
        print(
            f"Overall Multivariate - Mean: {np.mean(all_multivariate):.4f}, Std: {np.std(all_multivariate):.4f}"
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate predictability statistics by domain and create scatterplots"
    )

    # File paths
    parser.add_argument(
        "--domains-file",
        default="/home/caleb/synthefy-package/src/synthefy_pkg/fm_evals/scripts/domains.json",
        help="Path to domains.json file",
    )
    parser.add_argument(
        "--predictability-file",
        default="/workspace/fm_eval/predictability_statistics.json",
        help="Path to predictability statistics JSON file",
    )
    parser.add_argument(
        "--multivariate-file",
        default="/workspace/fm_eval/multivariate_predictability_statistics.json",
        help="Path to multivariate predictability statistics JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/caleb/synthefy-package",
        help="Output directory for results",
    )

    # Operations
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Run domain aggregation analysis",
    )
    parser.add_argument(
        "--scatterplot",
        action="store_true",
        help="Create scatterplots comparing predictability vs MAPE",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all operations (equivalent to --aggregate --scatterplot)",
    )

    # S3 options
    parser.add_argument(
        "--s3-bucket",
        default="synthefy-fm-dataset-forecasts",
        help="S3 bucket name for aggregated data",
    )
    parser.add_argument(
        "--s3-prefix",
        default="aggregated",
        help="S3 prefix/folder path for aggregated data",
    )

    return parser.parse_args()


def main():
    """Main function to run the aggregation."""
    args = parse_arguments()

    # If --all is specified, enable both operations
    if args.all:
        args.aggregate = True
        args.scatterplot = True

    # If no operations specified, run all by default
    if not args.aggregate and not args.scatterplot:
        args.aggregate = True
        args.scatterplot = True

    print("Loading domain mapping...")
    dataset_to_domain = load_domain_mapping(args.domains_file)
    print(f"Loaded {len(dataset_to_domain)} dataset-domain mappings")

    # Load statistics if needed for any operation
    predictability_stats = {}
    multivariate_stats = {}

    if args.aggregate or args.scatterplot:
        print("Loading predictability statistics...")
        predictability_stats = load_predictability_stats(
            args.predictability_file
        )
        print(f"Loaded {len(predictability_stats)} predictability statistics")

        print("Loading multivariate statistics...")
        multivariate_stats = load_multivariate_stats(args.multivariate_file)
        print(f"Loaded {len(multivariate_stats)} multivariate statistics")

    # Run domain aggregation if requested
    if args.aggregate:
        print("\n" + "=" * 60)
        print("DOMAIN AGGREGATION")
        print("=" * 60)

        print("Aggregating by domain...")
        domain_stats = aggregate_by_domain(
            dataset_to_domain, predictability_stats, multivariate_stats
        )

        print("Computing domain statistics...")
        results_df = compute_domain_statistics(domain_stats)

        # Save results to CSV
        output_file = os.path.join(
            args.output_dir, "domain_aggregation_results.csv"
        )
        save_domain_table(results_df, output_file)

    # Run scatterplot creation if requested
    if args.scatterplot:
        print("\n" + "=" * 60)
        print("CREATING SCATTERPLOTS")
        print("=" * 60)

        # Get all unique dataset names
        all_datasets = set(predictability_stats.keys()) | set(
            multivariate_stats.keys()
        )
        dataset_names = list(all_datasets)

        print(
            f"Downloading aggregated data for {len(dataset_names)} datasets..."
        )
        aggregated_data = download_s3_aggregated_data(
            dataset_names, args.s3_bucket, args.s3_prefix
        )

        if aggregated_data:
            print("Extracting MAPE values...")
            mape_values = extract_mape_values(aggregated_data)

            print("Extracting MAPE difference values...")
            mape_diff_values = extract_univariate_multivariate_mape_diff(
                aggregated_data
            )

            print("Creating scatterplots...")
            scatterplot_df = create_scatterplot(
                predictability_stats,
                multivariate_stats,
                mape_values,
                mape_diff_values,
                dataset_to_domain,
                args.output_dir,
            )

            # Print summary statistics
            print("\nSCATTERPLOT SUMMARY:")
            print(
                f"Datasets with predictability data: {len([x for x in scatterplot_df['predictability'] if not pd.isna(x)])}"
            )
            print(
                f"Datasets with multivariate data: {len([x for x in scatterplot_df['multivariate'] if not pd.isna(x)])}"
            )
            print(
                f"Datasets with MAPE data: {len([x for x in scatterplot_df['mape'] if not pd.isna(x)])}"
            )
            print(
                f"Datasets with MAPE difference data: {len([x for x in scatterplot_df['mape_diff'] if not pd.isna(x)])}"
            )

            # Correlation analysis
            valid_predictability = scatterplot_df.dropna(
                subset=["predictability", "mape"]
            )
            valid_multivariate = scatterplot_df.dropna(
                subset=["multivariate", "mape_diff"]
            )

            if not valid_predictability.empty:
                corr_pred = np.corrcoef(
                    valid_predictability["predictability"],
                    valid_predictability["mape"],
                )[0, 1]
                print(f"Predictability vs MAPE correlation: {corr_pred:.3f}")

            if not valid_multivariate.empty:
                corr_multi = np.corrcoef(
                    valid_multivariate["multivariate"],
                    valid_multivariate["mape_diff"],
                )[0, 1]
                print(
                    f"Multivariate vs MAPE Difference correlation: {corr_multi:.3f}"
                )

        else:
            print(
                "No aggregated data downloaded. Skipping scatterplot creation."
            )


if __name__ == "__main__":
    main()
