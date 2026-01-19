import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D

COMPILE = False


@dataclass
class AnomalyMetadata:
    """
    Metadata container for each detected anomaly.
    """

    timestamp: Any
    score: float
    original_value: float
    predicted_value: float
    group_metadata: Dict[str, Any]


def get_auto_window_size(freq_type: str) -> int:
    """
    Determine appropriate window size based on data frequency.
    """
    freq_num = (
        int("".join(filter(str.isdigit, freq_type)))
        if any(c.isdigit() for c in freq_type)
        else 1
    )
    freq_unit = "".join(filter(str.isalpha, freq_type))

    if freq_unit in ["T", "min"]:
        return (24 * 60 // freq_num) * 2  # 2 days worth of minutes
    elif freq_unit == "H":
        return 48  # 2 days worth of hours
    elif freq_unit == "D":
        return 14  # 2 weeks worth of days
    elif freq_unit == "W":
        return 26  # 6 months worth of weeks
    elif freq_unit == "M":
        return 24  # 2 years worth of months
    else:
        logger.warning(f"Unknown frequency type: {freq_type}, defaulting to 48 periods")
        return 48


def plot_anomaly_windows(
    df: pd.DataFrame,
    results: Dict[str, Dict[str, Dict[str, List[AnomalyMetadata]]]],
    timestamps_col: List[str],
    window_size: Optional[int] = None,
    max_plots_per_type: int = 5,
    plot_top_anomalies: bool = True,
    save_path: Optional[str] = None,
    freq_type: str = "H",
) -> None:
    """
    Create window plots around detected anomalies, organized by KPI and anomaly type.
    """
    if not save_path:
        raise ValueError("save_path is required for organized plot storage")

    if window_size is None:
        window_size = get_auto_window_size(freq_type)
        logger.info(
            f"Auto-determined window size: {window_size} periods for frequency {freq_type}"
        )

    half_window = window_size // 2
    anomaly_types = ["peak", "scattered", "out_of_pattern"]

    # Process each KPI
    for kpi in results.keys():
        logger.info(f"Creating plots for KPI: {kpi}")
        kpi_dir = os.path.join(save_path, kpi)
        os.makedirs(kpi_dir, exist_ok=True)

        # Process each anomaly type
        for anomaly_type in anomaly_types:
            logger.info(f"Processing {anomaly_type} anomalies for {kpi}")
            anomaly_dir = os.path.join(kpi_dir, anomaly_type)
            os.makedirs(anomaly_dir, exist_ok=True)

            # Get all anomalies for this type
            all_anomalies = []
            for group_key, group_anomalies in results[kpi][anomaly_type].items():
                for anomaly in group_anomalies:
                    all_anomalies.append((group_key, anomaly))

            if not all_anomalies:
                logger.warning(f"No anomalies found for {kpi} - {anomaly_type}")
                continue

            # Sort by score
            all_anomalies.sort(key=lambda x: x[1].score, reverse=plot_top_anomalies)
            plot_anomalies = all_anomalies[:max_plots_per_type]

            # Create plots for selected anomalies
            for plot_idx, (group_key, anomaly) in enumerate(plot_anomalies, 1):
                try:
                    # Create group filter
                    group_filter = pd.Series(True, index=df.index)
                    for col, val in anomaly.group_metadata.items():
                        group_filter &= df[col] == val

                    # Get window of data around anomaly
                    window_start = anomaly.timestamp - pd.Timedelta(hours=half_window)
                    window_end = anomaly.timestamp + pd.Timedelta(hours=half_window)

                    window_data = df[group_filter].copy()
                    window_data = window_data[
                        (window_data[timestamps_col[0]] >= window_start)
                        & (window_data[timestamps_col[0]] <= window_end)
                    ]

                    if len(window_data) < 2:
                        logger.warning(
                            f"Insufficient data points for anomaly at {anomaly.timestamp}"
                        )
                        continue

                    # Create plot
                    plt.figure(figsize=(12, 6))

                    # Plot the time series
                    plt.plot(
                        window_data[timestamps_col[0]],
                        window_data[kpi],
                        label="Actual",
                        color="blue",
                        alpha=0.7,
                    )

                    # Mark the anomaly point
                    plt.axvline(
                        x=anomaly.timestamp,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="Anomaly Time",
                    )
                    plt.scatter(
                        anomaly.timestamp,
                        anomaly.original_value,
                        color="red",
                        s=100,
                        zorder=5,
                        label=f"Anomaly (score: {anomaly.score:.2f})",
                    )

                    # Add predicted value if available (for peak anomalies)
                    if anomaly_type == "peak":
                        plt.scatter(
                            anomaly.timestamp,
                            anomaly.predicted_value,
                            color="green",
                            s=100,
                            zorder=4,
                            label="Predicted Value",
                        )

                    # Customize plot
                    group_label = " | ".join(
                        f"{k}: {v}" for k, v in anomaly.group_metadata.items()
                    )
                    plt.title(f"{kpi} - {anomaly_type.title()} Anomaly\n{group_label}")
                    plt.xlabel("Timestamp")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Save plot
                    timestamp_str = anomaly.timestamp.strftime("%Y%m%d_%H%M%S")
                    plot_filename = (
                        f"{'top' if plot_top_anomalies else 'bottom'}_"
                        f"rank_{plot_idx:02d}_score_{anomaly.score:.2f}_"
                        f"{timestamp_str}.png"
                    )
                    plot_path = os.path.join(anomaly_dir, plot_filename)

                    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
                    plt.close()

                    logger.info(
                        f"Saved plot {plot_idx}/{len(plot_anomalies)} "
                        f"for {kpi} - {anomaly_type}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error creating plot for {kpi} - {anomaly_type}: {str(e)}"
                    )
                    continue

    logger.info("Completed generating all anomaly plots")


def plot_concurrent_anomalies(
    df: pd.DataFrame,
    concurrent_results: Dict[str, List[Dict[str, Any]]],
    timestamps_col: List[str],
    window_size: Optional[int] = None,
    max_plots: int = 10,
    save_path: Optional[str] = None,
    freq_type: str = "H",
) -> None:
    """
    Create plots for concurrent anomalies showing all involved KPIs.
    """
    if not save_path:
        raise ValueError("save_path is required for plot storage")

    if not concurrent_results:
        logger.warning("No concurrent anomalies to plot")
        return

    if window_size is None:
        window_size = get_auto_window_size(freq_type)
        logger.info(
            f"Auto-determined window size: {window_size} periods for frequency {freq_type}"
        )

    half_window = window_size // 2

    # Sort clusters by total score and number of KPIs involved
    sorted_clusters = sorted(
        concurrent_results.items(),
        key=lambda x: (x[1]["distinct_kpis"], x[1]["total_score"]),
        reverse=True,
    )

    concurrent_dir = os.path.join(save_path, "concurrent_anomalies")
    os.makedirs(concurrent_dir, exist_ok=True)

    for cluster_idx, (timestamp_key, cluster) in enumerate(sorted_clusters[:max_plots]):
        try:
            cluster_time = pd.Timestamp(cluster["timestamp"])
            window_start = cluster_time - pd.Timedelta(hours=half_window)
            window_end = cluster_time + pd.Timedelta(hours=half_window)

            # Group anomalies by KPI and then by group
            kpi_group_anomalies = {}
            for kpi in cluster["kpis_involved"]:
                kpi_anomalies = [a for a in cluster["anomalies"] if a["kpi"] == kpi]
                # Group anomalies by group_key
                grouped_anomalies = {}
                for anomaly in kpi_anomalies:
                    group_key = str(anomaly["group_key"])
                    if group_key not in grouped_anomalies:
                        grouped_anomalies[group_key] = []
                    grouped_anomalies[group_key].append(anomaly)
                kpi_group_anomalies[kpi] = grouped_anomalies

            # Calculate total number of subplots needed
            total_subplots = sum(len(groups) for groups in kpi_group_anomalies.values())

            # Create figure with appropriate number of subplots
            fig, axes = plt.subplots(
                total_subplots, 1, figsize=(15, 5 * total_subplots), sharex=True
            )
            if total_subplots == 1:
                axes = [axes]

            current_ax = 0

            # Plot each KPI and group combination
            for kpi, grouped_anomalies in kpi_group_anomalies.items():
                for group_key, group_anomalies in grouped_anomalies.items():
                    ax = axes[current_ax]

                    # Filter data for this group
                    group_filter = pd.Series(True, index=df.index)
                    for col, val in group_anomalies[0]["group_metadata"].items():
                        group_filter &= df[col] == val

                    # Get KPI data within window for this group
                    kpi_data = df[
                        group_filter
                        & (df[timestamps_col[0]] >= window_start)
                        & (df[timestamps_col[0]] <= window_end)
                    ]

                    # Plot the time series
                    ax.plot(
                        kpi_data[timestamps_col[0]],
                        kpi_data[kpi],
                        color="blue",
                        alpha=0.7,
                    )[0]

                    # Plot all anomalies for this group
                    for anomaly in group_anomalies:
                        ax.axvline(
                            x=pd.Timestamp(anomaly["timestamp"]),
                            color="red",
                            linestyle="--",
                            alpha=0.5,
                        )
                        ax.scatter(
                            pd.Timestamp(anomaly["timestamp"]),
                            anomaly["original_value"],
                            color="red",
                            s=100,
                            zorder=5,
                        )
                        if anomaly["predicted_value"] is not None:
                            ax.scatter(
                                pd.Timestamp(anomaly["timestamp"]),
                                anomaly["predicted_value"],
                                color="green",
                                s=100,
                                zorder=4,
                            )

                    # Set title and labels with group information
                    group_label = " | ".join(
                        f"{k}: {v}"
                        for k, v in group_anomalies[0]["group_metadata"].items()
                    )
                    ax.set_title(f"{kpi} | {group_label}")  # Move group label to title
                    ax.set_ylabel("Value")  # Simplified y-label
                    ax.grid(True)

                    # Add a single legend for the first subplot only
                    if current_ax == 0:
                        legend_elements = [
                            Line2D([0], [0], color="blue", label="Actual Value"),
                            Line2D(
                                [0],
                                [0],
                                color="red",
                                linestyle="--",
                                label="Anomaly Time",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                label="Anomaly Point",
                                markerfacecolor="red",
                                markersize=10,
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                label="Predicted Value",
                                markerfacecolor="green",
                                markersize=10,
                            ),
                        ]
                        ax.legend(handles=legend_elements, loc="upper right")

                    current_ax += 1

            # Customize plot
            plt.suptitle(
                f"Concurrent Anomalies Cluster (Rank {cluster_idx + 1})\n"
                f"Time: {cluster_time}\n"
                f"KPIs Involved: {len(cluster['kpis_involved'])} | "
                f"Total Score: {cluster['total_score']:.2f}",
                fontsize=16,
            )
            plt.xlabel("Timestamp")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            timestamp_str = cluster_time.strftime("%Y%m%d_%H%M%S")
            plot_filename = (
                f"concurrent_rank_{cluster_idx + 1:02d}_"
                f"kpis_{len(cluster['kpis_involved'])}_"
                f"score_{cluster['total_score']:.2f}_"
                f"{timestamp_str}.png"
            )
            plot_path = os.path.join(concurrent_dir, plot_filename)

            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(
                f"Saved concurrent anomaly plot {cluster_idx + 1}/{min(max_plots, len(sorted_clusters))}"
            )

        except Exception as e:
            logger.error(
                f"Error creating concurrent anomaly plot for cluster {cluster_idx + 1}: {str(e)}"
            )
            continue

    logger.info("Completed generating concurrent anomaly plots")


def plot_group_anomalies(
    df: pd.DataFrame,
    results: Dict[str, Dict[str, Dict[str, List[AnomalyMetadata]]]],
    all_kpis: List[str],
    anomaly_type: str,
    group_name: str,
    timestamps_col: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (
        15,
        0,
    ),  # Height will be calculated based on number of KPIs
) -> None:
    """
    Plot anomalies for a specific group across all KPIs.

    Args:
        df: Input DataFrame containing the time series data
        results: Dictionary containing anomaly detection results
        anomaly_type: Type of anomalies to plot ('peak', 'scattered', 'out_of_pattern', or 'mixed')
        group_name: Name of the group to plot
        timestamps_col: Name of the timestamp column
        save_path: Optional path to save the plot
        figsize: Base figure size (width, height). Height will be adjusted based on number of KPIs.
    """
    # Get all KPIs that have anomalies for this group
    kpis_with_anomalies = []
    anomaly_types = (
        [anomaly_type]
        if anomaly_type != "mixed"
        else ["peak", "scattered", "out_of_pattern"]
    )

    for kpi in all_kpis:
        if kpi not in results:
            continue
        for atype in anomaly_types:
            if group_name in results[kpi][atype]:
                if kpi not in kpis_with_anomalies:
                    kpis_with_anomalies.append(kpi)

    if not kpis_with_anomalies:
        logger.warning(f"No anomalies found for group {group_name}")
        return

    # Calculate figure height based on number of KPIs
    n_kpis = len(all_kpis)
    height_per_subplot = 4  # Height in inches per subplot
    figsize = (figsize[0], n_kpis * height_per_subplot)

    # Create figure with subplots
    fig, axes = plt.subplots(n_kpis, 1, figsize=figsize, sharex=True)
    if n_kpis == 1:
        axes = [axes]

    # Create legend elements
    legend_elements = create_legend_elements()

    # Add anomaly type specific legend elements
    if anomaly_type == "mixed":
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=4,
                    label="Peak Anomaly",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="orange",
                    markersize=4,
                    label="Scattered Anomaly",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="purple",
                    markersize=4,
                    label="Pattern Anomaly",
                ),
            ]
        )

    # Get group metadata from any anomaly to parse group filter
    group_metadata = {}
    for atype in anomaly_types:
        if (
            group_name in results[kpis_with_anomalies[0]][atype]
            and results[kpis_with_anomalies[0]][atype][group_name]
        ):
            group_metadata = results[kpis_with_anomalies[0]][atype][group_name][
                0
            ].group_metadata
            break

    # Create group filter
    group_filter = pd.Series(True, index=df.index)
    for col, val in group_metadata.items():
        group_filter &= df[col] == val

    # Plot time series
    kpi_data = df[group_filter].copy()  # Create a copy to avoid SettingWithCopyWarning
    # Convert timestamp column to datetime if it's not already
    kpi_data[timestamps_col[0]] = pd.to_datetime(kpi_data[timestamps_col[0]])

    # Plot each KPI
    for idx, kpi in enumerate(all_kpis):
        ax = axes[idx]
        ax.plot(kpi_data[timestamps_col[0]], kpi_data[kpi], color="blue", alpha=0.7)

        if kpi in kpis_with_anomalies:
            # Plot anomalies for each type
            for atype in anomaly_types:
                if group_name in results[kpi][atype]:
                    anomalies = results[kpi][atype][group_name]

                    # # Define marker style based on anomaly type
                    # marker = (
                    #     "o"
                    #     if atype == "peak"
                    #     else ("s" if atype == "scattered" else "^")
                    # )
                    color = (
                        "red"
                        if atype == "peak"
                        else ("orange" if atype == "scattered" else "purple")
                    )

                    for anomaly in anomalies:
                        # Convert timestamp to datetime if it's not already
                        anomaly_timestamp = pd.to_datetime(anomaly.timestamp)

                        # Plot vertical line
                        ax.axvline(
                            x=anomaly_timestamp, color=color, linestyle="--", alpha=0.3
                        )

                        # # Plot anomaly point
                        # ax.scatter(
                        #     anomaly_timestamp,
                        #     anomaly.original_value,
                        #     color=color,
                        #     marker=marker,
                        #     s=40,  # Increased marker size for better visibility
                        #     zorder=5,
                        #     label=f"{atype.title()} (score: {anomaly.score:.2f})",
                        # )

                        # # Plot predicted value if available
                        # if anomaly.predicted_value is not None and atype == "peak":
                        #     ax.scatter(
                        #         anomaly_timestamp,
                        #         anomaly.predicted_value,
                        #         color="green",
                        #         s=40,  # Increased marker size for better visibility
                        #         zorder=4,
                        #     )

        # Customize subplot
        ax.set_title(f"{kpi}")
        ax.set_ylabel("Value")
        ax.grid(True)

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(handles=legend_elements, loc="upper right")

    # Add common xlabel
    plt.xlabel("Timestamp")

    # Add overall title
    plt.suptitle(
        f"Anomalies for Group: {group_name}\nType: {anomaly_type.title()}",
        fontsize=16,
        y=1.02,
    )

    # Adjust layout
    plt.tight_layout()

    # Save or show plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"group_{group_name}_{anomaly_type}_{timestamp_str}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved group anomaly plot to: {os.path.join(save_path, filename)}")
    else:
        plt.show()


def create_legend_elements() -> List[Line2D]:
    """
    Create standard legend elements for anomaly plots.
    """
    return [
        Line2D([0], [0], color="blue", label="Actual Value"),
        Line2D([0], [0], color="red", linestyle="--", label="Anomaly Time"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Anomaly Point",
            markerfacecolor="red",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Predicted Value",
            markerfacecolor="green",
            markersize=10,
        ),
    ]
