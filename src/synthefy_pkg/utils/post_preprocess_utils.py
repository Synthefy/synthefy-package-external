import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import layout, row
from bokeh.models import (
    BasicTickFormatter,
    ColorBar,
    ColumnDataSource,
    DataTable,
    Div,
    HTMLTemplateFormatter,
    LinearColorMapper,
    TableColumn,
    UIElement,
)
from bokeh.palettes import Viridis256

# Bokeh for interactive plots
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge, transform
from loguru import logger
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from scipy import signal

# For distance metrics
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, wasserstein_distance

# Local or custom scaling utility
from synthefy_pkg.postprocessing.utils import downsample_by_random_sampling
from synthefy_pkg.utils.scaling_utils import transform_using_scaler

# Constants
DEFAULT_COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
DEFAULT_BIN_COUNT = 10
DEFAULT_FIGURE_HEIGHT = 400
DEFAULT_TABLE_HEIGHT = 300
DEFAULT_TABLE_WIDTH = 1000


# ---------------------------------------------------------
#  Utility Functions
# ---------------------------------------------------------
def freedman_diaconis_bins(data: np.ndarray) -> int:
    """Calculate optimal number of bins using Freedman-Diaconis Rule.

    Args:
        data: Input array of numerical values.

    Returns:
        int: Optimal number of bins.

    Note:
        Returns DEFAULT_BIN_COUNT if data length < 2 or IQR = 0.
    """
    if len(data) < 2:
        return DEFAULT_BIN_COUNT
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return DEFAULT_BIN_COUNT
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    num_bins = int((data.max() - data.min()) / bin_width)
    return max(num_bins, 2)


def compute_jsd_continuous(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Jensen-Shannon distance between two continuous distributions.

    Args:
        data1: First array of values
        data2: Second array of values

    Returns:
        float: Jensen-Shannon distance

    Raises:
        ValueError: If input arrays are empty
    """
    if len(data1) == 0 or len(data2) == 0:
        raise ValueError("Input arrays cannot be empty")

    combined = np.concatenate([data1, data2])
    num_bins = freedman_diaconis_bins(combined)
    bins = np.histogram_bin_edges(combined, bins=num_bins)

    p_counts, _ = np.histogram(data1, bins=bins)
    q_counts, _ = np.histogram(data2, bins=bins)

    p_sum = p_counts.sum()
    q_sum = q_counts.sum()
    p = p_counts / p_sum if p_sum > 0 else p_counts
    q = q_counts / q_sum if q_sum > 0 else q_counts

    return float(jensenshannon(p, q))


def compute_emd_continuous(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Computes the 1D Earth Mover's distance (Wasserstein distance) for two 1D continuous arrays.
    Returns 0 if either array is empty.
    """
    if len(data1) == 0 or len(data2) == 0:
        return 0.0
    return wasserstein_distance(data1, data2)


def compute_jsd_discrete(
    discrete_vals1: np.ndarray, discrete_vals2: np.ndarray
) -> float:
    """
    Computes the Jensen-Shannon distance for discrete (categorical) data.
    Converts category counts to probability vectors before computing JSD.
    """
    discrete_vals1 = discrete_vals1.flatten().astype(str)
    discrete_vals2 = discrete_vals2.flatten().astype(str)

    all_cats = np.union1d(discrete_vals1, discrete_vals2)

    def get_counts(vals: np.ndarray) -> dict[str, int]:
        cats, counts = np.unique(vals, return_counts=True)
        return dict(zip(cats, counts))

    counts1 = get_counts(discrete_vals1)
    counts2 = get_counts(discrete_vals2)

    p = np.array([counts1.get(cat, 0) for cat in all_cats], dtype=float)
    q = np.array([counts2.get(cat, 0) for cat in all_cats], dtype=float)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p /= p_sum
    if q_sum > 0:
        q /= q_sum

    return float(jensenshannon(p, q))


def safe_load_npy_file(npy_name: str, dataset_dir: str) -> np.ndarray:
    """
    Safely loads a .npy file if it exists, otherwise returns an empty array.
    """
    path = os.path.join(dataset_dir, npy_name)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    else:
        logger.warning(f"File not found: {path}. Skipping.")
        return np.array([]).reshape(0, 0, 0)


def check_and_log_thresholds(
    col_type: str,
    col_name: str,
    jsd_val: float | None,
    jsd_test: float | None,
    emd_val: float | None,
    emd_test: float | None,
    jsd_threshold: float,
    emd_threshold: float,
) -> None:
    """
    Logs warnings if JSD or EMD values exceed specified thresholds.
    """
    if jsd_val is not None and jsd_val > jsd_threshold:
        logger.warning(
            f"[{col_type.capitalize()} -> {col_name}] "
            f"JSD(Train vs. Val)={jsd_val:.2f} exceeds threshold {jsd_threshold:.2f}."
        )
    if jsd_test is not None and jsd_test > jsd_threshold:
        logger.warning(
            f"[{col_type.capitalize()} -> {col_name}] "
            f"JSD(Train vs. Test)={jsd_test:.2f} exceeds threshold {jsd_threshold:.2f}."
        )
    if emd_val is not None and emd_val > emd_threshold:
        logger.warning(
            f"[{col_type.capitalize()} -> {col_name}] "
            f"EMD(Train vs. Val)={emd_val:.2f} exceeds threshold {emd_threshold:.2f}."
        )
    if emd_test is not None and emd_test > emd_threshold:
        logger.warning(
            f"[{col_type.capitalize()} -> {col_name}] "
            f"EMD(Train vs. Test)={emd_test:.2f} exceeds threshold {emd_threshold:.2f}."
        )


# ---------------------------------------------------------
#  Bokeh Plotting Functions
# ---------------------------------------------------------
def create_histogram_bokeh(
    data_list: List[np.ndarray],
    legend_labels: List[str],
    title_text: str,
    colors: Optional[List[str]] = None,
) -> figure:
    """Create overlaid histogram for multiple datasets using Bokeh.

    Args:
        data_list: List of 1D numpy arrays
        legend_labels: Labels for each dataset
        title_text: Plot title and x-axis label
        colors: Optional color list for datasets

    Returns:
        Bokeh figure with overlaid histograms

    Raises:
        ValueError: If data_list and legend_labels lengths don't match
                   or if colors list is too short
    """
    if len(data_list) != len(legend_labels):
        raise ValueError("data_list and legend_labels must have same length")

    colors = colors or DEFAULT_COLORS[: len(data_list)]
    if len(colors) < len(data_list):
        raise ValueError("Number of colors less than number of datasets")

    # Combine all data to compute common histogram bin edges
    combined = np.concatenate(data_list)
    num_bins = freedman_diaconis_bins(combined)
    data_min, data_max = combined.min(), combined.max()

    # Adjust range if all data points are identical
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5
    hist_range = (data_min, data_max)

    # Compute bin edges using the combined data
    _, edges = np.histogram(
        combined, bins=num_bins, range=hist_range, density=True
    )
    left = edges[:-1]
    right = edges[1:]

    # Create Bokeh figure
    p = figure(
        height=DEFAULT_FIGURE_HEIGHT,
        title=f"{title_text} - Overlaid Histogram",
        x_axis_label=title_text,
        y_axis_label="Density",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        sizing_mode="stretch_width",
    )

    # Loop over each dataset, compute its histogram and add a quad glyph
    for dataset, label, color in zip(data_list, legend_labels, colors):
        hist, _ = np.histogram(dataset, bins=edges, density=True)
        p.quad(
            top=hist,
            bottom=0,
            left=left,
            right=right,
            fill_color=color,
            fill_alpha=0.3,
            line_color=color,
            legend_label=label,
        )

    p.legend.location = "top_right"
    return p


def create_kde_bokeh(
    data_list: List[np.ndarray],
    legend_labels: List[str],
    title_text: str,
    colors: Optional[List[str]] = None,
) -> figure:
    """
    Creates an overlaid KDE plot for multiple datasets using Bokeh and scipy's gaussian_kde.

    Parameters:
        data_list (List[np.ndarray]): A list of 1D numpy arrays for which KDEs will be computed.
        legend_labels (List[str]): A list of labels corresponding to each dataset.
        title_text (str): The title for the plot and label for the x-axis.
        colors (Optional[List[str]]): Optional list of colors for each dataset. If not provided,
            a default palette will be used.

    Returns:
        figure: A Bokeh figure object with the overlaid KDE plots.
    """
    if len(data_list) != len(legend_labels):
        raise ValueError(
            "data_list and legend_labels must have the same length"
        )

    # Use a default color palette if colors not provided.
    colors = colors or DEFAULT_COLORS[: len(data_list)]
    if len(colors) < len(data_list):
        raise ValueError(
            "Number of colors provided is less than the number of datasets"
        )

    # Combine all data to define common x-axis limits.
    combined = np.concatenate(data_list)
    data_min, data_max = combined.min(), combined.max()

    # Adjust range if all data points are identical.
    if data_min == data_max:
        data_min -= 0.5
        data_max += 0.5

    x_values = np.linspace(data_min, data_max, 200)

    def safe_gaussian_kde(arr: np.ndarray) -> np.ndarray:
        if len(arr) < 2 or np.isclose(np.std(arr), 0.0):
            return np.zeros_like(x_values)
        try:
            kde_func = gaussian_kde(arr)
            return kde_func(x_values)
        except np.linalg.LinAlgError:
            return np.zeros_like(x_values)

    # Compute KDE values for each dataset.
    kde_values_list = [safe_gaussian_kde(arr) for arr in data_list]

    # Create Bokeh figure.
    p = figure(
        height=DEFAULT_FIGURE_HEIGHT,
        title=f"{title_text} - KDE",
        x_axis_label=title_text,
        y_axis_label="Density",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        sizing_mode="stretch_width",
    )

    # Overlay each KDE line.
    for kde_vals, label, color in zip(kde_values_list, legend_labels, colors):
        p.line(
            x_values,
            kde_vals,
            line_color=color,
            line_width=2,
            legend_label=label,
        )

    p.legend.location = "top_right"
    return p


def create_bar_overlay_bokeh(
    data_list: List[np.ndarray],
    legend_labels: List[str],
    title_text: str,
    colors: Optional[List[str]] = None,
    bar_width: float = 0.2,
) -> figure:
    """
    Creates an overlaid bar chart for discrete/categorical data from multiple datasets.

    Parameters:
        data_list (List[np.ndarray]): A list of 1D numpy arrays, each representing categorical data.
        legend_labels (List[str]): A list of labels corresponding to each dataset.
        title_text (str): The title for the plot.
        colors (Optional[List[str]]): Optional list of colors for each dataset. If not provided,
            a default palette is used.
        bar_width (float): The width of each bar (default is 0.2).

    Returns:
        figure: A Bokeh figure object with the overlaid bar chart.
    """
    # Ensure matching counts for datasets and labels.
    if len(data_list) != len(legend_labels):
        raise ValueError(
            "data_list and legend_labels must have the same length"
        )

    # If no colors provided, use a default palette.
    colors = colors or DEFAULT_COLORS[: len(data_list)]
    if len(colors) < len(data_list):
        raise ValueError(
            "Number of colors provided is less than the number of datasets"
        )

    # Convert each dataset to string type.
    data_list = [data.astype(str) for data in data_list]

    # Compute all categories across all datasets.
    all_data = np.concatenate(data_list)
    categories = np.unique(all_data)

    def get_counts(values: np.ndarray) -> dict[str, int]:
        cats, counts = np.unique(values, return_counts=True)
        return dict(zip(cats, counts))

    # Build a data dictionary for the ColumnDataSource.
    data = {"category": categories}
    for i, dataset in enumerate(data_list):
        counts = get_counts(dataset)
        data[f"data_{i}"] = np.array([counts.get(cat, 0) for cat in categories])

    source = ColumnDataSource(data=data)

    p = figure(
        x_range=categories,
        height=DEFAULT_FIGURE_HEIGHT,
        title=f"{title_text} - Categorical Distribution",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        sizing_mode="stretch_width",
    )

    # Compute dodge offsets for each dataset.
    n = len(data_list)
    gap = bar_width * 0.25  # maintain same relative gap as in original function
    total_width = n * bar_width + (n - 1) * gap
    offsets = [
        -total_width / 2 + bar_width / 2 + i * (bar_width + gap)
        for i in range(n)
    ]

    # Add a vbar glyph for each dataset.
    for i, (label, color, offset) in enumerate(
        zip(legend_labels, colors, offsets)
    ):
        p.vbar(
            x=dodge("category", offset, range=p.x_range),
            top=f"data_{i}",
            width=bar_width,
            source=source,
            fill_color=color,
            fill_alpha=0.6,
            line_color=color,
            legend_label=label,
        )

    p.x_range.range_padding = 0.05
    p.xgrid.grid_line_color = None
    p.legend.location = "top_right"
    p.xaxis.major_label_orientation = 0.8
    p.y_range.start = 0
    p.xaxis.axis_label = "Category"
    p.yaxis.axis_label = "Count"

    return p


def create_distance_header(
    col_type: str,
    col_name: str,
    jsd_val: float | None,
    jsd_test: float | None,
    emd_val: float | None,
    emd_test: float | None,
) -> Div:
    """
    Creates a small HTML header summarizing distance metrics for a given column.
    """
    text_parts = [
        f"<b>Column Type:</b> {col_type}",
        f"<b>Column Name:</b> {col_name}",
        (
            f"<b>JSD(Train vs. Val):</b> {jsd_val:.2f} &nbsp;|&nbsp; "
            f"<b>JSD(Train vs. Test):</b> {jsd_test:.2f}"
        ),
    ]
    # EMD is only for continuous/time-series
    if emd_val is not None and emd_test is not None:
        text_parts.append(
            f"<b>EMD(Train vs. Val):</b> {emd_val:.2f} &nbsp;|&nbsp; "
            f"<b>EMD(Train vs. Test):</b> {emd_test:.2f}"
        )

    combined_text = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(text_parts)
    return Div(text=f"<h3 style='margin-bottom: 4px;'>{combined_text}</h3>")


# ---------------------------------------------------------
#  Covariance Matrix Functions
# ---------------------------------------------------------
def compute_covariance_matrix(
    data1: np.ndarray,
    data2: np.ndarray,
) -> np.ndarray:
    """
    Computes the covariance matrix between data1 channels and data2 channels
    using combined data from the provided datasets.

    Each array in data1 and data2 is assumed to have shape (N, W, C),
      - N: number of windows (or samples)
      - W: number of time steps
      - C: number of channels for that modality.

    The function concatenates all arrays in each list along the window axis (axis=0)
    and computes a covariance matrix of shape (C_data1, C_data2) where each element [i, j] is the
    covariance between the i-th data1 channel and the j-th data2 channel.

    Parameters:
        data1 (np.ndarray): List of time-series arrays.
        data2 (np.ndarray): List of continuous metadata arrays.

    Returns:
        np.ndarray: Covariance matrix with shape (number of data1 channels, number of data2 channels).
    """
    num_channels_1 = data1.shape[2]
    num_channels_2 = data2.shape[2]
    cov_matrix = np.zeros((num_channels_1, num_channels_2))

    for i in range(num_channels_1):
        data1_flat = data1[:, :, i].flatten()
        # Normalize data1
        data1_flat = (data1_flat - np.mean(data1_flat)) / (
            np.std(data1_flat) + 1e-8
        )
        for j in range(num_channels_2):
            data2_flat = data2[:, :, j].flatten()
            # Normalize data2
            data2_flat = (data2_flat - np.mean(data2_flat)) / (
                np.std(data2_flat) + 1e-8
            )
            # ddof=0 => population covariance
            cov = np.cov(data1_flat, data2_flat, ddof=0)[0, 1]
            cov_matrix[i, j] = cov
    return cov_matrix


def plot_covariance_matrix(
    cov_matrix: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    same_columns_only: bool = False,
) -> List[UIElement]:
    """
    Plots the covariance matrix as a heatmap using Bokeh.

    By default, the y-axis represents time-series channels (displayed in reversed order for top-down indexing)
    and the x-axis represents continuous metadata channels. You can override these labels by providing
    row_names and col_names.

    Args:
        cov_matrix: Covariance matrix to be plotted
        row_names: Names for the rows (e.g. data1 channel names)
        col_names: Names for the columns (e.g. data2 channel names)
        title: Title of the plot
        x_axis_label: Label for the x-axis
        y_axis_label: Label for the y-axis
        same_columns_only: If True, only show cells where row and column names match
    """
    # Prepare data for the heatmap
    data = []
    for i, row_name in enumerate(row_names):
        for j, col_name in enumerate(col_names):
            # Skip if same_columns_only is True and names don't match
            if same_columns_only and row_name != col_name:
                continue

            data.append(
                {"row": row_name, "col": col_name, "cov": cov_matrix[i, j]}
            )

    df = pd.DataFrame(data)
    source = ColumnDataSource(df)

    # Set up the color mapper using Viridis256
    min_cov = df["cov"].min()
    max_cov = df["cov"].max()
    color_mapper = LinearColorMapper(
        palette=Viridis256, low=min_cov, high=max_cov
    )

    # Create the figure with reversed row ordering for top-down indexing
    p = figure(
        title=title,
        x_range=col_names,
        y_range=list(reversed(row_names)),
        x_axis_location="above",
        height=600,
        sizing_mode="stretch_width",
        tools="hover,save,pan,box_zoom,reset",
        tooltips=[
            (y_axis_label, "@row"),
            (x_axis_label, "@col"),
            ("Covariance", "@cov{0.2f}"),
        ],
    )

    p.rect(
        x="col",
        y="row",
        width=1,
        height=1,
        source=source,
        fill_color=transform("cov", color_mapper),
        line_color=None,
    )

    layout_elements = []

    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    p.add_layout(color_bar, "right")

    # Append a title Div and the plot to the provided layout elements.
    layout_elements.append(Div(text=f"<h1>{title}</h1>"))
    layout_elements.append(p)
    logger.info("Bokeh plots for covariance matrix created successfully!")
    return layout_elements


def plot_covariance_matrix_static(
    cov_matrix: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    cmap: Union[str, Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_percentage: bool = False,
) -> Figure:
    """
    Plots the covariance matrix as a heatmap using Matplotlib.

    By default, the y-axis represents time-series channels (displayed in reversed order
    for top-down indexing), and the x-axis represents continuous metadata channels.
    You can override these labels by providing row_names and col_names.

    This version is "production-ready" with:
        - Basic input validation checks
        - Clear, high-resolution figure
        - Annotated cells for every value
        - Adaptive text color for improved contrast

    Args:
        cov_matrix (np.ndarray): 2D array (M x N) representing the covariance matrix to be plotted.
        row_names (List[str]): Names for the rows (e.g., data1 channel names).
        col_names (List[str]): Names for the columns (e.g., data2 channel names).
        title (str): Title of the plot.
        x_axis_label (str): Label for the x-axis.
        y_axis_label (str): Label for the y-axis.
        cmap (Union[str, Colormap]): Colormap to use. If "viridis", uses a custom colormap that shows values between [-0.25, 0.25] as white.
        vmin (Optional[float]): Minimum value for colormap scaling.
        vmax (Optional[float]): Maximum value for colormap scaling.
        show_percentage (bool): Whether to display values as percentages (e.g., for difference matrix).

    Returns:
        Figure: A non-interactive Matplotlib Figure containing the heatmap.
    """
    # -----------------------
    # Validate inputs
    # -----------------------
    if cov_matrix.ndim != 2:
        raise ValueError("cov_matrix must be a 2D array.")

    rows, cols = cov_matrix.shape
    if rows != len(row_names) or cols != len(col_names):
        raise ValueError(
            f"cov_matrix shape ({rows}, {cols}) must match "
            f"len(row_names)={len(row_names)} and len(col_names)={len(col_names)}."
        )

    # Flip the matrix vertically so that the top row in the plot corresponds
    # to the last element in the original row_names (mimicking a reversed order).
    matrix_flipped = np.flipud(cov_matrix)
    flipped_row_names = row_names[::-1]

    # Create the figure and axis with higher DPI for better quality
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(16, 10))
    fig: Any = fig
    ax: Any = ax

    # Create custom colormap if none specified
    if isinstance(cmap, str) and cmap == "viridis":
        # Create a custom colormap that shows values between [-0.25, 0.25] as white
        colors = [(0, 0, 0.5), (1, 1, 1), (0.5, 0, 0)]  # blue -> white -> red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        norm = plt.Normalize(vmin=-1, vmax=1)  # type: ignore
        vmin, vmax = -1, 1
    else:
        # Use the specified colormap and normalization
        norm = plt.Normalize(  # type: ignore
            vmin=vmin if vmin is not None else cov_matrix.min(),
            vmax=vmax if vmax is not None else cov_matrix.max(),
        )

    # Plot the heatmap
    im = ax.imshow(
        matrix_flipped,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        norm=norm,
    )

    # Add a colorbar with a label
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.set_ylabel(
        "Covariance", rotation=-90, va="bottom", fontweight="bold", labelpad=15
    )
    cbar.outline.set_linewidth(1)  # type: ignore

    # Set tick labels for x and y axes
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(flipped_row_names)))
    ax.set_yticklabels(flipped_row_names, fontsize=12)

    # Add gridlines to separate cells
    ax.set_xticks(np.arange(-0.5, len(col_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(flipped_row_names), 1), minor=True)
    ax.grid(
        which="minor", color="white", linestyle="-", linewidth=0.7, alpha=0.3
    )

    # Annotate each cell with its value, adapting text color for contrast
    for i in range(len(flipped_row_names)):
        for j in range(len(col_names)):
            val = matrix_flipped[i, j]
            # Get normalized RGBA color for the cell background
            rgba = im.cmap(norm(val))  # type: ignore
            # Simple luminance formula to decide black/white text
            # (there are more precise formulas, but this is sufficient)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if luminance > 0.5 else "white"

            # Format the value based on whether we want to show percentage
            value_text = f"{val:.2f}%" if show_percentage else f"{val:.2f}"

            ax.text(
                j,
                i,
                value_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
                fontweight="bold",
            )

    # Set axis labels and title with improved formatting
    ax.set_xlabel(x_axis_label, fontweight="bold", fontsize=14, labelpad=10)
    ax.set_ylabel(y_axis_label, fontweight="bold", fontsize=14, labelpad=10)
    ax.set_title(title, fontweight="bold", fontsize=20, pad=20)

    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    # Adjust layout to ensure everything fits without overlapping
    fig.tight_layout()

    return fig


# ---------------------------------------------------------
#  Cross-Correlation Functions
# ---------------------------------------------------------
def compute_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    normalize: bool = True,
    normalize_to_01: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Computes cross-correlation (via scipy.signal.correlate) between two 1D signals, returning:
      - corr: the full cross-correlation array
      - lags: the corresponding lag values
      - max_corr: the maximum correlation value
      - lag_at_max: the lag (shift) at which that max occurs
    If normalize=True, we scale by the product of the signal norms.
    If normalize_to_01=True, we additionally scale to [0, 1] range.

    For input shapes:
      - signal1.shape = (n,)
      - signal2.shape = (m,)

    Output shapes:
      - corr.shape = (n+m-1,)  # Full cross-correlation array
      - lags.shape = (n+m-1,)  # Corresponding lag values array
    """
    s1 = signal1.flatten() - np.mean(signal1)
    s2 = signal2.flatten() - np.mean(signal2)

    corr = signal.correlate(s1, s2, mode="full", method="fft")
    lags = np.arange(-len(s1) + 1, len(s1))

    if normalize:
        norm_factor = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
        if norm_factor != 0:
            corr = corr / norm_factor

    # Optionally normalize to [0, 1] range
    if normalize_to_01:
        corr = (corr + 1) / 2

    max_idx = np.argmax(np.abs(corr))
    max_corr = corr[max_idx]
    lag_at_max = lags[max_idx]
    return corr, lags, max_corr, lag_at_max


def plot_cross_correlation(
    corr: np.ndarray,
    lags: np.ndarray,
    max_corr: float,
    best_lag: int,
    fig_title: str,
    downsample_factor: int = 10,
) -> figure:
    """
    Plots the full cross-correlation curve for two 1D signals, highlighting the max value.

    Parameters:
        corr: The cross-correlation values array
        lags: The corresponding lag values array
        max_corr: The maximum correlation value to highlight
        best_lag: The lag at which the maximum correlation occurs
        fig_title: Title for the figure
        downsample_factor: Factor by which to downsample the data for plotting efficiency

    Returns:
        A Bokeh figure object with the cross-correlation plot

    Note:
        Downsampling is applied to improve rendering performance for large arrays.
        The maximum correlation point is highlighted with a red marker.
    """

    corr_downsampled, lags_downsampled = downsample_by_random_sampling(
        corr, lags, downsample_factor
    )

    p = figure(
        title=fig_title,
        x_axis_label="Lag",
        y_axis_label="Cross Correlation",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        sizing_mode="stretch_width",
        height=DEFAULT_FIGURE_HEIGHT,
        output_backend="svg",
    )

    p.xaxis.formatter = BasicTickFormatter(use_scientific=False)

    p.line(
        lags_downsampled,
        corr_downsampled,
        line_width=2,
        legend_label="Cross-Correlation",
    )
    p.scatter(
        [best_lag],
        [max_corr],
        size=10,
        color="red",
        legend_label=f"Max: {max_corr:.2f} at lag {best_lag}",
    )
    p.legend.location = "top_left"
    p.hover.tooltips = [("Lag", "@x{0,0}"), ("Correlation", "@y{0.2f}")]

    return p


def compute_max_cross_correlation_matrix(
    data1: np.ndarray,
    data2: np.ndarray,
    data1_col_names: list[str],
    data2_col_names: list[str],
    pairwise_corr_figures: bool = False,
    downsample_factor: int = 10,
    same_columns_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[figure]]:
    """
    For each (data1 column, data2 column) pair, computes:
      - max_corr_matrix[i, j]: the maximum cross-correlation
      - lag_matrix[i, j]     : the lag at which it occurs
      - also returns a list of cross-correlation figures (one per pair).

    Args:
        data1: First dataset array with shape (N, W, C1)
        data2: Second dataset array with shape (N, W, C2)
        data1_col_names: Column names for first dataset
        data2_col_names: Column names for second dataset
        pairwise_corr_figures: Whether to generate individual correlation figures
        downsample_factor: Factor for downsampling large arrays
        same_columns_only: If True, only compute correlations between columns with same names.
                          This is useful for real-vs-predicted analysis where we only want
                          to compare corresponding columns.

    Returns:
        tuple: (max_corr_matrix, lag_matrix, figures)
    """
    num_data1_channels = len(data1_col_names)
    num_data2_channels = len(data2_col_names)

    max_corr_matrix = np.zeros((num_data1_channels, num_data2_channels))
    lag_matrix = np.zeros((num_data1_channels, num_data2_channels), dtype=int)
    figures = []

    for i in range(num_data1_channels):
        data1_flat = data1[:, :, i].flatten()
        for j in range(num_data2_channels):
            # Skip if same_columns_only is True and column names don't match
            if same_columns_only and data1_col_names[i] != data2_col_names[j]:
                continue

            data2_flat = data2[:, :, j].flatten()
            corr, lags, max_corr, best_lag = compute_cross_correlation(
                data1_flat, data2_flat, normalize=True
            )
            max_corr_matrix[i, j] = max_corr
            lag_matrix[i, j] = best_lag

            # Optional: create a figure for each pair
            if pairwise_corr_figures:
                fig_title = f"Cross-Correlation ({data1_col_names[i]}, {data2_col_names[j]})"
                cross_corr_fig = plot_cross_correlation(
                    corr, lags, max_corr, best_lag, fig_title, downsample_factor
                )
                figures.append(cross_corr_fig)
            logger.info(
                f"Bokeh plots for {data1_col_names[i]} vs {data2_col_names[j]} created successfully!"
            )

    return max_corr_matrix, lag_matrix, figures


def create_cross_corr_dataframe(
    max_corr_matrix: np.ndarray,
    lag_matrix: np.ndarray,
    data1_col_names: list[str],
    data2_col_names: list[str],
    same_columns_only: bool = False,
) -> pd.DataFrame:
    """
    Builds a DataFrame of [col1, col2, max_corr, lag].

    Args:
        max_corr_matrix: Matrix of maximum correlation values
        lag_matrix: Matrix of lags at maximum correlation
        col1_names: Column names for first dataset
        col2_names: Column names for second dataset
        same_columns_only: If True, only include rows where column names match
    """
    records = []
    for i, col1_name in enumerate(data1_col_names):
        for j, col2_name in enumerate(data2_col_names):
            # Skip if same_columns_only is True and column names don't match
            if same_columns_only and col1_name != col2_name:
                continue

            records.append(
                {
                    "col1": col1_name,
                    "col2": col2_name,
                    "max_corr": max_corr_matrix[i, j],
                    "lag": lag_matrix[i, j],
                }
            )
    return pd.DataFrame(records)


def create_cross_corr_datatable(
    df: pd.DataFrame, col1_name: str, col2_name: str
) -> DataTable:
    """
    Returns a Bokeh DataTable for cross-correlation results.
    """
    source = ColumnDataSource(df)
    columns = [
        TableColumn(field="col1", title=col1_name),
        TableColumn(field="col2", title=col2_name),
        TableColumn(field="max_corr", title="Max Correlation"),
        TableColumn(field="lag", title="Lag"),
    ]
    return DataTable(
        source=source,
        columns=columns,
        width=DEFAULT_TABLE_WIDTH,
        height=DEFAULT_TABLE_HEIGHT,
    )


def plot_cross_corr_heatmap(
    max_corr_matrix: np.ndarray,
    data1_col_names: list[str],
    data2_col_names: list[str],
    title: str = "Max Cross-Correlation Heatmap",
    same_columns_only: bool = False,
) -> figure:
    """
    Creates a heatmap to visualize the maximum cross-correlation values across data1 vs. data2 columns.
    """
    data = []
    for i, data1_name in enumerate(data1_col_names):
        for j, data2_name in enumerate(data2_col_names):
            # Skip if same_columns_only is True and names don't match
            if same_columns_only and data1_name != data2_name:
                continue

            data.append(
                {
                    "data1_col": data1_name,
                    "data2_col": data2_name,
                    "max_corr": max_corr_matrix[i, j],
                }
            )
    df = pd.DataFrame(data)
    source = ColumnDataSource(df)

    min_val = df["max_corr"].min()
    max_val = df["max_corr"].max()
    color_mapper = LinearColorMapper(
        palette=Viridis256, low=min_val, high=max_val
    )

    p = figure(
        title=title,
        x_range=data2_col_names,
        y_range=list(reversed(data1_col_names)),
        x_axis_location="above",
        height=600,
        sizing_mode="stretch_width",
        tools="hover,save,pan,box_zoom,reset",
        tooltips=[
            ("data1", "@data1_col"),
            ("data2", "@data2_col"),
            ("MaxCorr", "@max_corr{0.2f}"),
        ],
    )

    p.rect(
        x="data2_col",
        y="data1_col",
        width=1,
        height=1,
        source=source,
        fill_color=transform("max_corr", color_mapper),
        line_color=None,
    )

    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    p.add_layout(color_bar, "right")
    return p


def create_cross_corr_elements(
    data1: np.ndarray,
    data2: np.ndarray,
    data1_col_names: list[str],
    data2_col_names: list[str],
    pairwise_corr_figures: bool = False,
    downsample_factor: int = 10,
    csv_path: str | None = None,
    same_columns_only: bool = False,
    col1_name: str = "TS Column",
    col2_name: str = "CMD Column",
    title: str = "Max Cross-Correlation Analysis",
) -> list[UIElement]:
    """Creates cross-correlation analysis elements including table, heatmap, and optional figures.

    This function can be used in two ways:
    1. Feature-to-feature analysis: When data1 and data2 are different feature sets
       (e.g., time-series vs continuous metadata), and data1_col_names and data2_col_names
       are different. This helps understand relationships between different types of features.
    2. Real-vs-predicted analysis: When data1 and data2 are real and predicted data
       for the same features, and data1_col_names and data2_col_names are identical.
       This helps understand how well the model captures temporal relationships.

    Args:
        data1: First dataset array with shape (N, W, C1)
        data2: Second dataset array with shape (N, W, C2)
        data1_col_names: Column names for first dataset
        data2_col_names: Column names for second dataset
        pairwise_corr_figures: Whether to generate individual correlation figures
        downsample_factor: Factor for downsampling large arrays
        csv_path: Optional path to save CSV output
        same_columns_only: If True, only compute correlations between columns with same names.
                          This is useful for real-vs-predicted analysis where we only want
                          to compare corresponding columns.
        col1_name: Name of column for first dataset
        col2_name: Name of column for second dataset
        title: Title of the analysis
    Returns:
        list[UIElement]: List of Bokeh elements (tables, plots, etc.)
    """
    layout_elements = []

    # Compute cross-correlation matrices
    max_corr_matrix, lag_matrix, figures = compute_max_cross_correlation_matrix(
        data1,
        data2,
        data1_col_names,
        data2_col_names,
        pairwise_corr_figures=pairwise_corr_figures,
        downsample_factor=downsample_factor,
        same_columns_only=same_columns_only,
    )

    # Create cross-correlation dataframe and table
    df_cc = create_cross_corr_dataframe(
        max_corr_matrix,
        lag_matrix,
        data1_col_names,
        data2_col_names,
        same_columns_only=same_columns_only,
    )
    cc_table = create_cross_corr_datatable(df_cc, col1_name, col2_name)

    # Create heatmap visualization
    cc_heatmap = plot_cross_corr_heatmap(
        max_corr_matrix,
        data1_col_names,
        data2_col_names,
        title=title,
        same_columns_only=same_columns_only,
    )

    # Add elements to layout
    layout_elements.append(Div(text=f"<h1>{title}</h1>"))
    layout_elements.append(cc_table)
    layout_elements.append(cc_heatmap)

    # Optionally save to CSV if output directory and dataset name provided
    if csv_path:
        df_cc.to_csv(csv_path, index=False)
        logger.info(f"Cross-correlation summary saved to: {csv_path}")

    # Optionally add pairwise correlation figures
    if pairwise_corr_figures and figures:
        layout_elements.append(
            Div(
                text="<h2>Pairwise Cross-Correlation Figures</h2>",
                sizing_mode="stretch_width",
            )
        )
        # Add figures two per row
        for i in range(0, len(figures), 2):
            if i + 1 < len(figures):
                layout_elements.append(
                    row(list(figures[i : i + 2]), sizing_mode="stretch_width")
                )
            else:
                layout_elements.append(figures[i])

    return layout_elements


# ---------------------------------------------------------
#  Main Analysis Functions
# ---------------------------------------------------------
def process_continuous_columns(
    data_list: list[np.ndarray],
    col_names: list[str],
    col_type: str,
    results_records: list[dict[str, Any]],
    jsd_threshold: float,
    emd_threshold: float,
    data_labels: list[str],
) -> list[UIElement]:
    """
    Plots distributions (histogram + KDE) and computes JSD & EMD
    for time-series or continuous columns across multiple datasets.

    Parameters:
        data_list (list[np.ndarray]): A list of 3D numpy arrays (shape: samples x timepoints x columns)
            where each array corresponds to one dataset.
        col_names (list[str]): A list of column names (one per column in the third axis).
        col_type (str): Type of column (e.g., "continuous", "time-series") used for display.
        results_records (list[dict[str, Any]]): List to which results (distance metrics) will be recorded.
        jsd_threshold (float): Threshold value for Jensen-Shannon Divergence.
        emd_threshold (float): Threshold value for Earth Mover's Distance.
        data_labels (list[str]): List of legend labels for each dataset in data_list; the first dataset
            is used as the baseline for distance comparisons.
    """
    layout_elements = []

    # Add a header to the layout.
    header_title = f"<h1>{col_type.capitalize()} Distributions</h1>"
    layout_elements.append(Div(text=header_title, sizing_mode="stretch_width"))

    # Loop over each column.
    for idx, col_name in enumerate(col_names):
        # For the current column, extract & flatten data from each dataset.
        flattened_datasets = [data[:, :, idx].flatten() for data in data_list]

        # Create histogram and KDE plots using generalized plotting functions.
        # (Assumes create_overlaid_histogram_bokeh and create_kde_bokeh have been updated similarly.)
        hist_fig = create_histogram_bokeh(
            data_list=flattened_datasets,
            legend_labels=data_labels,
            title_text=f"[{col_type.capitalize()}] {col_name}",
        )
        kde_fig = create_kde_bokeh(
            data_list=flattened_datasets,
            legend_labels=data_labels,
            title_text=f"[{col_type.capitalize()}] {col_name}",
        )

        # Use the first dataset as the baseline.
        baseline = flattened_datasets[0]

        # Compute distance metrics for each additional dataset.
        for label, flat in zip(data_labels[1:], flattened_datasets[1:]):
            jsd_value = compute_jsd_continuous(baseline, flat)
            emd_value = compute_emd_continuous(baseline, flat)
            # Log and check thresholds for each comparison.
            check_and_log_thresholds(
                col_type,
                f"{col_name} vs {label}",
                jsd_value,
                jsd_value,
                emd_value,
                emd_value,
                jsd_threshold,
                emd_threshold,
            )

        # Record the results for this column.
        results_records.append(
            {
                "type": col_type,
                "column": col_name,
                "jsd_val": jsd_value,
                "jsd_test": jsd_value,
                "emd_val": emd_value,
                "emd_test": emd_value,
            }
        )

        # Create a header summarizing the distance metrics.
        # (Assumes create_distance_header has been generalized to accept the baseline label and dictionaries.)
        distance_hdr = create_distance_header(
            col_type,
            col_name,
            jsd_value,
            jsd_value,
            emd_value,
            emd_value,
        )
        layout_elements.append(distance_hdr)
        layout_elements.append(
            row([hist_fig, kde_fig], sizing_mode="stretch_width")
        )
        logger.info(f"Bokeh plots for {col_name} created successfully!")

    return layout_elements


def process_discrete_columns(
    data_list: list[np.ndarray],
    col_names: list[str],
    results_records: list[dict[str, Any]],
    jsd_threshold: float,
    legend_labels: list[str] | None = None,
) -> list[UIElement]:
    """
    Plots bar charts for discrete columns and computes JSD (reference dataset vs. others).

    For each discrete column, a bar chart is generated that overlays the distributions
    from each dataset provided in data_list. The first dataset is considered the reference (e.g. Train)
    and the JSD is computed between it and each of the remaining datasets. For the common case
    of three datasets, this function mimics the original behavior by computing and recording
    "jsd_val" (Train vs. Val) and "jsd_test" (Train vs. Test).

    Parameters:
        data_list (List[np.ndarray]): List of 3D numpy arrays (shape: N, W, num_columns) for each dataset.
        col_names (List[str]): List of column names.
        results_records (List[dict[str, Any]]): List to which the computed result records are appended.
        jsd_threshold (float): Threshold used for JSD checking.
        legend_labels (Optional[List[str]]): Labels for each dataset. If not provided, defaults to:
            - ["Train", "Val", "Test"] when len(data_list)==3,
            - Otherwise, generic names ("Dataset 1", "Dataset 2", ...).
    """
    if legend_labels is None:
        if len(data_list) == 3:
            legend_labels = ["Train", "Val", "Test"]
        else:
            legend_labels = [f"Dataset {i + 1}" for i in range(len(data_list))]
    if len(data_list) != len(legend_labels):
        raise ValueError(
            "data_list and legend_labels must have the same length"
        )

    layout_elements = []

    header_title = "<h1>Original Discrete Distributions</h1>"
    layout_elements.append(Div(text=header_title, sizing_mode="stretch_width"))

    # Process each discrete column
    for idx, col_name in enumerate(col_names):
        # For the given column, flatten each dataset's values
        flattened_vals = [d[:, :, idx].flatten() for d in data_list]

        # Create an overlaid bar chart for all datasets using the generalized function
        bar_fig = create_bar_overlay_bokeh(
            data_list=flattened_vals,
            legend_labels=legend_labels,
            title_text=f"[Discrete] {col_name}",
        )

        baseline_vals = flattened_vals[0]

        # If exactly three datasets, mimic original functionality (Train vs. Val/Test)
        if len(data_list) == 3:
            jsd_val = compute_jsd_discrete(baseline_vals, flattened_vals[1])
            jsd_test = compute_jsd_discrete(baseline_vals, flattened_vals[2])

            check_and_log_thresholds(
                col_type="discrete",
                col_name=col_name,
                jsd_val=jsd_val,
                jsd_test=jsd_test,
                emd_val=None,
                emd_test=None,
                jsd_threshold=jsd_threshold,
                emd_threshold=float("inf"),
            )

            results_records.append(
                {
                    "type": "discrete",
                    "column": col_name,
                    "jsd_val": jsd_val,
                    "jsd_test": jsd_test,
                    "emd_val": None,
                    "emd_test": None,
                }
            )

            distance_hdr = create_distance_header(
                col_type="discrete",
                col_name=col_name,
                jsd_val=jsd_val,
                jsd_test=jsd_test,
                emd_val=None,
                emd_test=None,
            )
            layout_elements.append(distance_hdr)
        else:
            # For more than three datasets, compute JSD for each non-baseline dataset
            jsd_results = {}
            for i, label in enumerate(legend_labels[1:], start=1):
                jsd = compute_jsd_discrete(baseline_vals, flattened_vals[i])
                jsd_results[label] = jsd

                check_and_log_thresholds(
                    col_type="discrete",
                    col_name=col_name,
                    jsd_val=jsd,
                    jsd_test=None,
                    emd_val=None,
                    emd_test=None,
                    jsd_threshold=jsd_threshold,
                    emd_threshold=float("inf"),
                )

                distance_hdr = create_distance_header(
                    col_type="discrete",
                    col_name=col_name,
                    jsd_val=jsd,
                    jsd_test=None,
                    emd_val=None,
                    emd_test=None,
                )
                layout_elements.append(distance_hdr)

            results_records.append(
                {
                    "type": "discrete",
                    "column": col_name,
                    "jsd": jsd_results,
                    "emd": None,
                }
            )

        layout_elements.append(bar_fig)
        logger.info(f"Bokeh plots for {col_name} created successfully!")

    return layout_elements


# -------------------------------------------------------
#  Create Bokeh DataTable of JSD/EMD
# -------------------------------------------------------
def create_metrics_table(
    results_records: list,
    jsd_threshold: float,
    emd_threshold: float,
    out_csv_path: str,
) -> list[UIElement]:
    """
    Create a Bokeh DataTable with JSD and EMD metrics.

    Args:
        results_records: List of dictionaries containing metrics data
        jsd_threshold: Threshold for JSD values highlighting
        emd_threshold: Threshold for EMD values highlighting
        out_csv_path: Path to save the CSV file with metrics

    Returns:
        tuple: (DataTable, threshold_text Div, table_header Div)
    """
    # Write JSD/EMD metrics to CSV
    df_results = pd.DataFrame(results_records)
    df_results.to_csv(out_csv_path, index=False)
    logger.info(f"Distance metrics CSV saved to: {out_csv_path}")

    # Create Bokeh DataTable
    cds = ColumnDataSource(df_results.fillna(value=np.nan))

    threshold_text = Div(
        text=f"<h3>Thresholds: JSD > {jsd_threshold:.2f}, EMD > {emd_threshold:.2f}</h3>",
        sizing_mode="stretch_width",
    )
    table_header = Div(
        text="<h1>Summary of JSD & EMD Distances</h1>",
        sizing_mode="stretch_width",
    )

    jsd_template = f"""
    <% if (value > {jsd_threshold}) {{ %>
        <div style="font-weight: bold; color: red;"><%= value.toFixed(2) %></div>
    <% }} else {{ %>
        <%= value.toFixed(2) %>
    <% }} %>
    """

    emd_template = f"""
    <% if (value > {emd_threshold}) {{ %>
        <div style="font-weight: bold; color: red;"><%= value.toFixed(2) %></div>
    <% }} else {{ %>
        <%= value.toFixed(2) %>
    <% }} %>
    """

    table_columns = [
        TableColumn(field="type", title="Type"),
        TableColumn(field="column", title="Column"),
        TableColumn(
            field="jsd_val",
            title="JSD(Train vs. Val)",
            formatter=HTMLTemplateFormatter(template=jsd_template),
        ),
        TableColumn(
            field="jsd_test",
            title="JSD(Train vs. Test)",
            formatter=HTMLTemplateFormatter(template=jsd_template),
        ),
        TableColumn(
            field="emd_val",
            title="EMD(Train vs. Val)",
            formatter=HTMLTemplateFormatter(template=emd_template),
        ),
        TableColumn(
            field="emd_test",
            title="EMD(Train vs. Test)",
            formatter=HTMLTemplateFormatter(template=emd_template),
        ),
    ]

    results_table = DataTable(
        source=cds,
        columns=table_columns,
        sizing_mode="stretch_width",
        sortable=True,
        selectable=True,
        width=DEFAULT_TABLE_WIDTH,
        height=DEFAULT_TABLE_HEIGHT,
    )

    return [table_header, threshold_text, results_table]


def post_preprocess_analysis(
    dataset_dir: str,
    jsd_threshold: float,
    emd_threshold: float,
    use_scaled_data: bool,
    output_dir: str,
    pairwise_corr_figures: bool = False,
    downsample_factor: int = 10,
) -> Tuple[str, str]:
    """Perform comprehensive analysis of preprocessed data splits.

    Args:
        dataset_dir: Directory containing dataset files
        jsd_threshold: Threshold for Jensen-Shannon Divergence warnings
        emd_threshold: Threshold for Earth Mover's Distance warnings
        use_scaled_data: Whether to use scaled or unscaled data
        output_dir: Directory for output files
        pairwise_corr_figures: Whether to generate pairwise correlation plots
        downsample_factor: Factor for downsampling large arrays

    Returns:
        Tuple[str, str]: Paths to generated HTML and CSV files

    Raises:
        FileNotFoundError: If required dataset files are missing
        ValueError: If data arrays have incompatible shapes
    """
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))

    out_plots_path = os.path.join(
        output_dir if output_dir else dataset_dir,
        f"{dataset_name}_post_preprocessing_{'scaled' if use_scaled_data else 'unscaled'}.html",
    )
    out_csv_path = os.path.join(
        output_dir if output_dir else dataset_dir,
        f"{dataset_name}_post_preprocessing_distances_{'scaled' if use_scaled_data else 'unscaled'}.csv",
    )

    results_records = []

    layout_elements: list[UIElement] = []

    # -------------------------------------------------------
    #  Load column names
    # -------------------------------------------------------
    timeseries_cols_path = os.path.join(
        dataset_dir, "timeseries_windows_columns.json"
    )
    continuous_cols_path = os.path.join(
        dataset_dir, "continuous_windows_columns.json"
    )
    colnames_json_path = os.path.join(dataset_dir, "colnames.json")

    timeseries_cols = []
    if os.path.exists(timeseries_cols_path):
        with open(timeseries_cols_path, "r") as f:
            timeseries_cols = json.load(f)

    continuous_cols = []
    if os.path.exists(continuous_cols_path):
        with open(continuous_cols_path, "r") as f:
            continuous_cols = json.load(f)

    original_discrete_colnames = []
    if os.path.exists(colnames_json_path):
        with open(colnames_json_path, "r") as f:
            cols_dict = json.load(f)
            original_discrete_colnames = cols_dict.get(
                "original_discrete_colnames", []
            )

    # -------------------------------------------------------
    #  Load train/val/test arrays
    # -------------------------------------------------------
    train_ts = safe_load_npy_file("train_timeseries.npy", dataset_dir)
    val_ts = safe_load_npy_file("val_timeseries.npy", dataset_dir)
    test_ts = safe_load_npy_file("test_timeseries.npy", dataset_dir)

    train_cont = safe_load_npy_file(
        "train_continuous_conditions.npy", dataset_dir
    )
    val_cont = safe_load_npy_file("val_continuous_conditions.npy", dataset_dir)
    test_cont = safe_load_npy_file(
        "test_continuous_conditions.npy", dataset_dir
    )

    train_orig_disc = safe_load_npy_file(
        "train_original_discrete_windows.npy", dataset_dir
    )
    val_orig_disc = safe_load_npy_file(
        "val_original_discrete_windows.npy", dataset_dir
    )
    test_orig_disc = safe_load_npy_file(
        "test_original_discrete_windows.npy", dataset_dir
    )

    # -------------------------------------------------------
    #  Optionally un-scale
    # -------------------------------------------------------
    if not use_scaled_data:
        if len(train_ts) > 0:
            train_ts = transform_using_scaler(
                windows=train_ts,
                timeseries_or_continuous="timeseries",
                dataset_name=dataset_name,
                original_discrete_windows=train_orig_disc,
                inverse_transform=True,
            )
        if len(val_ts) > 0:
            val_ts = transform_using_scaler(
                windows=val_ts,
                timeseries_or_continuous="timeseries",
                dataset_name=dataset_name,
                original_discrete_windows=val_orig_disc,
                inverse_transform=True,
            )
        if len(test_ts) > 0:
            test_ts = transform_using_scaler(
                windows=test_ts,
                timeseries_or_continuous="timeseries",
                dataset_name=dataset_name,
                original_discrete_windows=test_orig_disc,
                inverse_transform=True,
            )

        if len(train_cont) > 0:
            train_cont = transform_using_scaler(
                windows=train_cont,
                timeseries_or_continuous="continuous",
                dataset_name=dataset_name,
                original_discrete_windows=train_orig_disc,
                inverse_transform=True,
            )
        if len(val_cont) > 0:
            val_cont = transform_using_scaler(
                windows=val_cont,
                timeseries_or_continuous="continuous",
                dataset_name=dataset_name,
                original_discrete_windows=val_orig_disc,
                inverse_transform=True,
            )
        if len(test_cont) > 0:
            test_cont = transform_using_scaler(
                windows=test_cont,
                timeseries_or_continuous="continuous",
                dataset_name=dataset_name,
                original_discrete_windows=test_orig_disc,
                inverse_transform=True,
            )

    logger.info("All data were loaded successfully!")

    # -------------------------------------------------------
    #  Expand discrete arrays if shape is 2D => (N, 1, C)
    # -------------------------------------------------------
    if train_orig_disc.ndim == 2 and len(train_orig_disc) > 0:
        train_orig_disc = np.expand_dims(train_orig_disc, axis=1)
    if val_orig_disc.ndim == 2 and len(val_orig_disc) > 0:
        val_orig_disc = np.expand_dims(val_orig_disc, axis=1)
    if test_orig_disc.ndim == 2 and len(test_orig_disc) > 0:
        test_orig_disc = np.expand_dims(test_orig_disc, axis=1)

    # -------------------------------------------------------
    #  Rearrange TS shape to (N, W, C)
    # -------------------------------------------------------
    if len(train_ts) > 0:
        train_ts = einops.rearrange(train_ts, "b c t -> b t c")
    if len(val_ts) > 0:
        val_ts = einops.rearrange(val_ts, "b c t -> b t c")
    if len(test_ts) > 0:
        test_ts = einops.rearrange(test_ts, "b c t -> b t c")

    # -------------------------------------------------------
    #  Process Time-Series columns
    # -------------------------------------------------------
    if (
        len(train_ts) > 0
        and len(val_ts) > 0
        and len(test_ts) > 0
        and timeseries_cols
    ):
        ts_elements = process_continuous_columns(
            data_list=[train_ts, val_ts, test_ts],
            col_names=timeseries_cols,
            data_labels=["Train", "Val", "Test"],
            col_type="timeseries",
            results_records=results_records,
            jsd_threshold=jsd_threshold,
            emd_threshold=emd_threshold,
        )
        logger.info("Bokeh plots for timeseries columns created successfully!")
        layout_elements.extend(ts_elements)
    else:
        logger.warning(
            "Time-series data or columns not found; skipping time-series plots."
        )

    # -------------------------------------------------------
    #  Process Continuous columns
    # -------------------------------------------------------
    if (
        len(train_cont) > 0
        and len(val_cont) > 0
        and len(test_cont) > 0
        and continuous_cols
    ):
        continuous_elements = process_continuous_columns(
            data_list=[train_cont, val_cont, test_cont],
            col_names=continuous_cols,
            data_labels=["Train", "Val", "Test"],
            col_type="continuous",
            results_records=results_records,
            jsd_threshold=jsd_threshold,
            emd_threshold=emd_threshold,
        )
        logger.info("Bokeh plots for continuous columns created successfully!")
        layout_elements.extend(continuous_elements)
    else:
        logger.warning(
            "Continuous data or columns not found; skipping continuous plots."
        )

    # -------------------------------------------------------
    #  Process Discrete columns
    # -------------------------------------------------------
    if (
        len(train_orig_disc) > 0
        and len(val_orig_disc) > 0
        and len(test_orig_disc) > 0
        and original_discrete_colnames
    ):
        discrete_elements = process_discrete_columns(
            data_list=[train_orig_disc, val_orig_disc, test_orig_disc],
            col_names=original_discrete_colnames,
            results_records=results_records,
            jsd_threshold=jsd_threshold,
            legend_labels=["Train", "Val", "Test"],
        )
        logger.info(
            "Bokeh plots for original discrete columns created successfully!"
        )
        layout_elements.extend(discrete_elements)
    else:
        logger.warning(
            "Original discrete data not found; skipping discrete plots."
        )

    table_elements = create_metrics_table(
        results_records, jsd_threshold, emd_threshold, out_csv_path
    )
    layout_elements.extend(table_elements)

    # -------------------------------------------------------
    #  Covariance Matrix (TS vs. Continuous)
    # -------------------------------------------------------
    # Combine train/val/test => shape (N_total, W, C)
    ts_combined = np.concatenate([train_ts, val_ts, test_ts], axis=0)
    cont_combined = np.concatenate([train_cont, val_cont, test_cont], axis=0)
    cov_matrix = compute_covariance_matrix(ts_combined, cont_combined)
    cov_elements = plot_covariance_matrix(
        cov_matrix,
        timeseries_cols,
        continuous_cols,
        title="Covariance Matrix: Time Series Columns vs. Continuous Columns",
        x_axis_label="TS Columns",
        y_axis_label="CMD Columns",
    )
    layout_elements.extend(cov_elements)

    # -------------------------------------------------------
    #  Cross-Correlation (TS vs. Continuous)
    # -------------------------------------------------------
    if (
        ts_combined.size > 0
        and cont_combined.size > 0
        and timeseries_cols
        and continuous_cols
    ):
        cc_elements = create_cross_corr_elements(
            data1=ts_combined,
            data2=cont_combined,
            data1_col_names=timeseries_cols,
            data2_col_names=continuous_cols,
            pairwise_corr_figures=pairwise_corr_figures,
            downsample_factor=downsample_factor,
            csv_path=os.path.join(
                output_dir if output_dir else dataset_dir,
                f"{dataset_name}_cross_correlation_summary.csv",
            ),
        )
        layout_elements.extend(cc_elements)

    # -------------------------------------------------------
    #  Save Bokeh Layout
    # -------------------------------------------------------
    output_file(out_plots_path, title=f"{dataset_name} - Split Analysis")
    save(
        layout(children=layout_elements, sizing_mode="stretch_width", margin=20)
    )

    logger.info(f"Bokeh analysis HTML saved to: {out_plots_path}")
    logger.info(f"CSV file saved to: {out_csv_path}")

    return out_plots_path, out_csv_path
