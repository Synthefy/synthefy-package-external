import argparse
import glob
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _load_data(results_filepath: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load data from file path or use provided dictionary.

    Args:
        results_filepath: Path to JSON file or dictionary

    Returns:
        Dictionary containing the data
    """
    if isinstance(results_filepath, str):
        with open(results_filepath, "r") as f:
            return json.load(f)
    else:
        return results_filepath


def _group_data_by_column(
    data_list: List[Dict[str, Any]], column_key: str = "Column"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group data by column name from a list of dictionaries.

    Args:
        data_list: List of dictionaries containing data
        column_key: Key used to identify the column

    Returns:
        Dictionary mapping column names to lists of their data items
    """
    grouped_data = {}
    for item in data_list:
        col_name = item.get(column_key)
        if col_name:
            if col_name not in grouped_data:
                grouped_data[col_name] = []
            grouped_data[col_name].append(item)
    return grouped_data


def _process_column_features(
    column_data: Dict[str, List[Dict[str, Any]]],
    exclude_keys: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Process grouped column data to extract numeric features.

    Args:
        column_data: Dictionary mapping column names to lists of data items
        exclude_keys: Keys to exclude from feature extraction

    Returns:
        Dictionary mapping column names to numpy arrays of features
    """
    if exclude_keys is None:
        exclude_keys = []

    column_features = {}

    for col_name, data_items in column_data.items():
        features = []

        for item in data_items:
            item_features = _extract_numeric_features(
                item, exclude_keys=exclude_keys
            )
            features.extend(item_features)

        if features:
            column_features[col_name] = np.array(features, dtype=float)

    return column_features


def _create_aggregated_vectors(
    column_features: Dict[str, np.ndarray], output_dimension: int
) -> Dict[str, np.ndarray]:
    """
    Create aggregated vectors from column features using dimensionality reduction.

    Args:
        column_features: Dictionary mapping column names to feature arrays
        output_dimension: Target dimension for aggregated vectors

    Returns:
        Dictionary mapping column names to aggregated n-dimensional vectors
    """
    if len(column_features) == 0:
        return {}
    column_matrix = np.concatenate(
        [column_features[col_name] for col_name in column_features], axis=0
    )  # number of files x number of features
    aggregated_vector = _apply_dimensionality_reduction(
        column_matrix, output_dimension
    )
    aggregated_vectors = {
        col_name: aggregated_vector[i]
        for i, col_name in enumerate(column_features)
    }
    return aggregated_vectors


def _create_aggregated_vectors_statistics(
    column_features: OrderedDict[str, List[np.ndarray]],
    weights: Optional[Dict[str, float]] = None,
    statistic_type: str = "mean",  # options: mean, max, std
    combine_type: str = "sum",  # options: mean, concat
) -> Dict[str, Union[float, np.ndarray]] | float:
    """
    Create aggregated vectors from column features using statistics.
    """
    aggregated_vectors = {}
    aggregated_value = 0.0
    for col_name, features in column_features.items():
        if statistic_type == "mean":
            feature_value = float(np.mean(features))
        elif statistic_type == "max":
            feature_value = float(np.max(features))
        elif statistic_type == "std":
            feature_value = float(np.std(features))

        if combine_type == "concatenate":
            aggregated_vectors[col_name] = feature_value
        elif weights is not None:
            aggregated_value += feature_value * weights.get(col_name, 1.0)
        else:
            aggregated_value += feature_value

    return (
        aggregated_vectors
        if combine_type == "concatenate"
        else aggregated_value
    )


def _create_aggregated_vectors_from_matrix(
    column_features_matrix: Dict[str, np.ndarray], output_dimension: int
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Create aggregated vectors from a matrix of column features using PCA on the combined N x num_features matrix.

    The matrix is constructed as follows:
    - Rows: Each column from each JSON (e.g., if we have 3 JSONs with 2 columns each, we get 6 rows)
    - Columns: The features (which should be consistent across all JSONs)

    This allows PCA to work effectively on a proper matrix rather than individual vectors.

    Args:
        column_features_matrix: Dictionary mapping column names to lists of feature arrays (N samples x num_features)
        output_dimension: Target dimension for aggregated vectors

    Returns:
        Tuple of (Dictionary mapping column names to aggregated n-dimensional vectors, PCA matrix)
    """
    aggregated_vectors = {}

    # First, determine the maximum number of features across all columns
    all_features = []
    for features_list in column_features_matrix.values():
        all_features.extend(features_list)

    if not all_features:
        return aggregated_vectors, np.array([])

    max_features = max(len(features) for features in all_features)

    # Build the combined matrix: rows = all columns from all JSONs, columns = features
    combined_matrix = []
    column_names = []

    for col_name, features_list in column_features_matrix.items():
        for features in features_list:
            # Pad features to consistent length
            if len(features) < max_features:
                padded = np.zeros(max_features)
                padded[: len(features)] = features
                combined_matrix.append(padded)
            else:
                combined_matrix.append(features)
            column_names.append(col_name)

    if not combined_matrix:
        return aggregated_vectors, np.array([])

    # Stack into matrix: N x num_features where N = total number of columns across all JSONs
    features_matrix = np.vstack(combined_matrix)

    # Apply PCA to the combined matrix to get the feature space
    pca_matrix = None
    if (
        features_matrix.shape[0] > 1 and output_dimension != -1
    ):  # Only PCA if we have multiple rows
        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_matrix)

        # Apply PCA to get the principal components
        pca = PCA(
            n_components=min(
                output_dimension,
                features_matrix.shape[0],
                features_matrix.shape[1],
            )
        )
        pca_result = pca.fit_transform(features_scaled)
        pca_matrix = pca_result

        # Now project each column's features onto this PCA space
        for col_name, features_list in column_features_matrix.items():
            if len(features_list) == 0:
                continue

            # Get the first set of features for this column (representative)
            features = features_list[0]
            if len(features) < max_features:
                padded = np.zeros(max_features)
                padded[: len(features)] = features
                features = padded

            # Project onto PCA space
            features_scaled = scaler.transform(features.reshape(1, -1))
            projected = pca.transform(features_scaled).flatten()

            # Pad or truncate to output dimension
            if len(projected) < output_dimension:
                result = np.zeros(output_dimension)
                result[: len(projected)] = projected
            else:
                result = projected[:output_dimension]

            aggregated_vectors[col_name] = result
    else:
        # Single row case - just pad or truncate
        pca_matrix = features_matrix  # Return the original matrix as PCA matrix
        for col_name, features_list in column_features_matrix.items():
            if len(features_list) == 0:
                continue

            features = features_list[0]
            if output_dimension == -1:
                result = features
            elif len(features) < output_dimension:
                result = np.zeros(output_dimension)
                result[: len(features)] = features
            else:
                result = features[:output_dimension]

            aggregated_vectors[col_name] = result

    return aggregated_vectors, pca_matrix


def _extract_numeric_features(
    data_dict: Dict[str, Any], exclude_keys: Optional[List[str]] = None
) -> List[float]:
    """
    Extract numeric features from a dictionary, excluding specified keys and NaN values.

    Args:
        data_dict: Dictionary containing feature data
        exclude_keys: Keys to exclude from feature extraction

    Returns:
        List of numeric features (excluding NaN values)
    """
    if exclude_keys is None:
        exclude_keys = []

    features = []
    for key, value in data_dict.items():
        if key in exclude_keys:
            continue

        if isinstance(value, (int, float)):
            features.append(value)
        elif value is None:
            features.append(np.nan)
        elif (
            isinstance(value, str)
            and value.replace(".", "").replace("-", "").isdigit()
        ):
            features.append(float(value))

    return features


def _find_column_data(
    data_list: List[Dict[str, Any]], column_name: str, id_key: str = "Column"
) -> Optional[Dict[str, Any]]:
    """
    Find data for a specific column in a list of dictionaries.

    Args:
        data_list: List of dictionaries containing column data
        column_name: Name of the column to find
        id_key: Key used to identify the column

    Returns:
        Dictionary containing column data or None if not found
    """
    return next(
        (item for item in data_list if item.get(id_key) == column_name), None
    )


def _apply_dimensionality_reduction(
    features: np.ndarray, output_dimension: int
) -> np.ndarray:
    """
    Apply dimensionality reduction to features using PCA or zero-padding.

    Args:
        features: Input feature array
        output_dimension: Target dimension for output

    Returns:
        Aggregated vector with target dimension
    """
    if output_dimension == -1:
        return features
    else:
        if len(features) <= output_dimension:
            # Pad with zeros if features are fewer than target dimension
            aggregated_vector = np.zeros(output_dimension)
            aggregated_vector[: len(features)] = features
        else:
            # Use PCA if features are more than target dimension
            features_2d = features

            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_2d)

            # Apply PCA
            pca = PCA(n_components=output_dimension)
            aggregated_vector = pca.fit_transform(features_scaled).flatten()

        print("aggregated vector with high output dimension", aggregated_vector)
    print("aggregated vector", aggregated_vector)
    return aggregated_vector


def _extract_metric_features(
    data_by_target: Dict[str, List[Dict[str, Any]]],
    value_key: str,
    target_col: str,
) -> List[float]:
    """
    Extract metric values for a target column from grouped data.

    Args:
        data_by_target: Dictionary mapping target columns to lists of data items
        value_key: Key to extract the metric value from each data item
        target_col: Target column name

    Returns:
        List of metric values (excluding errors and invalid values)
    """
    if target_col not in data_by_target:
        return []

    values = []
    for item in data_by_target[target_col]:
        value = item.get(value_key, "0")
        if isinstance(value, str) and value != "Error":
            try:
                values.append(float(value))
            except (ValueError, TypeError):
                pass
        elif isinstance(value, (int, float)) and not np.isnan(value):
            values.append(float(value))

    return values


def _calculate_metric_statistics(values: List[float]) -> List[float]:
    """
    Calculate statistics (mean, max, std, count) for a list of values.

    Args:
        values: List of numeric values

    Returns:
        List of [mean, max, std, count] statistics
    """
    if values:
        mean_val = np.mean(values)
        max_val = np.max(values)
        std_val = np.std(values)
        count_val = len(values)

        return [
            float(mean_val),
            float(max_val),
            float(std_val),
            float(count_val),
        ]
    else:
        return [0.0, 0.0, 0.0, 0.0]


def _calculate_weighted_aggregate_statistics(
    features: List[float],
) -> List[float]:
    """
    Calculate weighted aggregate statistics across all metrics.

    Args:
        features: List of features where each metric contributes 4 values [mean, max, std, count]

    Returns:
        List of [weighted_mean, weighted_max, overall_std, num_metrics]
    """
    # Each metric contributes 4 values: [mean, max, std, count]
    # We need to extract means, maxs, and weights (counts) for each metric
    metric_weights = []
    metric_means = []
    metric_maxs = []

    # Process each metric's 4-value contribution
    for i in range(0, len(features), 4):
        if i + 3 < len(features):  # Ensure we have all 4 values
            mean_val = features[i]
            max_val = features[i + 1]
            weight = features[i + 3]  # Count as weight

            if weight > 0:  # Only include metrics with data
                metric_weights.append(weight)
                metric_means.append(mean_val)
                metric_maxs.append(max_val)

    if metric_weights and metric_means and metric_maxs:
        # Normalize weights to sum to 1
        total_weight = sum(metric_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in metric_weights]

            # Weighted mean across all metrics
            weighted_mean = sum(
                mean * weight
                for mean, weight in zip(metric_means, normalized_weights)
            )

            # Weighted max across all metrics
            weighted_max = sum(
                max_val * weight
                for max_val, weight in zip(metric_maxs, normalized_weights)
            )

            # Additional statistics
            overall_std = (
                float(np.std(metric_means)) if len(metric_means) > 1 else 0.0
            )
            num_metrics = len(metric_means)

            return [
                float(weighted_mean),
                float(weighted_max),
                float(overall_std),
                float(num_metrics),
            ]

    return [0.0, 0.0, 0.0, 0.0]


def _apply_dimensionality_reduction_from_matrix(
    features_matrix: np.ndarray, output_dimension: int
) -> np.ndarray:
    """
    Apply dimensionality reduction to a features matrix using PCA on the N x num_features matrix.

    Args:
        features_matrix: Input feature matrix (N samples x num_features)
        output_dimension: Target dimension for output

    Returns:
        Aggregated vector with target dimension
    """
    N, num_features = features_matrix.shape

    if N == 1:
        # Single sample case - pad or truncate
        if num_features <= output_dimension:
            aggregated_vector = np.zeros(output_dimension)
            aggregated_vector[:num_features] = features_matrix[0]
        else:
            aggregated_vector = features_matrix[0, :output_dimension]
    elif N < output_dimension:
        # Fewer samples than target dimension - pad with zeros
        aggregated_vector = np.zeros(output_dimension)
        # Use the first N components from PCA, pad the rest
        if num_features > 1:
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_matrix)

            # Apply PCA with min(N, num_features) components
            pca_components = min(N, num_features)
            pca = PCA(n_components=pca_components)
            pca_result = pca.fit_transform(features_scaled)

            # Take the first sample's PCA result and pad
            aggregated_vector[:pca_components] = pca_result[0]
        else:
            # Single feature case
            aggregated_vector[:N] = features_matrix[:, 0]
    else:
        # N >= output_dimension - use PCA to reduce to target dimension
        if num_features > 1:
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_matrix)

            # Apply PCA
            pca = PCA(n_components=output_dimension)
            pca_result = pca.fit_transform(features_scaled)

            # Take the first sample's PCA result
            aggregated_vector = pca_result[0]
        else:
            # Single feature case - take first output_dimension samples
            aggregated_vector = features_matrix[:output_dimension, 0]

    return aggregated_vector


basic_statistics = OrderedDict(
    [
        (
            "Kurtosis",
            1 / -2.0,
        ),  # basic (smaller better, unnormalized, high is 2)
        (
            "Skewness",
            1 / -2.0,
        ),  # basic (smaller better, unnormalized, high is 2)
    ]
)

outlier_analysis = OrderedDict(
    [
        ("Outlier Percentage", -1.0),  # basic (smaller better, normalized)
    ]
)

tsfeatures_statistics = OrderedDict(
    [
        (
            "x_acf10",
            1.0 / 10.0,
        ),  # ts_features (higher better, normalized to 10)
        ("diff2_acf1", 1.0),  # ts_features (higher better, normalized)
        (
            "x_pacf5",
            1 / 2.0,
        ),  # ts_features (higher better, unnormalized, high is 2)
        ("trend", 1.0),  # ts_features (higher better, normalized)
        # ("seasonal_strength", 1.0),  # ts_features (higher better, normalized)
        ("spike", -1.0),  # ts_features (lower better, normalized)
        ("linearity", 1.0),  # ts_features (higher better, normalized)
        ("curvature", -1.0),  # ts_features (lower better, normalized)
        # ("peak", 1.0),  # ts_features (higher better, normalized)
        # ("trough", 1.0),  # ts_features (higher better, normalized)
        (
            "nperiods",
            1.0,
        ),  # ts_features (higher better, count relative to length/seasonality)
        ("entropy", -1.0),  # ts_features (lower better, range (0, log(n)))
        ("hurst", 2.0),  # ts_features (distance from 0.5 better, normalized)
        ("nonlinearity", -1.0),  # ts_features (lower better, normalized)
        ("stability", -1.0),  # ts_features (lower better, normalized)
        ("lumpiness", -1.0),  # ts_features (lower better, normalized)
        # ("max_level_shift", -1.0),  # ts_features (lower better, normalized)
        # ("max_var_shift", -1.0),  # ts_features (lower better, normalized)
        # ("max_kl_shift", -1.0),  # ts_features (lower better, normalized)
        (
            "crossing_points",
            1.0,
        ),  # ts_features (moderate better, count relative to length/seasonality)
        (
            "unitroot_kpss",
            -10.0,
        ),  # ts_features (lower better, unnormalized range high 0.1)
        (
            "unitroot_pp",
            1 / 2.0,
        ),  # ts_features (higher better, unnormalized range high 2)
        ("flat_spots", -1.0),  # ts_features (lower better, normalized)
    ]
)

decomposition_residuals = OrderedDict(
    [
        ("Fourier", -1.0),
        ("SSA", -1.0),
        ("SINDy", -1.0),
        ("STL", -1.0),
    ]
)


def aggregate_predictability_statistics(
    results_inputs: List[str],
) -> OrderedDict[str, float]:
    """
    Combine predictability statistics into a single number for each key in the domain.
    """
    # Load all data sources
    all_data = []
    for input_item in results_inputs:
        name = "_".join(input_item.split("/")[-1].split(".")[0].split("_")[:-2])
        data = _load_data(input_item)
        all_data.append((name, data))

    # Extract and combine features across all data sources
    aggregated_values = OrderedDict()
    for data_idx, (name, data) in enumerate(all_data):
        aggregated_values[name] = 0.0
        series_length = data["basic_statistics"][0]["Count"] / len(
            data["basic_statistics"]
        )
        print("trying sum for", name, len(data.keys()))
        for key, value in data.items():
            if key == "outlier_analysis":
                outlier_analysis_values = value[0]
                for k in outlier_analysis:
                    print(
                        "trying sum for outlier_analysis",
                        k,
                        outlier_analysis_values[k],
                        outlier_analysis[k],
                    )
                    if outlier_analysis_values[k] is not None and not np.isnan(
                        float(outlier_analysis_values[k][:-1])
                    ):
                        aggregated_values[name] += np.clip(
                            float(outlier_analysis_values[k][:-1])
                            * outlier_analysis[k],
                            -1,
                            1,
                        )
            if key == "basic_statistics":
                basic_statistics_values = value[0]
                for k in basic_statistics.keys():
                    print(
                        "trying sum for basic_statistics",
                        k,
                        basic_statistics_values[k],
                        basic_statistics[k],
                    )
                    if basic_statistics_values[k] is not None and not np.isnan(
                        float(basic_statistics_values[k])
                    ):
                        aggregated_values[name] += np.clip(
                            float(basic_statistics_values[k])
                            * basic_statistics[k],
                            -1,
                            1,
                        )
            elif key == "ts_features":
                tsfeatures_statistics_values = value[0]
                for k in tsfeatures_statistics:
                    if k in ["nperiods", "crossing_points"]:
                        print(
                            "trying sum for tsfeatures_statistics",
                            k,
                            tsfeatures_statistics_values[k],
                            tsfeatures_statistics[k] / series_length,
                        )
                        if tsfeatures_statistics_values[
                            k
                        ] is not None and not np.isnan(
                            float(tsfeatures_statistics_values[k])
                        ):
                            aggregated_values[name] += np.clip(
                                float(tsfeatures_statistics_values[k])
                                * tsfeatures_statistics[k]
                                / series_length,
                                -1,
                                1,
                            )
                    else:
                        print(
                            "trying sum for tsfeatures_statistics",
                            k,
                            tsfeatures_statistics_values[k],
                            tsfeatures_statistics[k],
                        )
                        if tsfeatures_statistics_values[
                            k
                        ] is not None and not np.isnan(
                            float(tsfeatures_statistics_values[k])
                        ):
                            aggregated_values[name] += np.clip(
                                float(tsfeatures_statistics_values[k])
                                * tsfeatures_statistics[k],
                                -1,
                                1,
                            )
            elif key == "decomposition_analysis":
                for vdict in value:
                    if vdict["Decomposition Type"] in decomposition_residuals:
                        aggregated_values[name] += (
                            np.clip(float(vdict["Residual"]), 0, 1)
                            * decomposition_residuals[
                                vdict["Decomposition Type"]
                            ]
                        )
                    print(
                        "trying sum for decomposition_residuals",
                        vdict["Decomposition Type"],
                        decomposition_residuals[vdict["Decomposition Type"]],
                        vdict["Residual"],
                    )
                    aggregated_values[name] += (
                        float(vdict["Residual"])
                        * decomposition_residuals[vdict["Decomposition Type"]]
                    )
        aggregated_values[name] /= (
            len(decomposition_residuals.keys())
            + len(tsfeatures_statistics.keys())
            + len(basic_statistics.keys())
            + len(outlier_analysis.keys())
        )
    return aggregated_values


def _matrix_and_clean_nan(
    column_features_matrix: Dict[str, List[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Combine basic statistics into a fixed dimension vector.
    Loads results from multiple JSON files or dictionaries and creates aggregated vectors for each column.
    """
    new_column_features_matrix: OrderedDict[str, np.ndarray] = OrderedDict()
    for col_name, features in column_features_matrix.items():
        new_column_features_matrix[col_name] = np.stack(
            features, axis=0
        )  # number of files x number of features

    if len(new_column_features_matrix) > 0:
        # interpolate nan values
        full_matrix = np.concatenate(
            [features for features in new_column_features_matrix.values()],
            axis=0,
        )
        # first take the non-nan mean of the columns:
        non_nan_mean = np.nanmean(full_matrix, axis=0)

        for i in range(full_matrix.shape[1]):
            full_matrix[:, i] = np.nan_to_num(
                full_matrix[:, i], non_nan_mean[i]
            )

        # then reassign
        for i, (col_name, features) in enumerate(
            new_column_features_matrix.items()
        ):
            new_column_features_matrix[col_name] = np.expand_dims(
                full_matrix[i], axis=0
            )

    return new_column_features_matrix


def aggregate_basic_statistics(
    results_inputs: Union[
        str, Dict[str, Any], List[Union[str, Dict[str, Any]]]
    ],
    output_dimension: int = -1,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Combine basic statistics into a fixed dimension vector.
    Loads results from multiple JSON files or dictionaries and creates aggregated vectors for each column.

    The function constructs a matrix where:
    - Rows represent each dataset from each JSON (e.g., 3 JSONs with 2 columns = 6 rows)
    - Columns represent the features (consistent across all JSONs)

    PCA is then applied to this combined matrix for effective dimensionality reduction.

    Args:
        results_inputs: Single filepath/dict, or list of filepaths/dicts
        output_dimension: Target dimension for the aggregated vectors

    Returns:
        Tuple of (Dictionary mapping column names to aggregated n-dimensional vectors, PCA matrix)
    """
    # Normalize input to list
    if not isinstance(results_inputs, list):
        results_inputs = [results_inputs]

    # Validate input
    if not results_inputs:
        raise ValueError("Input list cannot be empty")

    # Load all data sources
    all_data = []
    for input_item in results_inputs:
        data = _load_data(input_item)
        all_data.append(data)

    # Extract and combine features across all data sources
    column_features_matrix = OrderedDict()

    for data_idx, data in enumerate(all_data):
        # Extract the relevant data sections
        ts_features = data.get("ts_features", [])
        basic_statistics = data.get("basic_statistics", [])
        quantile_analysis = data.get("quantile_analysis", [])
        outlier_analysis = data.get("outlier_analysis", [])

        # Process each column
        for col_data in ts_features:
            if not col_data:
                continue

            col_name = col_data.get("unique_id", "unknown")
            if col_name == "unknown":
                continue

            # Initialize feature vector for this column if not exists
            if col_name not in column_features_matrix:
                column_features_matrix[col_name] = []

            # Extract features for this column from this data source
            features = []

            # Extract tsfeatures (excluding unique_id)
            features.extend(
                _extract_numeric_features(col_data, exclude_keys=["unique_id"])
            )

            # Add basic statistics for this column
            basic_stats = _find_column_data(basic_statistics, col_name)
            if basic_stats:
                features.extend(
                    _extract_numeric_features(
                        basic_stats, exclude_keys=["Column"]
                    )
                )

            # Add quantile analysis for this column
            quantile_stats = _find_column_data(quantile_analysis, col_name)
            if quantile_stats:
                features.extend(
                    _extract_numeric_features(
                        quantile_stats, exclude_keys=["Column"]
                    )
                )
            # Add outlier analysis for this column
            outlier_stats = _find_column_data(outlier_analysis, col_name)
            if outlier_stats:
                features.extend(
                    _extract_numeric_features(
                        outlier_stats, exclude_keys=["Column"]
                    )
                )
            # Store the features for this column from this data source
            column_features_matrix[col_name].append(
                np.array(features, dtype=float)
            )

    column_features_matrix = _matrix_and_clean_nan(column_features_matrix)
    # Create aggregated vectors using the combined matrix approach
    if output_dimension == -1:
        # use a manual computation from all of the values rather than PCA
        aggregated_vectors = _create_aggregated_vectors(
            column_features_matrix, output_dimension
        )
        return aggregated_vectors, np.array([])
    else:
        aggregated_vectors, pca_matrix = _create_aggregated_vectors_from_matrix(
            column_features_matrix, output_dimension
        )
        return aggregated_vectors, pca_matrix


def aggregate_decomposition_statistics(
    results_inputs: Union[
        str, Dict[str, Any], List[Union[str, Dict[str, Any]]]
    ],
    output_dimension: int = -1,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Combine decomposition statistics into a fixed dimension vector.
    Loads results from multiple JSON files or dictionaries and creates aggregated vectors for each column.
    Groups by column and combines different decomposition types (Fourier, SSA, SINDy, STL).

    The function constructs a matrix where:
    - Rows represent each column from each JSON (e.g., 3 JSONs with 2 columns = 6 rows)
    - Columns represent the features (consistent across all JSONs)

    PCA is then applied to this combined matrix for effective dimensionality reduction.

    Args:
        results_inputs: Single filepath/dict, or list of filepaths/dicts
        output_dimension: Target dimension for the aggregated vectors

    Returns:
        Tuple of (Dictionary mapping column names to aggregated n-dimensional vectors, PCA matrix)
    """
    # Normalize input to list
    if not isinstance(results_inputs, list):
        results_inputs = [results_inputs]

    # Validate input
    if not results_inputs:
        raise ValueError("Input list cannot be empty")

    # Load all data sources
    all_data = []
    for input_item in results_inputs:
        if isinstance(input_item, str):
            with open(input_item, "r") as f:
                data = json.load(f)
        else:
            data = input_item
        all_data.append(data)

    # Extract and combine features across all data sources
    column_features_matrix = OrderedDict()

    for data_idx, data in enumerate(all_data):
        # Extract the relevant data sections
        decomposition_analysis = data.get("decomposition_analysis", [])

        # Group decomposition analysis by column and process features
        column_decompositions = _group_data_by_column(
            decomposition_analysis, column_key="Column"
        )

        # Process each column
        for col_name, data_items in column_decompositions.items():
            # Initialize feature vector for this column if not exists
            if col_name not in column_features_matrix:
                column_features_matrix[col_name] = []

            # Extract features for this column from this data source
            features = []
            for item in data_items:
                item_features = _extract_numeric_features(
                    item, exclude_keys=["Column", "Decomposition Type", "Type"]
                )
                features.extend(item_features)

            # Store the features for this column from this data source
            column_features_matrix[col_name].append(
                np.array(features, dtype=float)
            )

    column_features_matrix = _matrix_and_clean_nan(column_features_matrix)
    # Create aggregated vectors using the combined matrix approach
    aggregated_vectors, pca_matrix = _create_aggregated_vectors_from_matrix(
        column_features_matrix, output_dimension
    )
    return aggregated_vectors, pca_matrix


# equally weight the mean and max values for each relation
relational_statistics = OrderedDict(
    [
        ("cross_correlation_analysis", (0.5, 0.5)),
        ("granger_causality_analysis", (0.5, 0.5)),
        ("convergent_cross_mapping_analysis", (0.5, 0.5)),
        # ("mutual_information_analysis", (0.5, 0.5)), # we aren't using mutual information for now
        ("transfer_entropy_analysis", (0.5, 0.5)),
        ("dlinear_analysis", (0.5, 0.5)),
    ]
)

relational_statistics_subkeys = OrderedDict(
    [
        ("cross_correlation_analysis", "Max Correlation"),
        ("granger_causality_analysis", "Granger Causality"),
        ("convergent_cross_mapping_analysis", "Convergent Cross-Mapping"),
        # ("mutual_information", 1.0), # we aren't using mutual information for now
        ("transfer_entropy_analysis", "Max Transfer Entropy"),
        ("dlinear_analysis", "DLinear Causality"),
    ]
)

relational_column_mapping = OrderedDict(
    [
        ("cross_correlation_analysis", "Series 2"),
        ("granger_causality_analysis", "Target"),
        ("convergent_cross_mapping_analysis", "Target"),
        ("transfer_entropy_analysis", "Target"),
        ("dlinear_analysis", "Target"),
    ]
)

relational_source_column_mapping = OrderedDict(
    [
        ("cross_correlation_analysis", "Series 1"),
        ("granger_causality_analysis", "Source"),
        ("convergent_cross_mapping_analysis", "Source"),
        ("transfer_entropy_analysis", "Source"),
        ("dlinear_analysis", "Source"),
    ]
)


def _identify_padding_columns(data: Dict[str, Any]) -> set:
    """
    Identify padding columns from metadata_summary that have range '0.0 to 0.0'.

    Args:
        data: Dictionary containing the loaded data

    Returns:
        Set of column names that are padding columns
    """
    padding_columns = set()

    if "metadata_summary" in data:
        for column_info in data["metadata_summary"]:
            if column_info.get("Range") == "0.0 to 0.0":
                padding_columns.add(column_info["Column Name"])

    return padding_columns


def aggregate_multivariate_predictability_statistics(
    results_inputs: List[str],
) -> OrderedDict[str, float]:
    """
    Combine the relational statistics into a single number for each key in the domain.
    Excludes padding columns (columns with range '0.0 to 0.0') when they appear as source columns.
    """
    # Load all data sources
    all_data = []
    for input_item in results_inputs:
        name = "_".join(input_item.split("/")[-1].split(".")[0].split("_")[:-1])
        data = _load_data(input_item)
        all_data.append((name, data))

    # Extract and combine features across all data sources
    aggregated_values = OrderedDict()
    for data_idx, (name, data) in enumerate(all_data):
        aggregated_values[name] = 0.0

        # Identify padding columns for this dataset
        padding_columns = _identify_padding_columns(data)

        # get the target column first
        for key, value in data.items():
            if key == "ts_features":
                target_col = value[0]["unique_id"]
        for key, value in data.items():
            if key in relational_statistics:
                all_value_row = list()
                for relation_dict in value:
                    # Check if this relation involves the target column
                    target_match = relation_dict[
                        relational_column_mapping[key]
                    ] == target_col or relation_dict[
                        relational_column_mapping[key]
                    ] in ["target", "Target", "target_col"]

                    # Check if the source column is a padding column
                    source_col = relation_dict.get(
                        relational_source_column_mapping[key]
                    )
                    is_padding_source = source_col in padding_columns

                    # Only include if target matches and source is not a padding column
                    if target_match and not is_padding_source:
                        # print(f"Added {relational_statistics_subkeys[key]} for {key} with value {relation_dict[relational_statistics_subkeys[key]]}")
                        all_value_row.append(
                            np.clip(
                                float(
                                    relation_dict[
                                        relational_statistics_subkeys[key]
                                    ]
                                ),
                                0,
                                1,
                            )
                        )
                    # elif target_match and is_padding_source:
                    #     print(f"Excluded padding column {source_col} from {key} analysis")

                values = np.array(all_value_row)
                # values[values == 1] = 0 # 1.0 is probably a bug
                if len(values) > 0:
                    computed_max_min_mean = (
                        np.max(values) * relational_statistics[key][0]
                        + np.mean(values) * relational_statistics[key][1]
                    )
                    aggregated_values[name] += computed_max_min_mean
                    print(values)
                    print(
                        f"Added {key} with {values.shape} value {np.max(values)} * {relational_statistics[key][0]} + {np.mean(values)} * {relational_statistics[key][1]} = {computed_max_min_mean}"
                    )
        print(f"Aggregated values for {name}: {aggregated_values[name]}")
        aggregated_values[name] /= len(list(relational_statistics.keys()))
    return aggregated_values


def aggregate_relational_statistics(
    results_filepath: str,
    output_dimension: int = 10,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Combine relational statistics into a fixed dimension vector.
    Loads results from the results JSON and aggregates relational metrics for each target column.

    For each target column, extracts:
    - Mean correlation across all metrics (transfer entropy, mutual information, CCM, granger causality, cross correlation)
    - Maximum correlation across all metrics
    - Individual metric statistics

    Args:
        results_filepath: Path to JSON file containing relational analysis results
        output_dimension: Target dimension for the aggregated vectors

    Returns:
        Tuple of (Dictionary mapping column names to aggregated n-dimensional vectors, PCA matrix)
    """
    # Load data from JSON file
    data = _load_data(results_filepath)

    # Extract relational analysis data
    cross_corr_data = data.get("cross_correlation_analysis", [])
    granger_causality_data = data.get("granger_causality_analysis", [])
    ccm_data = data.get("convergent_cross_mapping_analysis", [])
    mutual_info_data = data.get("mutual_information_analysis", [])
    transfer_entropy_data = data.get("transfer_entropy_analysis", [])

    # Group data by target column for each analysis type
    cross_corr_by_target = _group_data_by_column(cross_corr_data, "Series 2")
    granger_by_target = _group_data_by_column(granger_causality_data, "Target")
    ccm_by_target = _group_data_by_column(ccm_data, "target")
    mutual_info_by_target = _group_data_by_column(mutual_info_data, "Column 2")
    transfer_entropy_by_target = _group_data_by_column(
        transfer_entropy_data, "Target"
    )

    # Get all unique target columns
    all_target_columns = set()
    all_target_columns.update(cross_corr_by_target.keys())
    all_target_columns.update(granger_by_target.keys())
    all_target_columns.update(ccm_by_target.keys())
    all_target_columns.update(mutual_info_by_target.keys())
    all_target_columns.update(transfer_entropy_by_target.keys())

    if not all_target_columns:
        return {}, np.array([])

    # Build features matrix for each target column
    column_features_matrix = OrderedDict()

    # Define metric configurations for cleaner processing
    metric_configs = [
        (cross_corr_by_target, "Max Correlation", "cross_correlation"),
        (granger_by_target, "Granger Causality", "granger_causality"),
        (ccm_by_target, "ccm_score", "ccm"),
        (mutual_info_by_target, "Mutual Information", "mutual_information"),
        (transfer_entropy_by_target, "Transfer Entropy", "transfer_entropy"),
    ]

    for target_col in all_target_columns:
        features = []

        # Process each metric type using helper functions
        for data_by_target, value_key, metric_name in metric_configs:
            values = _extract_metric_features(
                data_by_target, value_key, target_col
            )
            stats = _calculate_metric_statistics(values)
            features.extend(stats)

        # Calculate weighted aggregate statistics across all metrics
        weighted_stats = _calculate_weighted_aggregate_statistics(features)
        features.extend(weighted_stats)

        # Store features for this target column
        if features:
            column_features_matrix[target_col] = [
                np.array(features, dtype=float)
            ]

    # Create aggregated vectors using the combined matrix approach
    aggregated_vectors, pca_matrix = _create_aggregated_vectors_from_matrix(
        column_features_matrix, output_dimension
    )

    return aggregated_vectors, pca_matrix


def aggregate_forecasting_statistics(
    results_filepath: str,
    output_dimension: int = 10,
):
    """
    Combine forecasting statistics into a fixed dimension vector
    loads results from the results json in the results folder
    TODO: not implemented because the performance of the models is poor already
    """


def find_json_files(folder_path: str) -> List[str]:
    """
    Find all JSON files in a specified folder.

    Args:
        folder_path: Path to the folder to search for JSON files

    Returns:
        List of paths to JSON files found in the folder
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    # Search for JSON files recursively
    json_pattern = os.path.join(folder_path, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)

    if not json_files:
        print(f"Warning: No JSON files found in {folder_path}")
        return []

    print(f"Found {len(json_files)} JSON files:")
    for file_path in json_files:
        print(f"  - {file_path}")

    return json_files


def write_json(data: Any, filepath: str) -> None:
    """
    Write data to a JSON file with proper formatting.

    Args:
        data: Data to write to JSON file
        filepath: Path to the output JSON file
    """

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    converted_data = convert_numpy(data)

    with open(filepath, "w") as f:
        json.dump(converted_data, f, indent=2)


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Aggregate statistics from JSON files and save results"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="data",
        help="Folder containing JSON files to process (default: data)",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="*",
        help="Specific JSON files to process (overrides --input-folder)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="data/aggregated_metrics",
        help="Output folder for aggregated results (default: data/aggregated_metrics)",
    )
    parser.add_argument(
        "--output-dimension",
        type=int,
        default=10,
        help="Target dimension for aggregated vectors (default: 10)",
    )
    parser.add_argument(
        "--multivariate-only",
        action="store_true",
        help="Only compute multivariate predictability statistics",
    )
    parser.add_argument(
        "--predictability-only",
        action="store_true",
        help="Only compute predictability statistics",
    )

    args = parser.parse_args()

    # Determine input files
    if args.input_files:
        results_filepaths = args.input_files
        print(f"Using specified input files: {results_filepaths}")
    else:
        results_filepaths = find_json_files(args.input_folder)
        if not results_filepaths:
            print("No JSON files found. Exiting.")
            return

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Process multivariate predictability statistics
    if not args.predictability_only:
        print("\nProcessing multivariate predictability statistics...")
        aggregated_vectors = aggregate_multivariate_predictability_statistics(
            results_filepaths
        )
        output_file = os.path.join(
            args.output_folder, "multivariate_predictability_statistics.json"
        )
        write_json(aggregated_vectors, output_file)
        print(f"Multivariate predictability statistics saved to: {output_file}")
        print(f"Results: {aggregated_vectors}")

    # Process predictability statistics
    if not args.multivariate_only:
        print("\nProcessing predictability statistics...")
        aggregated_vectors = aggregate_predictability_statistics(
            results_filepaths
        )
        output_file = os.path.join(
            args.output_folder, "predictability_statistics.json"
        )
        write_json(aggregated_vectors, output_file)
        print(f"Predictability statistics saved to: {output_file}")
        print(f"Results: {aggregated_vectors}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
