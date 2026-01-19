import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.utils.dataset_utils import load_column_names

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE", "")


def load_columns(dataset_name: str, continuous_or_discrete: str) -> List[str]:
    """
    Load the continuous/discrete columns from js.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        List[str]: A list of continuous/discrete column names.
    """
    colnames_dict = load_column_names(dataset_name)
    colnames = colnames_dict[f"{continuous_or_discrete}_colnames"]
    return colnames


def extract_col_indices(
    tstr_dataset_config: DictConfig,
    colnames_to_extract: List[str],
    continuous_or_discrete: str,
) -> List[int]:
    """
    Extract indices for classification labels from continuous/discrete columns.

    Args:
        tstr_dataset_config (DictConfig): Configuration containing dataset
            - datset_name (str): Name of the dataset.
            - classification_start_index (int): Start index for classification labels in discrete conditions array 3rd dim.
            - classification_inclusive_end_index (int): End index for classification labels in discrete conditions array 3rd dim.
            - classification_original_discrete_labels (Optional[List[str]]): List of classification labels (original discrete column name) to extract indices for.

    Returns:
        List[int]: A list of indices from the continuous/discrete colnames list.
    """
    if len(colnames_to_extract) == 0:
        return []

    dataset_name = tstr_dataset_config["dataset_name"]
    # Get indices for classification labels
    colnames = load_columns(dataset_name, continuous_or_discrete)
    extracted_indices = []
    selected_colnames = []  # Track selected column names
    for label in colnames_to_extract:
        # Check if label contains ellipsis
        if "..." in label:
            # Remove ellipsis and use substring matching
            clean_label = label.replace("...", "")
            label_indices = [
                i
                for i, col in enumerate(colnames)
                if col.startswith(clean_label)
            ]
            # Track matched column names
            selected_colnames.extend([colnames[i] for i in label_indices])
        else:
            # Use exact matching for labels without ellipsis
            label_indices = [
                i for i, col in enumerate(colnames) if label == col
            ]
            # Track matched column names
            selected_colnames.extend([colnames[i] for i in label_indices])

        if not label_indices:
            raise ValueError(
                f"No matching classification labels found for {label} in {continuous_or_discrete} colnames"
            )
        extracted_indices.extend(label_indices)

    extracted_indices.sort()
    logger.info(
        f"Selected {continuous_or_discrete} columns: {selected_colnames}"
    )

    if not extracted_indices:
        raise ValueError("No matching classification labels found in colnames")

    return extracted_indices


def extract_classification_indices_with_start_end_index(
    tstr_dataset_config: DictConfig,
) -> Union[List[int], None]:
    """
    Extract classification indices using start and end indices from the configuration.

    Args:
        tstr_dataset_config (DictConfig): Configuration containing dataset
            - datset_name (str): Name of the dataset.
            - classification_start_index (int): Start index for classification labels in discrete conditions array 3rd dim.
            - classification_inclusive_end_index (int): End index for classification labels in discrete conditions array 3rd dim.
            - classification_original_discrete_labels (Optional[List[str]]): List of classification labels (original discrete column name) to extract indices for.

    Returns:
        Optional[List[int]]: A list of classification indices if start and end indices are provided, otherwise None.
    """
    logger.info("Extracting classification indices with start/end index")
    dataset_start_index = tstr_dataset_config.get(
        "classification_start_index", None
    )
    dataset_end_index = tstr_dataset_config.get(
        "classification_inclusive_end_index", None
    )
    indices_provided = (
        dataset_start_index is not None and dataset_end_index is not None
    )
    if indices_provided:
        classification_indices = list(
            range(dataset_start_index, dataset_end_index + 1)
        )
    else:
        classification_indices = None

    return classification_indices


def get_classification_indices(tstr_dataset_config: DictConfig) -> List[int]:
    """
    Retrieve classification indices (from discrete windows array) from the configuration, prioritizing start/end indices over discrete labels.

    Args:
        tstr_dataset_config (DictConfig): Configuration containing dataset
            - datset_name (str): Name of the dataset.
            - classification_start_index (int): Start index for classification labels in discrete conditions array 3rd dim.
            - classification_inclusive_end_index (int): End index for classification labels in discrete conditions array 3rd dim.
            - classification_original_discrete_labels (Optional[List[str]]): List of classification labels (original discrete column name) to extract indices for.

    Returns:
        List[int]: A list of classification indices to use.
    """
    # First try getting indices from start/end index
    classification_indices = (
        extract_classification_indices_with_start_end_index(tstr_dataset_config)
    )

    # If that returns None, fall back to discrete labels method
    if classification_indices is None:
        logger.warning("Extracting classification indices with discrete labels")
        classification_original_discrete_labels = tstr_dataset_config.get(
            "classification_original_discrete_labels", []
        )
        if len(classification_original_discrete_labels) == 0:
            raise ValueError(
                "classification_original_discrete_labels not provided in dataset config"
            )

        classification_indices = extract_col_indices(
            tstr_dataset_config,
            classification_original_discrete_labels,
            "discrete",
        )

    return classification_indices


def get_regression_index(tstr_dataset_config: DictConfig) -> int:
    """
    Retrieve regression index (from continuous windows array) from the configuration.

    Args:
        tstr_dataset_config (DictConfig): Configuration containing dataset
            - datset_name (str): Name of the dataset.
            - regression_continuous_input_cols (Optional[List[str]]): List of regression labels (original continuous column name) to extract indices for.

    Returns:
        int: A regression index to use.
    """
    regression_index = tstr_dataset_config.get("index_of_interest", None)
    if regression_index is None:
        regression_continuous_interest_cols = tstr_dataset_config.get(
            "continuous_col_of_interest", []
        )
        if len(regression_continuous_interest_cols) == 0:
            raise ValueError(
                "continuous_col_of_interest not provided in dataset config"
            )
        logger.warning("Extracting regression index with continuous input cols")
        regression_index = extract_col_indices(
            tstr_dataset_config,
            regression_continuous_interest_cols,
            "continuous",
        )
        regression_index = regression_index[0]

    return regression_index


def has_multiple_labels(
    discrete_label_embedding: np.ndarray, classification_indices: List[int]
) -> bool:
    """Check if there are multiple labels active simultaneously in any timstamp.

    Args:
        discrete_label_embedding: ndarray of shape (batch_size, time_steps, num_labels)
        classification_indices: List of indices for the classification labels

    Returns:
        bool: True if any time step has multiple active labels simultaneously
    """
    # Get relevant labels using indices
    relevant_labels = discrete_label_embedding[:, :, classification_indices]

    # Sum the number of active labels per time step
    # Shape: (batch_size, window_size)
    active_labels_per_timestep = np.sum(relevant_labels, axis=2)

    # Check if any timestamp has more than one active label
    return bool(np.any(active_labels_per_timestep > 1))


def convert_h5_to_npy(directory: str, split: str) -> None:
    """
    Converts the H5 file in the specified directory to npy files.

    The H5 file is expected to contain the following datasets:
      - "original_timeseries"
      - "synthetic_timeseries"
      - "discrete_conditions"
      - "continuous_conditions"

    The function will search for an H5 file whose name contains the given
    split identifier. If found, it loads the datasets and saves them as npy files.

    Args:
        directory (str): Directory where the H5 file is stored.
        split (str): Identifier string (e.g., "train", "val", or "test") to select the file and name output files.
    """

    h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]
    if not h5_files:
        raise FileNotFoundError("No H5 file found in the specified directory.")
    if len(h5_files) > 1:
        raise ValueError(
            "More than one H5 file found in the specified directory."
        )

    h5_file_name = h5_files[0]
    h5_file_path = os.path.join(directory, h5_file_name)
    logger.info(
        f"Converting H5 file '{h5_file_name}' to NPY format in directory: {directory}"
    )

    with h5py.File(h5_file_path, "r") as f:
        original_timeseries = np.array(f["original_timeseries"])
        synthetic_timeseries = np.array(f["synthetic_timeseries"])
        discrete_conditions = np.array(f["discrete_conditions"])
        continuous_conditions = np.array(f["continuous_conditions"])

    np.save(
        os.path.join(directory, f"{split}_original_timeseries.npy"),
        original_timeseries,
    )
    np.save(
        os.path.join(directory, f"{split}_synthetic_timeseries.npy"),
        synthetic_timeseries,
    )
    np.save(
        os.path.join(directory, f"{split}_discrete_conditions.npy"),
        discrete_conditions,
    )
    np.save(
        os.path.join(directory, f"{split}_continuous_conditions.npy"),
        continuous_conditions,
    )
    logger.info(
        f"Successfully converted '{h5_file_name}' to NPY files in {directory}"
    )


def construct_dataset_paths_by_config(
    config, split: str, synthetic_or_original_or_custom: str
):
    """
    Constructs dataset paths for timeseries and discrete conditions based on configuration.

    Args:
        config (dict): Configuration dictionary
        split (str): Dataset split ('train', 'val', or 'test')
        synthetic_or_original_or_custom (str): Type of dataset ('original', 'synthetic', or 'custom')

    Returns:
        tuple: (timeseries_dataset_locations, discrete_conditions_locations)
    """
    timeseries_dataset_loc = []
    discrete_conditions_loc = []
    continuous_conditions_loc = []
    path_types = {}

    if synthetic_or_original_or_custom in ["original", "synthetic"]:
        # The test data must always be original - this is a requirement for the TSTR model. If you want to use synthetic data for eval (which doesnt make sense), you can do the "custom" option
        if split == "test":
            synthetic_or_original_or_custom = "original"
            logger.info("Using original test data")

        # Common path components
        base_path = os.path.join(
            SYNTHEFY_DATASETS_BASE,
            config["execution_config"]["generation_save_path"],
            config["dataset_config"]["dataset_name"],
            config["execution_config"]["experiment_name"],
            config["execution_config"]["run_name"],
            f"{split}_dataset",
        )
        base_path = os.path.expandvars(base_path)
        timeseries_file = f"{split}_{'original' if synthetic_or_original_or_custom == 'original' else 'synthetic'}_timeseries.npy"
        timeseries_dataset_loc = [os.path.join(base_path, timeseries_file)]
        discrete_conditions_loc = [
            os.path.join(base_path, f"{split}_discrete_conditions.npy")
        ]
        continuous_conditions_loc = [
            os.path.join(base_path, f"{split}_continuous_conditions.npy")
        ]

        path_types[timeseries_dataset_loc[0]] = synthetic_or_original_or_custom
        path_types[discrete_conditions_loc[0] + "_0"] = (
            synthetic_or_original_or_custom
        )
        path_types[continuous_conditions_loc[0] + "_0"] = (
            synthetic_or_original_or_custom
        )

    elif synthetic_or_original_or_custom == "custom":
        dataset_paths = config["tstr_config"].get(f"{split}_dataset_paths", [])
        if not dataset_paths:
            raise ValueError(
                "No dataset paths found for custom dataset - "
                "must include the train_dataset_paths and test_dataset_paths (optional val_dataset_paths)"
            )

        for dataset_dict in dataset_paths:
            dataset_path = dataset_dict["path"]
            dataset_path = os.path.expandvars(dataset_path)
            synthetic_or_original = dataset_dict["synthetic_or_original"]
            if synthetic_or_original not in ["original", "synthetic"]:
                raise ValueError(
                    "synthetic_or_original must be one of ('original', 'synthetic')"
                )

            timeseries_file = f"{split}_{'original' if synthetic_or_original == 'original' else 'synthetic'}_timeseries.npy"
            timeseries_dataset_loc.append(
                os.path.join(dataset_path, timeseries_file)
            )
            discrete_conditions_loc.append(
                os.path.join(
                    dataset_path,
                    f"{split}_discrete_conditions.npy",
                )
            )
            continuous_conditions_loc.append(
                os.path.join(
                    dataset_path,
                    f"{split}_continuous_conditions.npy",
                )
            )
            # Add a postfix number to the key if it already exists
            ts_key = timeseries_dataset_loc[-1]
            disc_key = discrete_conditions_loc[-1]
            cont_key = continuous_conditions_loc[-1]

            # Add paths with unique keys
            path_types[ts_key] = synthetic_or_original
            _add_path_with_unique_key(
                path_types, disc_key, synthetic_or_original
            )
            _add_path_with_unique_key(
                path_types, cont_key, synthetic_or_original
            )
    else:
        raise ValueError(
            "Invalid value for synthetic_or_original_or_custom - "
            "must be one of ('original', 'synthetic', 'custom')"
        )

    return (
        timeseries_dataset_loc,
        discrete_conditions_loc,
        continuous_conditions_loc,
        path_types,
    )


def _add_path_with_unique_key(path_types, key, value):
    """Add a path to path_types with a unique key by adding a postfix if needed."""
    if f"{key}_0" in path_types:
        postfix = 1
        while f"{key}_0_{postfix}" in path_types:
            postfix += 1
        path_types[f"{key}_{postfix}"] = value
    else:
        path_types[f"{key}_0"] = value


def construct_dataset_paths_by_task_mode(
    config: Dict[str, Any], split: str, task_mode: str
) -> Tuple[List[str], List[str], List[str]]:
    """Constructs dataset paths for time-series and condition arrays."""
    timeseries_paths: List[str] = []
    discrete_conditions_paths: List[str] = []
    continuous_conditions_paths: List[str] = []
    data_types: List[str] = []

    if split == "test" or task_mode == "TRTR":
        data_types = ["original"]
        logger.info("Test dataset will use original data.")
    elif task_mode == "TSTR":
        data_types = ["synthetic"]
        logger.info("TSTR dataset will use synthetic data.")
    elif task_mode == "TRSTR":
        data_types = ["original", "synthetic"]
        logger.info("TRSTR dataset will use both original and synthetic data.")
    else:
        raise ValueError(
            "Invalid task mode. Must be one of ('TRTR', 'TSTR', 'TRSTR')."
        )
    base_path = os.path.join(
        SYNTHEFY_DATASETS_BASE,
        config["execution_config"]["generation_save_path"],
        config["dataset_config"]["dataset_name"],
        config["execution_config"]["experiment_name"],
        config["execution_config"]["run_name"],
        f"{split}_dataset",
    )
    base_path = os.path.expandvars(base_path)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")

    for data_type in data_types:
        ts_file = f"{split}_{data_type}_timeseries.npy"
        timeseries_paths.append(os.path.join(base_path, ts_file))
        discrete_conditions_paths.append(
            os.path.join(base_path, f"{split}_discrete_conditions.npy")
        )
        continuous_conditions_paths.append(
            os.path.join(base_path, f"{split}_continuous_conditions.npy")
        )

    return (
        timeseries_paths,
        discrete_conditions_paths,
        continuous_conditions_paths,
    )


def convert_pkls_to_npy(directory: str, split: str) -> None:
    """
    input:
    directory (train_dataset, val_dataset, test_dataset) from generated logs
    split - one of ("train", "val", "test")
    output: None,
    Description: Saves the numpy arrays for original_timeseries, synthetic_timeseries, discrete_conditions
    """

    original_timeseries = []
    synthetic_timeseries = []
    discrete_conditions = []
    continuous_conditions = []

    pkl_files = [f for f in sorted(os.listdir(directory)) if f.endswith(".pkl")]
    if not pkl_files:
        raise ValueError("No PKL files found in the specified directory.")

    for file in pkl_files:
        try:
            data = torch.load(os.path.join(directory, file))
            missing_keys = []
            for key in [
                "original_timeseries",
                "synthetic_timeseries",
                "discrete_conditions",
                "continuous_conditions",
            ]:
                if key not in data:
                    missing_keys.append(key)

            if not missing_keys:
                original_timeseries.append(data["original_timeseries"])
                synthetic_timeseries.append(data["synthetic_timeseries"])
                discrete_conditions.append(data["discrete_conditions"])
                continuous_conditions.append(data["continuous_conditions"])
            else:
                logger.error(
                    f"Error loading {file}: Missing required keys: {', '.join(missing_keys)}"
                )
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    if not original_timeseries:
        raise ValueError("No valid PKL files found with the required keys.")

    # Convert lists of arrays to a single numpy array by concatenating along axis 0
    original_timeseries = np.concatenate(original_timeseries, axis=0)
    synthetic_timeseries = np.concatenate(synthetic_timeseries, axis=0)
    discrete_conditions = np.concatenate(discrete_conditions, axis=0)
    continuous_conditions = np.concatenate(continuous_conditions, axis=0)

    np.save(
        os.path.join(directory, f"{split}_original_timeseries.npy"),
        original_timeseries,
    )
    np.save(
        os.path.join(directory, f"{split}_synthetic_timeseries.npy"),
        synthetic_timeseries,
    )
    np.save(
        os.path.join(directory, f"{split}_discrete_conditions.npy"),
        discrete_conditions,
    )
    np.save(
        os.path.join(directory, f"{split}_continuous_conditions.npy"),
        continuous_conditions,
    )
    logger.info(f"Converted {split} dataset to .npy - output to {directory}")


def sample_synthetic_data(
    data: np.ndarray,
    synthetic_percentage: int,
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample a percentage of synthetic data based on the specified parameters.

    Args:
        data (np.ndarray): The data array to sample from.
        synthetic_percentage (int): Percentage of data to keep (0-100).
        indices (Optional[np.ndarray], optional): If provided, use these indices for consistent sampling.

    Returns:
        np.ndarray: The sampled data.
    """
    if synthetic_percentage < 100:
        n_samples = data.shape[0]
        n_to_keep = max(
            1, int(n_samples * synthetic_percentage / 100)
        )  # Keep at least 1 sample

        if indices is None:
            # Generate consistent sampling indices
            indices = get_sampling_indices(n_samples, n_to_keep)
            logger.info(
                f"Using {synthetic_percentage}% of synthetic data: {n_to_keep}/{n_samples} samples"
            )

        return data[indices]

    # Return original data if no sampling needed
    return data


def get_sampling_indices(n_samples, n_to_keep, seed=None):
    """
    Generate consistent sampling indices for a given number of samples and keep count.

    Args:
        n_samples (int): Total number of samples.
        n_to_keep (int): Number of samples to keep.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of indices to keep.
    """
    # Use a deterministic hash of the inputs as the seed if none provided
    if seed is None:
        # Create a deterministic seed based on input parameters
        seed_str = f"{n_samples}_{n_to_keep}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate the indices
    indices = np.random.choice(n_samples, n_to_keep, replace=False)

    return indices
