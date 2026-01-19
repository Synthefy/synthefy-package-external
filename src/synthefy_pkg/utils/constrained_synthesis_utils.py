import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, Literal, Optional, Union
import copy

import cvxpy as cp
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from synthefy_pkg.utils.dataset_utils import load_timeseries_colnames
from synthefy_pkg.utils.scaling_utils import transform_timeseries_constraints


def scale_equality_constraints(
    equality_constraints: Dict[str, Dict[str, Any]], dataset_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Scale the values in equality constraints using the transform_timeseries_constraints function.

    Args:
        equality_constraints: Dict mapping channel names to their constraints
        dataset_name: Name of the dataset for scaling reference

    Returns:
        Dict with the same structure but with scaled constraint values
    """

    # Create a copy to avoid modifying the original
    scaled_constraints = copy.deepcopy(equality_constraints)

    # For each constraint type that needs scaling (min, max, etc.)
    for constraint_type in ["min", "max"]:
        # Collect values across channels for this constraint type
        values_dict = {}
        for channel_name, constraints in scaled_constraints.items():
            if constraint_type in constraints:
                values_dict[channel_name] = constraints[constraint_type]

        # If we have values to scale
        if values_dict:
            # Scale the values
            scaled_values = transform_timeseries_constraints(
                values_dict, dataset_name=dataset_name
            )

            # Update the constraints with scaled values
            for channel_name, scaled_value in scaled_values.items():
                scaled_constraints[channel_name][constraint_type] = scaled_value

    # Handle nested constraints like "min and argmin", "max and argmax"
    for nested_constraint_type in ["min and argmin", "max and argmax"]:
        for channel_name, constraints in scaled_constraints.items():
            if nested_constraint_type in constraints and isinstance(
                constraints[nested_constraint_type], dict
            ):
                value_key = "min" if "min" in nested_constraint_type else "max"
                if value_key in constraints[nested_constraint_type]:
                    # Create a temporary dict with just this value
                    temp_dict = {
                        channel_name: constraints[nested_constraint_type][
                            value_key
                        ]
                    }
                    # Scale it
                    scaled_temp = transform_timeseries_constraints(
                        temp_dict,
                        dataset_name=dataset_name,
                    )
                    # Update the nested constraint
                    scaled_constraints[channel_name][nested_constraint_type][
                        value_key
                    ] = scaled_temp[channel_name]

    return scaled_constraints


def get_equality_constraints(
    dataset_config: DictConfig,
    sample: torch.Tensor,
) -> Dict[str, Dict[str, Any]]:
    """
    Get equality constraints from dataset config and sample.
    
    This function retrieves constraints in one of three ways:
    1. Extract constraints directly from the provided sample windows
    2. Use user-provided constraints from the config
    3. Load constraints from a file based on the dataset name
    
    Args:
        dataset_config: DictConfig containing dataset configuration parameters from synthesis config
            Must include fields:
            - constraints: List of constraint types to extract
            - extract_equality_constraints_from_windows: Boolean flag to determine constraint source
            - dataset_name: Name of the dataset
            - user_provided_constraints: Optional dict of user-defined constraints
        sample: torch.Tensor of shape (batch_size, channels, window_size) containing the sample data
            
    Returns:
        Dict mapping channel names to their constraints: {channel_name: {constraint_type: value, ...}, ...}
        If constraints are extracted from windows, the key "all_channels" will be used.
    """
    constraints_to_extract = dataset_config.constraints

    if dataset_config.extract_equality_constraints_from_windows:
        # Extract constraints from windows for all channels
        # Save it under key "all_channels" to match the common format of return
        equality_constraints = {
            "all_channels": extract_equality_constraints_from_batch(
                sample,
                constraints_to_extract,
            )
        }
    else:
        # Check if user-defined constraints are provided
        if dataset_config.user_provided_constraints is not None:
            equality_constraints = extract_channel_constraints(
                dataset_name=dataset_config.dataset_name,
                constraints_dict=dataset_config.user_provided_constraints,
            )
            equality_constraints = scale_equality_constraints(
                equality_constraints, dataset_config.dataset_name
            )

        else:
            # Use constraints from file
            equality_constraints = extract_channel_constraints(
                dataset_name=dataset_config.dataset_name,
                constraints_to_extract=constraints_to_extract,
            )

    return equality_constraints


def extract_channel_constraints(
    dataset_name: str,
    constraints_to_extract: Optional[List[str]] = None,
    constraints_dict: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract channel-specific constraints from
        - a file from preprocessing step, where we store min and max from train+val dataset for each channel
        - for synthesis API case, from user provided constraints (can be channel specific or not)

    Args:
        dataset_name (str): Name of dataset to load channel names from.
        constraints_to_extract (Optional[List[str]]): List of constraint types to extract when loading from file.
            Only used if constraints_dict is None.
        constraints_dict (Optional[Dict[str, Dict[str, Union[int, float]]]]): Dictionary mapping
            channel names to their constraints. Each channel maps to a dict of constraint types and values.
            Example: {"channel1": {"min": 0, "max": 10}, "channel2": {"min": -1}}
            If provided, all constraint types in the dictionary will be extracted.

    Returns:
        Dict mapping channel names to their constraints: {channel_name: {constraint_type: value, ...}, ...}
    """
    # If constraints_dict is provided, use it directly
    if constraints_dict is not None:
        # Convert argmin and argmax values to integers if present
        for channel_name, channel_dict in constraints_dict.items():
            if "argmin" in channel_dict:
                channel_dict["argmin"] = int(channel_dict["argmin"])
            if "argmax" in channel_dict:
                channel_dict["argmax"] = int(channel_dict["argmax"])
            if "max and argmax" in channel_dict and isinstance(
                channel_dict["max and argmax"], dict
            ):
                if "argmax" in channel_dict["max and argmax"]:
                    channel_dict["max and argmax"]["argmax"] = int(
                        channel_dict["max and argmax"]["argmax"]
                    )
            if "min and argmin" in channel_dict and isinstance(
                channel_dict["min and argmin"], dict
            ):
                if "argmin" in channel_dict["min and argmin"]:
                    channel_dict["min and argmin"]["argmin"] = int(
                        channel_dict["min and argmin"]["argmin"]
                    )
        return constraints_dict

    assert (
        constraints_to_extract is not None
    ), "constraints_to_extract must be provided if constraints_dict is None"
    assert (
        constraints_dict is None
    ), "Only one of constraints_dict or constraints_to_extract should be provided"

    # Get ordered channel names
    channel_names = load_timeseries_colnames(dataset_name)

    # Otherwise, load constraints from file
    filepath = os.path.join(
        os.getenv("SYNTHEFY_DATASETS_BASE", ""),
        dataset_name,
        "constraints_channel_minmax_values.json",
    )
    with open(filepath, "r") as f:
        file_constraints = json.load(f)

    # Extract only the requested constraint types
    channel_constraints = {}
    for channel_name in channel_names:
        if channel_name in file_constraints:
            channel_dict = {}
            for constraint_type in constraints_to_extract:
                if constraint_type in file_constraints[channel_name]:
                    channel_dict[constraint_type] = file_constraints[
                        channel_name
                    ][constraint_type]
                else:
                    logger.warning(
                        f"Constraint {constraint_type} not found for channel {channel_name}"
                    )

            if channel_dict:  # Only add if there are constraints
                channel_constraints[channel_name] = channel_dict

    return channel_constraints


def extract_equality_constraints_from_batch(
    batch_samples: Union[torch.Tensor, np.ndarray],
    constraints_to_extract: List[str],
) -> Dict[str, Any]:
    """
    Extract equality constraints from the batch samples.
    input: batch_samples: torch.tensor or np.array of shape (batch_size, channels, window_size)
    output: equality_constraints: dictionary of constraints. generally Dict[str, Dict[str, float]] or Dict[str, float]
    """

    # Convert to numpy if needed
    batch_samples_numpy = (
        batch_samples.detach().cpu().numpy()
        if isinstance(batch_samples, torch.Tensor)
        else batch_samples
    )

    B, C, T = batch_samples_numpy.shape

    constraints = {}
    for constraint_name in constraints_to_extract:
        if constraint_name == "min":
            constraints[constraint_name] = np.min(batch_samples_numpy, axis=-1)
        elif constraint_name == "max":
            constraints[constraint_name] = np.max(batch_samples_numpy, axis=-1)
        elif constraint_name == "argmax":
            constraints[constraint_name] = np.argmax(
                batch_samples_numpy, axis=-1
            )
        elif constraint_name == "max and argmax":
            constraints[constraint_name] = {
                "max": np.max(batch_samples_numpy, axis=-1),
                "argmax": np.argmax(batch_samples_numpy, axis=-1),
            }

        elif constraint_name == "argmin":
            constraints[constraint_name] = np.argmin(
                batch_samples_numpy, axis=-1
            )

        elif constraint_name == "min and argmin":
            constraints[constraint_name] = {
                "min": np.min(batch_samples_numpy, axis=-1),
                "argmin": np.argmin(batch_samples_numpy, axis=-1),
            }

        elif constraint_name == "mean":
            constraints[constraint_name] = np.mean(batch_samples_numpy, axis=-1)

        elif constraint_name == "mean change":
            constraints[constraint_name] = np.mean(
                np.diff(batch_samples_numpy, axis=-1), axis=-1
            )

        elif "autocorr" in constraint_name:
            lag = int(constraint_name.split("_")[-1])
            autocorr_values = np.zeros((B, C))
            for sample_idx in range(B):
                for channel_idx in range(C):
                    timeseries = batch_samples_numpy[sample_idx, channel_idx]
                    channel_mean = np.mean(timeseries)
                    channel_variance = np.var(timeseries)
                    autocorr_values[sample_idx, channel_idx] = (
                        np.mean(
                            (timeseries[:-lag] - channel_mean)
                            * (timeseries[lag:] - channel_mean)
                        )
                        / channel_variance
                    )
            constraints[constraint_name] = autocorr_values

        else:
            raise ValueError(f"Invalid constraint: {constraint_name}")

    return constraints


def _add_constraint_to_list(
    constraints: list,
    constraint_key: str,
    constraint_data: Any,
    opt_var: cp.Variable,
    start_idx: int,
    end_idx: int,
    sample_idx: int = None,
    channel_idx: int = None,
    tolerance: float = 5e-3,
):
    """
    Helper function to add a specific constraint to the constraints list.

    Args:
        constraints: List of constraints to append to
        constraint_key: Type of constraint (min, max, argmin, etc.)
        constraint_data: The constraint data (could be a value or dict with nested structure)
        opt_var: CVXPY optimization variable
        start_idx: Start index for the channel in the flattened variable
        end_idx: End index for the channel in the flattened variable
        sample_idx: Index of the sample in the batch (for batch constraints)
        channel_idx: Index of the channel (for batch constraints)
        tolerance: Tolerance for equality constraints
    """
    if constraint_key == "min":
        minval = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(opt_var[start_idx:end_idx] >= minval)

    elif constraint_key == "max":
        maxval = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(opt_var[start_idx:end_idx] <= maxval)

    elif constraint_key == "argmax":
        argmax = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(
            opt_var[start_idx:end_idx] <= opt_var[start_idx + argmax]
        )

    elif constraint_key == "max and argmax":
        if sample_idx is None:
            argmax = constraint_data["argmax"]
            maxval = constraint_data["max"]
        else:
            argmax = constraint_data["argmax"][sample_idx][channel_idx]
            maxval = constraint_data["max"][sample_idx][channel_idx]

        constraints.append(opt_var[start_idx + argmax] <= maxval + tolerance)
        constraints.append(opt_var[start_idx + argmax] >= maxval - tolerance)

    elif constraint_key == "argmin":
        argmin = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(
            opt_var[start_idx:end_idx] >= opt_var[start_idx + argmin]
        )

    elif constraint_key == "min and argmin":
        if sample_idx is None:
            argmin = constraint_data["argmin"]
            minval = constraint_data["min"]
        else:
            argmin = constraint_data["argmin"][sample_idx][channel_idx]
            minval = constraint_data["min"][sample_idx][channel_idx]

        constraints.append(opt_var[start_idx + argmin] <= minval + tolerance)
        constraints.append(opt_var[start_idx + argmin] >= minval - tolerance)

    elif constraint_key == "mean":
        meanval = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(
            cp.mean(opt_var[start_idx:end_idx]) <= meanval + tolerance
        )
        constraints.append(
            cp.mean(opt_var[start_idx:end_idx]) >= meanval - tolerance
        )

    elif constraint_key == "mean change":
        mean_change = (
            constraint_data
            if sample_idx is None
            else constraint_data[sample_idx][channel_idx]
        )
        constraints.append(
            cp.mean(cp.diff(opt_var[start_idx:end_idx]))
            <= mean_change + tolerance
        )
        constraints.append(
            cp.mean(cp.diff(opt_var[start_idx:end_idx]))
            >= mean_change - tolerance
        )

    elif "autocorr" in constraint_key:
        # Handle autocorrelation constraints if needed
        raise NotImplementedError(
            f"Autocorrelation constraints not implemented in helper function"
        )

    else:
        raise ValueError(f"Invalid constraint: {constraint_key}")


# Function to project the sample to the convex equality constraints using CVXPY
def project_sample_to_equality_constraints_cvxpy(
    per_sample_projection_input_dict,
):
    """
    Project a sample to the convex equality constraints using CVXPY.

    Args:
        per_sample_projection_input_dict: Dictionary containing sample and constraints
        {
            "sample": np.ndarray, # shape (channels, horizon)
            "sample_idx": int,
            "constraints_to_be_executed": Dict[str, Any],
            "warm_start_sample": Optional[np.ndarray],
            "penalty_coefficient": float,
            "dataset_name": Optional[str],
        }
    """

    # TODO - We should unit test the part until the actual optimization solving
    sample_idx = per_sample_projection_input_dict["sample_idx"]
    constraints_to_be_executed = per_sample_projection_input_dict[
        "constraints_to_be_executed"
    ]
    warm_start_sample = per_sample_projection_input_dict["warm_start_sample"]
    sample = per_sample_projection_input_dict["sample"]
    dataset_name = per_sample_projection_input_dict.get("dataset_name")

    horizon = sample.shape[-1]
    num_channels = sample.shape[-2]

    flattened_sample = sample.flatten()
    random_channel_idx = np.random.randint(num_channels)
    assert (
        sample[random_channel_idx]
        == flattened_sample[
            random_channel_idx * horizon : (random_channel_idx + 1) * horizon
        ]
    ).all()
    opt_var = cp.Variable(flattened_sample.shape)
    if warm_start_sample is not None:
        opt_var.value = warm_start_sample.flatten()
    else:
        opt_var.value = flattened_sample

    objective_function = cp.Minimize(cp.norm(flattened_sample - opt_var) ** 2.0)

    tolerance = 5e-3

    constraints = []

    # Check if we're using the "all_channels" format
    if "all_channels" in constraints_to_be_executed:
        all_channel_constraints = constraints_to_be_executed["all_channels"]
        constraints_keys = list(all_channel_constraints.keys())

        for channel_idx in range(num_channels):
            start_idx = channel_idx * horizon
            end_idx = (channel_idx + 1) * horizon
            for constraint_key in constraints_keys:
                _add_constraint_to_list(
                    constraints,
                    constraint_key,
                    all_channel_constraints[constraint_key],
                    opt_var,
                    start_idx,
                    end_idx,
                    sample_idx,
                    channel_idx,
                    tolerance,
                )

    # Handle channel-specific constraints
    else:
        if dataset_name is None:
            raise ValueError(
                "dataset_name must be provided when using channel-specific constraints"
            )

        # Get ordered channel names to map names to indices
        ordered_channel_names = load_timeseries_colnames(dataset_name)
        channel_indices = {
            name: idx for idx, name in enumerate(ordered_channel_names)
        }

        # Process each channel's constraints
        for (
            channel_name,
            channel_constraints,
        ) in constraints_to_be_executed.items():
            if channel_name not in channel_indices:
                logger.warning(
                    f"Channel {channel_name} not found in dataset {dataset_name}"
                )
                continue

            channel_idx = channel_indices[channel_name]
            start_idx = channel_idx * horizon
            end_idx = (channel_idx + 1) * horizon

            for constraint_key, constraint_value in channel_constraints.items():
                _add_constraint_to_list(
                    constraints,
                    constraint_key,
                    constraint_value,
                    opt_var,
                    start_idx,
                    end_idx,
                    tolerance=tolerance,
                )

    problem = cp.Problem(objective_function, constraints)

    try:
        problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        # Check if the solver found a solution
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            sol = opt_var.value
            projected_sample = sol.reshape(sample.shape)
            projected_sample = projected_sample.astype(np.float32)
            return (projected_sample, sample_idx)
        else:
            # If optimization failed, log the issue and return the original sample
            logger.warning(
                f"Optimization failed with status {problem.status} for sample {sample_idx}. Using original sample."
            )
            return (sample, sample_idx)
    except Exception as e:
        # If any exception occurs during optimization, log it and return the original sample
        logger.warning(
            f"Exception during optimization for sample {sample_idx}: {str(e)}. Using original sample."
        )
        return (sample, sample_idx)


def apply_min_max_constraints_by_clipping(
    sample_estimate_batch: np.ndarray,
    constraints: Dict[str, Dict[str, Union[int, float, np.ndarray]]],
    dataset_name: Optional[str] = None,
):
    """
    Apply min/max/argmin/argmax constraints by clipping.

    Args:
        sample_estimate_batch: Batch of samples to constrain, shape (B, C, H)
        constraints: Dictionary with constraints in one of two formats:
            1. {"all_channels": {"min": np.ndarray(B,C), "max": np.ndarray(B,C),
                                "argmin": np.ndarray(B,C), "argmax": np.ndarray(B,C)}}
            2. {"channel_name1": {"min": value, "max": value}, "channel_name2": {...}}
        dataset_name: Name of the dataset to load ordered channel names (required for channel-specific constraints)

    Returns:
        Samples with constraints applied

    Raises:
        ValueError: If inconsistent constraints are detected (e.g., min > max)
    """
    projected_samples = sample_estimate_batch.copy()
    batch_size, num_channels, horizon = projected_samples.shape

    # Check if we're using the "all_channels" format
    if "all_channels" in constraints:
        all_channel_constraints = constraints["all_channels"]

        # Validate min/max constraints consistency
        if (
            "min" in all_channel_constraints
            and "max" in all_channel_constraints
        ):
            min_vals = all_channel_constraints["min"]
            max_vals = all_channel_constraints["max"]

            # Check if min values are less than max values
            if np.any(min_vals > max_vals):
                inconsistent_indices = np.where(min_vals > max_vals)
                raise ValueError(
                    f"Inconsistent min/max constraints detected: min > max at indices {inconsistent_indices}"
                )

        # Apply all constraints using a common masking approach
        for constraint_type, operation in [
            ("min", np.maximum),
            ("max", np.minimum),
            ("argmin", np.maximum),
            ("argmax", np.minimum),
        ]:
            # Skip argmax/argmin if min/max constraints exist
            if (
                constraint_type == "argmax" and "max" in all_channel_constraints
            ) or (
                constraint_type == "argmin" and "min" in all_channel_constraints
            ):
                continue

            if constraint_type in all_channel_constraints:
                # Get constraint values or indices
                constraint_data = all_channel_constraints[constraint_type]

                # For min/max, use values directly; for argmin/argmax, extract values at indices
                if constraint_type in ["min", "max"]:
                    constraint_vals = constraint_data
                else:  # argmin or argmax
                    raise NotImplementedError(
                        "argmin/argmax constraints not supported for all_channels format"
                    )
                    # TODO: Implement argmin/argmax for all_channels format after determining the use case
                    # arg_vals = constraint_data

                    # # Extract values at the argmin/argmax positions
                    # constraint_vals = np.zeros_like(constraint_data)
                    # for b in range(batch_size):
                    #     for c in range(num_channels):
                    #         arg_val = int(arg_vals[b, c])
                    #         # Get the value at the argmin/argmax position
                    #         val = projected_samples[b, c, arg_val]
                    #         constraint_vals[b, c] = val

                    #         # Validate consistency with min/max constraints if they exist
                    #         if (
                    #             constraint_type == "argmax"
                    #             and "min" in all_channel_constraints
                    #         ):
                    #             min_val = all_channel_constraints["min"][b, c]
                    #             if val < min_val:
                    #                 raise ValueError(
                    #                     f"Inconsistent constraints: argmax value {val} < min value {min_val} at batch {b}, channel {c}"
                    #                 )

                    #         elif (
                    #             constraint_type == "argmin"
                    #             and "max" in all_channel_constraints
                    #         ):
                    #             max_val = all_channel_constraints["max"][b, c]
                    #             if val > max_val:
                    #                 raise ValueError(
                    #                     f"Inconsistent constraints: argmin value {val} > max value {max_val} at batch {b}, channel {c}"
                    #                 )

                # Reshape for broadcasting: (batch, channel, 1) to match (batch, channel, window_size)
                constraint_vals_reshaped = constraint_vals[:, :, np.newaxis]
                # Apply operation to all samples
                projected_samples = operation(
                    projected_samples, constraint_vals_reshaped
                )

    # Handle channel-specific constraints
    else:
        if dataset_name is None:
            raise ValueError(
                "dataset_name must be provided when using channel-specific constraints"
            )

        # Get ordered channel names to map names to indices
        ordered_channel_names = load_timeseries_colnames(dataset_name)
        channel_indices = {
            name: idx for idx, name in enumerate(ordered_channel_names)
        }

        # Apply constraints for each channel
        for channel_name, channel_constraints in constraints.items():
            if channel_name not in channel_indices:
                raise ValueError(
                    f"Channel {channel_name} not found in dataset {dataset_name}"
                )

            channel_idx = channel_indices[channel_name]

            # Validate min/max constraints consistency
            if "min" in channel_constraints and "max" in channel_constraints:
                min_val = channel_constraints["min"]
                max_val = channel_constraints["max"]

                # Check if min value is less than max value
                if min_val > max_val:
                    raise ValueError(
                        f"Inconsistent min/max constraints for channel {channel_name}: min ({min_val}) > max ({max_val})"
                    )

            # Apply all constraints using a common masking approach
            for constraint_type, operation in [
                ("min", np.maximum),
                ("max", np.minimum),
                ("argmin", np.maximum),
                ("argmax", np.minimum),
            ]:
                # Skip if the corresponding min/max constraint exists for argmax/argmin
                if (
                    constraint_type == "argmax" and "max" in channel_constraints
                ) or (
                    constraint_type == "argmin" and "min" in channel_constraints
                ):
                    continue

                if constraint_type in channel_constraints:
                    # Get constraint values or indices
                    constraint_data = channel_constraints[constraint_type]

                    # For min/max, use values directly; for argmin/argmax, extract values at indices
                    if constraint_type in ["min", "max"]:
                        # For min/max, create a batch-sized array with the same value for all samples
                        constraint_vals = np.full(batch_size, constraint_data)
                    elif constraint_type in ["argmin", "argmax"]:
                        raise NotImplementedError(
                            "argmin/argmax constraints not supported for all_channels format"
                        )
                        # # For argmin/argmax, constraint_data is the index position
                        # idx = int(constraint_data)
                        # constraint_vals = np.zeros(batch_size)

                        # # For each sample in the batch, get the value at the argmin/argmax position
                        # for b in range(batch_size):
                        #     # Get the value at the argmin/argmax position for this specific channel
                        #     val = projected_samples[b, channel_idx, idx]
                        #     constraint_vals[b] = val

                        #     # For argmin/argmax constraints, we need to ensure the value at that position
                        #     # is the minimum/maximum in the sequence
                        #     if constraint_type == "argmin":
                        #         # For argmin, all other values should be >= this value
                        #         projected_samples[b, channel_idx, :] = (
                        #             np.maximum(
                        #                 projected_samples[b, channel_idx, :],
                        #                 np.full(horizon, val),
                        #             )
                        #         )
                        #         # Restore the original value at the argmin position
                        #         projected_samples[b, channel_idx, idx] = val
                        #     elif constraint_type == "argmax":
                        #         # For argmax, all other values should be <= this value
                        #         projected_samples[b, channel_idx, :] = (
                        #             np.minimum(
                        #                 projected_samples[b, channel_idx, :],
                        #                 np.full(horizon, val),
                        #             )
                        #         )
                        #         # Restore the original value at the argmax position
                        #         projected_samples[b, channel_idx, idx] = val
                    else:
                        raise NotImplementedError(
                            f"Invalid constraint type: {constraint_type} - only min, max are supported for now"
                        )

                    # For min/max constraints, apply the operation to the entire channel
                    if constraint_type in ["min", "max"]:
                        # Reshape for broadcasting: (batch, 1) to match (batch, window_size)
                        constraint_vals_reshaped = constraint_vals[
                            :, np.newaxis
                        ]
                        # Apply operation only to the current channel
                        projected_samples[:, channel_idx, :] = operation(
                            projected_samples[:, channel_idx, :],
                            constraint_vals_reshaped,
                        )

    return projected_samples


def project_all_samples_to_equality_constraints(
    sample_estimate_batch: np.ndarray,
    constraints: Dict[str, Dict[str, Union[int, float]]],
    warm_start_samples: Optional[np.ndarray],
    penalty_coefficient: float,
    projection_method: Literal["strict", "clipping"],
    dataset_name: Optional[str] = None,
):
    """
    Project samples to satisfy equality constraints using CVXPY optimization.

    Args:
        sample_estimate_batch: Batch of samples to project, shape (B, C, H)
        constraints: Dictionary of constraints to apply, either:
            - {"all_channels": {"constraint_type": values, ...}}
            - {"channel_name1": {"constraint_type": value, ...}, "channel_name2": {...}}
        warm_start_samples: Initial guess for optimization (can be None for clipping method) shape (B, C, H)
        penalty_coefficient: Coefficient for penalty-based projection (not currently used)
        projection_method: Method for projection ("strict", "penalty_based", or "clipping")
        dataset_name: Name of dataset (required for channel-specific constraints
                                       which is used mainly in synthesis inference API)

    Returns:
        Projected samples satisfying the constraints
    """
    # Fast path for clipping method which doesn't need per-sample processing
    if projection_method == "clipping":
        # Check if constraints only contain min/max (suitable for clipping)
        if "all_channels" in constraints:
            constraint_keys = set(constraints["all_channels"].keys())
        else:
            # For channel-specific constraints, check all constraint types across all channels
            constraint_keys = set()
            for channel_constraints in constraints.values():
                # Add all constraint types from this channel to our set
                constraint_keys.update(channel_constraints.keys())

        if constraint_keys.issubset({"min", "max"}):
            return apply_min_max_constraints_by_clipping(
                sample_estimate_batch, constraints, dataset_name
            )
        else:
            logger.warning(
                "Clipping method requested but constraints contain more than min/max. "
                "Falling back to strict projection method."
            )
            projection_method = "strict"

    assert warm_start_samples is not None, "Warm start samples must be provided"
    # For other methods, prepare per-sample processing
    per_sample_projection_inputs_list = []
    for sample_idx in range(sample_estimate_batch.shape[0]):
        sample_estimate = sample_estimate_batch[sample_idx]
        per_sample_projection_input_dict = {
            "sample": sample_estimate,
            "constraints_to_be_executed": constraints,
            "sample_idx": sample_idx,
            "penalty_coefficient": penalty_coefficient,
            "warm_start_sample": warm_start_samples[sample_idx],
            "dataset_name": dataset_name,
        }
        per_sample_projection_inputs_list.append(
            per_sample_projection_input_dict
        )

    if sample_estimate_batch.shape[0] == 1:
        if projection_method == "strict":
            result = project_sample_to_equality_constraints_cvxpy(
                per_sample_projection_inputs_list[0]
            )

        # Comment this out for now since we are not using penalty based projection - can get it from UT repo if needed
        elif projection_method == "penalty_based":
            raise NotImplementedError(
                "Penalty based projection not supported yet"
            )
            # list_of_constraints = list(constraints.keys())
            # if "autocorr_12" in list_of_constraints:
            #     result = project_sample_to_minimize_scipy_penalty(per_sample_projection_inputs_list[0])
            # else:
            #     result = project_sample_to_minimize_penalty(per_sample_projection_inputs_list[0])

        else:
            raise ValueError(
                f"Invalid projection method: {projection_method} - only strict and clipping are supported for now"
            )
        return np.expand_dims(result[0], axis=0)

    else:
        pool = mp.Pool(int(mp.cpu_count() / 4))

        if projection_method == "strict":
            project_sample_to_equality_constraints_fn = (
                project_sample_to_equality_constraints_cvxpy
            )
        elif projection_method == "penalty_based":
            raise NotImplementedError(
                "Penalty based projection not supported yet"
            )
            # list_of_constraints = list(constraints.keys())
            # if "autocorr_12" in list_of_constraints:
            #     project_sample_to_equality_constraints_fn = (
            #         project_sample_to_minimize_scipy_penalty
            #     )
            # else:
            #     project_sample_to_equality_constraints_fn = (
            #         project_sample_to_minimize_penalty
            #     )
        else:
            raise ValueError(
                f"Invalid projection method: {projection_method} - only strict and clipping are supported for now"
            )
        results = pool.map(
            project_sample_to_equality_constraints_fn,
            per_sample_projection_inputs_list,
        )
        pool.close()
        # sort the results based on the second element of the tuple
        results = sorted(results, key=lambda x: x[1])
        projected_timeseries_list = [result[0] for result in results]
        # constraint_violation_list = [result[2] for result in results]

        # if penalty_coefficient > 999:
        #     print(f"Constraint violation: {constraint_violation_list}")

        projected_timeseries = np.stack(projected_timeseries_list)

        return projected_timeseries
