"""
Utility functions for logging curriculum parameters to MLflow and other logging systems.
"""

import time
from typing import Any, Dict, Optional

import mlflow
from loguru import logger


def log_curriculum_parameters_to_mlflow(
    updated_values: Dict[str, Any],
    global_step: int,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Log curriculum parameters to MLflow.

    Args:
        updated_values: Dictionary containing curriculum parameter updates
        global_step: Current global training step
        experiment_name: MLflow experiment name (optional)
        tracking_uri: MLflow tracking URI (optional)
        run_id: MLflow run ID (optional, for use with Lightning)
    """
    try:
        # If we have a run_id, use it directly without setting experiment/tracking URI
        if run_id:
            logger.debug(
                f"Logging curriculum parameters to MLflow at step {global_step} using run_id {run_id}: {updated_values}"
            )

            for param_name, param_value in updated_values.items():
                if isinstance(param_value, dict):
                    # Handle nested dictionary parameters (e.g., distribution parameters)
                    for nested_key, nested_value in param_value.items():
                        if isinstance(nested_value, (int, float)):
                            metric_name = (
                                f"curriculum_{param_name}_{nested_key}"
                            )
                            mlflow.log_metric(
                                metric_name,
                                nested_value,
                                step=global_step,
                                run_id=run_id,
                            )
                            logger.debug(
                                f"Logged curriculum metric: {metric_name} = {nested_value}"
                            )
                elif isinstance(param_value, (int, float)):
                    # Handle simple scalar parameters
                    metric_name = f"curriculum_{param_name}"
                    mlflow.log_metric(
                        metric_name,
                        param_value,
                        step=global_step,
                        run_id=run_id,
                    )
                    logger.debug(
                        f"Logged curriculum metric: {metric_name} = {param_value}"
                    )

            logger.debug(
                f"Successfully logged {len(updated_values)} curriculum parameters to MLflow"
            )
            return

        # Fallback to setting experiment and tracking URI (for non-Lightning usage)
        # Set experiment and tracking URI if provided
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Check if we're in an active MLflow run or have a run_id
        active_run = mlflow.active_run()
        if not active_run and not run_id:
            logger.warning(
                "No active MLflow run found and no run_id provided. Curriculum parameters will not be logged."
            )
            return

        # If we have a run_id but no active run, set the run_id
        if run_id and not active_run:
            mlflow.set_tracking_uri(tracking_uri or "http://localhost:5000")
            mlflow.set_experiment(experiment_name or "Default")

        logger.debug(
            f"Logging curriculum parameters to MLflow at step {global_step}: {updated_values}"
        )

        for param_name, param_value in updated_values.items():
            if isinstance(param_value, dict):
                # Handle nested dictionary parameters (e.g., distribution parameters)
                for nested_key, nested_value in param_value.items():
                    if isinstance(nested_value, (int, float)):
                        metric_name = f"curriculum_{param_name}_{nested_key}"
                        if run_id:
                            mlflow.log_metric(
                                metric_name,
                                nested_value,
                                step=global_step,
                                run_id=run_id,
                            )
                        else:
                            mlflow.log_metric(
                                metric_name, nested_value, step=global_step
                            )
                        logger.debug(
                            f"Logged curriculum metric: {metric_name} = {nested_value}"
                        )
            elif isinstance(param_value, (int, float)):
                # Handle simple scalar parameters
                metric_name = f"curriculum_{param_name}"
                if run_id:
                    mlflow.log_metric(
                        metric_name,
                        param_value,
                        step=global_step,
                        run_id=run_id,
                    )
                else:
                    mlflow.log_metric(
                        metric_name, param_value, step=global_step
                    )
                logger.debug(
                    f"Logged curriculum metric: {metric_name} = {param_value}"
                )

        logger.debug(
            f"Successfully logged {len(updated_values)} curriculum parameters to MLflow"
        )
    except Exception as e:
        logger.error(f"Failed to log curriculum parameters to MLflow: {e}")
        logger.error(f"Updated values: {updated_values}")
        logger.error(f"Global step: {global_step}")
        logger.error(f"Run ID: {run_id}")


def get_mlflow_run_id_from_trainer(trainer) -> Optional[str]:
    """
    Get the MLflow run ID from a Lightning trainer.

    Args:
        trainer: Lightning trainer instance

    Returns:
        MLflow run ID if available, None otherwise
    """
    try:
        if trainer and trainer.loggers:
            for logger_instance in trainer.loggers:
                if (
                    hasattr(logger_instance, "run_id")
                    and logger_instance.run_id
                ):
                    return logger_instance.run_id
        return None
    except Exception as e:
        logger.warning(f"Failed to get MLflow run ID from trainer: {e}")
        return None


def format_curriculum_parameters_for_logging(
    updated_values: Dict[str, Any],
) -> Dict[str, float]:
    """
    Format curriculum parameters into a flat dictionary suitable for logging.

    Args:
        updated_values: Dictionary containing curriculum parameter updates

    Returns:
        Flattened dictionary with metric names as keys and values as floats
    """
    formatted_params = {}

    for param_name, param_value in updated_values.items():
        if isinstance(param_value, dict):
            # Handle nested dictionary parameters
            for nested_key, nested_value in param_value.items():
                if isinstance(nested_value, (int, float)):
                    metric_name = f"curriculum_{param_name}_{nested_key}"
                    formatted_params[metric_name] = float(nested_value)
        elif isinstance(param_value, (int, float)):
            # Handle simple scalar parameters
            metric_name = f"curriculum_{param_name}"
            formatted_params[metric_name] = float(param_value)

    return formatted_params


def log_curriculum_parameters_to_mlflow_with_retry(
    updated_values: Dict[str, Any],
    global_step: int,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> None:
    """
    Log curriculum parameters to MLflow with retry logic for Lightning compatibility.

    This function will retry logging if no active MLflow run is found, which can happen
    when Lightning hasn't started the MLflow run yet.

    Args:
        updated_values: Dictionary containing curriculum parameter updates
        global_step: Current global training step
        experiment_name: MLflow experiment name (optional)
        tracking_uri: MLflow tracking URI (optional)
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            # Set experiment and tracking URI if provided
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            # Check if we're in an active MLflow run
            active_run = mlflow.active_run()
            if active_run:
                logger.info(
                    f"Found active MLflow run: {active_run.info.run_id}"
                )
                log_curriculum_parameters_to_mlflow(
                    updated_values, global_step, experiment_name, tracking_uri
                )
                return
            else:
                if attempt < max_retries - 1:
                    logger.trace(
                        f"No active MLflow run found (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.warning(
                        f"No active MLflow run found after {max_retries} attempts. Curriculum parameters will not be logged."
                    )
                    return

        except Exception as e:
            if attempt < max_retries - 1:
                logger.trace(
                    f"Error logging curriculum parameters (attempt {attempt + 1}/{max_retries}): {e}, retrying..."
                )
                time.sleep(retry_delay)
            else:
                logger.exception(
                    f"Failed to log curriculum parameters after {max_retries} attempts: {e}"
                )
                return
