import json
import os
from typing import Any, Dict, List, Tuple

from loguru import logger

COMPILE = False


def save_anomaly_results(
    results: Dict[str, Dict[str, Dict[str, List[Any]]]],
    concurrent_results: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    filename: str,
) -> Tuple[str, str]:
    """
    Save anomaly detection results and concurrent anomalies to JSON files.

    Args:
        results: Dictionary containing individual anomaly results by KPI/type/group
        concurrent_results: Dictionary containing concurrent anomaly results
        output_path: Directory path where to save the results
        filename: Base filename for the results (without extension)

    Returns:
        Tuple containing:
        - Path to the main results file
        - Path to the concurrent results file

    Example:
        >>> results = {...}  # Your anomaly detection results
        >>> concurrent_results = {...}  # Your concurrent anomalies
        >>> output_path = "path/to/output"
        >>> filename = "anomaly_detection_results"
        >>> main_file, concurrent_file = save_anomaly_results(
        ...     results, concurrent_results, output_path, filename
        ... )
    """
    # Create timestamp for the filename
    output_file = os.path.join(output_path, f"{filename}.json")
    output_file_concurrent = os.path.join(output_path, f"{filename}_concurrent.json")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for kpi, kpi_results in results.items():
        json_results[kpi] = {}
        for anomaly_type, type_results in kpi_results.items():
            json_results[kpi][anomaly_type] = {}
            for group_key, anomalies in type_results.items():
                json_results[kpi][anomaly_type][group_key] = [
                    {
                        "timestamp": anomaly.timestamp.isoformat(),
                        "score": float(anomaly.score),
                        "original_value": float(anomaly.original_value),
                        "predicted_value": float(anomaly.predicted_value),
                        "group_metadata": anomaly.group_metadata,
                    }
                    for anomaly in anomalies
                ]

    # Save main results
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved anomaly detection results to: {output_file}")

    # Save concurrent anomalies
    with open(output_file_concurrent, "w") as f:
        json.dump(concurrent_results, f, indent=2)
    logger.info(f"Saved concurrent anomaly results to: {output_file_concurrent}")

    return output_file, output_file_concurrent
