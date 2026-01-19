import argparse
import json
import subprocess
import sys

from loguru import logger

from synthefy_pkg.utils.basic_utils import (
    compare_system_specs,
    get_system_specs,
)

COMPILE = False

# Default threshold for significant differences
DEFAULT_THRESHOLD = 0.10  # 10% increase in any metric considered significant


def load_metrics(file_path):
    """Load metrics from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file at {file_path}.")
        sys.exit(1)


def compare_metrics(base_metrics, new_metrics, threshold):
    """Compare two sets of metrics and report all differences."""
    report = []
    has_significant_change = False

    for dataset in new_metrics.keys():
        if dataset not in base_metrics:
            print(f"Warning: Dataset '{dataset}' not found in base metrics.")
            continue

        for variable, metrics in base_metrics[dataset].items():
            if variable not in new_metrics[dataset]:
                print(
                    f"Warning: Variable '{variable}' not found in new metrics for dataset '{dataset}'."
                )
                continue

            for metric, base_value in metrics.items():
                if metric not in new_metrics[dataset][variable]:
                    print(
                        f"Warning: Metric '{metric}' not found for variable '{variable}' in dataset '{dataset}'."
                    )
                    continue

                new_value = new_metrics[dataset][variable][metric]

                # If base_value is zero, avoid division by zero
                if base_value == 0:
                    base_value = 1e-10

                # Calculate the relative difference
                diff = (new_value - base_value) / base_value

                # Check if the difference is significant
                if diff == 0:
                    direction = "unchanged"
                else:
                    direction = "improved" if diff < 0 else "worsened"
                is_significant = abs(diff) > threshold

                # Add all differences to the report
                report.append(
                    {
                        "dataset": dataset,
                        "variable": variable,
                        "metric": metric,
                        "base_value": base_value,
                        "new_value": new_value,
                        "difference_percent": diff * 100,
                        "direction": direction,
                        "is_significant": is_significant,
                    }
                )

                # Track if any significant changes exist
                if is_significant:
                    has_significant_change = True

    return report, has_significant_change


def print_report(report):
    """Print a report of all differences."""
    if report:
        print("Metric changes compared to base results:\n")
        for item in report:
            significance_marker = "(!)" if item["is_significant"] else ""
            print(
                f"Dataset: {item['dataset']}, Variable: {item['variable']}, Metric: {item['metric']} {significance_marker}"
            )
            print(f"  Base Value: {item['base_value']}")
            print(f"  New Value: {item['new_value']}")
            print(
                f"  Difference: {item['difference_percent']:.2f}% ({item['direction']})\n"
            )
    else:
        print("No metric changes detected compared to base results.")


def main(args):
    compare_system_specs(args.base_file)

    # Load base and new metrics from command line arguments
    base_metrics = load_metrics(args.base_file)
    new_metrics = load_metrics(args.new_file)

    # Use provided threshold or default
    threshold = DEFAULT_THRESHOLD
    if args.threshold is not None:
        threshold = float(args.threshold)

    # Compare and generate report
    report, has_significant_change = compare_metrics(
        base_metrics, new_metrics, threshold
    )
    print_report(report)

    # Exit with non-zero code only if significant differences are found
    if has_significant_change:
        logger.error(
            "Significant metric changes detected compared to base results."
        )
        sys.exit(1)
    else:
        logger.info(
            "No significant metric changes detected compared to base results."
        )


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Compare metrics between base and new JSON files."
    )
    parser.add_argument(
        "base_file", type=str, help="Path to the base metrics JSON file"
    )
    parser.add_argument(
        "new_file", type=str, help="Path to the new metrics JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=f"Threshold for significant differences (default: {DEFAULT_THRESHOLD} or {DEFAULT_THRESHOLD * 100}%%)",
    )
    args = parser.parse_args()
    main(args)

# Sample command to run this script:
"""
python3 compare_error_metrics_with_base.py \
    base_results/air_quality_all_metrics.json \
    <path_to_new_metrics_json_file> \
    --threshold 0.05
"""
