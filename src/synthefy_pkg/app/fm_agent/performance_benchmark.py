#!/usr/bin/env python3
"""
Performance benchmark script for comparing whatif tool performance
between regular LLM (gemini-2.5-flash-preview-05-20) and fast LLM (gemini-2.0-flash-lite).

This script runs actual LLM calls without mocking to measure:
1. Execution time
2. Accuracy of perturbation_type extraction
3. Success/failure rates
"""

import json
import os
import time
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

from synthefy_pkg.app.fm_agent.agent import (
    UIState,
    fast_llm,
    what_if_parameter_modification_tool_fast,
    what_if_parameter_modification_tool_for_testing,
)
from synthefy_pkg.app.fm_agent.agent import llm as regular_llm


class PerformanceBenchmark:
    """Performance benchmark for whatif tool comparing regular vs fast LLM."""

    def __init__(self):
        """Initialize the benchmark with test cases."""
        self.test_ui_state = UIState(
            user_id="test_user",
            file_path_key="test_file",
            dataset_columns=[
                "temperature",
                "unemployment_rate",
                "sales",
                "revenue",
                "price",
                "demand",
                "cost",
                "profit",
                "store_name",
                "region",
            ],
        )

        # Test cases with expected outcomes
        self.test_cases = [
            # Percentage increases
            {
                "prompt": "What if the sales increase 30%?",
                "expected_perturbation_type": "multiply",
                "expected_value": 1.3,
                "expected_column": "sales",
                "category": "percentage_increase",
            },
            {
                "prompt": "If revenue grows by 25 percent, what happens?",
                "expected_perturbation_type": "multiply",
                "expected_value": 1.25,
                "expected_column": "revenue",
                "category": "percentage_increase",
            },
            # Percentage decreases
            {
                "prompt": "If sales drop by 20 percent, what happens?",
                "expected_perturbation_type": "multiply",
                "expected_value": 0.8,
                "expected_column": "sales",
                "category": "percentage_decrease",
            },
            {
                "prompt": "What if revenue decreases by 15%?",
                "expected_perturbation_type": "multiply",
                "expected_value": 0.85,
                "expected_column": "revenue",
                "category": "percentage_decrease",
            },
            # Arithmetic operations
            {
                "prompt": "Assume revenue increases by 1000.",
                "expected_perturbation_type": "add",
                "expected_value": 1000,
                "expected_column": "revenue",
                "category": "arithmetic_add",
            },
            {
                "prompt": "What if we subtract 100 from the revenue?",
                "expected_perturbation_type": "subtract",
                "expected_value": 100,
                "expected_column": "revenue",
                "category": "arithmetic_subtract",
            },
            {
                "prompt": "If the cost doubles, what happens?",
                "expected_perturbation_type": "multiply",
                "expected_value": 2,
                "expected_column": "cost",
                "category": "arithmetic_multiply",
            },
            {
                "prompt": "What if the profit is divided by 2?",
                "expected_perturbation_type": "divide",
                "expected_value": 2,
                "expected_column": "profit",
                "category": "arithmetic_divide",
            },
            # Exact values
            {
                "prompt": "Set the price to 10 and the demand to 500.",
                "expected_perturbation_type": None,
                "expected_value": [10, 500],
                "expected_column": ["price", "demand"],
                "category": "exact_values",
            },
            {
                "prompt": "Set store_name to 'New York Store' and region to 'Northeast'.",
                "expected_perturbation_type": None,
                "expected_value": ["New York Store", "Northeast"],
                "expected_column": ["store_name", "region"],
                "category": "exact_string_values",
            },
            # Multiple modifications
            {
                "prompt": "If cost increases by 5 and profit drops by 10%, what happens?",
                "expected_perturbation_type": ["add", "multiply"],
                "expected_value": [5, 0.9],
                "expected_column": ["cost", "profit"],
                "category": "multiple_modifications",
            },
        ]

    def validate_result(
        self, result: Dict[str, Any], expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate if result matches expected outcome."""
        validation = {"is_valid": True, "errors": [], "score": 0.0}

        if "error" in result:
            validation["is_valid"] = False
            validation["errors"].append(f"Error in result: {result['error']}")
            return validation

        if (
            "parameter_updates" not in result
            or "full_modifications" not in result["parameter_updates"]
        ):
            validation["is_valid"] = False
            validation["errors"].append(
                "Missing parameter_updates or full_modifications"
            )
            return validation

        modifications = result["parameter_updates"]["full_modifications"]

        # Handle multiple modifications
        if isinstance(expected["expected_column"], list):
            expected_columns = expected["expected_column"]
            expected_types = (
                expected["expected_perturbation_type"]
                if isinstance(expected["expected_perturbation_type"], list)
                else [expected["expected_perturbation_type"]]
                * len(expected_columns)
            )
            if len(modifications) != len(expected_columns):
                validation["errors"].append(
                    f"Expected {len(expected_columns)} modifications, got {len(modifications)}"
                )
                validation["score"] = 0.0
                return validation

            score_components = []
            for i, expected_col in enumerate(expected_columns):
                found_match = False
                for mod in modifications:
                    if mod["name"] == expected_col:
                        found_match = True
                        # Check perturbation type
                        if mod["perturbation_type"] == expected_types[i]:
                            score_components.append(1.0)
                        else:
                            score_components.append(0.5)
                            validation["errors"].append(
                                f"Wrong perturbation_type for {expected_col}: expected {expected_types[i]}, got {mod['perturbation_type']}"
                            )
                        break

                if not found_match:
                    score_components.append(0.0)
                    validation["errors"].append(
                        f"Column {expected_col} not found in modifications"
                    )

            validation["score"] = (
                mean(score_components) if score_components else 0.0
            )
        else:
            # Single modification
            if len(modifications) != 1:
                validation["errors"].append(
                    f"Expected 1 modification, got {len(modifications)}"
                )
                validation["score"] = 0.0
                return validation

            mod = modifications[0]
            score = 0.0

            # Check column name
            if mod["name"] == expected["expected_column"]:
                score += 0.5
            else:
                validation["errors"].append(
                    f"Wrong column: expected {expected['expected_column']}, got {mod['name']}"
                )

            # Check perturbation type
            if (
                mod["perturbation_type"]
                == expected["expected_perturbation_type"]
            ):
                score += 0.5
            else:
                validation["errors"].append(
                    f"Wrong perturbation_type: expected {expected['expected_perturbation_type']}, got {mod['perturbation_type']}"
                )

            validation["score"] = score

        validation["is_valid"] = len(validation["errors"]) == 0
        return validation

    def run_single_test(
        self, test_case: Dict[str, Any], llm_instance, is_fast: bool
    ) -> Dict[str, Any]:
        """Run a single test case and measure performance."""
        start_time = time.time()

        try:
            if is_fast:
                result = what_if_parameter_modification_tool_for_testing(
                    test_case["prompt"], self.test_ui_state, llm_instance
                )
            else:
                result = what_if_parameter_modification_tool_for_testing(
                    test_case["prompt"], self.test_ui_state, llm_instance
                )

            execution_time = time.time() - start_time
            validation = self.validate_result(result, test_case)

            return {
                "prompt": test_case["prompt"],
                "category": test_case["category"],
                "execution_time": execution_time,
                "success": "error" not in result,
                "validation": validation,
                "raw_result": result,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "prompt": test_case["prompt"],
                "category": test_case["category"],
                "execution_time": execution_time,
                "success": False,
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "score": 0.0,
                },
                "raw_result": {"error": str(e)},
            }

    def run_benchmark(self, num_runs: int = 3) -> Dict[str, Any]:
        """Run the complete benchmark comparing regular vs fast LLM."""
        print("Starting Performance Benchmark...")
        print(f"Running {num_runs} iterations for each test case")
        print("=" * 60)

        # Check if LLMs are available
        if regular_llm is None:
            print(
                "ERROR: Regular LLM not available. Please check AI service configuration."
            )
            return {}

        if fast_llm is None:
            print(
                "ERROR: Fast LLM not available. Please check AI service configuration."
            )
            return {}

        results = {
            "regular_llm": {"results": [], "stats": {}},
            "fast_llm": {"results": [], "stats": {}},
        }

        # Run tests for regular LLM
        print("\n1. Testing Regular LLM (gemini-2.5-flash-preview-05-20)")
        print("-" * 50)
        for i, test_case in enumerate(self.test_cases):
            print(
                f"Test {i + 1}/{len(self.test_cases)}: {test_case['category']}"
            )

            run_results = []
            for run in range(num_runs):
                result = self.run_single_test(test_case, regular_llm, False)
                run_results.append(result)
                print(
                    f"  Run {run + 1}: {result['execution_time']:.2f}s, Score: {result['validation']['score']:.2f}"
                )

            results["regular_llm"]["results"].extend(run_results)

        # Run tests for fast LLM
        print("\n2. Testing Fast LLM (gemini-2.0-flash-lite)")
        print("-" * 50)
        for i, test_case in enumerate(self.test_cases):
            print(
                f"Test {i + 1}/{len(self.test_cases)}: {test_case['category']}"
            )

            run_results = []
            for run in range(num_runs):
                result = self.run_single_test(test_case, fast_llm, True)
                run_results.append(result)
                print(
                    f"  Run {run + 1}: {result['execution_time']:.2f}s, Score: {result['validation']['score']:.2f}"
                )

            results["fast_llm"]["results"].extend(run_results)

        # Calculate statistics
        for llm_type in ["regular_llm", "fast_llm"]:
            test_results = results[llm_type]["results"]

            execution_times = [r["execution_time"] for r in test_results]
            scores = [r["validation"]["score"] for r in test_results]
            success_rate = sum(1 for r in test_results if r["success"]) / len(
                test_results
            )

            results[llm_type]["stats"] = {
                "avg_execution_time": mean(execution_times),
                "std_execution_time": stdev(execution_times)
                if len(execution_times) > 1
                else 0,
                "avg_accuracy_score": mean(scores),
                "std_accuracy_score": stdev(scores) if len(scores) > 1 else 0,
                "success_rate": success_rate,
                "total_tests": len(test_results),
            }

        return results

    def print_comparison(self, results: Dict[str, Any]):
        """Print a detailed comparison of the results."""
        if not results:
            print("No results to compare.")
            return

        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON RESULTS")
        print("=" * 60)

        regular_stats = results["regular_llm"]["stats"]
        fast_stats = results["fast_llm"]["stats"]

        print(
            f"\n{'Metric':<25} {'Regular LLM':<20} {'Fast LLM':<20} {'Speedup':<15}"
        )
        print("-" * 80)

        # Execution time comparison
        regular_time = regular_stats["avg_execution_time"]
        fast_time = fast_stats["avg_execution_time"]
        speedup = regular_time / fast_time if fast_time > 0 else float("inf")

        print(
            f"{'Avg Execution Time':<25} {regular_time:.3f}s ± {regular_stats['std_execution_time']:.3f} {fast_time:.3f}s ± {fast_stats['std_execution_time']:.3f} {speedup:.2f}x"
        )

        # Accuracy comparison
        regular_acc = regular_stats["avg_accuracy_score"]
        fast_acc = fast_stats["avg_accuracy_score"]
        acc_diff = fast_acc - regular_acc

        print(
            f"{'Avg Accuracy Score':<25} {regular_acc:.3f} ± {regular_stats['std_accuracy_score']:.3f} {fast_acc:.3f} ± {fast_stats['std_accuracy_score']:.3f} {acc_diff:+.3f}"
        )

        # Success rate comparison
        regular_success_rate_str = f"{regular_stats['success_rate']:.1%}"
        fast_success_rate_str = f"{fast_stats['success_rate']:.1%}"
        success_rate_diff_str = (
            f"{fast_stats['success_rate'] - regular_stats['success_rate']:+.1%}"
        )

        print(
            f"{'Success Rate':<25} {regular_success_rate_str:<19} {fast_success_rate_str:<19} {success_rate_diff_str}"
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if speedup > 1:
            print(f"✅ Fast LLM is {speedup:.2f}x faster than regular LLM")
        else:
            print(f"❌ Fast LLM is {1 / speedup:.2f}x slower than regular LLM")

        if acc_diff >= -0.05:  # Allow 5% accuracy drop
            print(
                f"✅ Fast LLM maintains accuracy (difference: {acc_diff:+.1%})"
            )
        else:
            print(
                f"⚠️  Fast LLM has lower accuracy (difference: {acc_diff:+.1%})"
            )

        if fast_stats["success_rate"] >= regular_stats["success_rate"]:
            print("✅ Fast LLM maintains success rate")
        else:
            print("⚠️  Fast LLM has lower success rate")

    def save_results(
        self,
        results: Dict[str, Any],
        filename: str = "whatif_benchmark_results.json",
    ):
        """Save detailed results to a JSON file."""
        try:
            # Convert results to JSON-serializable format
            json_results = json.dumps(results, indent=2, default=str)
            with open(filename, "w") as f:
                f.write(json_results)
            print(f"\n📁 Detailed results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main function to run the benchmark."""
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ AI service configuration not set.")
        print("Please configure the service before running the benchmark.")
        return

    benchmark = PerformanceBenchmark()

    # Run the benchmark
    results = benchmark.run_benchmark(num_runs=3)

    if results:
        # Print comparison
        benchmark.print_comparison(results)

        # Save detailed results
        benchmark.save_results(results)
    else:
        print("❌ Benchmark failed to run. Check LLM configuration.")


if __name__ == "__main__":
    main()
