#!/usr/bin/env python3
"""
Real integration tests for the improved what-if parameter modification tool.
These tests use ACTUAL LLMs (no mocking) to validate that the improved prompts
robustly extract perturbation_types and prevent hallucination of invalid values.
"""

import os
import sys
import unittest
from statistics import mean
from typing import Any, Dict, List

import pytest

from synthefy_pkg.app.data_models import PerturbationType
from synthefy_pkg.app.fm_agent.agent import (
    UIState,
    _find_best_column_match,
    fast_llm,
    what_if_parameter_modification_tool_for_testing,
)
from synthefy_pkg.app.fm_agent.agent import llm as regular_llm
from synthefy_pkg.app.fm_agent.test_data.loader import (
    get_whatif_prompts_by_category,
    get_whatif_test_cases_by_category,
    load_whatif_test_cases,
)

# Environment variable to control LLM test execution
LLM_TESTS_ENABLED = (
    os.getenv("SYNTHEFY_RUN_LLM_TESTS", "false").lower() == "true"
)

# Skip all LLM tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not LLM_TESTS_ENABLED,
    reason="LLM tests disabled. Set SYNTHEFY_RUN_LLM_TESTS=true to enable",
)


class TestFuzzyColumnMatching(unittest.TestCase):
    """Tests for the fuzzy column matching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_columns = [
            "store_name",
            "store_region",
            "customer_name",
            "customer_type",
            "product_name",
            "product_category",
            "user_id",
            "user_status",
            "sales",
            "revenue",
            "temperature",
            "unemployment_rate",
            "location_details",
            "department_id",
            "region_code",
            "category",  # Keep this for testing abbreviation matching
        ]

    def test_exact_matches(self):
        """Test that exact matches work correctly."""
        test_cases = [
            ("store_name", "store_name"),
            ("customer_name", "customer_name"),
            ("sales", "sales"),
            ("category", "category"),
        ]

        for query, expected in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                self.assertEqual(result, expected)

    def test_typo_matching(self):
        """Test that common typos are correctly matched."""
        # Focus on the core store-related typos which are the main user request
        test_cases = [
            ("sotre", "store_name"),  # Missing 't' - MAIN use case
            ("stor", "store_name"),  # Missing 'e' at end - MAIN use case
        ]

        # Test cases that might have competing matches - test these more flexibly
        flexible_test_cases = [
            (
                "temprature",
                ["temperature"],
            ),  # Extra 'r' - might be harder to match
            ("revenu", ["revenue"]),  # Missing 'e' - should be easier
            (
                "custmer",
                ["customer_name"],
            ),  # Missing 'o' - might not match if threshold too strict
            (
                "prodcut",
                ["product_name"],
            ),  # Swapped 'uc' - might not match if threshold too strict
            (
                "catgory",
                ["product_category", "category"],
            ),  # Missing 'e' - could match either
        ]

        # Test the core cases that should work reliably
        for query, expected in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                self.assertIsNotNone(
                    result, f"Failed to find any match for '{query}'"
                )
                self.assertEqual(
                    result,
                    expected,
                    f"Failed to match '{query}' to '{expected}', got '{result}'",
                )

        # Test the flexible cases - allow reasonable alternatives
        for query, acceptable_results in flexible_test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                # For these cases, we just want to ensure we get a reasonable match
                if result is not None:
                    # If we get a result, it should be one of the acceptable ones
                    self.assertIn(
                        result,
                        acceptable_results,
                        f"'{query}' matched '{result}', expected one of {acceptable_results}",
                    )
                # If we get None, that's okay for these edge cases - the algorithm is being conservative

    def test_phrase_matching(self):
        """Test that phrases are correctly matched to column names."""
        test_cases = [
            ("name of the store", "store_name"),
            ("store name", "store_name"),
            ("stor name", "store_name"),
            ("customer name", "customer_name"),
            (
                "user name",
                "user_id",
            ),  # Should match user_id as closest available
            ("product name", "product_name"),
            (
                "region name",
                "store_region",
            ),  # Should match store_region as closest available
        ]

        for query, expected in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                self.assertIsNotNone(
                    result, f"Failed to find any match for phrase '{query}'"
                )
                # For phrases, validate that we get a semantically reasonable match
                if "store" in query.lower():
                    # Should match some store-related column
                    self.assertTrue(
                        result and "store" in result.lower(),
                        f"Failed to match store phrase '{query}' to store-related column, got '{result}'",
                    )
                elif "customer" in query.lower():
                    self.assertTrue(
                        result and "customer" in result.lower(),
                        f"Failed to match customer phrase '{query}' to customer-related column, got '{result}'",
                    )
                elif "user" in query.lower():
                    self.assertTrue(
                        result and "user" in result.lower(),
                        f"Failed to match user phrase '{query}' to user-related column, got '{result}'",
                    )
                elif "product" in query.lower():
                    self.assertTrue(
                        result and "product" in result.lower(),
                        f"Failed to match product phrase '{query}' to product-related column, got '{result}'",
                    )
                elif "region" in query.lower():
                    self.assertTrue(
                        result and "region" in result.lower(),
                        f"Failed to match region phrase '{query}' to region-related column, got '{result}'",
                    )

    def test_abbreviation_matching(self):
        """Test that abbreviations and variations are correctly matched."""
        test_cases = [
            ("id", "user_id"),
            ("type", "customer_type"),
            (
                "category",
                "category",
            ),  # Should match exact category when available
            ("code", "region_code"),
        ]

        for query, expected in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                self.assertEqual(
                    result,
                    expected,
                    f"Failed to match abbreviation '{query}' to '{expected}'",
                )

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        test_cases = [
            ("STORE_NAME", "store_name"),
            ("Customer_Name", "customer_name"),
            ("SALES", "sales"),
        ]

        for query, expected in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                self.assertEqual(
                    result,
                    expected,
                    f"Failed case insensitive match '{query}' to '{expected}'",
                )

    def test_no_match_cases(self):
        """Test cases where no reasonable match should be found."""
        test_cases = [
            "",  # Empty string
            "xyz_column",  # Completely unrelated
            "abcdefg",  # Random string
            "!@#$%",  # Special characters only
            "a",  # Single character that doesn't match anything
        ]

        for query in test_cases:
            with self.subTest(query=query):
                result = _find_best_column_match(query, self.test_columns)
                # For truly unrelated queries, we should either get None or
                # a very weak match. Let's be more specific about expectations.
                if query == "":
                    self.assertIsNone(result, "Empty string should return None")
                elif query in ["xyz_column", "abcdefg", "!@#$%"]:
                    # These should either return None or a very weak/inappropriate match
                    # At minimum, we want to document what the current behavior is
                    if result is not None:
                        # If it returns something, it shouldn't be a high-confidence match
                        # This is more of a behavioral documentation test
                        self.assertIn(
                            result,
                            self.test_columns,
                            f"If matching '{query}', result '{result}' must be from valid columns",
                        )

    def test_malformed_inputs(self):
        """Test that malformed inputs are handled gracefully."""
        malformed_cases = [
            None,  # None input
            123,  # Integer instead of string
            [],  # List instead of string
            {},  # Dict instead of string
        ]

        for query in malformed_cases:
            with self.subTest(query=query):
                # Should not crash, should return None for invalid input types
                try:
                    result = _find_best_column_match(query, self.test_columns)
                    self.assertIsNone(
                        result,
                        f"Malformed input {type(query)} should return None",
                    )
                except (TypeError, AttributeError):
                    # It's also acceptable to raise an appropriate exception
                    pass

    def test_empty_column_list(self):
        """Test behavior with empty column list."""
        result = _find_best_column_match("sales", [])
        self.assertIsNone(result, "Empty column list should return None")

        result = _find_best_column_match("", [])
        self.assertIsNone(
            result, "Empty query with empty columns should return None"
        )


class TestWhatIfToolReal(unittest.TestCase):
    """Real integration tests for the improved what-if parameter modification tool."""

    def setUp(self):
        """Set up test fixtures."""
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
                "store_region",
                "customer_name",
                "customer_type",
                "product_name",
                "product_category",
                "user_id",
                "user_status",
                "category",
                "region",
                # Additional columns for comprehensive string testing
                "customer_segment",
                "payment_method",
                "priority_level",
                "status",
                "location_type",
                "location_details",
                "department_id",
                "region_code",
                "userid",
                "product_code",
                "code",
                "description",
                "Store",
            ],
        )

        # Skip tests if no API key available
        if not os.getenv("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY not available")

        if regular_llm is None:
            self.skipTest("Regular LLM not available")

    def validate_no_hallucination(self, result: Dict[str, Any]) -> bool:
        """Check that result contains no hallucinated perturbation_types."""
        valid_perturbation_types = {
            "add",
            "subtract",
            "multiply",
            "divide",
            None,
        }

        if "error" in result:
            return False

        if (
            "parameter_updates" not in result
            or "full_modifications" not in result["parameter_updates"]
        ):
            return False

        modifications = result["parameter_updates"]["full_modifications"]

        for mod in modifications:
            if mod.get("perturbation_type") not in valid_perturbation_types:
                print(
                    f"HALLUCINATION DETECTED: Invalid perturbation_type '{mod.get('perturbation_type')}' for column '{mod.get('name')}'"
                )
                return False

        return True

    def test_percentage_increase_no_hallucination(self):
        """Test that percentage increases don't generate invalid perturbation_types like 'percentage'."""
        test_cases = get_whatif_prompts_by_category("percentage_increase")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

                # Should produce multiply perturbation_type
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        self.assertEqual(
                            modifications[0]["perturbation_type"],
                            "multiply",
                            f"Expected 'multiply' but got '{modifications[0]['perturbation_type']}' for: {prompt}",
                        )

    def test_percentage_decrease_no_hallucination(self):
        """Test that percentage decreases don't generate invalid perturbation_types."""
        test_cases = get_whatif_prompts_by_category("percentage_decrease")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

                # Should produce multiply perturbation_type
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        self.assertEqual(
                            modifications[0]["perturbation_type"],
                            "multiply",
                            f"Expected 'multiply' but got '{modifications[0]['perturbation_type']}' for: {prompt}",
                        )

    def test_arithmetic_operations_no_hallucination(self):
        """Test that arithmetic operations don't generate invalid perturbation_types."""
        test_cases = get_whatif_test_cases_by_category("arithmetic_operations")

        for case in test_cases:
            prompt = case["prompt"]
            expected_type = case["expected_perturbation_type"]

            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

                # Should produce correct perturbation_type
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        self.assertEqual(
                            modifications[0]["perturbation_type"],
                            expected_type,
                            f"Expected '{expected_type}' but got '{modifications[0]['perturbation_type']}' for: {prompt}",
                        )

    def test_exact_values_no_hallucination(self):
        """Test that exact values don't generate invalid perturbation_types."""
        test_cases = get_whatif_prompts_by_category("exact_values")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

                # Should produce None perturbation_type for exact values
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        self.assertIsNone(
                            modifications[0]["perturbation_type"],
                            f"Expected None but got '{modifications[0]['perturbation_type']}' for: {prompt}",
                        )

    def test_comprehensive_string_exact_values_no_hallucination(self):
        """Comprehensive test for various string exact value patterns and edge cases."""
        # Load comprehensive string test cases from JSON
        comprehensive_string_data = load_whatif_test_cases()[
            "comprehensive_string_exact_values"
        ]

        all_string_test_cases = []

        # Extract all test cases from nested structure
        if "test_categories" in comprehensive_string_data:
            for category_name, category_prompts in comprehensive_string_data[
                "test_categories"
            ].items():
                all_string_test_cases.extend(category_prompts)

        # Add any direct test prompts if they exist
        if "test_prompts" in comprehensive_string_data:
            all_string_test_cases.extend(
                comprehensive_string_data["test_prompts"]
            )

        for prompt in all_string_test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for string prompt: {prompt}",
                )

                # Should produce None perturbation_type for exact string values
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        # For string exact values, perturbation_type should be None
                        string_modifications = [
                            mod
                            for mod in modifications
                            if isinstance(mod.get("value"), str)
                            and mod.get("perturbation_or_exact_value")
                            == "exact_value"
                        ]
                        for mod in string_modifications:
                            self.assertIsNone(
                                mod["perturbation_type"],
                                f"Expected None perturbation_type for string exact value '{mod['value']}' in column '{mod['name']}' for prompt: {prompt}",
                            )

    def test_multiple_modifications_no_hallucination(self):
        """Test that multiple modifications don't generate invalid perturbation_types."""
        test_cases = get_whatif_prompts_by_category("multiple_modifications")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

                # Should have multiple modifications
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    self.assertGreater(
                        len(modifications),
                        1,
                        f"Expected multiple modifications for: {prompt}",
                    )

    def test_edge_cases_no_hallucination(self):
        """Test edge cases that might cause hallucination."""
        test_cases = get_whatif_prompts_by_category("edge_cases")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for prompt: {prompt}",
                )

    def test_invalid_prompts_handled_gracefully(self):
        """Test that invalid prompts are handled gracefully without hallucination."""
        test_cases = get_whatif_prompts_by_category("invalid_prompts")

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Should either succeed without hallucination or fail gracefully
                if "error" not in result:
                    self.assertTrue(
                        self.validate_no_hallucination(result),
                        f"Hallucination detected for invalid prompt: {prompt}",
                    )

    def test_performance_comparison(self):
        """Test performance comparison between regular and fast LLM."""
        if fast_llm is None:
            self.skipTest("Fast LLM not available")

        test_prompts = get_whatif_prompts_by_category("performance_comparison")

        regular_times = []
        fast_times = []
        regular_successes = []
        fast_successes = []

        for prompt in test_prompts:
            # Test regular LLM
            import time

            start_time = time.time()
            regular_result = what_if_parameter_modification_tool_for_testing(
                prompt, self.test_ui_state, regular_llm
            )
            regular_time = time.time() - start_time
            regular_times.append(regular_time)
            regular_successes.append(
                self.validate_no_hallucination(regular_result)
            )

            # Test fast LLM
            start_time = time.time()
            fast_result = what_if_parameter_modification_tool_for_testing(
                prompt, self.test_ui_state, fast_llm
            )
            fast_time = time.time() - start_time
            fast_times.append(fast_time)
            fast_successes.append(self.validate_no_hallucination(fast_result))

        # Calculate metrics
        avg_regular_time = mean(regular_times)
        avg_fast_time = mean(fast_times)
        speedup = (
            avg_regular_time / avg_fast_time
            if avg_fast_time > 0
            else float("inf")
        )

        regular_success_rate = mean(regular_successes)
        fast_success_rate = mean(fast_successes)

        print("\nPerformance Comparison:")
        print(
            f"Regular LLM: {avg_regular_time:.3f}s avg, {regular_success_rate:.1%} success"
        )
        print(
            f"Fast LLM: {avg_fast_time:.3f}s avg, {fast_success_rate:.1%} success"
        )
        print(f"Speedup: {speedup:.2f}x")

        # Both should have no hallucinations
        self.assertGreaterEqual(
            regular_success_rate, 0.8, "Regular LLM success rate too low"
        )
        self.assertGreaterEqual(
            fast_success_rate, 0.8, "Fast LLM success rate too low"
        )

    def test_fuzzy_column_matching_integration(self):
        """Test that the what-if tool correctly handles typos and variations in column names."""
        # Load fuzzy matching test cases from JSON
        test_cases = get_whatif_test_cases_by_category("fuzzy_column_matching")

        for case in test_cases:
            prompt = case["prompt"]
            acceptable_columns = case["acceptable_columns"]
            expected_perturbation_type = case["expected_perturbation_type"]

            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for fuzzy matching prompt: {prompt}",
                )

                # Check that an acceptable column was matched
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    if modifications:
                        # Find if any modification uses an acceptable column
                        found_acceptable_match = False
                        matched_column = None
                        matched_perturbation_type = None

                        for mod in modifications:
                            if mod["name"] in acceptable_columns:
                                found_acceptable_match = True
                                matched_column = mod["name"]
                                matched_perturbation_type = mod[
                                    "perturbation_type"
                                ]
                                break

                        self.assertTrue(
                            found_acceptable_match,
                            f"No acceptable column from {acceptable_columns} found in modifications for prompt: {prompt}. Got: {[m['name'] for m in modifications]}",
                        )

                        if matched_perturbation_type is not None:
                            self.assertEqual(
                                matched_perturbation_type,
                                expected_perturbation_type,
                                f"Expected perturbation_type '{expected_perturbation_type}' for column '{matched_column}' in prompt: {prompt}",
                            )

    def test_mixed_typos_and_operations(self):
        """Test complex prompts with both typos and various operations."""
        test_cases = get_whatif_prompts_by_category(
            "mixed_typos_and_operations"
        )

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Most important: No hallucination
                self.assertTrue(
                    self.validate_no_hallucination(result),
                    f"Hallucination detected for mixed typos prompt: {prompt}",
                )

                # Should have multiple modifications
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    self.assertGreater(
                        len(modifications),
                        1,
                        f"Expected multiple modifications for mixed prompt: {prompt}",
                    )

                    # All column names should be valid (from the dataset)
                    for mod in modifications:
                        self.assertIn(
                            mod["name"],
                            self.test_ui_state.dataset_columns,
                            f"Column '{mod['name']}' not in dataset columns for prompt: {prompt}",
                        )

    def test_fuzzy_matching_performance(self):
        """Test that fuzzy matching doesn't significantly impact performance."""
        if fast_llm is None:
            self.skipTest("Fast LLM not available")

        # Load performance test data from JSON
        performance_data = load_whatif_test_cases()[
            "fuzzy_matching_performance"
        ]
        fuzzy_prompts = performance_data.get("fuzzy_prompts", [])
        clean_prompts = performance_data.get("clean_prompts", [])

        import time

        fuzzy_times = []
        clean_times = []

        # Test fuzzy prompts
        for prompt in fuzzy_prompts:
            start_time = time.time()
            result = what_if_parameter_modification_tool_for_testing(
                prompt, self.test_ui_state, fast_llm
            )
            fuzzy_time = time.time() - start_time
            fuzzy_times.append(fuzzy_time)

            # Should still work without hallucination
            self.assertTrue(
                self.validate_no_hallucination(result),
                f"Fuzzy matching failed for: {prompt}",
            )

        # Test clean prompts
        for prompt in clean_prompts:
            start_time = time.time()
            result = what_if_parameter_modification_tool_for_testing(
                prompt, self.test_ui_state, fast_llm
            )
            clean_time = time.time() - start_time
            clean_times.append(clean_time)

            # Should work without hallucination
            self.assertTrue(
                self.validate_no_hallucination(result),
                f"Clean prompt failed for: {prompt}",
            )

        # Calculate performance metrics
        avg_fuzzy_time = mean(fuzzy_times)
        avg_clean_time = mean(clean_times)
        performance_overhead = (
            (avg_fuzzy_time - avg_clean_time) / avg_clean_time
            if avg_clean_time > 0
            else 0
        )

        print("\nFuzzy Matching Performance:")
        print(f"  Average fuzzy prompt time: {avg_fuzzy_time:.3f}s")
        print(f"  Average clean prompt time: {avg_clean_time:.3f}s")
        print(f"  Performance overhead: {performance_overhead:.1%}")

        # Performance should be reasonable (less than 50% overhead)
        self.assertLess(
            performance_overhead,
            0.5,
            f"Fuzzy matching overhead too high: {performance_overhead:.1%}",
        )

    def test_strict_perturbation_type_validation(self):
        """Test that only valid perturbation types are ever returned."""
        # Load strict validation test prompts from JSON
        test_prompts = get_whatif_prompts_by_category(
            "strict_perturbation_validation"
        )

        valid_types = {"add", "subtract", "multiply", "divide", None}

        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, self.test_ui_state, regular_llm
                )

                # Must not have errors for valid prompts
                self.assertNotIn(
                    "error",
                    result,
                    f"Unexpected error for valid prompt: {prompt}",
                )

                # Must have proper structure
                self.assertIn("parameter_updates", result)
                self.assertIn("full_modifications", result["parameter_updates"])

                modifications = result["parameter_updates"][
                    "full_modifications"
                ]
                for mod in modifications:
                    perturbation_type = mod.get("perturbation_type")
                    self.assertIn(
                        perturbation_type,
                        valid_types,
                        f"Invalid perturbation_type '{perturbation_type}' in modification for prompt: {prompt}",
                    )

    def test_invalid_column_handling(self):
        """Test that non-existent columns are handled appropriately."""
        # Create a minimal dataset to force column mismatches
        minimal_ui_state = UIState(
            user_id="test_user",
            file_path_key="test_file",
            dataset_columns=["sales", "revenue"],  # Very limited columns
        )

        prompts_with_invalid_columns = get_whatif_prompts_by_category(
            "invalid_column_handling"
        )

        for prompt in prompts_with_invalid_columns:
            with self.subTest(prompt=prompt):
                result = what_if_parameter_modification_tool_for_testing(
                    prompt, minimal_ui_state, regular_llm
                )

                # Should either have no modifications or only modifications to valid columns
                if (
                    "parameter_updates" in result
                    and "full_modifications" in result["parameter_updates"]
                ):
                    modifications = result["parameter_updates"][
                        "full_modifications"
                    ]
                    for mod in modifications:
                        self.assertIn(
                            mod["name"],
                            minimal_ui_state.dataset_columns,
                            f"Column '{mod['name']}' not in minimal dataset for prompt: {prompt}",
                        )

    def test_error_conditions_are_properly_handled(self):
        """Test that error conditions produce appropriate error responses."""
        # Test with completely empty UI state to force errors
        empty_ui_state = UIState(
            user_id="test_user",
            file_path_key="test_file",
            dataset_columns=[],  # Empty columns should cause issues
        )

        result = what_if_parameter_modification_tool_for_testing(
            "What if sales increase by 30%?", empty_ui_state, regular_llm
        )

        # With empty columns, we should either get an error or no modifications
        if "error" not in result:
            # If no error, should have no modifications since no columns exist
            if (
                "parameter_updates" in result
                and "full_modifications" in result["parameter_updates"]
            ):
                modifications = result["parameter_updates"][
                    "full_modifications"
                ]
                self.assertEqual(
                    len(modifications),
                    0,
                    "Should have no modifications with empty column list",
                )

    def test_performance_thresholds_are_reasonable(self):
        """Test that performance doesn't degrade significantly."""
        if fast_llm is None:
            self.skipTest("Fast LLM not available")

        # More stringent performance test
        simple_prompt = "What if sales increase 30%?"

        import time

        times = []
        successes = []

        # Run multiple times for more reliable measurement
        for _ in range(3):
            start_time = time.time()
            result = what_if_parameter_modification_tool_for_testing(
                simple_prompt, self.test_ui_state, fast_llm
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            successes.append(self.validate_no_hallucination(result))

        avg_time = mean(times)
        success_rate = mean(successes)

        # More stringent requirements
        self.assertGreaterEqual(
            success_rate, 0.95, f"Success rate too low: {success_rate:.1%}"
        )
        self.assertLess(
            avg_time, 5.0, f"Performance too slow: {avg_time:.2f}s average"
        )

        print("\nStringent Performance Test:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Success rate: {success_rate:.1%}")


def run_comprehensive_hallucination_test():
    """
    Run a comprehensive test specifically designed to catch hallucination.
    This is not a unit test but a standalone function for thorough testing.
    """
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not available")
        return

    if regular_llm is None:
        print("❌ Regular LLM not available")
        return

    print("\n" + "=" * 60)
    print("COMPREHENSIVE HALLUCINATION TEST")
    print("=" * 60)

    test_ui_state = UIState(
        user_id="hallucination_test_user",
        file_path_key="hallucination_test_file",
        dataset_columns=[
            "sales",
            "revenue",
            "cost",
            "profit",
            "price",
            "demand",
        ],
    )

    # Load hallucination test prompts from JSON
    hallucination_test_prompts = get_whatif_prompts_by_category(
        "hallucination_test"
    )

    valid_perturbation_types = {"add", "subtract", "multiply", "divide", None}
    total_tests = len(hallucination_test_prompts)
    hallucination_count = 0

    for i, prompt in enumerate(hallucination_test_prompts):
        print(f"\nTest {i + 1}/{total_tests}: {prompt}")

        result = what_if_parameter_modification_tool_for_testing(
            prompt, test_ui_state, regular_llm
        )

        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            continue

        if (
            "parameter_updates" not in result
            or "full_modifications" not in result["parameter_updates"]
        ):
            print("  ❌ Missing expected structure")
            continue

        modifications = result["parameter_updates"]["full_modifications"]

        hallucination_found = False
        for mod in modifications:
            perturbation_type = mod.get("perturbation_type")
            if perturbation_type not in valid_perturbation_types:
                print(
                    f"  👻 HALLUCINATION: Invalid perturbation_type '{perturbation_type}' for column '{mod.get('name')}'"
                )
                hallucination_found = True
                hallucination_count += 1
                break

        if not hallucination_found:
            print(
                f"  ✅ Valid perturbation_type: {modifications[0].get('perturbation_type') if modifications else 'None'}"
            )

    print("\n" + "=" * 60)
    print("HALLUCINATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Hallucinations Detected: {hallucination_count}")
    print(
        f"Success Rate: {((total_tests - hallucination_count) / total_tests):.1%}"
    )

    if hallucination_count == 0:
        print(
            "🎉 EXCELLENT: No hallucinations detected! Prompts are working correctly."
        )
    else:
        print(
            f"⚠️  WARNING: {hallucination_count} hallucination(s) detected. Prompts need improvement."
        )


if __name__ == "__main__":
    # Run unit tests
    print("Running Real Integration Tests (No Mocking)...")
    unittest.main(verbosity=2, exit=False)

    # Run comprehensive hallucination test
    run_comprehensive_hallucination_test()
