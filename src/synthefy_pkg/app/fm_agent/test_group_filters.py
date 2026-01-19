#!/usr/bin/env python3
"""
Parameterized tests for improved group filters extraction with categorical features.
"""

import os
import sys

import pytest

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from synthefy_pkg.app.data_models import CategoricalFeatureValues
from synthefy_pkg.app.fm_agent.agent import (
    UIState,
    get_state_parameters_to_update_tool,
)
from synthefy_pkg.app.fm_agent.test_data.loader import (
    load_group_filters_test_cases,
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


@pytest.fixture
def categorical_features():
    """Fixture providing diverse categorical features for testing."""
    return [
        CategoricalFeatureValues(
            feature_name="customer_segment",
            values=[
                "enterprise",
                "small_business",
                "individual",
                "government",
                "non_profit",
            ],
        ),
        CategoricalFeatureValues(
            feature_name="product_category",
            values=[
                "electronics",
                "clothing",
                "books",
                "home_garden",
                "sports",
                "automotive",
            ],
        ),
        CategoricalFeatureValues(
            feature_name="region_code",
            values=["NA", "EU", "APAC", "LATAM", "MEA"],
        ),
        CategoricalFeatureValues(
            feature_name="priority_level",
            values=["low", "medium", "high", "critical"],
        ),
        CategoricalFeatureValues(
            feature_name="department_id",
            values=["D001", "D002", "D003", "D004", "D005"],
        ),
        CategoricalFeatureValues(
            feature_name="Store",
            values=["store_1", "store_2", "store_3", "store_15", "store_20"],
        ),
        CategoricalFeatureValues(
            feature_name="userid",
            values=["user_001", "user_123", "raimi", "john_doe", "alice_smith"],
        ),
        CategoricalFeatureValues(
            feature_name="status",
            values=["active", "inactive", "pending", "archived"],
        ),
        CategoricalFeatureValues(
            feature_name="location_type",
            values=["urban", "suburban", "rural", "metropolitan"],
        ),
        CategoricalFeatureValues(
            feature_name="payment_method",
            values=[
                "credit_card",
                "debit_card",
                "paypal",
                "bank_transfer",
                "cash",
            ],
        ),
    ]


@pytest.fixture
def ui_state():
    """Fixture providing a sample UI state with diverse columns."""
    return UIState(
        user_id="test_user",
        file_path_key="test_file.parquet",
        dataset_columns=[
            "customer_segment",
            "revenue",
            "product_category",
            "region_code",
            "priority_level",
            "department_id",
            "transaction_date",
            "quantity_sold",
            "Store",
            "userid",
            "status",
            "location_type",
            "payment_method",
            "sales",
            "profit",
            "cost",
            "weekly_sales",
            "monthly_sales",
            "daily_sales",
            "temperature",
            "humidity",
            "price",
            "discount",
            "inventory_level",
        ],
        possible_timestamp_columns=[
            "transaction_date",
            "created_at",
            "date",
            "datetime",
        ],
        group_filters={},
    )


# Load test cases from JSON file
def load_test_cases():
    """Load test cases from JSON file and convert to pytest format."""
    test_cases_data = load_group_filters_test_cases()
    return [
        (case["prompt"], case["expected"], case["description"])
        for case in test_cases_data
    ]


# Get test cases for parametrization
test_cases = load_test_cases()


@pytest.mark.parametrize("prompt,expected,description", test_cases)
def test_parameter_extraction(
    prompt, expected, description, ui_state, categorical_features
):
    """Test parameter extraction with diverse scenarios."""

    result = get_state_parameters_to_update_tool.func(  # type: ignore
        user_input=prompt,
        ui_state=ui_state,
        categorical_features=categorical_features,
    )

    parameter_updates = result.get("parameter_updates", {})

    # Check if the expected parameters are extracted
    for key, expected_value in expected.items():
        assert key in parameter_updates, (
            f"Missing key: {key} in result: {parameter_updates}. Description: {description}"
        )
        assert parameter_updates[key] == expected_value, (
            f"Wrong value for {key}. Expected: {expected_value}, Got: {parameter_updates[key]}. Description: {description}"
        )


# Additional individual test methods for specific scenarios
class TestGroupFiltersExtraction:
    """Additional specific test cases for edge scenarios."""

    def test_fuzzy_matching_abbreviations(self, ui_state, categorical_features):
        """Test that abbreviations are properly matched to full terms."""
        result = get_state_parameters_to_update_tool.func(  # type: ignore
            user_input="show data for gov customers",
            ui_state=ui_state,
            categorical_features=categorical_features,
        )

        parameter_updates = result.get("parameter_updates", {})
        assert parameter_updates.get("group_filters") == {
            "customer_segment": ["government"]
        }

    def test_multiple_parameter_extraction(
        self, ui_state, categorical_features
    ):
        """Test that multiple parameters are extracted correctly."""
        result = get_state_parameters_to_update_tool.func(  # type: ignore
            user_input="forecast revenue for small business using transaction_date as timestamp for 30 days",
            ui_state=ui_state,
            categorical_features=categorical_features,
        )

        parameter_updates = result.get("parameter_updates", {})
        assert "target_columns" in parameter_updates
        assert "timestamp_column" in parameter_updates
        assert "group_filters" in parameter_updates
        assert "forecast_length" in parameter_updates

    def test_boolean_parameter_handling(self, ui_state, categorical_features):
        """Test that boolean parameters like leak_metadata are handled correctly."""
        result = get_state_parameters_to_update_tool.func(  # type: ignore
            user_input="enable metadata leakage and forecast revenue",
            ui_state=ui_state,
            categorical_features=categorical_features,
        )

        parameter_updates = result.get("parameter_updates", {})
        assert parameter_updates.get("leak_metadata") is True
        assert parameter_updates.get("target_columns") == ["revenue"]

    def test_store_number_transformation(self, ui_state, categorical_features):
        """Test specific store number transformation scenarios."""
        test_cases = [
            ("show sales for store 1", {"Store": ["store_1"]}),
            ("display data for store 15", {"Store": ["store_15"]}),
            (
                "filter by store 2 and store 3",
                {"Store": ["store_2", "store_3"]},
            ),
            ("show me shop 1 data", {"Store": ["store_1"]}),  # shop -> store
        ]

        for prompt, expected_filters in test_cases:
            result = get_state_parameters_to_update_tool.func(  # type: ignore
                user_input=prompt,
                ui_state=ui_state,
                categorical_features=categorical_features,
            )
            parameter_updates = result.get("parameter_updates", {})
            assert parameter_updates.get("group_filters") == expected_filters, (
                f"Failed for prompt: {prompt}"
            )

    def test_error_handling(self, ui_state, categorical_features):
        """Test that the function handles errors gracefully."""
        # Test with empty prompt
        result = get_state_parameters_to_update_tool.func(  # type: ignore
            user_input="",
            ui_state=ui_state,
            categorical_features=categorical_features,
        )
        assert "parameter_updates" in result

        # Test with nonsensical prompt
        result = get_state_parameters_to_update_tool.func(  # type: ignore
            user_input="xyz abc 123 random nonsense",
            ui_state=ui_state,
            categorical_features=categorical_features,
        )
        assert "parameter_updates" in result

    def test_underscore_overfitting_prevention(
        self, ui_state, categorical_features
    ):
        """
        Test that the system doesn't overfit to underscores shown in categorical context.

        This test ensures that even though the categorical features context shows values
        with underscores (e.g., 'credit_card', 'small_business'), the system can still
        correctly match user inputs that use spaces instead of underscores.
        """
        test_cases = [
            # Payment methods - context shows 'credit_card', 'debit_card'
            ("show credit card sales", {"payment_method": ["credit_card"]}),
            ("filter debit card users", {"payment_method": ["debit_card"]}),
            ("bank transfer payments", {"payment_method": ["bank_transfer"]}),
            # Customer segments - context shows 'small_business', 'non_profit'
            (
                "small business customers",
                {"customer_segment": ["small_business"]},
            ),
            ("non profit segment", {"customer_segment": ["non_profit"]}),
            # User IDs - context shows 'john_doe', 'alice_smith'
            ("user john doe", {"userid": ["john_doe"]}),
            ("alice smith data", {"userid": ["alice_smith"]}),
            # Location types - context shows values without spaces typically
            ("urban location", {"location_type": ["urban"]}),
            ("suburban area", {"location_type": ["suburban"]}),
        ]

        for prompt, expected_filters in test_cases:
            result = get_state_parameters_to_update_tool.func(  # type: ignore
                user_input=prompt,
                ui_state=ui_state,
                categorical_features=categorical_features,
            )
            parameter_updates = result.get("parameter_updates", {})
            actual_filters = parameter_updates.get("group_filters", {})

            assert actual_filters == expected_filters, (
                f"Underscore overfitting detected! "
                f"Input: '{prompt}' -> Expected: {expected_filters}, Got: {actual_filters}. "
                f"The system should handle space-separated input correctly even when "
                f"categorical context shows underscored values."
            )

    def test_categorical_value_format_variations(self):
        """
        Test various combinations of database value formats vs user input formats.

        This tests the cross-matrix of:
        - Database values: underscore, space, hyphen, different casing
        - User input: underscore, space, hyphen, different casing

        Ensures the fuzzy matching can handle format mismatches.
        """
        # Test different database value formats
        test_scenarios = [
            # Scenario 1: Database has underscores, user types various formats
            {
                "name": "Database_Underscores_vs_User_Variations",
                "categorical_features": [
                    CategoricalFeatureValues(
                        feature_name="Store",
                        values=["store_1", "store_2", "store_15"],
                    ),
                    CategoricalFeatureValues(
                        feature_name="payment_method",
                        values=["credit_card", "debit_card", "bank_transfer"],
                    ),
                ],
                "test_cases": [
                    ("store 1", {"Store": ["store_1"]}),  # space → underscore
                    ("store-1", {"Store": ["store_1"]}),  # hyphen → underscore
                    (
                        "Store 1",
                        {"Store": ["store_1"]},
                    ),  # space + caps → underscore
                    ("STORE_1", {"Store": ["store_1"]}),  # caps → underscore
                    (
                        "credit card",
                        {"payment_method": ["credit_card"]},
                    ),  # space → underscore
                    (
                        "credit-card",
                        {"payment_method": ["credit_card"]},
                    ),  # hyphen → underscore
                ],
            },
            # Scenario 2: Database has spaces, user types various formats
            {
                "name": "Database_Spaces_vs_User_Variations",
                "categorical_features": [
                    CategoricalFeatureValues(
                        feature_name="Store",
                        values=["store 1", "store 2", "store 15"],
                    ),
                    CategoricalFeatureValues(
                        feature_name="payment_method",
                        values=["credit card", "debit card", "bank transfer"],
                    ),
                ],
                "test_cases": [
                    ("store_1", {"Store": ["store 1"]}),  # underscore → space
                    ("store-1", {"Store": ["store 1"]}),  # hyphen → space
                    (
                        "Store_1",
                        {"Store": ["store 1"]},
                    ),  # underscore + caps → space
                    ("STORE 1", {"Store": ["store 1"]}),  # caps → space
                    (
                        "credit_card",
                        {"payment_method": ["credit card"]},
                    ),  # underscore → space
                    (
                        "credit-card",
                        {"payment_method": ["credit card"]},
                    ),  # hyphen → space
                ],
            },
            # Scenario 3: Database has hyphens, user types various formats
            {
                "name": "Database_Hyphens_vs_User_Variations",
                "categorical_features": [
                    CategoricalFeatureValues(
                        feature_name="Store",
                        values=["store-1", "store-2", "store-15"],
                    ),
                    CategoricalFeatureValues(
                        feature_name="payment_method",
                        values=["credit-card", "debit-card", "bank-transfer"],
                    ),
                ],
                "test_cases": [
                    ("store_1", {"Store": ["store-1"]}),  # underscore → hyphen
                    ("store 1", {"Store": ["store-1"]}),  # space → hyphen
                    (
                        "Store_1",
                        {"Store": ["store-1"]},
                    ),  # underscore + caps → hyphen
                    ("STORE-1", {"Store": ["store-1"]}),  # caps → hyphen
                    (
                        "credit_card",
                        {"payment_method": ["credit-card"]},
                    ),  # underscore → hyphen
                    (
                        "credit card",
                        {"payment_method": ["credit-card"]},
                    ),  # space → hyphen
                ],
            },
            # Scenario 4: Database has mixed casing, user types various formats
            {
                "name": "Database_MixedCase_vs_User_Variations",
                "categorical_features": [
                    CategoricalFeatureValues(
                        feature_name="Store",
                        values=["Store_1", "Store_2", "Store_15"],
                    ),
                    CategoricalFeatureValues(
                        feature_name="customer_segment",
                        values=["Small_Business", "Enterprise", "Non_Profit"],
                    ),
                ],
                "test_cases": [
                    (
                        "store 1",
                        {"Store": ["Store_1"]},
                    ),  # lowercase space → Mixed_Case
                    (
                        "store-1",
                        {"Store": ["Store_1"]},
                    ),  # lowercase hyphen → Mixed_Case
                    (
                        "store_1",
                        {"Store": ["Store_1"]},
                    ),  # lowercase underscore → Mixed_Case
                    (
                        "STORE_1",
                        {"Store": ["Store_1"]},
                    ),  # uppercase → Mixed_Case
                    (
                        "small business",
                        {"customer_segment": ["Small_Business"]},
                    ),  # lowercase space → Mixed_Case
                    (
                        "small_business",
                        {"customer_segment": ["Small_Business"]},
                    ),  # lowercase underscore → Mixed_Case
                ],
            },
        ]

        for scenario in test_scenarios:
            print(f"\n--- Testing Scenario: {scenario['name']} ---")

            # Create UI state for this scenario
            ui_state = UIState(
                user_id="test_user",
                file_path_key="test_file.parquet",
                dataset_columns=[
                    "Store",
                    "payment_method",
                    "customer_segment",
                    "sales",
                    "revenue",
                ],
                possible_timestamp_columns=["date"],
                group_filters={},
            )

            for user_input, expected_filters in scenario["test_cases"]:
                result = get_state_parameters_to_update_tool.func(  # type: ignore
                    user_input=f"show sales for {user_input}",
                    ui_state=ui_state,
                    categorical_features=scenario["categorical_features"],
                )

                parameter_updates = result.get("parameter_updates", {})
                actual_filters = parameter_updates.get("group_filters", {})

                assert actual_filters == expected_filters, (
                    f"❌ Format mismatch in {scenario['name']}!\n"
                    f"User input: '{user_input}'\n"
                    f"Database values: {[f.values for f in scenario['categorical_features']]}\n"
                    f"Expected: {expected_filters}\n"
                    f"Got: {actual_filters}\n"
                    f"The system should handle format conversion between database and user input!"
                )


# Legacy function for backwards compatibility or manual testing
def test_group_filters_extraction_manual():
    """Legacy manual test function - kept for backwards compatibility."""
    pass


if __name__ == "__main__":
    # Run the tests when executed directly
    pytest.main([__file__, "-v"])
