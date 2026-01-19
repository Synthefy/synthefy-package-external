import copy
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np
import pandas as pd
import pytest

# from synthefy_pkg.app.data_models import FMDataPointCard
from synthefy_pkg.app.fm_agent.agent import (
    MetaDataVariationList,
    UIState,
    _extract_metadata_dataset_number_from_input,
    what_if_parameter_modification_tool_for_testing,
)

skip_in_ci = pytest.mark.skipif(
    os.environ.get("SKIP_CERTAIN_TESTS") == "true",
    reason="Test skipped based on environment variable",
)


@pytest.fixture
def sample_columns():
    return ["Weekly_Sales", "Temperature", "Unemployment", "label"]


@pytest.fixture
def sample_data_point_cards():
    # Each card is a dict with timestamp and values
    return [
        {
            "timestamp": ts,
            "values": {
                "Weekly_Sales": np.random.randint(100, 1000),
                "Temperature": np.random.randint(10, 30),
                "Unemployment": np.random.randint(1, 10),
                "label": "None",
            },
        }
        for ts in [
            "2010-02-12",
            "2010-02-13",
            "2010-02-14",
            "2010-02-15",
            "2010-02-16",
            "2010-02-19",
            "2010-02-22",
            "2010-02-26",
            "2010-03-01",
            "2010-03-05",
            "2010-03-12",
            "2010-03-19",
            "2010-03-26",
            "2010-04-02",
            "2010-04-09",
            "2010-04-16",
            "2010-04-23",
            "2010-04-30",
            "2010-05-01",
            "2010-06-01",
            "2010-07-01",
            "2010-08-01",
        ]
    ]


def make_ui_state(data_point_cards):
    return UIState(
        user_id="test_user",
        file_path_key="test_file",
        target_columns=["Weekly_Sales", "Temperature", "Unemployment", "label"],
        timestamp_column="timestamp",
        covariates=None,
        min_timestamp=None,
        forecast_timestamp=None,
        group_filters=None,
        forecast_length=None,
        backtest_stride=None,
        leak_metadata=None,
        dataset_columns=[
            "Weekly_Sales",
            "Temperature",
            "Unemployment",
            "label",
        ],
        possible_timestamp_columns=["timestamp"],
    )


# class TestGetChangedDataPointCards:
#     def test_all_cards_changed(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 4},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 10, "b": 20},
#             {"timestamp": "2024-01-02", "a": 30, "b": 40},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 2
#         assert changed[0]["card_index"] == 0
#         assert changed[1]["card_index"] == 1
#         assert changed[0]["values_to_change"] == {"a": 10, "b": 20}
#         assert changed[1]["values_to_change"] == {"a": 30, "b": 40}

#     def test_no_cards_changed(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 4},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 4},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert changed == []

#     def test_some_cards_changed(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 4},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 10, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 40},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 2
#         assert changed[0]["card_index"] == 0
#         assert changed[1]["card_index"] == 1
#         assert changed[0]["values_to_change"] == {"a": 10}
#         assert changed[1]["values_to_change"] == {"b": 40}

#     def test_empty_input(self):
#         df_orig = pd.DataFrame([])
#         df_new = pd.DataFrame([])
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert changed == []

#     def test_different_timestamp_column(self):
#         orig = [
#             {"date": "2024-01-01", "a": 1},
#             {"date": "2024-01-02", "a": 2},
#         ]
#         new = [
#             {"date": "2024-01-01", "a": 10},
#             {"date": "2024-01-02", "a": 2},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(df_orig, df_new, "date")
#         assert len(changed) == 1
#         assert changed[0]["card_index"] == 0
#         assert changed[0]["values_to_change"] == {"a": 10}

#     def test_non_overlapping_timestamps(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1},
#             {"timestamp": "2024-01-02", "a": 2},
#         ]
#         new = [
#             {"timestamp": "2024-01-03", "a": 3},
#             {"timestamp": "2024-01-04", "a": 4},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert changed == []

#     def test_multiple_columns_changed_varied(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2, "c": 3},
#             {"timestamp": "2024-01-02", "a": 4, "b": 5, "c": 6},
#             {"timestamp": "2024-01-03", "a": 7, "b": 8, "c": 9},
#         ]
#         new = [
#             {
#                 "timestamp": "2024-01-01",
#                 "a": 10,
#                 "b": 2,
#                 "c": 30,
#             },  # a and c changed
#             {"timestamp": "2024-01-02", "a": 4, "b": 50, "c": 6},  # b changed
#             {"timestamp": "2024-01-03", "a": 7, "b": 8, "c": 9},  # no change
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 2
#         assert changed[0]["card_index"] == 0
#         assert changed[0]["values_to_change"] == {"a": 10, "c": 30}
#         assert changed[1]["card_index"] == 1
#         assert changed[1]["values_to_change"] == {"b": 50}

#     def test_nan_and_none_handling(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": np.nan},
#             {"timestamp": "2024-01-02", "a": None, "b": 2},
#             {"timestamp": "2024-01-03", "a": 3, "b": 4},
#         ]
#         new = [
#             {
#                 "timestamp": "2024-01-01",
#                 "a": np.nan,
#                 "b": np.nan,
#             },  # a: 1 -> nan
#             {"timestamp": "2024-01-02", "a": 10, "b": 2},  # a: None -> 10
#             {"timestamp": "2024-01-03", "a": 3, "b": None},  # b: 4 -> None
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 3
#         # 2024-01-01: a changed from 1 to nan
#         assert changed[0]["card_index"] == 0
#         assert "a" in changed[0]["values_to_change"]
#         assert pd.isna(changed[0]["values_to_change"]["a"])
#         # 2024-01-02: a changed from None to 10
#         assert changed[1]["card_index"] == 1
#         assert changed[1]["values_to_change"] == {"a": 10}
#         # 2024-01-03: b changed from 4 to None
#         assert changed[2]["card_index"] == 2
#         assert changed[2]["values_to_change"] == {"b": None}  # TODO fix this

#     def test_new_column_added(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1},
#             {"timestamp": "2024-01-02", "a": 2},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 100},  # new column b
#             {"timestamp": "2024-01-02", "a": 2, "b": 200},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert changed == []

#     def test_removed_column_in_mod(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": 2},
#             {"timestamp": "2024-01-02", "a": 3, "b": 4},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 1},  # b removed
#             {"timestamp": "2024-01-02", "a": 30},  # a changed, b removed
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert changed == []

#     def test_mixed_types(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1, "b": "x"},
#             {"timestamp": "2024-01-02", "a": 2.0, "b": "y"},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": "1", "b": "x"},  # a: int -> str
#             {"timestamp": "2024-01-02", "a": 2.0, "b": "z"},  # b: y -> z
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 2
#         assert changed[0]["card_index"] == 0
#         assert changed[0]["values_to_change"] == {"a": "1"}
#         assert changed[1]["card_index"] == 1
#         assert changed[1]["values_to_change"] == {"b": "z"}

#     def test_index_mismatch(self):
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1},
#             {"timestamp": "2024-01-02", "a": 2},
#             {"timestamp": "2024-01-03", "a": 3},
#         ]
#         new = [
#             {"timestamp": "2024-01-03", "a": 30},
#             {"timestamp": "2024-01-01", "a": 10},
#             {"timestamp": "2024-01-02", "a": 2},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         assert len(changed) == 2
#         # Find by card_index
#         card_indices = sorted([c["card_index"] for c in changed])
#         assert card_indices == [0, 2]
#         for c in changed:
#             if c["card_index"] == 0:
#                 assert c["values_to_change"] == {"a": 10}
#             elif c["card_index"] == 2:
#                 assert c["values_to_change"] == {"a": 30}

#     def test_messed_up_index(self):
#         # DataFrames with non-sequential, non-unique, and unsorted indices
#         orig = [
#             {"timestamp": "2024-01-01", "a": 1},
#             {"timestamp": "2024-01-02", "a": 2},
#             {"timestamp": "2024-01-03", "a": 3},
#         ]
#         new = [
#             {"timestamp": "2024-01-01", "a": 10},
#             {"timestamp": "2024-01-02", "a": 2},
#             {"timestamp": "2024-01-03", "a": 30},
#         ]
#         df_orig = pd.DataFrame(orig)
#         df_new = pd.DataFrame(new)
#         # Mess up the index using set_index (simulate non-sequential, unsorted index)
#         df_orig = df_orig.set_index(pd.Index([100, 50, 25]))
#         df_new = df_new.set_index(pd.Index([9, 8, 7]))
#         changed = get_changed_data_point_cards_from_dfs(
#             df_orig, df_new, "timestamp"
#         )
#         # Should match by timestamp, not by index
#         assert len(changed) == 2
#         card_indices = sorted([c["card_index"] for c in changed])
#         assert card_indices == [0, 2]
#         for c in changed:
#             if c["card_index"] == 0:
#                 assert c["values_to_change"] == {"a": 10}
#             elif c["card_index"] == 2:
#                 assert c["values_to_change"] == {"a": 30}


class DummyUIState(UIState):
    def __init__(self, dataset_columns):
        super().__init__(
            user_id="test_user",
            file_path_key="test_file",
            target_columns=None,
            timestamp_column=None,
            covariates=None,
            min_timestamp=None,
            forecast_timestamp=None,
            group_filters=None,
            forecast_length=None,
            backtest_stride=None,
            leak_metadata=None,
            dataset_columns=dataset_columns,
            possible_timestamp_columns=[],
        )


@skip_in_ci
class TestWhatIfParameterModificationTool:
    def test_single_modification(self):
        ui_state = DummyUIState(["Temperature", "sales"])
        result = what_if_parameter_modification_tool_for_testing(
            "What if the temperature is increased by 10",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert isinstance(mods.variation_list, list)
        assert len(mods.variation_list) == 1
        assert mods.variation_list[0].name == "Temperature"
        assert (
            mods.variation_list[0].perturbation_or_exact_value == "perturbation"
        )
        assert mods.variation_list[0].perturbation_type == "add"
        assert mods.variation_list[0].value == 10

    def test_multiple_modifications(self):
        ui_state = DummyUIState(["temperature", "cost", "profit_MTD"])
        result = what_if_parameter_modification_tool_for_testing(
            "set the temperature to 5 and profit to 100 and forecast the cost again",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert isinstance(mods.variation_list, list)
        assert len(mods.variation_list) == 2
        names = {m.name for m in mods.variation_list}
        assert "temperature" in names
        assert "profit_MTD" in names
        for m in mods.variation_list:
            assert m.perturbation_or_exact_value == "exact_value"
            assert m.perturbation_type is None
            if m.name == "temperature":
                assert m.value == 5
            if m.name == "profit_MTD":
                assert m.value == 100

    def test_typos_and_capitalization(self):
        ui_state = DummyUIState(["Temperature", "SALES", "profit"])
        result = what_if_parameter_modification_tool_for_testing(
            "set the temperture to 20 and sakles to 2000",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert len(mods.variation_list) == 2
        names = {m.name for m in mods.variation_list}
        assert "Temperature" in names
        assert "SALES" in names
        for m in mods.variation_list:
            assert m.perturbation_or_exact_value == "exact_value"
            assert m.perturbation_type is None
            if m.name.lower() == "temperature":
                assert m.value == 20
            if m.name.lower() == "sales":
                assert m.value == 2000

    def test_special_characters(self):
        ui_state = DummyUIState(["sales", "cost"])
        result = what_if_parameter_modification_tool_for_testing(
            "If ssles drop by 50 perecent and cost increases by 5$",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert len(mods.variation_list) == 2
        names = {m.name for m in mods.variation_list}
        assert "sales" in names
        assert "cost" in names

    def test_irrelevant_prompt(self):
        ui_state = DummyUIState(["temperature", "sales"])
        result = what_if_parameter_modification_tool_for_testing(
            "Show me a plot of the data",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        # Should be empty or not present
        assert mods.variation_list == [] or mods.variation_list is None

    def test_only_available_columns(self):
        ui_state = DummyUIState(["temperature", "sales"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set temperature to 10 and revenue to 1000",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        # Only temperature should   be present
        assert any(m.name == "temperature" for m in mods.variation_list)
        assert not any(m.name == "revenue" for m in mods.variation_list)

    def test_set_exact_value_number(self):
        ui_state = DummyUIState(["temperature", "category"])
        result = what_if_parameter_modification_tool_for_testing(
            "what if set temperature to 42",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert len(mods.variation_list) == 1
        mod = mods.variation_list[0]
        assert mod.name == "temperature"
        assert mod.perturbation_or_exact_value == "exact_value"
        assert mod.perturbation_type is None
        assert mod.value == 42

    def test_set_exact_value_string(self):
        ui_state = DummyUIState(["status", "temperature"])
        result = what_if_parameter_modification_tool_for_testing(
            "simultate when status is 'closed' and temperature is 15",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert len(mods.variation_list) == 2
        found_status = False
        found_temperature = False
        for mod in mods.variation_list:
            if mod.name == "status":
                found_status = True
                assert mod.perturbation_or_exact_value == "exact_value"
                assert mod.perturbation_type is None
                assert mod.value == "closed" or mod.value == "'closed'"
            if mod.name == "temperature":
                found_temperature = True
                assert mod.perturbation_or_exact_value == "exact_value"
                assert mod.perturbation_type is None
                assert mod.value == 15
        assert found_status and found_temperature

    def test_set_string_with_spaces_and_special_chars(self):
        ui_state = DummyUIState(["status", "temperature"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set status to 'in progress - urgent!' and temperature to 25",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "status"
            and (
                m.value == "in progress - urgent!"
                or m.value == "'in progress - urgent!'"
            )
            for m in mods.variation_list
        )
        assert any(
            m.name == "temperature" and m.value == 25
            for m in mods.variation_list
        )

    def test_set_numeric_with_unit(self):
        ui_state = DummyUIState(["temperature"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set temperature to 30C",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        # Should extract 30 as the value, ignoring the 'C'
        assert any(
            m.name == "temperature" and (m.value == 30 or m.value == "30")
            for m in mods.variation_list
        )

    def test_set_value_with_typo_in_column_and_value(self):
        ui_state = DummyUIState(["temperature", "status"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set temperture to 'clsoed'",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        # Should match 'temperature' and value 'clsoed'
        assert any(
            m.name.lower() == "temperature"
            and (m.value == "clsoed" or m.value == "'clsoed'")
            for m in mods.variation_list
        )

    def test_set_multiple_columns_mixed_types(self):
        ui_state = DummyUIState(["status", "temperature", "score"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set status to 'done', temperature to 18, and score to 99.5",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "status" and (m.value == "done" or m.value == "'done'")
            for m in mods.variation_list
        )
        assert any(
            m.name == "temperature" and m.value == 18
            for m in mods.variation_list
        )
        assert any(
            m.name == "score" and (m.value == 99.5 or m.value == "99.5")
            for m in mods.variation_list
        )

    def test_set_value_to_none_or_empty(self):
        ui_state = DummyUIState(["status"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set status to None",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "status"
            and (
                m.value is None
                or m.value == "None"
                or m.value == "null"
                or m.value == ""
            )
            for m in mods.variation_list
        )

    def test_set_value_with_quoted_string(self):
        ui_state = DummyUIState(["status"])
        result = what_if_parameter_modification_tool_for_testing(
            'Set status to "approved"',
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "status"
            and (m.value == "approved" or m.value == '"approved"')
            for m in mods.variation_list
        )

    def test_set_value_with_percent(self):
        ui_state = DummyUIState(["discount"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set discount to 15%",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        # Should extract 15 or 0.15 as value
        assert any(
            m.name == "discount"
            and (
                m.value == 15
                or m.value == 0.15
                or m.value == "15"
                or m.value == "0.15"
            )
            for m in mods.variation_list
        )

    def test_set_value_with_negative_number(self):
        ui_state = DummyUIState(["temperature"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set temperature to -5",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "temperature" and (m.value == -5 or m.value == "-5")
            for m in mods.variation_list
        )

    def test_set_value_with_scientific_notation(self):
        ui_state = DummyUIState(["score"])
        result = what_if_parameter_modification_tool_for_testing(
            "Set score to 1e-3",
            ui_state,
        )
        mods = MetaDataVariationList(
            variation_list=result["parameter_updates"]["full_modifications"]
        )
        assert any(
            m.name == "score" and (m.value == 1e-3) for m in mods.variation_list
        )


class TestParseGroupFiltersString:
    """Test cases for _parse_group_filters_string function"""

    @pytest.fixture
    def sample_categorical_features(self):
        from synthefy_pkg.app.data_models import CategoricalFeatureValues

        return [
            CategoricalFeatureValues(
                feature_name="Store",
                values=["store_1", "store_2", "store_15", "store_20"],
            ),
            CategoricalFeatureValues(
                feature_name="Region", values=["US", "Europe", "Asia"]
            ),
            CategoricalFeatureValues(
                feature_name="user_id",
                values=["1234567890", "raimi", "john_doe"],
            ),
            CategoricalFeatureValues(
                feature_name="category",
                values=["Electronics", "Clothing", "Books"],
            ),
        ]

    @pytest.fixture
    def sample_dataset_columns(self):
        return [
            "Store",
            "Region",
            "user_id",
            "category",
            "sales",
            "revenue",
            "timestamp",
        ]

    def test_empty_string(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "", sample_categorical_features, sample_dataset_columns
        )
        assert result == {}

    def test_single_filter_equals(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Store": ["store_1"]}

    def test_single_filter_colon(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Region:US", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Region": ["US"]}

    def test_single_filter_is(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store is store_1",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"]}

    def test_multiple_filters_comma_separated(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1,Region=US",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"], "Region": ["US"]}

    def test_multiple_filters_and_separated(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1 and Region=US",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"], "Region": ["US"]}

    def test_multiple_values_single_filter(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1,store_2",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1", "store_2"]}

    def test_quoted_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Region='US'", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Region": ["US"]}

    def test_double_quoted_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            'Region="US"', sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Region": ["US"]}

    def test_bracketed_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=[store_1]",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"]}

    def test_fuzzy_column_matching(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        # Test typo in column name
        result = _parse_group_filters_string(
            "stor=store_1", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Store": ["store_1"]}

    def test_fuzzy_value_matching(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        # Test space to underscore conversion in value
        result = _parse_group_filters_string(
            "Store=store 1", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Store": ["store_1"]}

    def test_case_insensitive_matching(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "store=STORE_1", sample_categorical_features, sample_dataset_columns
        )
        assert result == {"Store": ["store_1"]}

    def test_numeric_user_id(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "user_id=1234567890",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"user_id": ["1234567890"]}

    def test_pipe_separator(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1|Region=US",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"], "Region": ["US"]}

    def test_semicolon_separator(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1;Region=US",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"], "Region": ["US"]}

    def test_equals_operator(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store equals store_1",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"]}

    def test_eq_operator(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store eq store_1",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"]}

    def test_mixed_separators(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1;Region:US|user_id=raimi",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {
            "Store": ["store_1"],
            "Region": ["US"],
            "user_id": ["raimi"],
        }

    def test_whitespace_handling(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "  Store = store_1  ,  Region = US  ",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {"Store": ["store_1"], "Region": ["US"]}

    def test_no_matching_column(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "NonExistentColumn=value",
            sample_categorical_features,
            sample_dataset_columns,
        )
        # Should return empty dict since no column matches
        assert result == {}

    def test_no_matching_value(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=non_existent_store",
            sample_categorical_features,
            sample_dataset_columns,
        )
        # Should keep original value if no categorical match found
        assert result == {"Store": ["non_existent_store"]}

    def test_partial_invalid_filters(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        # Mix of valid and invalid filters
        result = _parse_group_filters_string(
            "Store=store_1,InvalidColumn=value",
            sample_categorical_features,
            sample_dataset_columns,
        )
        assert result == {
            "Store": ["store_1"]
        }  # Only valid filter should remain

    def test_malformed_filter_no_value(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=", sample_categorical_features, sample_dataset_columns
        )
        # Should handle gracefully and return empty
        assert result == {}

    def test_malformed_filter_no_key(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "=store_1", sample_categorical_features, sample_dataset_columns
        )
        # Should handle gracefully and return empty
        assert result == {}

    def test_complex_mixed_format(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        filter_str = "Store='store_1',store_2;Region is US AND user_id:raimi|category equals Electronics"
        result = _parse_group_filters_string(
            filter_str, sample_categorical_features, sample_dataset_columns
        )
        expected = {
            "Store": ["store_1", "store_2"],
            "Region": ["US"],
            "user_id": ["raimi"],
            "category": ["Electronics"],
        }
        assert result == expected

    def test_none_categorical_features(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1", None, sample_dataset_columns
        )
        assert result == {
            "Store": ["store_1"]
        }  # Should still work without categorical features

    def test_empty_categorical_features(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=store_1", [], sample_dataset_columns
        )
        assert result == {
            "Store": ["store_1"]
        }  # Should still work with empty categorical features

    def test_special_characters_in_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        # Add a categorical feature with special characters for testing
        from synthefy_pkg.app.data_models import CategoricalFeatureValues
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        special_features = sample_categorical_features + [
            CategoricalFeatureValues(
                feature_name="special",
                values=[
                    "value-with-hyphens",
                    "value_with_underscores",
                    "value with spaces",
                ],
            )
        ]
        dataset_cols = sample_dataset_columns + ["special"]

        result = _parse_group_filters_string(
            "special=value-with-hyphens", special_features, dataset_cols
        )
        assert result == {"special": ["value-with-hyphens"]}

    def test_numeric_values_as_strings(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        result = _parse_group_filters_string(
            "Store=15", sample_categorical_features, sample_dataset_columns
        )
        # Should match store_15 via fuzzy matching
        assert result == {"Store": ["store_15"]}

    @pytest.mark.skipif(
        os.environ.get("SKIP_CERTAIN_TESTS") == "true",
        reason="Test skipped based on environment variable",
    )
    def test_performance_with_large_input(
        self, sample_categorical_features, sample_dataset_columns
    ):
        import time

        from synthefy_pkg.app.fm_agent.agent import _parse_group_filters_string

        # Create a large filter string
        large_filter = ",".join([f"Store=store_{i}" for i in range(100)])

        start_time = time.time()
        result = _parse_group_filters_string(
            large_filter, sample_categorical_features, sample_dataset_columns
        )
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        # Should still parse correctly
        assert "Store" in result
        assert len(result["Store"]) > 0


class TestFindBestColumnMatch:
    """Test cases for _find_best_column_match function"""

    @pytest.fixture
    def sample_dataset_columns(self):
        return [
            "Store",
            "Region",
            "user_id",
            "category",
            "sales",
            "revenue",
            "timestamp",
            "store_name",
            "customer_name",
        ]

    def test_exact_match_case_sensitive(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match("Store", sample_dataset_columns)
        assert result == "Store"

    def test_exact_match_case_insensitive(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match("store", sample_dataset_columns)
        assert result == "Store"

    def test_typo_matching(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        # Test common typos
        result = _find_best_column_match("stor", sample_dataset_columns)
        # Note: "stor" could match both "Store" and "store_name", and the algorithm
        # prefers more descriptive column names, so it returns "store_name"
        assert result == "store_name"

        result = _find_best_column_match("regoin", sample_dataset_columns)
        assert result == "Region"

    def test_phrase_matching(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        # Test phrase to column mapping
        result = _find_best_column_match(
            "name of the store", sample_dataset_columns
        )
        assert result == "store_name"

        result = _find_best_column_match(
            "customer name", sample_dataset_columns
        )
        assert result == "customer_name"

    def test_substring_matching(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match("user", sample_dataset_columns)
        assert result == "user_id"

    def test_no_match_found(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match(
            "completely_unrelated_column", sample_dataset_columns
        )
        assert result is None

    def test_empty_query(self, sample_dataset_columns):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match("", sample_dataset_columns)
        assert result is None

    def test_empty_columns(self):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        result = _find_best_column_match("Store", [])
        assert result is None

    def test_preference_for_descriptive_columns(self):
        from synthefy_pkg.app.fm_agent.agent import _find_best_column_match

        # When multiple columns could match, prefer more descriptive ones
        columns = ["s", "store", "store_name", "store_location_name"]
        result = _find_best_column_match("store", columns)
        # Should prefer exact match first, but if fuzzy matching, prefer more descriptive
        assert result == "store"  # Exact match takes precedence


class TestFindBestCategoricalValues:
    """Test cases for _find_best_categorical_values function"""

    @pytest.fixture
    def sample_categorical_features(self):
        from synthefy_pkg.app.data_models import CategoricalFeatureValues

        return [
            CategoricalFeatureValues(
                feature_name="Store",
                values=["store_1", "store_2", "store_15", "store_20"],
            ),
            CategoricalFeatureValues(
                feature_name="Region", values=["US", "Europe", "Asia"]
            ),
            CategoricalFeatureValues(
                feature_name="status", values=["active", "inactive", "pending"]
            ),
        ]

    def test_exact_value_match(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Store", ["store_1"], sample_categorical_features
        )
        assert result == ["store_1"]

    def test_case_insensitive_match(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Region", ["us"], sample_categorical_features
        )
        assert result == ["US"]

    def test_space_to_underscore_conversion(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Store", ["store 1"], sample_categorical_features
        )
        assert result == ["store_1"]

    def test_partial_match(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Store", ["15"], sample_categorical_features
        )
        assert result == ["store_15"]

    def test_multiple_values(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Store", ["store_1", "store 2"], sample_categorical_features
        )
        assert result == ["store_1", "store_2"]

    def test_no_matching_feature(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "NonExistentColumn", ["value"], sample_categorical_features
        )
        assert result == [
            "value"
        ]  # Should return original values if no feature found

    def test_no_matching_values(self, sample_categorical_features):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values(
            "Store", ["non_existent_store"], sample_categorical_features
        )
        assert result == [
            "non_existent_store"
        ]  # Should return original values if no match

    def test_empty_categorical_features(self):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values("Store", ["store_1"], [])
        assert result == ["store_1"]

    def test_none_categorical_features(self):
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        result = _find_best_categorical_values("Store", ["store_1"], None)
        assert result == ["store_1"]

    def test_hyphen_to_underscore_conversion(self, sample_categorical_features):
        # Add test data with hyphens
        from synthefy_pkg.app.data_models import CategoricalFeatureValues
        from synthefy_pkg.app.fm_agent.agent import (
            _find_best_categorical_values,
        )

        special_features = sample_categorical_features + [
            CategoricalFeatureValues(
                feature_name="product",
                values=["product-name-1", "product_name_2"],
            )
        ]

        result = _find_best_categorical_values(
            "product", ["product name 1"], special_features
        )
        assert result == ["product-name-1"]


class TestImproveGroupFiltersDict:
    """Test cases for _improve_group_filters_dict function"""

    @pytest.fixture
    def sample_categorical_features(self):
        from synthefy_pkg.app.data_models import CategoricalFeatureValues

        return [
            CategoricalFeatureValues(
                feature_name="Store",
                values=["store_1", "store_2", "store_15", "store_20"],
            ),
            CategoricalFeatureValues(
                feature_name="Region", values=["US", "Europe", "Asia"]
            ),
        ]

    @pytest.fixture
    def sample_dataset_columns(self):
        return ["Store", "Region", "user_id", "category", "sales", "revenue"]

    def test_improve_column_names(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _improve_group_filters_dict

        # Test with typos in column names
        input_dict = {"stor": ["store_1"], "regoin": ["US"]}
        result = _improve_group_filters_dict(
            input_dict, sample_categorical_features, sample_dataset_columns
        )

        assert "Store" in result
        assert "Region" in result
        assert result["Store"] == ["store_1"]
        assert result["Region"] == ["US"]

    def test_improve_categorical_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _improve_group_filters_dict

        # Test with variations in values
        input_dict = {"Store": ["store 1"], "Region": ["us"]}
        result = _improve_group_filters_dict(
            input_dict, sample_categorical_features, sample_dataset_columns
        )

        assert result["Store"] == ["store_1"]
        assert result["Region"] == ["US"]

    def test_handle_non_list_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _improve_group_filters_dict

        # Test with string values instead of lists
        input_dict: Dict[str, Any] = {"Store": "store_1", "Region": "US"}
        result = _improve_group_filters_dict(
            input_dict, sample_categorical_features, sample_dataset_columns
        )

        assert result["Store"] == ["store_1"]
        assert result["Region"] == ["US"]

    def test_preserve_unmatched_values(
        self, sample_categorical_features, sample_dataset_columns
    ):
        from synthefy_pkg.app.fm_agent.agent import _improve_group_filters_dict

        # Test with values that don't match any categorical values
        input_dict = {"Store": ["custom_store"], "unknown_column": ["value"]}
        result = _improve_group_filters_dict(
            input_dict, sample_categorical_features, sample_dataset_columns
        )

        assert result["Store"] == [
            "custom_store"
        ]  # Should preserve unmatched values
        assert result["unknown_column"] == [
            "value"
        ]  # Should preserve unknown columns


class TestExtractMetadataDatasetNumberFromInput:
    """Test cases for _extract_metadata_dataset_number_from_input function"""

    def test_explicit_number_with_datasets_keyword(self):
        """Test explicit numbers with 'datasets' keyword"""
        assert (
            _extract_metadata_dataset_number_from_input("Give me 20 datasets")
            == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Find 5 metadata datasets"
            )
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show 100 metadatas")
            == 100
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Get 50 recommendations"
            )
            == 50
        )
        assert (
            _extract_metadata_dataset_number_from_input("I need 75 results")
            == 75
        )
        assert (
            _extract_metadata_dataset_number_from_input("Want 10 matches") == 10
        )

    def test_explicit_number_with_give_find_show_keywords(self):
        """Test patterns like 'give me 20', 'find 10', etc."""
        assert _extract_metadata_dataset_number_from_input("Give me 30") == 30
        assert _extract_metadata_dataset_number_from_input("Find me 15") == 15
        assert _extract_metadata_dataset_number_from_input("Show 25") == 25
        assert _extract_metadata_dataset_number_from_input("Get 40") == 40
        assert (
            _extract_metadata_dataset_number_from_input("give me 45") == 45
        )  # lowercase
        assert (
            _extract_metadata_dataset_number_from_input("find 35") == 35
        )  # lowercase

    def test_top_and_first_patterns(self):
        """Test 'top N' and 'first N' patterns"""
        assert (
            _extract_metadata_dataset_number_from_input("Show me the top 10")
            == 10
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Give me top 5 datasets"
            )
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input("Find the first 20")
            == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input("first 8 results") == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input("TOP 12") == 12
        )  # uppercase

    def test_best_and_relevant_patterns(self):
        """Test patterns with 'best', 'top', 'most relevant'"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "15 of the best datasets"
            )
            == 15
        )
        assert (
            _extract_metadata_dataset_number_from_input("25 most relevant")
            == 25
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "30 top recommendations"
            )
            == 30
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "10 of the most relevant datasets"
            )
            == 10
        )

    def test_external_data_sources_pattern(self):
        """Test patterns with 'external data sources'"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "Find 12 external data sources"
            )
            == 12
        )
        assert (
            _extract_metadata_dataset_number_from_input("20 data sources") == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Get 8 external sources"
            )
            == 8
        )

    def test_qualitative_terms_mapping(self):
        """Test qualitative terms mapped to specific numbers"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "Give me a few datasets"
            )
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show me few datasets")
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input("Find some datasets")
            == 7
        )
        assert (
            _extract_metadata_dataset_number_from_input("Get several datasets")
            == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show me many datasets")
            == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input("lots of datasets")
            == 25
        )
        assert (
            _extract_metadata_dataset_number_from_input("numerous datasets")
            == 30
        )
        assert (
            _extract_metadata_dataset_number_from_input("plenty of datasets")
            == 35
        )
        assert (
            _extract_metadata_dataset_number_from_input("extensive datasets")
            == 40
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "comprehensive datasets"
            )
            == 50
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "enormous amount of datasets"
            )
            == 100
        )
        assert (
            _extract_metadata_dataset_number_from_input("massive datasets")
            == 150
        )

    def test_case_insensitive_qualitative_terms(self):
        """Test that qualitative terms work case-insensitively"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "Give me A FEW datasets"
            )
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show me MANY datasets")
            == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input("Find SEVERAL datasets")
            == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Comprehensive DATASETS"
            )
            == 50
        )

    def test_boundary_values(self):
        """Test boundary values (1 and 200)"""
        assert (
            _extract_metadata_dataset_number_from_input("Give me 1 dataset")
            == 1
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show me 200 datasets")
            == 200
        )
        # Out of bounds should return None
        assert (
            _extract_metadata_dataset_number_from_input("Give me 0 datasets")
            is None
        )
        assert (
            _extract_metadata_dataset_number_from_input("Show me 201 datasets")
            is None
        )

    def test_no_matches(self):
        """Test inputs that should not match any patterns"""
        assert (
            _extract_metadata_dataset_number_from_input("Show me the data")
            is None
        )
        assert (
            _extract_metadata_dataset_number_from_input("Get the forecast")
            is None
        )
        assert (
            _extract_metadata_dataset_number_from_input("Display the chart")
            is None
        )
        assert _extract_metadata_dataset_number_from_input("") is None
        assert _extract_metadata_dataset_number_from_input("   ") is None

    def test_complex_sentences_with_numbers(self):
        """Test complex sentences that contain the target patterns"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "Please give me 25 metadata datasets about stock prices"
            )
            == 25
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "I need to find the top 15 economic indicators for analysis"
            )
            == 15
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Can you show me several datasets related to unemployment?"
            )
            == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "For my research, I need many external data sources about inflation"
            )
            == 20
        )

    def test_multiple_numbers_in_input(self):
        """Test inputs with multiple numbers - should return the first match"""
        # Should return 10 (first match from "give me 10")
        assert (
            _extract_metadata_dataset_number_from_input(
                "Give me 10 datasets but not 20 results"
            )
            == 10
        )
        # Should return 5 (first match from "top 5")
        assert (
            _extract_metadata_dataset_number_from_input(
                "Show me top 5 and also 15 more datasets"
            )
            == 5
        )

    def test_numbers_with_context(self):
        """Test numbers that appear with relevant context words"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "Find 30 datasets concerning precipitation in Canada"
            )
            == 30
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Retrieve 45 metadata recommendations for machine learning"
            )
            == 45
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Locate 12 external sources about economic indicators"
            )
            == 12
        )

    def test_edge_case_formatting(self):
        """Test edge cases with unusual formatting"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "give    me     20    datasets"
            )
            == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input("top\t15\nresults")
            == 15
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Show me\n\nseveral\ndatasets"
            )
            == 8
        )

    def test_numeric_strings_vs_numbers(self):
        """Test that both numeric patterns work correctly"""
        assert (
            _extract_metadata_dataset_number_from_input("twenty datasets")
            is None
        )  # spelled out numbers not supported
        assert (
            _extract_metadata_dataset_number_from_input("5 datasets") == 5
        )  # integers work

    def test_qualitative_terms_in_sentences(self):
        """Test qualitative terms used in natural sentences"""
        assert (
            _extract_metadata_dataset_number_from_input(
                "I need a few datasets for my analysis"
            )
            == 5
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Can you find several metadata sources about inflation?"
            )
            == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Show me many relevant datasets for forecasting"
            )
            == 20
        )

    def test_regex_pattern_priorities(self):
        """Test that the regex patterns work in their defined order"""
        # This should match the first pattern (explicit number with datasets keyword)
        assert _extract_metadata_dataset_number_from_input("50 datasets") == 50
        # This should match the second pattern (give/find/show me number)
        assert _extract_metadata_dataset_number_from_input("show me 25") == 25
        # This should match the top pattern
        assert _extract_metadata_dataset_number_from_input("top 10") == 10

    def test_mixed_case_inputs(self):
        """Test mixed case inputs"""
        assert (
            _extract_metadata_dataset_number_from_input("Give Me 15 DataSets")
            == 15
        )
        assert (
            _extract_metadata_dataset_number_from_input("TOP 20 RESULTS") == 20
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "Show Me SEVERAL datasets"
            )
            == 8
        )
        assert (
            _extract_metadata_dataset_number_from_input(
                "FIND Many METADATA sources"
            )
            == 20
        )
