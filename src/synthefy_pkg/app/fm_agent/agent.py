import ast
import asyncio
import functools
import json
import os
import time
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import aioboto3
import pandas as pd
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import BaseModel, Field

from synthefy_pkg.app.data_models import (
    ApiForecastRequest,
    CategoricalFeatureValues,
    FMDataPointModification,
    FoundationModelChatRequest,
    FoundationModelConfig,
    HaverDatasetMatch,
    MetaDataVariation,
    SynthefyTimeSeriesModelType,
)
from synthefy_pkg.app.metadata.metadata_rag_service import (
    get_metadata_recommendations,
)

COMPILE = True

GEMINI_MODELS = {
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
}

# Configuration constants
DEFAULT_MAX_RETRIES = 2
LLM_MAX_RETRIES = 3
MAX_CATEGORICAL_VALUES_PREVIEW = (
    10  # Maximum number of categorical values to show in prompts
)


class UIUpdateType(Enum):
    PARAMETERS_UPDATE = "parameters_update"
    FILTER_UPDATE = "filter_update"
    EXPLAIN_UPDATE = "explain_update"
    ANOMALY_DETECTION_UPDATE = "anomaly_detection_update"
    FORECAST_UPDATE = "forecast_update"
    METADATA_DATASETS_UPDATE = "metadata_datasets_update"


# --- Custom State for Agent ---
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    user_input: str = ""
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    fastapi_response: Optional[List[Dict[str, Any]]] = None
    categorical_features: Optional[List[CategoricalFeatureValues]] = None


# --- UIState Model ---
class UIState(BaseModel):
    user_id: str
    file_path_key: str
    target_columns: Optional[List[str]] = None
    timestamp_column: Optional[str] = None
    covariates: Optional[List[str]] = None
    min_timestamp: Optional[str] = None
    forecast_timestamp: Optional[str] = None
    group_filters: Optional[Dict[str, List[str | int | float]]] = None
    forecast_length: Optional[int] = None
    backtest_stride: Optional[str] = None
    leak_metadata: Optional[bool] = None
    point_modifications: Optional[List[FMDataPointModification]] = None
    full_and_range_modifications: Optional[List[MetaDataVariation]] = None
    dataset_columns: List[str] = Field(default_factory=list)
    possible_timestamp_columns: List[str] = Field(default_factory=list)
    metadata_datasets: Optional[List[HaverDatasetMatch]] = None
    number_of_recommended_metadata_datasets: Optional[int] = None


@tool
def search_metadata_tool(user_input: str) -> Dict[str, Any]:
    """
    Search and retrieve metadata information about available data columns and fields.
    Use this tool when the user asks about data structure, columns, or metadata.
    """
    return {"result": f"[STUB] Searching metadata for: {user_input}"}


@tool
def explain_tool(user_input: str) -> Dict[str, Any]:
    """
    Provide explanations for data patterns, model behavior, or system operations.
    Use this tool when the user asks for explanations or wants to understand something.
    """
    return {"result": f"[STUB] Explaining: {user_input}"}


@tool
def anomaly_detection_tool(user_input: str) -> Dict[str, Any]:
    """
    Identify and analyze unusual patterns or outliers in the data.
    Use this tool when the user wants to detect anomalies or outliers.
    """
    return {"result": f"[STUB] Anomaly detection for: {user_input}"}


def safe_code_check(code: str) -> bool:
    """
    Check if the provided code is safe to execute by validating against dangerous patterns.

    This function performs basic security checks to prevent execution of potentially
    harmful code by checking for dangerous imports and restricting to safe operations.

    Args:
        code (str): The code string to validate for safety

    Returns:
        bool: True if the code is considered safe to execute, False otherwise

    Note:
        This is a basic security check and should not be considered comprehensive.
        Additional security measures should be implemented for production use.
    """
    # Check for dangerous imports
    dangerous_imports = [
        "import os",
        "import subprocess",
        "import shutil",
        "import sys",
    ]
    if any(imp in code.lower() for imp in dangerous_imports):
        return False

    # Allow only pandas import
    if "import" in code.lower() and "import pandas" not in code.lower():
        return False

    return True


# def get_changed_data_point_cards_from_dfs(
#     df_orig: pd.DataFrame, df_mod: pd.DataFrame, timestamp_column: str
# ) -> List[Dict[str, Any]]:
#     """
#     Compare two DataFrames and return a list of data_point_cards (dicts)
#     for rows that have changed (excluding the timestamp column).
#
#     returns:
#         Dict[str, Any] list of dicts with keys: card_index:int, values_to_change:dict (only changed values)
#         List[int] list of original indices that were changed
#     """
#     if not len(df_orig) or not len(df_mod):
#         return []
#
#     if set(df_orig.columns) != set(df_mod.columns):
#         logger.error(
#             f"Columns in df_orig and df_mod must match: {set(df_orig.columns)} != {set(df_mod.columns)}"
#         )
#         return []
#
#     # Reset the index to get the original index
#     df_orig = df_orig.reset_index(drop=True)
#     df_mod = df_mod.reset_index(drop=True)
#
#     # Save the original index mapping (timestamp -> orig index)
#     orig_index_map = {
#         row[timestamp_column]: idx
#         for idx, row in df_orig.reset_index().iterrows()
#     }
#
#     # Ensure both DataFrames are sorted by timestamp for alignment
#     df_orig_indexed = df_orig.set_index(timestamp_column).sort_index()
#     df_mod_indexed = df_mod.set_index(timestamp_column).sort_index()
#
#     # Align indices
#     common_idx = df_orig_indexed.index.intersection(df_mod_indexed.index)
#     # Compare only on common indices
#     changed_mask = (
#         df_orig_indexed.loc[common_idx] != df_mod_indexed.loc[common_idx]
#     ).any(axis=1)
#     changed_idx = common_idx[changed_mask]
#
#     cards_to_change = []
#     for ts in changed_idx:
#         orig_row = df_orig_indexed.loc[ts]
#         mod_row = df_mod_indexed.loc[ts]
#         # Only include changed columns
#         changed_values = {
#             col: mod_row[col]
#             for col in df_mod_indexed.columns
#             if col in orig_row
#             and not pd.isna(orig_row[col])
#             and not pd.isna(mod_row[col])
#             and orig_row[col] != mod_row[col]
#             or (pd.isna(orig_row[col]) != pd.isna(mod_row[col]))
#         }
#         orig_idx = orig_index_map.get(ts, None)
#         cards_to_change.append(
#             {
#                 "card_index": orig_idx,
#                 "values_to_change": changed_values,
#             }
#         )
#
#     return cards_to_change


class MetaDataVariationList(BaseModel):
    variation_list: List[MetaDataVariation] = Field(
        description="A list of MetaDataVariations that indicate the instructions for changing data."
    )


def _what_if_parameter_modification_core(
    user_input: str,
    ui_state: UIState,
    llm_to_use: Optional[ChatGoogleGenerativeAI] = None,
) -> Dict[str, Any]:
    """
    Core logic for what-if parameter modification tool.

    This function analyzes user input to extract data modification instructions and
    converts them into MetaDataVariation objects for scenario analysis.

    Args:
        user_input (str): The user's natural language input describing modifications
        ui_state (UIState): The current UI state containing dataset information
        llm_to_use (Optional[ChatGoogleGenerativeAI]): Optional LLM instance to use.
            If None, uses the global llm instance.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - parameter_updates (Dict): Updates to apply to UI state
            - fastapi_response (Dict): Response data for the API
            - error (str, optional): Error message if processing failed

    Note:
        This function contains the shared implementation used by both the @tool decorated
        version and the testing version.
    """
    llm_to_use = llm_to_use or llm

    if llm_to_use is None:
        return {
            "error": "AI service temporarily unavailable. Please try again later.",
            "parameter_updates": {},
            "fastapi_response": {},
        }

    parser = PydanticOutputParser(pydantic_object=MetaDataVariationList)
    format_instructions = parser.get_format_instructions()

    # Enhanced few-shot examples with more comprehensive coverage
    few_shot_examples = """
        Examples (CRITICAL: perturbation_type MUST be one of: add, subtract, multiply, divide, or None):
        
        User prompt: What if the temperature is 10 degrees higher and the cost goes down to 53?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'temperature', 'value': 10, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'add'},
        {'name': 'cost', 'value': 53, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: What if the sales increase 30%?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'sales', 'value': 1.3, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: If sales drop by 20 percent, what happens?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'sales', 'value': 0.8, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: What if revenue decreases by 15%?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'revenue', 'value': 0.85, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: Set the Unemployment_Rate to 5.1 for the next month.
        Available columns: ['Temperature', 'Unemployment_Rate', 'Sales', 'Revenue', 'Price', 'Demand', 'Cost', 'Profit']
        Output: [
        {'name': 'Unemployment_Rate', 'value': 5.1, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: If the s@les drop by 20%, what happens?
        Available columns: ['temperature', 'unemployment_rate', 's@les', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 's@les', 'value': 0.8, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: Assume revenue increases by 1000.
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'REVENUE', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'REVENUE', 'value': 1000, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'add'}
        ]

        User prompt: What if the price is reduced by 5 percent?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'store_price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'store_price', 'value': 0.95, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: If the cost doubles, and the acquisition_cost halves, what's the effect?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit', 'acquisition_cost']
        Output: [
        {'name': 'cost', 'value': 2, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'},
        {'name': 'acquisition_cost', 'value': 0.5, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: What if the profit is divided by 2?
        Available columns: ['temperature', 'unemployment_rate', 'sales', 'revenue', 'price', 'demand', 'cost', 'profit']
        Output: [
        {'name': 'profit', 'value': 2, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'divide'}
        ]

        User prompt: Set the price to 10 and the demand to 500.
        Available columns: ['price', 'demand', 'sales']
        Output: [
        {'name': 'price', 'value': 10, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'demand', 'value': 500, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: If cost increases by 5 and profit drops by 10%.
        Available columns: ['cost', 'profit', 'revenue']
        Output: [
        {'name': 'cost', 'value': 5, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'add'},
        {'name': 'profit', 'value': 0.9, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: set the sales to 1000, and the revenue to 2000!
        Available columns: ['SALES', 'REVENUE', 'profit']
        Output: [
        {'name': 'SALES', 'value': 1000, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'REVENUE', 'value': 2000, 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: What if we subtract 100 from the revenue?
        Available columns: ['sales', 'revenue', 'profit']
        Output: [
        {'name': 'revenue', 'value': 100, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'subtract'}
        ]

        User prompt: Increase the price by 50%.
        Available columns: ['price', 'quantity', 'sales']
        Output: [
        {'name': 'price', 'value': 1.5, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: What if sales grow 25 percent?
        Available columns: ['sales', 'revenue', 'customers']
        Output: [
        {'name': 'sales', 'value': 1.25, 'perturbation_or_exact_value': 'perturbation', 'perturbation_type': 'multiply'}
        ]

        User prompt: Set store_name to "New York Store" and region to "Northeast".
        Available columns: ['store_name', 'region', 'sales', 'revenue']
        Output: [
        {'name': 'store_name', 'value': 'New York Store', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'region', 'value': 'Northeast', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: Set the sotre to "Main Street" and location info.
        Available columns: ['store_name', 'location_details', 'sales', 'revenue']
        Output: [
        {'name': 'store_name', 'value': 'Main Street', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: Change the name of the store to "Downtown Branch" and stor region to "Central".
        Available columns: ['store_name', 'store_region', 'sales', 'revenue']
        Output: [
        {'name': 'store_name', 'value': 'Downtown Branch', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'store_region', 'value': 'Central', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: Update the custmer name to "John Doe" and customer typ to "Premium".
        Available columns: ['customer_name', 'customer_type', 'sales', 'revenue']
        Output: [
        {'name': 'customer_name', 'value': 'John Doe', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'customer_type', 'value': 'Premium', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: Set product nam to "Widget A" and the catgory to "Electronics".
        Available columns: ['product_name', 'category', 'sales', 'revenue']
        Output: [
        {'name': 'product_name', 'value': 'Widget A', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'category', 'value': 'Electronics', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]

        User prompt: Change the user id to "12345" and user staus to "Active".
        Available columns: ['user_id', 'user_status', 'activity_count']
        Output: [
        {'name': 'user_id', 'value': '12345', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None},
        {'name': 'user_status', 'value': 'Active', 'perturbation_or_exact_value': 'exact_value', 'perturbation_type': None}
        ]
        """

    columns = ui_state.dataset_columns
    prompt = f"""
        You are an expert at extracting data modification instructions for a time series forecasting UI.

        CRITICAL RULES:
        1. perturbation_type MUST be one of: add, subtract, multiply, divide, or None
        2. Do NOT use any other values like "percentage", "increase", "decrease", etc.
        3. For percentage changes, use multiply with the appropriate factor (e.g., +30% = 1.3, -20% = 0.8)
        4. For exact values, set perturbation_type to None
        5. For additions/subtractions, use the raw number (e.g., "increase by 100" = add 100)

        COLUMN MATCHING RULES:
        1. Be VERY FLEXIBLE with column names - handle typos, abbreviations, and variations
        2. Examples of flexible matching:
           - "sotre" should match "store_name"
           - "stor name" should match "store_name" 
           - "name of the store" should match "store_name"
           - "custmer" should match "customer_name"
           - "product nam" should match "product_name"
           - "catgory" should match "category"
        3. Match based on semantic meaning, not exact spelling
        4. Handle missing letters, extra letters, and common typos
        5. Consider word boundaries and partial matches

        Given:
        - The user prompt: "{user_input}"
        - The available columns are: {columns}

        Your task:
        - Identify all columns to modify (use FLEXIBLE matching from available columns, handle typos/capitalization)
        - For each column, determine:
          * Exact value (perturbation_type: None) OR perturbation (add/subtract/multiply/divide)
          * The numeric value or string value
        - For percentages: convert to multiplication factors
          * "increase by X%" = multiply by (1 + X/100)
          * "decrease by X%" = multiply by (1 - X/100)  
          * "double" = multiply by 2
          * "halve" = multiply by 0.5

        {few_shot_examples}

        VALIDATION CHECKLIST:
        - perturbation_type is ONLY: add, subtract, multiply, divide, or None
        - No other perturbation_type values allowed
        - Percentage changes converted to multiply operations
        - String values use exact_value with perturbation_type: None
        - Column names matched FLEXIBLY from available columns (handle typos!)

        {format_instructions}
    """

    chain = llm_to_use | parser
    max_retries = LLM_MAX_RETRIES
    for attempt in range(max_retries):
        try:
            meta_variation_list = chain.invoke(prompt)

            # Validate and correct column names using fuzzy matching
            corrected_variations = []
            for variation in meta_variation_list.variation_list:
                # Find the best matching column name
                best_column_match = _find_best_column_match(
                    variation.name, columns
                )
                if best_column_match:
                    # Create a new variation with the corrected column name
                    corrected_variation = MetaDataVariation(
                        name=best_column_match,
                        value=variation.value,
                        perturbation_or_exact_value=variation.perturbation_or_exact_value,
                        perturbation_type=variation.perturbation_type,
                        order=variation.order,
                    )
                    corrected_variations.append(corrected_variation)
                else:
                    # If no match found, log a warning but keep the original
                    logger.warning(
                        f"Could not find matching column for '{variation.name}' in columns: {columns}"
                    )
                    corrected_variations.append(variation)

            # Update the variation list with corrected column names
            meta_variation_list.variation_list = corrected_variations

            # Validate perturbation_types
            valid_perturbation_types = {
                "add",
                "subtract",
                "multiply",
                "divide",
                None,
            }
            invalid_variations = []

            for variation in meta_variation_list.variation_list:
                if variation.perturbation_type not in valid_perturbation_types:
                    invalid_variations.append(
                        f"'{variation.name}' has invalid perturbation_type: '{variation.perturbation_type}'"
                    )

            if invalid_variations:
                if attempt < max_retries - 1:
                    error_msg = f"Invalid perturbation_types found: {', '.join(invalid_variations)}. Must be one of: add, subtract, multiply, divide, or None."
                    prompt_with_feedback = (
                        prompt + f"\n\n[VALIDATION ERROR]: {error_msg}\n"
                        "Please correct your output to use ONLY the allowed perturbation_types."
                    )
                    prompt = prompt_with_feedback
                    continue
                else:
                    return {
                        "error": f"Failed validation after {max_retries} attempts: {', '.join(invalid_variations)}",
                        "parameter_updates": {},
                        "fastapi_response": {},
                    }

            # Success - all perturbation_types are valid
            break

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                prompt_with_feedback = (
                    prompt + f"\n\n[ERROR FEEDBACK]: {error_msg}\n"
                    "Please ensure perturbation_type is one of: add, subtract, multiply, divide, or None."
                    "Also ensure column names are matched to available columns with flexible matching."
                )
                prompt = prompt_with_feedback
            else:
                return {
                    "error": f"Failed to extract what-if modifications after {max_retries} attempts: {error_msg}",
                    "parameter_updates": {},
                    "fastapi_response": {},
                }

    # Convert MetaDataVariation to MetaDataVariation (as full modifications)
    full_and_range_modifications = []
    for i, mv in enumerate(meta_variation_list.variation_list):
        full_mod = MetaDataVariation(
            name=mv.name,
            value=mv.value,
            perturbation_or_exact_value=mv.perturbation_or_exact_value,
            perturbation_type=mv.perturbation_type,
            order=i + 1,  # Start from 1, apply in the order generated
            # No timestamps = full modification
            min_timestamp=None,
            max_timestamp=None,
        )
        full_and_range_modifications.append(full_mod)

    return {
        "parameter_updates": {
            "full_and_range_modifications": [
                mod.model_dump() for mod in full_and_range_modifications
            ],
        },
        "fastapi_response": {
            "update_type": UIUpdateType.PARAMETERS_UPDATE,
            "update": {
                "full_and_range_modifications": [
                    mod.model_dump() for mod in full_and_range_modifications
                ],
            },
        },
    }


@tool
def what_if_parameter_modification_tool(
    user_input: str, ui_state: UIState
) -> Dict[str, Any]:
    """
    Analyze the user prompt and determines which data should be modified.
    This tool is used to do what-if analysis on the data, scenario simulation, etc.
    Returns a list of MetaDataVariations that indicate the instructions for changing data.

    Uses the faster Gemini 2.0 Flash Lite model for efficient what-if analysis.
    """
    return _what_if_parameter_modification_core(user_input, ui_state, fast_llm)


# Unit test version without @tool decorator for testing
def what_if_parameter_modification_tool_for_testing(
    user_input: str, ui_state: UIState, test_llm=None
) -> Dict[str, Any]:
    """
    Unit test version of what_if_parameter_modification_tool without @tool decorator.
    """
    return _what_if_parameter_modification_core(user_input, ui_state, test_llm)


# Performance test version using gemini-2.0-flash-lite
def what_if_parameter_modification_tool_fast(
    user_input: str, ui_state: UIState
) -> Dict[str, Any]:
    """
    Fast version using gemini-2.0-flash-lite for performance testing.
    """
    return _what_if_parameter_modification_core(user_input, ui_state, fast_llm)


@tool
def get_state_parameters_to_update_tool(
    user_input: str,
    ui_state: UIState,
    categorical_features: Optional[List[CategoricalFeatureValues]] = None,
) -> Dict[str, Any]:
    """
    Analyze the user prompt and the current UI state, and determine which UI parameters should be updated.
    Returns a dict with the parameter_updates (only updated keys) and a result message.

    Uses the more powerful Gemini 2.5 model for complex parameter extraction and fuzzy matching.
    """

    # Use the main LLM for complex parameter extraction with fuzzy matching
    if llm is None:
        return {
            "error": "AI service temporarily unavailable. Please try again later.",
            "parameter_updates": {},
            "fastapi_response": {},
        }

    # TODO - use outputParser to parse the response (mainly group_filters causes problems since the data structure is complex)
    output_parser = PydanticOutputParser(pydantic_object=UIState)
    format_instructions = output_parser.get_format_instructions()

    # Build categorical features context for better filter matching
    categorical_context = _build_categorical_features_context(
        categorical_features
    )

    # Few-shot examples for better LLM extraction with categorical values
    few_shot_examples = """
        Examples:
        User prompt: forecast for the next 30 days
        Current UI state: forecast_length: 10
        Output: {{"forecast_length": 30}}

        User prompt: Change the target columns to ["profit", "revenue"] and show data for region = US
        Current UI state: target_columns: ["sales"], group_filters: {{}} | possible options are: ["Profit_NET", "sales", "revenue", "region"]
        Output: {{"target_columns": ["Profit_NET", "revenue"], "group_filters": {{"region": ["US"]}}}}

        User prompt: show me the sales for store 1
        Current UI state: group_filters: {{}} | Available categorical features: Store: ["store_1", "store_2", "store_3", ...]
        Output: {{"target_columns": ["sales"], "group_filters": {{"Store": ["store_1"]}}}}

        User prompt: show me data for raimi
        Current UI state: group_filters: {{}} | possible column values are: ["userid", "region", "sales"]
        Output: {{"group_filters": {{"userid": ["raimi"]}}}}

        User prompt: Set the timestamp column to "date", leak metadata, and filter by store in [store_1, store_2]
        Current UI state: timestamp_column: "datetime", leak_metadata: False, group_filters: {{}}
        Output: {{"timestamp_column": "date", "leak_metadata": true, "group_filters": {{"store": ["store_1", "store_2"]}}}}

        User prompt: forecast for user_id = 1234567890 | possible column values are: ["userid", "heart_rate"]
        Current UI state: group_filters: {}
        Output: {"group_filters": {"userid": ["1234567890"]}}

        User prompt: set the filter to store is store_1 and plot the timeseris data for revenue
        Current UI state: (any) | possible target columns are: ["weekly_sales", "daily_sales", "revenue_USD", "monthly_sales", "yearly_sales"]
        Output: {{"target_columns": ["revenue_USD"], "group_filters": {{"Store": ["store_1"]}}}}

        User prompt: show me data for store 15
        Current UI state: group_filters: {{}} | Available categorical features: Store: ["store_1", "store_15", "store_20", ...]
        Output: {{"group_filters": {{"Store": ["store_15"]}}}}

        User prompt: show me weekly sales data for store 15
        Current UI state: target_columns: [], group_filters: {{}} | Available categorical features: Store: ["store_1", "store_15", "store_20", ...] | possible target columns are: ["Weekly_Sales", "Daily_Sales", "Monthly_Sales"]
        Output: {{"target_columns": ["Weekly_Sales"], "group_filters": {{"Store": ["store_15"]}}}}

        User prompt: update the timestamp column to "date" and look at the weekly sales
        Current UI state: timestamp_column: "datetime"
        Output: {{"timestamp_column": "date", "target_columns": ["weekly_sales"]}}

        User prompt: Show me a plot of the data
        Current UI state: (any)
        Output: {{}}

        User prompt: Set leak_metadata to true and forecast again for the heart rate.
        Current UI state: leak_metadata: False, forecast_length: 10
        Output: {{"target_columns": ["heart_rate"], "leak_metadata": true, "forecast_length": 10}}

        User prompt: change the forecast horizon to 15 and the timestamp column to "ds", then show me the data.
        Current UI state: forecast_length: 10, timestamp_column: "date"
        Output: {{"forecast_length": 15, "timestamp_column": "ds"}}

        User prompt: add filter for store: store_1 and show the weekly sales data
        Current UI state: group_filters: {{}} | Possible column values are: ["Store", "weekly_sales", "date", "region"]
        Output: {{"target_columns": ["weekly_sales"], "group_filters": {{"Store": ["store_1"]}}}}

        User prompt: Set the number of metadata datasets to 50
        Current UI state: number_of_recommended_metadata_datasets: None
        Output: {{"number_of_recommended_metadata_datasets": 50}}
        
        User prompt: Configure 25 metadata datasets for the next analysis
        Current UI state: number_of_recommended_metadata_datasets: None
        Output: {{"number_of_recommended_metadata_datasets": 25}}
    """

    # Prepare the extraction prompt for the LLM
    extraction_prompt = (
        "You are an expert at configuring time series forecasting UI parameters. You must match the prompt parameter to the possible parameters "
        "for the target columns, timestamp column, covariates, group filters, etc. "
        "You are given the current UI state and a new user request. "
        f"{few_shot_examples}\n"
        f"{categorical_context}\n"
        "The current UI state is as follows:\n"
        f"- target_columns: {ui_state.target_columns}\n | possible options are: [{', '.join(ui_state.dataset_columns)}]"
        f"- timestamp_column: {ui_state.timestamp_column}\n | possible options are: [{', '.join(ui_state.possible_timestamp_columns)}]"
        f"- covariates: {ui_state.covariates}\n | possible options are: [{', '.join(ui_state.dataset_columns)}]"
        f"- min_timestamp: {ui_state.min_timestamp}\n"
        f"- forecast_timestamp: {ui_state.forecast_timestamp}\n"
        f"- group_filters: {ui_state.group_filters}\n | possible column values are: [{', '.join(ui_state.dataset_columns)}], the return structure must be a Dict[str: List[str]]. Note that you may need to infer the column name from the user prompt (only the value may be mentioned)."
        f"- forecast_length: {ui_state.forecast_length}\n"
        f"- backtest_stride: {ui_state.backtest_stride}\n"
        f"- leak_metadata: {ui_state.leak_metadata} (boolean - be True or False)\n"
        f"- number_of_recommended_metadata_datasets: {ui_state.number_of_recommended_metadata_datasets}\n"
        f"User prompt: {user_input}\n\n"
        "Extract ALL relevant parameters that need to be changed or set, not just one."
        "If the user prompt mentions multiple parameters, make sure to include every parameter that is present in the prompt."
        "For group_filters, pay special attention to the categorical features available. When the user mentions values like 'store 1', match them to the closest available value like 'store_1'."
        "For number_of_recommended_metadata_datasets, extract numbers when users configure a quantity of datasets (e.g., 'show me a few' = 5, 'several' = 7, 'many' = 20)."
        "Return them as a Python dictionary with these possible keys: "
        "target_columns, timestamp_column, covariates, min_timestamp, forecast_timestamp, group_filters, forecast_length, backtest_stride, leak_metadata, number_of_recommended_metadata_datasets.\n\n"
        "Return only a Python dictionary with the relevant keys and values. "
        "**If the column type is not specified, then the mentioned columns are likely target columns.**"
        "<Important>: There may be typos/mistakes in the user prompt. So have a bias to match the user prompt to as many of the possible parameters as possible. </Important>"
        "<Important>: group_filters MUST BE A Dict[str: List[str]] format. Make sure this is a proper Python dictionary with string keys and list values. </Important>"
        "<Important>: When extracting group_filters, use fuzzy matching to find the best categorical values. For example, 'store 1' should match 'store_1', 'Store 15' should match 'store_15'. </Important>"
        "<Important>: For boolean values, use lowercase true/false in the JSON output which will be converted to Python True/False. </Important>"
        "<Important>: For number_of_recommended_metadata_datasets, extract explicit numbers (1-200 range) or interpret qualitative terms: 'few'=5, 'several'=7, 'many'=20, 'top X'=X. </Important>"
        f"{format_instructions}"
    )

    # Main extraction with retry logic for format errors
    max_retries = LLM_MAX_RETRIES
    knobs_dict = {}

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = llm.invoke(extraction_prompt)
            end_time = time.time()
            logger.info(
                f"response: {response}, time taken: {end_time - start_time} seconds"
            )

            content = response.content
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            else:
                content = str(content)

            # replace true/false with True/False for boolean values
            content = content.replace(": true", ": True").replace(
                ": false", ": False"
            )

            # Find the first code block or dict in the response
            if "```" in content:
                code = content.split("```", 2)[1].strip()
                # Remove language specifier if present
                if code.startswith("python"):
                    code = code[len("python") :].strip()
                if code.startswith("json"):
                    code = code[len("json") :].strip()
                knobs_dict = ast.literal_eval(code)
            else:
                knobs_dict = ast.literal_eval(content.strip())

            # Validate that group_filters is in correct format if present
            if "group_filters" in knobs_dict:
                if not isinstance(knobs_dict["group_filters"], dict):
                    raise ValueError(
                        f"group_filters must be a dictionary, got {type(knobs_dict['group_filters'])}"
                    )
                for key, value in knobs_dict["group_filters"].items():
                    if not isinstance(value, list):
                        raise ValueError(
                            f"group_filters values must be lists, got {type(value)} for key {key}"
                        )

            # Validate number_of_recommended_metadata_datasets if present
            if "number_of_recommended_metadata_datasets" in knobs_dict:
                num_datasets = knobs_dict[
                    "number_of_recommended_metadata_datasets"
                ]
                if not isinstance(num_datasets, int):
                    raise ValueError(
                        f"number_of_recommended_metadata_datasets must be an integer, got {type(num_datasets)}"
                    )
                if not (1 <= num_datasets <= 200):
                    raise ValueError(
                        f"number_of_recommended_metadata_datasets must be between 1 and 200, got {num_datasets}"
                    )

            # Success - break out of retry loop
            break

        except (ValueError, SyntaxError, TypeError) as e:
            error_msg = str(e)
            logger.warning(
                f"Attempt {attempt + 1} failed with error: {error_msg}"
            )

            if attempt < max_retries - 1:
                # Add error feedback to the prompt for retry
                error_feedback = (
                    f"\n\n[ERROR FEEDBACK]: Your previous response failed to parse correctly. "
                    f"Error: {error_msg}\n"
                    f"Please ensure you return a valid Python dictionary with the exact format specified. "
                    f"For group_filters, use the format: {{'column_name': ['value1', 'value2']}}. "
                    f"For boolean values, use lowercase true/false."
                )
                extraction_prompt = extraction_prompt + error_feedback
            else:
                # Final attempt failed - return error
                return {
                    "error": f"Failed to extract UI parameters after {max_retries} attempts: {error_msg}",
                    "raw_response": str(response.content)
                    if "response" in locals()
                    else "No response",
                    "parameter_updates": {},
                    "fastapi_response": {},
                }

    # --- Reflection step: Ask LLM if all relevant parameters were extracted ---
    reflection_prompt = (
        "You are reviewing the following extraction of UI parameters from a user prompt. "
        "Given the user prompt, the current UI state, and the extracted parameters, answer YES or NO: Did the extraction capture ALL relevant parameters that the user wanted to change? "
        "If not, explain what was missed.\n\n"
        f"User prompt: {user_input}\n"
        f"Current UI state: {ui_state.model_dump()}\n"
        f"Extracted parameters: {knobs_dict}\n\n"
        "Answer with 'YES' if all relevant parameters were extracted, or 'NO' if any were missed, and explain."
    )
    reflection_response = llm.invoke(reflection_prompt)
    reflection_content = str(reflection_response.content).strip().lower()
    logger.info(f"Reflection response: {reflection_content}")

    if reflection_content.startswith("no"):
        # Feedback: try again with explicit feedback
        feedback_prompt = (
            extraction_prompt
            + "\n\nYou previously missed some parameters. Carefully review the user prompt and extract ALL relevant parameters that need to be changed, not just one."
        )
        logger.info("Re-invoking extraction due to reflection feedback.")
        retry_response = llm.invoke(feedback_prompt)
        retry_content = str(retry_response.content)

        try:
            # Parse retry response
            if "```" in retry_content:
                code = retry_content.split("```", 2)[1].strip()
                if code.startswith("python"):
                    code = code[len("python") :].strip()
                if code.startswith("json"):
                    code = code[len("json") :].strip()
                knobs_dict = ast.literal_eval(code)
            else:
                knobs_dict = ast.literal_eval(retry_content.strip())
        except Exception as e:
            logger.warning(f"Failed to parse reflection retry response: {e}")
            # Keep the original knobs_dict from the main extraction

    # --- Improved group_filters post-processing ---
    if "group_filters" in knobs_dict:
        if isinstance(knobs_dict["group_filters"], str):
            # Handle string responses that should be dictionaries
            group_filters_str = knobs_dict["group_filters"].strip()
            knobs_dict["group_filters"] = _parse_group_filters_string(
                group_filters_str,
                categorical_features,
                ui_state.dataset_columns,
            )
        elif isinstance(knobs_dict["group_filters"], dict):
            # Improve existing dictionary with fuzzy matching
            knobs_dict["group_filters"] = _improve_group_filters_dict(
                knobs_dict["group_filters"],
                categorical_features,
                ui_state.dataset_columns,
            )

    return {
        "parameter_updates": knobs_dict or {},
        "fastapi_response": {
            "update_type": UIUpdateType.PARAMETERS_UPDATE,
            "update": knobs_dict or {},
        },
    }


def _parse_group_filters_string(
    group_filters_str: str,
    categorical_features: Optional[List[CategoricalFeatureValues]],
    dataset_columns: List[str],
) -> Dict[str, List[str]]:
    """
    Parse a string representation of group filters into proper dictionary format.

    Handles various input formats including 'Store = store_1', 'region:US',
    'Store=store_1,region=US', and other common filter expressions.

    Args:
        group_filters_str (str): String representation of filters to parse
        categorical_features (Optional[List[CategoricalFeatureValues]]): Available
            categorical features for value matching
        dataset_columns (List[str]): Available dataset column names

    Returns:
        Dict[str, List[str]]: Parsed filters as dictionary with column names as keys
            and lists of filter values as values

    Note:
        Uses fuzzy matching to handle typos and variations in column names and values.
    """
    group_filters_dict = {}

    try:
        s = group_filters_str.strip()
        if not s:
            return {}

        # Handle multiple filters separated by comma, semicolon, or pipe
        # But be careful not to split on commas that are part of value lists
        separators = [";", "|", " and ", " AND "]
        filters = [s]  # Start with the whole string

        # First split by major separators (but not comma yet)
        for sep in separators:
            new_filters = []
            for f in filters:
                new_filters.extend(
                    [x.strip() for x in f.split(sep) if x.strip()]
                )
            filters = new_filters

        # Now handle comma separation more carefully
        # We need to distinguish between filter separators and value list separators
        final_filters = []
        for f in filters:
            # Split by comma only if we can identify multiple key=value pairs
            if "," in f:
                # Check if this looks like multiple filters vs multiple values
                comma_parts = [x.strip() for x in f.split(",")]

                # If all parts have = : is equals or eq, treat as separate filters
                has_operators = []
                for part in comma_parts:
                    has_op = any(
                        op in part
                        for op in ["=", ":", " is ", " equals ", " eq "]
                    )
                    has_operators.append(has_op)

                # If first part has operator and others don't, it's likely multiple values for one key
                if has_operators[0] and not any(has_operators[1:]):
                    final_filters.append(
                        f
                    )  # Keep as single filter with multiple values
                elif all(has_operators):
                    final_filters.extend(
                        comma_parts
                    )  # Split into separate filters
                else:
                    final_filters.append(f)  # Keep as single filter
            else:
                final_filters.append(f)

        # Parse each individual filter
        for f in final_filters:
            if not f:
                continue

            # Try different separators: =, :, 'is', 'equals'
            key, value = None, None

            for sep in ["=", ":", " is ", " equals ", " eq "]:
                if sep in f:
                    parts = f.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        break

            if key and value:
                # Clean up quotes and brackets
                value = value.strip("'\"[]")

                # Handle multiple values like "store_1,store_2"
                # But only if we determined this is a single filter with multiple values
                if "," in value:
                    values = [v.strip().strip("'\"") for v in value.split(",")]
                else:
                    values = [value]

                # Find best matching column name with stricter matching
                best_key = _find_best_column_match(key, dataset_columns)
                if best_key:
                    # Find best matching values using categorical features
                    best_values = _find_best_categorical_values(
                        best_key, values, categorical_features
                    )
                    if best_values:
                        # If we already have this key, extend the values
                        if best_key in group_filters_dict:
                            group_filters_dict[best_key].extend(best_values)
                        else:
                            group_filters_dict[best_key] = best_values

    except Exception as e:
        logger.warning(
            f"Failed to parse group_filters string: {group_filters_str} ({e})"
        )

    return group_filters_dict


def _improve_group_filters_dict(
    group_filters: Dict[str, List[str]],
    categorical_features: Optional[List[CategoricalFeatureValues]],
    dataset_columns: List[str],
) -> Dict[str, List[str]]:
    """
    Improve existing group_filters dictionary by applying fuzzy matching for keys and values.

    Takes an existing group filters dictionary and enhances it by matching column names
    and values to the closest available options in the dataset.

    Args:
        group_filters (Dict[str, List[str]]): Existing group filters to improve
        categorical_features (Optional[List[CategoricalFeatureValues]]): Available
            categorical features for value matching
        dataset_columns (List[str]): Available dataset column names

    Returns:
        Dict[str, List[str]]: Improved filters with better matched column names and values
    """
    improved_filters = {}

    for key, values in group_filters.items():
        # Find best matching column name
        best_key = _find_best_column_match(key, dataset_columns)
        if not best_key:
            best_key = key  # Keep original if no match found

        # Find best matching values
        if isinstance(values, list):
            best_values = _find_best_categorical_values(
                best_key, values, categorical_features
            )
        else:
            best_values = _find_best_categorical_values(
                best_key, [str(values)], categorical_features
            )

        if best_values:
            improved_filters[best_key] = best_values
        else:
            improved_filters[best_key] = (
                values if isinstance(values, list) else [str(values)]
            )

    return improved_filters


def _get_semantic_similarity_score(query_str: str, col_str: str) -> float:
    """Calculate semantic similarity considering context and meaning."""
    query_str = query_str.lower()
    col_str = col_str.lower()

    # Higher score for more descriptive matches
    descriptiveness_bonus = len(col_str.split("_")) * 0.1

    # Penalty for single-character columns when query is longer
    if len(col_str) == 1 and len(query_str) > 1:
        return 0.1

    # Bonus for containing the query as a word/part
    if query_str in col_str:
        # Even higher bonus if it's a word boundary
        if (
            f"_{query_str}_" in f"_{col_str}_"
            or col_str.startswith(query_str)
            or col_str.endswith(query_str)
        ):
            return 0.9 + descriptiveness_bonus
        return 0.8 + descriptiveness_bonus

    # Check if query words are in column
    query_words = query_str.replace("_", " ").replace("-", " ").split()
    col_words = col_str.replace("_", " ").replace("-", " ").split()

    # Special handling for common semantic relationships
    semantic_relationships = {
        "customer": ["customer", "client", "user"],
        "store": ["store", "shop", "location", "outlet"],
        "product": ["product", "item", "goods"],
        "user": ["user", "customer", "userid"],
        "region": ["region", "area", "location", "zone"],
        "category": ["category", "type", "class", "group"],
        "status": ["status", "state", "condition"],
        "name": ["name", "title", "label", "identifier"],
    }

    # Calculate word matches with semantic relationships AND character similarity
    word_matches = 0
    total_query_words = len(query_words)

    for qw in query_words:
        found_match = False
        for cw in col_words:
            # Direct word match (exact or substring)
            if len(qw) >= 3 and len(cw) >= 3:  # Only for meaningful words
                if qw in cw or cw in qw:
                    word_matches += 1
                    found_match = True
                    break

            # Character similarity match for typos (like "sotre" vs "store")
            if not found_match and len(qw) >= 3 and len(cw) >= 3:
                char_similarity = sum(
                    1 for a, b in zip(qw, cw) if a == b
                ) / max(len(qw), len(cw))
                # Lower threshold for better typo handling, but require similar lengths
                if char_similarity >= 0.6 and abs(len(qw) - len(cw)) <= 2:
                    word_matches += 1
                    found_match = True
                    break

            # Semantic relationship match (only for exact matches in relationships)
            if not found_match:
                for key, synonyms in semantic_relationships.items():
                    if qw in synonyms and cw in synonyms:
                        word_matches += 1
                        found_match = True
                        break

            if found_match:
                break

    # Require at least 50% of query words to match and at least one word match
    if word_matches > 0 and word_matches >= total_query_words * 0.5:
        return (word_matches / total_query_words) * 0.7 + descriptiveness_bonus

    # Check for partial word matches (handles typos like "sotre" in "store_name")
    # This is now secondary since we handle character similarity above
    for qw in query_words:
        if len(qw) >= 4:  # Only check partial matches for longer words
            for cw in col_words:
                if len(cw) >= 4:  # Only match against longer words
                    # Calculate character similarity for individual words
                    char_similarity = sum(
                        1 for a, b in zip(qw, cw) if a == b
                    ) / max(len(qw), len(cw))
                    # Keep this as fallback for very similar words
                    if char_similarity >= 0.8 and abs(len(qw) - len(cw)) <= 1:
                        return 0.6 + descriptiveness_bonus

    return 0.0


def _calculate_string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using Levenshtein-like approach"""
    s1, s2 = s1.lower(), s2.lower()

    # If one string is much shorter, check if it's contained in the other
    if len(s1) < len(s2) * 0.5:
        return 0.8 if s1 in s2 else 0.0
    if len(s2) < len(s1) * 0.5:
        return 0.8 if s2 in s1 else 0.0

    # Simple character-based similarity
    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    max_len = max(len(s1), len(s2))
    return matches / max_len if max_len > 0 else 0.0


def _find_best_column_match(
    query: str, dataset_columns: List[str]
) -> Optional[str]:
    """
    Find the best matching column name using fuzzy matching techniques.

    Attempts to match a query string to available dataset columns using various
    matching strategies including exact match, substring matching, typo handling,
    and common abbreviation handling.

    Args:
        query (str): The column name query to match
        dataset_columns (List[str]): List of available column names in the dataset

    Returns:
        Optional[str]: The best matching column name, or None if no match found

    Note:
        Uses multiple matching strategies in order of preference:
        1. Exact case-insensitive match
        2. Semantic and contextual matching
        3. Fuzzy matching with typo tolerance
        4. Substring matching
        5. Common abbreviation mapping
        6. Keyword-based matching for complex phrases
    """
    if not query or not dataset_columns:
        return None

    query_lower = query.lower().strip()

    # Exact match first (case-sensitive, then case-insensitive)
    for col in dataset_columns:
        if col == query.strip():  # Exact case-sensitive match
            logger.debug(f"🎯 Exact case-sensitive match: '{query}' -> '{col}'")
            return col

    for col in dataset_columns:
        if col.lower() == query_lower:  # Exact case-insensitive match
            logger.debug(
                f"🎯 Exact case-insensitive match: '{query}' -> '{col}'"
            )
            return col

    # If we reach here, we need fuzzy matching - log this for frequency tracking
    logger.info(
        f"🔍 FUZZY MATCHING NEEDED: '{query}' not found exactly in columns: {dataset_columns}"
    )

    # Handle phrases like "name of the store" -> "store_name" BEFORE other matching
    phrase_mappings = {
        "name of the store": ["store_name", "shop_name"],
        "store name": ["store_name", "shop_name"],
        "stor name": ["store_name", "shop_name"],
        "customer name": ["customer_name", "client_name"],
        "user name": ["user_name", "username", "user_id"],
        "product name": ["product_name", "item_name"],
        "category name": ["category_name", "category"],
        "region name": ["region_name", "region"],
    }

    # Check if the query matches any known phrases
    for phrase, possible_cols in phrase_mappings.items():
        if query_lower == phrase or phrase in query_lower:
            for possible_col in possible_cols:
                for col in dataset_columns:
                    if (
                        col.lower() == possible_col
                        or possible_col in col.lower()
                    ):
                        logger.info(
                            f"✅ PHRASE MATCH: '{query}' -> '{col}' (via phrase mapping)"
                        )
                        return col

    # Semantic matching with context awareness - using extracted function
    semantic_candidates = []
    for col in dataset_columns:
        score = _get_semantic_similarity_score(query_lower, col)
        if (
            score > 0.6
        ):  # Raised threshold from 0.4 to 0.6 for stricter matching
            semantic_candidates.append((col, score))

    if semantic_candidates:
        # Sort by score (descending) and return the best
        semantic_candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = semantic_candidates[0][0]
        logger.info(
            f"✅ SEMANTIC MATCH: '{query}' -> '{best_match}' (score: {semantic_candidates[0][1]:.3f})"
        )
        return best_match

    # Fuzzy matching with typo tolerance - using extracted function
    fuzzy_candidates = []

    for col in dataset_columns:
        col_lower = col.lower()

        # Direct similarity
        score = _calculate_string_similarity(query_lower, col_lower)
        if (
            score >= 0.7
        ):  # Raised threshold from 0.5 to 0.7 for stricter matching
            # Bonus for more descriptive column names
            descriptiveness_bonus = len(col.split("_")) * 0.05
            fuzzy_candidates.append((col, score + descriptiveness_bonus))

        # Try without underscores and separators
        query_clean = (
            query_lower.replace("_", "").replace("-", "").replace(" ", "")
        )
        col_clean = col_lower.replace("_", "").replace("-", "").replace(" ", "")
        score = _calculate_string_similarity(query_clean, col_clean)
        if (
            score >= 0.7
        ):  # Raised threshold from 0.5 to 0.7 for stricter matching
            descriptiveness_bonus = len(col.split("_")) * 0.05
            fuzzy_candidates.append((col, score + descriptiveness_bonus))

    if fuzzy_candidates:
        # Sort by score and return best match
        fuzzy_candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = fuzzy_candidates[0][0]
        logger.info(
            f"✅ FUZZY MATCH: '{query}' -> '{best_match}' (score: {fuzzy_candidates[0][1]:.3f})"
        )
        return best_match

    # Partial match - query is substring of column (with preference for longer columns)
    substring_candidates = []
    for col in dataset_columns:
        if query_lower in col.lower():
            # Prefer longer, more descriptive column names
            score = len(col) + len(col.split("_")) * 2
            substring_candidates.append((col, score))

    if substring_candidates:
        substring_candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = substring_candidates[0][0]
        logger.info(
            f"✅ SUBSTRING MATCH: '{query}' -> '{best_match}' (query in column)"
        )
        return best_match

    # Partial match - column is substring of query
    for col in dataset_columns:
        if col.lower() in query_lower:
            logger.info(
                f"✅ REVERSE SUBSTRING MATCH: '{query}' -> '{col}' (column in query)"
            )
            return col

    # Handle common abbreviations and variations with more flexibility
    common_mappings = {
        "store": ["store", "shop", "location", "outlet", "branch"],
        "name": ["name", "title", "label", "identifier"],
        "user": [
            "user",
            "customer",
            "userid",
            "user_id",
            "customer_id",
            "client",
        ],
        "region": ["region", "area", "location", "zone", "territory"],
        "product": ["product", "item", "goods", "merchandise"],
        "category": ["category", "type", "class", "group"],
        "priority": ["priority", "level", "importance", "rank"],
        "department": ["department", "dept", "division", "section"],
        "segment": ["segment", "group", "cluster", "category"],
        "status": ["status", "state", "condition"],
        "id": ["id", "identifier", "key", "number"],
        "type": ["type", "kind", "category", "class"],
    }

    # Extract words from query for keyword matching
    query_words = query_lower.replace("_", " ").replace("-", " ").split()

    keyword_candidates = []
    for col in dataset_columns:
        col_lower = col.lower()
        col_words = col_lower.replace("_", " ").replace("-", " ").split()

        # Check if query words match column words (with abbreviations)
        matches = 0
        total_query_words = len(query_words)

        for query_word in query_words:
            # Skip very short words that could cause false matches
            if len(query_word) < 3:
                continue

            found_match = False
            for col_word in col_words:
                # Direct exact match (case insensitive)
                if query_word == col_word:
                    matches += 1
                    found_match = True
                    break

                # Semantic relationship match - but only for exact matches
                if not found_match:
                    for canonical, variations in common_mappings.items():
                        # Both words must be exactly in the variations list
                        if query_word in variations and col_word in variations:
                            # Additional check: don't match very different words
                            # "column" and "user" should not match even if both are in some category
                            word_similarity = sum(
                                1
                                for a, b in zip(query_word, col_word)
                                if a == b
                            ) / max(len(query_word), len(col_word))
                            if (
                                word_similarity >= 0.3
                            ):  # Require some character similarity
                                matches += 1
                                found_match = True
                                break

                if found_match:
                    break

        # Much stricter requirements for keyword matching
        # Require at least 80% of query words to match AND at least 2 word matches
        # OR if single word query, require exact match
        if total_query_words == 1:
            # For single word queries, require exact match or very high similarity
            if matches >= 1:
                # Additional validation: check overall similarity
                overall_similarity = sum(
                    1 for a, b in zip(query_lower, col_lower) if a == b
                ) / max(len(query_lower), len(col_lower))
                if overall_similarity >= 0.6:
                    score = 1.0 + len(col.split("_")) * 0.1
                    keyword_candidates.append((col, score))
        else:
            # For multi-word queries, require high match ratio AND multiple matches
            match_ratio = matches / total_query_words
            if matches >= 2 and match_ratio >= 0.8:
                score = match_ratio + len(col.split("_")) * 0.1
                keyword_candidates.append((col, score))

    if keyword_candidates:
        keyword_candidates.sort(key=lambda x: x[1], reverse=True)
        best_match = keyword_candidates[0][0]

        # Final validation: ensure the match makes semantic sense
        # Check if the best match has reasonable similarity to the original query
        best_match_lower = best_match.lower()

        # Calculate a comprehensive similarity score
        char_similarity = sum(
            1 for a, b in zip(query_lower, best_match_lower) if a == b
        ) / max(len(query_lower), len(best_match_lower))

        # Check if query is a reasonable substring or the column is reasonable substring
        substring_match = (
            query_lower in best_match_lower
            or best_match_lower in query_lower
            or any(qw in best_match_lower for qw in query_words if len(qw) >= 4)
        )

        # Only return match if it passes final validation
        if char_similarity >= 0.3 or substring_match:
            logger.info(
                f"✅ KEYWORD MATCH: '{query}' -> '{best_match}' (score: {keyword_candidates[0][1]:.3f}, char_sim: {char_similarity:.3f})"
            )
            return best_match
        else:
            logger.info(
                f"❌ KEYWORD MATCH REJECTED: '{query}' -> '{best_match}' failed final validation (char_sim: {char_similarity:.3f})"
            )

    # If we reach here, no match was found
    logger.warning(
        f"❌ NO MATCH FOUND: '{query}' could not be matched to any column in {dataset_columns}"
    )
    return None


def _find_best_categorical_values(
    column_name: str,
    query_values: List[str],
    categorical_features: Optional[List[CategoricalFeatureValues]],
) -> List[str]:
    """
    Find the best matching categorical values using fuzzy matching.

    Matches a list of query values to the closest available categorical values
    for a specific column, handling common variations and transformations.

    Args:
        column_name (str): Name of the column to find values for
        query_values (List[str]): List of values to match against available options
        categorical_features (Optional[List[CategoricalFeatureValues]]): Available
            categorical features with their possible values

    Returns:
        List[str]: List of best matching values from the available categorical values.
            If no match is found for a query value, the original value is preserved.

    Note:
        Handles common transformations like space-to-underscore conversion and
        case variations to find the best matches.
    """
    if not categorical_features or not query_values:
        return query_values

    # Find the categorical feature for this column
    categorical_feature = None
    for feature in categorical_features:
        if (
            feature.feature_name
            and feature.feature_name.lower() == column_name.lower()
        ):
            categorical_feature = feature
            break

    if not categorical_feature or not categorical_feature.values:
        return query_values

    available_values = categorical_feature.values
    matched_values = []

    for query_val in query_values:
        query_val_lower = str(query_val).lower().strip()
        best_match = None

        # Exact match first
        for val in available_values:
            if str(val).lower() == query_val_lower:
                best_match = val
                break

        # Partial matches
        if not best_match:
            for val in available_values:
                val_lower = str(val).lower()
                if query_val_lower in val_lower or val_lower in query_val_lower:
                    best_match = val
                    break

        # Handle common transformations
        if not best_match:
            # Handle space to underscore conversion
            query_underscore = query_val_lower.replace(" ", "_")
            query_no_space = query_val_lower.replace(" ", "")
            # Handle hyphen to underscore conversion
            query_hyphen_to_underscore = query_val_lower.replace("-", "_")

            for val in available_values:
                val_lower = str(val).lower()
                val_no_underscore = val_lower.replace("_", "")
                val_no_hyphen = val_lower.replace("-", "")

                # Try various transformations
                if (
                    val_lower == query_underscore
                    or val_lower == query_no_space
                    or val_lower == query_hyphen_to_underscore
                    or val_no_underscore == query_no_space
                    or val_no_hyphen == query_no_space
                    or val_lower.replace("_", "") == query_no_space
                    or val_lower.replace("-", "") == query_no_space
                ):
                    best_match = val
                    break

        if best_match:
            matched_values.append(best_match)
        else:
            # Keep original if no match found
            matched_values.append(query_val)

    return matched_values


def _build_categorical_features_context(
    categorical_features: Optional[List[CategoricalFeatureValues]],
) -> str:
    """
    Build a formatted context string for categorical features to include in LLM prompts.

    This function creates a structured text representation of available categorical features
    and their values, limiting the number of displayed values to prevent overwhelming
    the LLM prompt.

    Args:
        categorical_features (Optional[List[CategoricalFeatureValues]]): List of categorical
            features with their possible values. Can be None or empty.

    Returns:
        str: A formatted context string. Returns empty string if no categorical features
            are provided. Otherwise returns a string in the format:

            ```

            Categorical Features Available:
            - feature_name_1: ['value1', 'value2', 'value3', '...']
            - feature_name_2: ['valueA', 'valueB', 'valueC']
            ```

            The '...' is added when a feature has more than MAX_CATEGORICAL_VALUES_PREVIEW
            values to indicate truncation.

    Example:
        >>> features = [
        ...     CategoricalFeatureValues(feature_name="Store", values=["store_1", "store_2", "store_3"]),
        ...     CategoricalFeatureValues(feature_name="Region", values=["North", "South"])
        ... ]
        >>> result = _build_categorical_features_context(features)
        >>> print(result)

        Categorical Features Available:
        - Store: ['store_1', 'store_2', 'store_3']
        - Region: ['North', 'South']

    Note:
        Only features with both a valid feature_name and non-empty values list are included.
        The number of values shown per feature is limited by MAX_CATEGORICAL_VALUES_PREVIEW.
    """
    if not categorical_features:
        return ""

    categorical_context = "\n\nCategorical Features Available:\n"
    for feature in categorical_features:
        feature_name = feature.feature_name
        values = feature.values
        if feature_name and values:
            # Show only first few values to avoid overwhelming the prompt
            values_preview = values[:MAX_CATEGORICAL_VALUES_PREVIEW] + (
                ["..."] if len(values) > MAX_CATEGORICAL_VALUES_PREVIEW else []
            )
            categorical_context += f"- {feature_name}: {values_preview}\n"

    return categorical_context


# Create a faster LLM instance using gemini-2.0-flash-lite for performance testing
def create_fast_llm() -> Optional[ChatGoogleGenerativeAI]:
    """
    Create a faster LLM instance using gemini-2.0-flash-lite for performance testing.

    Initializes a ChatGoogleGenerativeAI instance configured with the lightweight
    gemini-2.0-flash-lite model for scenarios requiring faster response times.

    Returns:
        Optional[ChatGoogleGenerativeAI]: Configured LLM instance, or None if
            initialization fails due to missing API key or other errors

    Raises:
        Logs warnings for initialization failures but does not raise exceptions

    Note:
        Requires GEMINI_API_KEY environment variable to be set.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODELS["gemini-2.0-flash-lite"],
            temperature=0,
            max_retries=DEFAULT_MAX_RETRIES,
            google_api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"Error initializing fast LLM: {e}")
        return None


@tool
async def filter_and_visualize_tool(
    user_input: str, ui_state: UIState
) -> Dict[str, Any]:
    """
    PREREQUISITE: You must call get_state_parameters_to_update_tool FIRST to set target_columns and group_filters.

    Filter and visualize a dataset based on UI state parameters.
    This tool requires that target_columns and group_filters are already set in the UI state.

    IMPORTANT: This tool will fail if group_filters is None or empty. Always call get_state_parameters_to_update_tool first.

    Description:
    - Filter a dataset based on the current UI state (not user input directly).
        the function: run_filter_core_code is defined in foundation_models.py and returns a FilterResponse;
        After running the filter, the UIState is updated with the new filtered dataset.
        UIState.file_path_key is updated with FilterResponse.filtered_dataset_key.

        This will be automatically visualized on the UI from the file_path_key.

        FilterResponse:
            status: StatusCode
            message: str
            filtered_dataset_key: str

    Inputs:
    - user_input: str - the user's input/prompt
    - ui_state: UIState - the current UI state

    Outputs:
    - Dict[str, Any] - dict with keys:
        - fastapi_response: Dict[str, Any] - the fastapi response to be sent to the user
        - error: str | None - the error message if any
        - raw_response: str | None - the raw response from the LLM (for debugging)

    """

    from synthefy_pkg.app.routers.foundation_models import (
        FilterRequest,
        GroupLabelColumnFilters,
        run_filter_core_code,
    )

    logger.info(f"filter_and_visualize_tool: {ui_state}")

    if not ui_state.group_filters:
        return {
            "fastapi_response": {
                "update_type": UIUpdateType.FILTER_UPDATE,
                "update": {
                    "status": 400,
                    "message": "Cannot filter data: no group_filters set. You must call get_state_parameters_to_update_tool FIRST to set the required parameters (target_columns and group_filters) before calling this filter tool.",
                },
            }
        }

    try:
        # Create the FilterRequest
        filter_request = FilterRequest(
            user_id=ui_state.user_id,
            file_key=ui_state.file_path_key,
            filters=GroupLabelColumnFilters(
                filter=[ui_state.group_filters]
                if ui_state.group_filters
                else [{}],
            ),
            timestamp_column=ui_state.timestamp_column,  # type: ignore
        )
        session = aioboto3.Session()
        # TODO: BUG: we must do the filter on the original dataset, not the filtered dataset.
        # so we need to store/save the original dataset key somewhere.
        # can we do it in the session?
        filter_response = await run_filter_core_code(
            filter_request=filter_request,
            aioboto3_session=session,
        )
        return {
            "parameter_updates": {
                "file_path_key": filter_response.filtered_dataset_key
            },
            "fastapi_response": {
                "update_type": UIUpdateType.FILTER_UPDATE,
                "update": filter_response.model_dump(),
            },
        }
    except Exception as e:
        return {
            "fastapi_response": {
                "update_type": UIUpdateType.FILTER_UPDATE,
                "update": {
                    "status": 500,
                    "message": f"Error filtering dataset: {str(e)}",
                },
            }
        }


# --- Tool Definitions ---
@tool
async def forecast_tool(user_input: str, ui_state: UIState) -> Dict[str, Any]:
    """
    Analyze input and run forecasting models when prediction-related queries are detected.
    Use this tool when the user wants to forecast, predict, or see future values.
    Construct the input to the do_forecast_core_code function from the user input and the UI state.

    Return:
        dict with keys:
        fastapi_response: dict with keys:
            update_type: UIUpdateType.FORECAST_UPDATE
            update: dict with keys:
                status: 200
                message: str
                forecast_response: dict

    """
    from synthefy_pkg.app.routers.foundation_models import do_forecast_core_code

    missing_fields = []
    if ui_state.user_id is None:
        missing_fields.append("user_id")
    if ui_state.file_path_key is None:
        missing_fields.append("file_path_key")
    if ui_state.target_columns is None:
        missing_fields.append("target_columns")
    if ui_state.timestamp_column is None:
        missing_fields.append("timestamp_column")
    if ui_state.forecast_length is None:
        missing_fields.append("forecast_length")
    if ui_state.min_timestamp is None:
        missing_fields.append("min_timestamp")
    if ui_state.forecast_timestamp is None:
        missing_fields.append("forecast_timestamp")

    if missing_fields:
        return {
            "fastapi_response": {
                "update_type": UIUpdateType.FORECAST_UPDATE,
                "update": {
                    "status": 400,
                    "message": f"Please set the following required fields: {', '.join(missing_fields)}",
                    "forecast_response": None,
                },
            }
        }
    else:
        try:
            request = ApiForecastRequest(
                user_id=ui_state.user_id,
                config=FoundationModelConfig(
                    file_path_key=ui_state.file_path_key,
                    model_type=SynthefyTimeSeriesModelType.DEFAULT,
                    timeseries_columns=ui_state.target_columns,  # type:ignore
                    timestamp_column=ui_state.timestamp_column,  # type:ignore
                    forecast_length=ui_state.forecast_length,  # type:ignore
                    covariate_columns=ui_state.covariates or [],
                    min_timestamp=ui_state.min_timestamp,  # type:ignore
                    forecasting_timestamp=ui_state.forecast_timestamp,  # type:ignore
                    point_modifications=ui_state.point_modifications,
                    full_and_range_modifications=ui_state.full_and_range_modifications,
                ),
            )
            session = aioboto3.Session()

            forecast_response = await do_forecast_core_code(
                request,
                session,
            )
            return {
                "fastapi_response": {
                    "update_type": UIUpdateType.FORECAST_UPDATE,
                    "update": forecast_response.model_dump(),
                }
            }
        except Exception as e:
            logger.error(f"Error forecasting: {str(e)}")
            return {
                "error": "Sorry, something went wrong internally. Please try again later.",
                "parameter_updates": {},
                "fastapi_response": {},
            }


@tool
async def find_metadata_datasets(
    user_input: str,
    ui_state: UIState,  # type: ignore
) -> Dict[str, Any]:
    """
    Find and retrieve metadata datasets based on the user's prompt.
    Use this tool when the user asks about finding external data sources or metadata datasets.
    The number of datasets to return can be specified directly in the user prompt (e.g., "Give me 20 datasets")
    or set via get_state_parameters_to_update_tool in the UI state.
    """
    try:
        # Primary: Extract number directly from user input for joint queries like "Give me 20 datasets about..."
        extracted_number = _extract_metadata_dataset_number_from_input(
            user_input
        )

        # Secondary: Check UI state for pre-set values (from get_state_parameters_to_update_tool)
        ui_state_number = None
        if ui_state.number_of_recommended_metadata_datasets is not None:
            if 1 <= ui_state.number_of_recommended_metadata_datasets <= 200:
                ui_state_number = (
                    ui_state.number_of_recommended_metadata_datasets
                )
            else:
                logger.warning(
                    f"UI state number_of_recommended_metadata_datasets ({ui_state.number_of_recommended_metadata_datasets}) "
                    f"is outside valid range (1-200). Ignoring."
                )

        # Determine final number with priority: extracted > UI state > default
        if extracted_number is not None:
            num_datasets_to_return = extracted_number
            source = "extracted from user input"
        elif ui_state_number is not None:
            num_datasets_to_return = ui_state_number
            source = "from UI state"
        else:
            num_datasets_to_return = 50  # Default value
            source = "default"

        logger.info(
            f"find_metadata_datasets: Using {num_datasets_to_return} datasets ({source}) for query: '{user_input}'"
        )

        # Get metadata recommendations using the user input and the determined number
        metadata_datasets = await get_metadata_recommendations(
            user_input, num_datasets_to_return
        )

        return {
            "fastapi_response": {
                "update_type": UIUpdateType.METADATA_DATASETS_UPDATE,
                "update": metadata_datasets,
            }
        }
    except Exception as e:
        return {
            "fastapi_response": {
                "update_type": UIUpdateType.METADATA_DATASETS_UPDATE,
                "update": {
                    "status": 500,
                    "message": f"Error finding metadata datasets: {str(e)}",
                },
            }
        }


# --- TOOLS List ---
TOOLS = [
    get_state_parameters_to_update_tool,
    forecast_tool,
    search_metadata_tool,
    explain_tool,
    anomaly_detection_tool,
    filter_and_visualize_tool,
    what_if_parameter_modification_tool,
    find_metadata_datasets,
]
# --- LLM Setup ---
# Model Usage Strategy:
# - Gemini 2.5 Flash (main LLM): Complex tasks like parameter extraction with fuzzy matching, agent decisions, conversations
# - Gemini 2.0 Flash Lite (fast LLM): Simple tasks like what-if data modifications, basic operations
try:
    api_key = os.getenv("GEMINI_API_KEY")

    # Initialize main LLM (2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODELS["gemini-2.5-flash-preview-05-20"],
        temperature=0,
        max_retries=DEFAULT_MAX_RETRIES,
        google_api_key=api_key,
    )

    # Initialize fast LLM (2.0 Flash Lite)
    fast_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODELS["gemini-2.0-flash-lite"],
        temperature=0,
        max_retries=DEFAULT_MAX_RETRIES,
        google_api_key=api_key,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

except Exception as e:
    logger.warning(f"Error initializing LLMs: {e}")
    llm = None
    fast_llm = None
    llm_with_tools = None


# --- Agent Nodes ---
async def agent_node(state: AgentState) -> AgentState:
    """
    The main agent node that decides which tool to call next based on the conversation.

    This is the core decision-making component of the agent that analyzes the current
    state and conversation history to determine appropriate tool usage.

    Args:
        state (AgentState): Current state of the agent including messages, UI state,
            and previous tool outputs

    Returns:
        AgentState: Updated agent state with new messages and potential tool calls

    Note:
        If the LLM is not available, returns an error state with an appropriate message.
        The function adds system context about the current UI state to guide tool selection.
    """

    if llm_with_tools is None:
        # Return error state if llm_with_tools is None
        error_message = AIMessage(
            content="AI service temporarily unavailable. Please try again later."
        )
        messages = state.messages + [error_message]
        return AgentState(
            messages=messages,
            ui_state=state.ui_state,
            user_input=state.user_input,
            tool_outputs=state.tool_outputs,
            fastapi_response=state.fastapi_response,
            categorical_features=state.categorical_features,
        )

    messages = state.messages
    ui_state_dict = state.ui_state

    # Add current UI state as context to the system message
    ui_context = f"Current UI State: {ui_state_dict}"
    system_message = f"""You are a "Synthefy Agent" - an assistant for a GenAI Time Series Forecasting and Analysis Platform for the company Synthefy.

    {ui_context}

    **CRITICAL: You MUST follow the correct tool ordering rules below!**

    **MANDATORY TOOL ORDERING:**
    
    For ANY request that involves any combination of filtering, showing data, changing parameters, or forecasting:
    1. **ALWAYS call get_state_parameters_to_update_tool FIRST** to set the required parameters
    2. **ONLY THEN call filter_and_visualize_tool** to apply the filters
    3. **ONLY THEN call forecast_tool** to generate forecasts
    
    For what-if scenarios and data modifications:
    1. **Call what_if_parameter_modification_tool** to set up data modifications
    2. **THEN call get_state_parameters_to_update_tool** to set target_columns and group_filters if needed
    3. **FINALLY call filter_and_visualize_tool** to visualize the modified data
    
    **CRITICAL FOR WHAT-IF SCENARIOS**: Always call get_state_parameters_to_update_tool after what_if_parameter_modification_tool 
    and before filter_and_visualize_tool. This is mandatory - filter_and_visualize_tool will fail without proper parameters set.
    
    **Example: "show me weekly sales data for store 15"**
    ✅ CORRECT ORDER:
    1. Call get_state_parameters_to_update_tool (sets target_columns=["Weekly_Sales"], group_filters={{"Store": ["store_15"]}})
    2. Call filter_and_visualize_tool (applies the filters)
    
    **Example: "simulate sales increase by 30%"**
    ✅ CORRECT ORDER:
    1. Call what_if_parameter_modification_tool (sets up 30% increase modification)
    2. Call get_state_parameters_to_update_tool (sets target_columns=["Weekly_Sales"] if needed)
    3. Call filter_and_visualize_tool (visualizes the modified data)

    **Example: "set the timeseries column to weekly sales, filter by store on store_1, forecast length to 30, 
    find me 10 metadata datasets concerning GDP, then forecast again with leaking data"
    ✅ CORRECT ORDER:
    1. Call get_state_parameters_to_update_tool (sets leak_metadata=True, target_columns=["Weekly_Sales"], group_filters={{"Store": ["store_15"]}}, 
    forecast_length=30, number_of_recommended_metadata_datasets=10)
    2. Call filter_and_visualize_tool (applies the filters)
    3. Call find_metadata_datasets (finds metadata datasets)
    4. Call forecast_tool (generates forecasts with leaking data)

    
    ❌ WRONG ORDER: 
    - Never call filter_and_visualize_tool when group_filters is empty/null
    - Never call filter_and_visualize_tool before setting parameters
    - Never skip get_state_parameters_to_update_tool when doing what-if scenarios
    
    **Tool Usage Guidelines:**
    - get_state_parameters_to_update_tool: Set UI parameters when user wants to change settings, filter data, or select columns. Also use for PURE parameter setting (e.g., "set the number of metadata datasets to 30")
    - filter_and_visualize_tool: Filter and display data (ONLY after parameters are set)
    - forecast_tool: Generate predictions (requires parameters to be set first)
    - search_metadata_tool: Add context/metadata to dataset
    - find_metadata_datasets: Find external data sources. Handles JOINT queries directly (e.g., "Give me 20 datasets about inflation") - no need to call get_state_parameters_to_update_tool first for these
    - explain_tool: Explain data patterns or forecasts
    - anomaly_detection_tool: Detect outliers
    - what_if_parameter_modification_tool: Modify actual data values for scenarios

    **OPTIMIZED METADATA DATASET WORKFLOW:**
    - For joint queries like "Give me 15 datasets about stock prices": Call find_metadata_datasets DIRECTLY (it extracts the number)
    - For pure parameter setting like "set metadata count to 30": Call get_state_parameters_to_update_tool
    - For subsequent queries without numbers: find_metadata_datasets will use the UI state value

    **You may call multiple tools in sequence, but ALWAYS follow the ordering rules.**

    At each step, include a user-friendly summary of what you are doing, but DON'T MENTION THE TOOL BEING USED.
    Don't stop early until you have finished the entire task from the user!
    """

    # Add system message if not already present
    if not messages or messages[0].content != system_message:
        messages = [HumanMessage(content=system_message)] + messages

    # Get LLM response with tool calling
    try:
        response = await llm_with_tools.ainvoke(messages)  # type: ignore
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        # Return a simple response without tool calls to continue the conversation
        error_response = AIMessage(
            content="I've updated the parameters. Let me proceed with filtering the data for you."
        )
        messages.append(error_response)

        return AgentState(
            messages=messages,
            ui_state=state.ui_state,
            user_input=state.user_input,
            tool_outputs=state.tool_outputs,
            fastapi_response=state.fastapi_response,
            categorical_features=state.categorical_features,
        )

    logger.info(f"Agent response: {response}")

    # Add the response to messages
    messages.append(response)

    fastapi_response = state.fastapi_response or []
    if response.content and fastapi_response:
        fastapi_response[-1]["message_to_user"] = response.content

    return AgentState(
        messages=messages,
        ui_state=state.ui_state,
        user_input=state.user_input,
        tool_outputs=state.tool_outputs,
        fastapi_response=state.fastapi_response,
        categorical_features=state.categorical_features,
    )


def _sanitize_tool_result_for_conversation(result: Any) -> Dict[str, Any]:
    """
    Sanitize tool results before adding to conversation history.

    Removes complex objects like CategoricalFeatureValues that might cause
    serialization issues with the Gemini API.

    Args:
        result: The tool result to sanitize

    Returns:
        Dict[str, Any]: Sanitized result safe for conversation history
    """
    if not isinstance(result, dict):
        return {"result": "Tool execution completed"}

    sanitized = {}
    for key, value in result.items():
        if key == "parameter_updates":
            # Include parameter updates but exclude complex objects
            sanitized[key] = value
        elif key == "fastapi_response":
            # Include fastapi_response info but exclude complex nested data
            if isinstance(value, dict):
                sanitized[key] = {
                    "update_type": value.get("update_type", "unknown"),
                    "status": "success" if "error" not in result else "error",
                }
            else:
                sanitized[key] = "Response generated"
        elif key == "error":
            sanitized[key] = value
        elif key == "message":
            sanitized[key] = value
        else:
            # For other keys, include simple values but exclude complex objects
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                # Simplified representation for complex data structures
                sanitized[key] = (
                    f"<{type(value).__name__} with {len(value)} items>"
                )
            else:
                sanitized[key] = f"<{type(value).__name__}>"

    return sanitized


def _make_json_safe(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-safe representations.

    Handles conversion of complex Python objects including Enums, Pydantic models,
    and nested data structures to ensure they can be serialized to JSON.

    Args:
        obj (Any): The object to convert to JSON-safe format

    Returns:
        Any: JSON-safe representation of the input object

    Note:
        - Converts Enums to their .value attribute
        - Converts Pydantic models using model_dump()
        - Recursively processes nested structures (dicts, lists, tuples)
    """
    # Recursively convert Enums to their .value for JSON serialization
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, BaseModel):
        # Handle Pydantic models by converting to dict
        return _make_json_safe(obj.model_dump())
    elif isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_safe(x) for x in obj)
    else:
        return obj


async def tool_node(state: AgentState) -> AgentState:
    """
    Execute the tools that the agent decided to call.

    Processes tool calls from the agent's last message, executes the appropriate tools,
    and updates the agent state with results and UI parameter changes.

    Args:
        state (AgentState): Current agent state containing tool calls to execute

    Returns:
        AgentState: Updated state with tool execution results, updated UI state,
            and new messages from tool outputs

    Note:
        Handles both synchronous and asynchronous tool execution. Updates the UI state
        automatically when tools return parameter_updates. Includes error handling
        for UIState validation failures.
    """
    messages = state.messages
    ui_state_dict = state.ui_state
    tool_outputs = state.tool_outputs

    fastapi_response = state.fastapi_response or []

    # Get the last message (should be from the agent with tool calls)
    last_message = messages[-1] if messages else None

    tool_calls = (
        getattr(last_message, "tool_calls", None) if last_message else None
    )
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Find and execute the tool
            tool_func = None
            for tool in TOOLS:
                if tool.name == tool_name:
                    tool_func = tool
                    break

            if tool_func:
                # Prepare arguments for tool execution (copy to avoid modifying conversation history)
                execution_args = tool_args.copy()

                # Add categorical_features to tools that need them
                if tool_name == "get_state_parameters_to_update_tool":
                    execution_args["categorical_features"] = (
                        state.categorical_features
                    )

                # Execute the tool (await if async)
                try:
                    result = await tool_func.ainvoke(execution_args)

                    # Update UI state if the tool returned parameter updates
                    if (
                        isinstance(result, dict)
                        and "parameter_updates" in result
                    ):
                        ui_state = UIState(**ui_state_dict)
                        for key, value in result["parameter_updates"].items():
                            if key == "data_point_cards":
                                # TODO later
                                pass
                                # # Only update the card_indexes and only the changed fields
                                # for card_to_update in value:
                                #     card_index = card_to_update.get(
                                #         "card_index"
                                #     )
                                #     values_to_change = card_to_update.get(
                                #         "values_to_change", {}
                                #     )
                                #     if (
                                #         ui_state.data_point_cards
                                #         and card_index is not None
                                #         and card_index >= 0
                                #         and card_index
                                #         < len(ui_state.data_point_cards)
                                #     ):
                                #         orig_card = ui_state.data_point_cards[
                                #             card_index
                                #         ]
                                #         # Update only the changed fields in the 'values' dict
                                #         for k, v in values_to_change.items():
                                #             orig_card["values"][k] = v
                            else:
                                setattr(ui_state, key, value)

                        ui_state_dict = ui_state.model_dump()
                except Exception as e:
                    # Catch pydantic validation errors and return a user-friendly fastapi_response
                    logger.error(f"UIState validation error: {e}")
                    fastapi_response.append(
                        {
                            "update_type": UIUpdateType.PARAMETERS_UPDATE.value,
                            "update": {},
                            "status": "error",
                            "message": "Something went wrong with your request. Please rephrase and try again.",
                        }
                    )
                    continue

                if isinstance(result, dict) and "fastapi_response" in result:
                    fastapi_response.append(result["fastapi_response"])

                # Sanitize result for conversation history (remove non-serializable objects)
                sanitized_result = _sanitize_tool_result_for_conversation(
                    result
                )

                # Add tool result to messages
                tool_message = ToolMessage(
                    content=str(sanitized_result), tool_call_id=tool_id
                )
                messages.append(tool_message)

                # Track tool outputs
                tool_outputs.append({"tool": tool_name, "output": result})

    return AgentState(
        messages=messages,
        ui_state=ui_state_dict,
        user_input=state.user_input,
        tool_outputs=tool_outputs,
        fastapi_response=fastapi_response,
        categorical_features=state.categorical_features,
    )


def should_continue(state: AgentState) -> str:
    """
    Determine if the agent should continue calling tools or end the conversation.

    Analyzes the current agent state to decide the next step in the workflow.

    Args:
        state (AgentState): Current agent state to analyze

    Returns:
        str: Either "tools" to continue with tool execution, or END to terminate
            the agent workflow

    Note:
        Decision is based on whether the last message contains tool calls that
        need to be executed.
    """
    messages = state.messages
    last_message = messages[-1]

    # If the last message has tool calls, go to tools
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    # Otherwise, end
    return END


# --- Build the Agent Graph ---
def create_agent_graph():
    """
    Create the agent workflow graph with dynamic tool selection.

    Builds a StateGraph that defines the agent's workflow, including nodes for
    the main agent logic and tool execution, with conditional edges for flow control.

    Returns:
        Any: Compiled StateGraph representing the complete agent workflow

    Note:
        The graph includes:
        - agent node: Main decision-making logic
        - tools node: Tool execution
        - Conditional edges based on tool call requirements
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# --- Main Agent Runner ---
async def run_agent(
    request: FoundationModelChatRequest, **kwargs
) -> List[Dict[str, Any]]:
    """
    Run the agent with dynamic tool selection and UIState management.

    Main entry point for executing the agent workflow with a user request,
    handling the complete conversation flow from initial request to final response.

    Args:
        request (FoundationModelChatRequest): The user's request containing prompt
            and configuration parameters
        **kwargs: Additional keyword arguments (currently unused)

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing FastAPI response data,
            with each dictionary representing an update or result from the agent's execution

    Note:
        - Initializes UIState from the request configuration
        - Creates and executes the agent graph
        - Returns only the fastapi_response data, converted to JSON-safe format
        - Logs execution start and completion
    """
    logger.info(f"run_agent: Called with user_input='{request.prompt}'")

    # Create UIState object from request
    ui_state = UIState(
        user_id=request.user_id,
        file_path_key=request.config.file_path_key,
        target_columns=request.config.timeseries_columns,
        timestamp_column=request.config.timestamp_column,
        covariates=request.config.covariate_columns,
        min_timestamp=request.config.min_timestamp,
        forecast_timestamp=request.config.forecasting_timestamp,
        group_filters=None,  # Can be set later by tools
        forecast_length=request.config.forecast_length,
        backtest_stride=None,  # Can be set later by tools
        leak_metadata=bool(
            (
                request.config.covariate_columns_to_leak
                and len(request.config.covariate_columns_to_leak) > 0
            )
            or (
                request.config.metadata_dataframes_leak_idxs
                and len(request.config.metadata_dataframes_leak_idxs) > 0
            )
        ),
        point_modifications=request.config.point_modifications,
        full_and_range_modifications=request.config.full_and_range_modifications,
        dataset_columns=request.dataset_columns,
        possible_timestamp_columns=request.timestamp_columns,
        number_of_recommended_metadata_datasets=request.number_of_recommended_metadata_datasets,
    )

    # Initialize state
    initial_state = AgentState(
        messages=[HumanMessage(content=request.prompt)],
        ui_state=ui_state.model_dump(),
        user_input=request.prompt,
        tool_outputs=[],
        fastapi_response=None,
        categorical_features=request.categorical_features,
    )

    # Create and run the agent
    agent_graph = create_agent_graph()
    final_state = await agent_graph.ainvoke(initial_state)

    # Extract fastapi_response only
    if isinstance(final_state, dict):
        fastapi_response = final_state.get("fastapi_response")
    else:
        fastapi_response = final_state.fastapi_response

    logger.info("run_agent: Agent execution complete.")

    # Patch for JSON safety (convert Enums to .value)
    return [_make_json_safe(resp) for resp in (fastapi_response or [])]


# --- Streaming Support ---
async def run_agent_streaming(
    request: FoundationModelChatRequest,
) -> AsyncGenerator[str, None]:
    """
    Run the agent with streaming support for real-time updates.

    Executes the agent workflow with streaming capabilities, yielding updates
    as they become available for real-time user feedback.

    Args:
        request (FoundationModelChatRequest): The user's request containing prompt
            and configuration parameters

    Yields:
        str: JSON-encoded strings representing either:
            - chatbox_update_to_user: Messages for the user interface
            - fastapi_response updates: Tool execution results and state changes

    Note:
        - Streams both agent messages and tool execution results
        - Prevents duplicate message sending by tracking sent messages
        - Yields newline-delimited JSON for easy parsing by clients
        - Initializes UIState from request configuration similar to run_agent
    """
    logger.info(
        f"run_agent_streaming: Called with user_input='{request.prompt}'"
    )

    # Create UIState object from request
    ui_state = UIState(
        user_id=request.user_id,
        file_path_key=request.config.file_path_key,
        target_columns=request.config.timeseries_columns,
        timestamp_column=request.config.timestamp_column,
        covariates=request.config.covariate_columns,
        min_timestamp=request.config.min_timestamp,
        forecast_timestamp=request.config.forecasting_timestamp,
        group_filters=None,  # Can be set later by tools
        forecast_length=request.config.forecast_length,
        backtest_stride=None,  # Can be set later by tools
        leak_metadata=bool(
            (
                request.config.covariate_columns_to_leak
                and len(request.config.covariate_columns_to_leak) > 0
            )
            or (
                request.config.metadata_dataframes_leak_idxs
                and len(request.config.metadata_dataframes_leak_idxs) > 0
            )
        ),
        point_modifications=request.config.point_modifications,
        full_and_range_modifications=request.config.full_and_range_modifications,
        dataset_columns=request.dataset_columns,
        possible_timestamp_columns=request.timestamp_columns,
        number_of_recommended_metadata_datasets=request.number_of_recommended_metadata_datasets,
    )

    # Initialize state
    initial_state = AgentState(
        messages=[HumanMessage(content=request.prompt)],
        ui_state=ui_state.model_dump(),
        user_input=request.prompt,
        tool_outputs=[],
        fastapi_response=None,
        categorical_features=request.categorical_features,
    )

    # Create and stream the agent
    agent_graph = create_agent_graph()
    sent_count = 0
    messages_already_sent = []
    async for update in agent_graph.astream(initial_state):
        # loop through the Agent's message to find the message_to_user
        if "agent" in update:
            if "messages" in update["agent"]:
                for message in update["agent"]["messages"]:
                    if isinstance(message, AIMessage):
                        message_to_user = message.content
                        if message_to_user:
                            if message_to_user not in messages_already_sent:
                                messages_already_sent.append(message_to_user)
                                yield (
                                    json.dumps(
                                        {
                                            "chatbox_update_to_user": message_to_user
                                        }
                                    )
                                    + "\n"
                                )

        if "tools" in update:
            fastapi_response: List[Dict[str, Any]] = update["tools"].get(
                "fastapi_response"
            )
            if fastapi_response and len(fastapi_response) > sent_count:
                # Only yield new responses
                for resp in fastapi_response[sent_count:]:
                    resp_safe = _make_json_safe(resp)
                    yield json.dumps(resp_safe) + "\n"
                sent_count = len(fastapi_response)


# --- Memory Management ---
# AGENT_MEMORY_DIR = "/home/raimi/agent_memory"
# os.makedirs(AGENT_MEMORY_DIR, exist_ok=True)


# def get_agent_history(user_id: str, thread_id: str) -> List[dict]:
#     path = os.path.join(AGENT_MEMORY_DIR, f"{user_id}_{thread_id}.json")
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             return json.load(f)
#     return []


# def save_agent_history(user_id: str, thread_id: str, messages: List[dict]):
#     path = os.path.join(AGENT_MEMORY_DIR, f"{user_id}_{thread_id}.json")
#     with open(path, "w") as f:
#         json.dump(messages, f, default=str, indent=2)


def _extract_metadata_dataset_number_from_input(
    user_input: str,
) -> Optional[int]:
    """
    Extract the requested number of metadata datasets from user input.

    Args:
        user_input: The user's input string

    Returns:
        The number of datasets requested, or None if not specified
    """
    import re

    user_input_lower = user_input.lower()

    # Look for explicit numbers with dataset-related keywords
    patterns = [
        r"(\d+)\s*(?:metadata\s*)?(?:datasets?|metadatas?|recommendations?|results?|matches?)",
        r"(?:give|find|show|get)\s*(?:me\s*)?(\d+)",
        r"(\d+)\s*(?:of\s*)?(?:the\s*)?(?:best|top|most\s*relevant)",
        r"top\s*(\d+)",
        r"first\s*(\d+)",
        r"(\d+)\s*(?:external\s*)?(?:data\s*)?(?:sources?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            try:
                num = int(match.group(1))
                # Reasonable bounds check
                if 1 <= num <= 200:
                    return num
            except (ValueError, IndexError):
                continue

    # Look for qualitative terms and convert to numbers
    qualitative_mappings = {
        "few": 5,
        "a few": 5,
        "some": 7,
        "several": 8,
        "many": 20,
        "lots": 25,
        "numerous": 30,
        "plenty": 35,
        "extensive": 40,
        "comprehensive": 50,
        "enormous": 100,
        "massive": 150,
    }

    # Check for qualitative terms
    for term, number in qualitative_mappings.items():
        # Use word boundaries to avoid partial matches
        if re.search(r"\b" + re.escape(term) + r"\b", user_input_lower):
            return number
