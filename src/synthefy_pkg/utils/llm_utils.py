"""
LLM Utilities for Dataset Description Processing

This module provides functions to extract structured information from dataset descriptions
using Large Language Models (LLMs). It leverages the Azure OpenAI API as configured in
the application to parse unstructured text into structured data points.

It also includes functions for using free/local LLMs as alternatives.
"""

import json
import re
from time import sleep
from typing import Any, Dict, List, Literal, Union

import requests
from loguru import logger


def process_dataset_description(
    description: str, model: str = "llama3.2"
) -> Dict[str, Any]:
    """
    Process a dataset description using an LLM to extract structured information.

    Can use either cloud-based LLM (Azure OpenAI) or local LLM via Ollama.

    Args:
        description (str): The dataset description text
        model (str): The model name to use with Ollama (if use_local_llm is True)

    Returns:
        Dict[str, Any]: The extracted structured information
    """
    # Process with Ollama
    result_description = process_dataset_with_local_llm(
        description, model=model, extract_or_expand="extract"
    )
    # Return in same format as the cloud-based version
    return {
        "description": description,
        "processed_description": result_description,
        "processed_with": f"ollama-{model}",
    }


def process_dataset_with_local_llm(
    dataset_info: Union[Dict[str, Any], str],
    model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/generate",
    max_retries: int = 3,
    retry_delay: int = 2,
    extract_or_expand: Literal["extract", "expand"] = "expand",
) -> str:
    """
    Process dataset information using a local LLM via Ollama to generate a comprehensive description.

    This function creates a detailed description of a dataset based on its metadata using
    a free, locally-hosted LLM like Llama 3 through Ollama. This approach doesn't require
    expensive commercial API access while still leveraging LLM capabilities.

    Prerequisites:
    - Ollama installed (https://ollama.ai)
    - Desired model pulled (e.g., `ollama pull llama3`)

    Args:
        dataset_info (Union[Dict[str, Any], str]): Dataset metadata as a dictionary or string
        model (str): Name of the Ollama model to use (default: "llama3")
        ollama_url (str): URL of the Ollama API (default: "http://localhost:11434/api/generate")
        max_retries (int): Maximum number of retries on failure
        retry_delay (int): Seconds to wait between retries

    Returns:
        str: A comprehensive textual description of the dataset
    """
    # Convert to string if dictionary is provided
    if isinstance(dataset_info, dict):
        # Make a copy to avoid modifying the original
        dataset_info_copy = dataset_info.copy()

        # If we have start and end dates but no time_period, create it
        start_date = dataset_info_copy.get("time_period_start", "")
        end_date = dataset_info_copy.get("time_period_end", "")
        if start_date and end_date and not dataset_info_copy.get("time_period"):
            dataset_info_copy["time_period"] = f"{start_date} to {end_date}"

        clean_info = json.dumps(dataset_info_copy, ensure_ascii=False)
    else:
        clean_info = str(dataset_info).strip()

    if extract_or_expand == "extract":
        prompt = f"""
You are a specialized data extraction assistant. Your task is to extract structured information from financial dataset metadata.

## INSTRUCTIONS
1. Extract the following key fields from the metadata:
   - location: Geographic location or country covered by the dataset
   - time_period: The exact time range covered by the dataset
   - subject: What the dataset is measuring or studying
   - source: The organization or provider of the data
   - frequency: How often data is collected (daily, monthly, quarterly, etc.)
   - units: Measurement units used in the dataset

2. IMPORTANT TIME PERIOD RULES:
   - If the metadata includes "time_period", "time_period_start", or "time_period_end" fields, ONLY use these exact values
   - Do NOT infer or create a time period if these fields are present
   - Format as: "time_period: [time_period_start] to [time_period_end]"

3. FORMAT RULES:
   - Return as simple comma-separated key-value pairs
   - Example: "location: Japan, time_period: 1965-01-01 to 2025-03-01, frequency: Monthly"
   - Do NOT use quotes, braces, JSON syntax, or newlines
   - Separate each key-value pair with a comma

4. OMISSION RULES:
   - If information is not available, omit the entire key-value pair
   - Do NOT include placeholder values like "unknown" or "N/A"
   - Do NOT include technical details about file size, format, or storage

## METADATA TO PROCESS:
{clean_info}
        """
    elif extract_or_expand == "expand":
        # Create the prompt for the LLM
        prompt = (
            "You are an expert in financial and economic datasets."
            "Based on the following dataset metadata, create a clear, comprehensive description that could be used to understand what this dataset contains."
            "Focus on explaining what the data represents, its source, geographic coverage, time period, and any other relevant characteristics."
            "Do NOT include information about file size, format (such as CSV or Parquet), or technical storage details."
            "Do NOT include newlines in your response meaning \\n or \\r."
            "Respond with ONLY the description text, without any introductions, explanations, or additional comments.\n"
            "Dataset metadata:\n"
            f"{clean_info}\n"
            "Description:"
        )

    # Payload for the Ollama API - using the /api/generate endpoint
    payload = {"model": model, "prompt": prompt, "stream": False}

    # Try to get a response from the local LLM
    for attempt in range(max_retries):
        try:
            response = requests.post(ollama_url, json=payload)

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                # Explicitly remove any newline characters from the response
                if isinstance(dataset_info, dict):
                    # Add title if available
                    response_text = response_text + (
                        f", title: {dataset_info.get('description')}"
                        if dataset_info.get("description") is not None
                        else ""
                    )

                    # Ensure time period is included
                    if extract_or_expand == "extract":
                        # Add the time period if available
                        if "time_period:" not in response_text.lower():
                            if dataset_info.get("time_period"):
                                response_text += f", time_period: {dataset_info.get('time_period')}"
                            elif dataset_info.get(
                                "time_period_start"
                            ) and dataset_info.get("time_period_end"):
                                response_text += f", time_period: {dataset_info.get('time_period_start')} to {dataset_info.get('time_period_end')}"

                response_text = (
                    response_text.replace("\n", " ")
                    .replace("\r", " ")
                    .replace(",,", ",")
                )
                # Remove any unnecessary quotes from the response
                if extract_or_expand == "extract":
                    # Strip outer quotes if present
                    if (
                        response_text.startswith('"')
                        and response_text.endswith('"')
                    ) or (
                        response_text.startswith("'")
                        and response_text.endswith("'")
                    ):
                        response_text = response_text[1:-1]

                    # Replace escaped quotes
                    response_text = response_text.replace('\\"', "").replace(
                        "\\'", ""
                    )

                    # Remove any remaining quotes
                    response_text = response_text.replace('"', "").replace(
                        "'", ""
                    )

                return response_text
            else:
                logger.warning(
                    f"Attempt {attempt + 1}: Local LLM request failed with status {response.status_code}"
                )

            # Wait before retrying
            if attempt < max_retries - 1:
                sleep(retry_delay)

        except Exception as e:
            logger.error(
                f"Error in local LLM processing (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries - 1:
                sleep(retry_delay)

    # Fallback to a basic extraction if all attempts fail
    try:
        # If input was already a dictionary, use it directly
        data = (
            dataset_info
            if isinstance(dataset_info, dict)
            else json.loads(clean_info)
        )

        description = data.get("description", "")
        name = data.get("name", "")
        source = data.get("sourceName", "")

        if description:
            fallback_text = f"Dataset {name if name else 'unnamed'}: {description} {f'from {source}' if source else ''}"
            # Remove newlines from fallback text
            return fallback_text.replace("\n", " ").replace("\r", " ")

        return "Dataset information available but could not be processed through the local LLM."

    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.error(f"Failed to process dataset information: {e}")
        return "Dataset information available but could not be processed completely."


def batch_process_with_local_llm(
    dataset_info_list: List[Union[Dict[str, Any], str]],
    model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/generate",
) -> List[str]:
    """
    Process multiple dataset information items using a local LLM.

    Args:
        dataset_info_list (List[Union[Dict[str, Any], str]]): List of dataset information as dictionaries or strings
        model (str): Name of the Ollama model to use
        ollama_url (str): URL of the Ollama API

    Returns:
        List[str]: List of comprehensive textual descriptions
    """
    descriptions = []
    for info in dataset_info_list:
        result = process_dataset_with_local_llm(
            info, model, ollama_url, extract_or_expand="expand"
        )
        descriptions.append(result)

    return descriptions
