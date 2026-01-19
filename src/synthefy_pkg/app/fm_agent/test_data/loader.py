#!/usr/bin/env python3
"""
Utility functions for loading and using JSON test data files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_test_data_dir() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent


def load_test_data(filename: str) -> Dict[str, Any]:
    """Load test data from JSON file.

    Args:
        filename: Name of the JSON file to load

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    test_data_dir = get_test_data_dir()
    file_path = test_data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_group_filters_test_cases() -> List[Dict[str, Any]]:
    """Load group filters test cases.

    Returns:
        List of test cases from group_filters_test_cases.json
    """
    data = load_test_data("group_filters_test_cases.json")
    return data["test_cases"]


def load_whatif_test_cases() -> Dict[str, Any]:
    """Load what-if test cases.

    Returns:
        Dictionary of test categories from whatif_test_cases.json
    """
    data = load_test_data("whatif_test_cases.json")
    return data["test_categories"]


def get_whatif_prompts_by_category(category: str) -> List[str]:
    """Get what-if test prompts for a specific category.

    Args:
        category: Name of the test category

    Returns:
        List of test prompts for the category

    Raises:
        KeyError: If the category doesn't exist
    """
    categories = load_whatif_test_cases()

    if category not in categories:
        available = list(categories.keys())
        raise KeyError(
            f"Category '{category}' not found. Available: {available}"
        )

    category_data = categories[category]

    if "test_prompts" in category_data:
        return category_data["test_prompts"]
    elif "test_cases" in category_data:
        return [case["prompt"] for case in category_data["test_cases"]]
    else:
        return []


def get_whatif_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Get what-if test cases for a specific category.

    Args:
        category: Name of the test category

    Returns:
        List of test cases for the category

    Raises:
        KeyError: If the category doesn't exist
    """
    categories = load_whatif_test_cases()

    if category not in categories:
        available = list(categories.keys())
        raise KeyError(
            f"Category '{category}' not found. Available: {available}"
        )

    category_data = categories[category]

    if "test_cases" in category_data:
        return category_data["test_cases"]
    elif "test_prompts" in category_data:
        # Convert simple prompts to test case format
        return [{"prompt": prompt} for prompt in category_data["test_prompts"]]
    else:
        return []


def get_all_whatif_prompts() -> List[str]:
    """Get all what-if test prompts from all categories.

    Returns:
        List of all test prompts
    """
    all_prompts = []
    categories = load_whatif_test_cases()

    for category_name in categories:
        try:
            prompts = get_whatif_prompts_by_category(category_name)
            all_prompts.extend(prompts)
        except KeyError:
            continue

    return all_prompts


def validate_test_data_files() -> Dict[str, bool]:
    """Validate all test data JSON files.

    Returns:
        Dictionary mapping filename to validation success
    """
    results = {}

    json_files = ["group_filters_test_cases.json", "whatif_test_cases.json"]

    for filename in json_files:
        try:
            load_test_data(filename)
            results[filename] = True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Validation failed for {filename}: {e}")
            results[filename] = False

    return results


def save_test_data(filename: str, data: Dict[str, Any]) -> None:
    """Save test data to JSON file.

    Args:
        filename: Name of the JSON file to save
        data: Data to save as JSON
    """
    test_data_dir = get_test_data_dir()
    file_path = test_data_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_group_filters_test_case(
    prompt: str, expected: Dict[str, Any], description: str
) -> None:
    """Add a new group filters test case.

    Args:
        prompt: User input prompt
        expected: Expected output parameters
        description: Description of what this test validates
    """
    data = load_test_data("group_filters_test_cases.json")

    new_case = {
        "prompt": prompt,
        "expected": expected,
        "description": description,
    }

    data["test_cases"].append(new_case)
    save_test_data("group_filters_test_cases.json", data)


def add_whatif_test_prompt(category: str, prompt: str) -> None:
    """Add a new what-if test prompt to a category.

    Args:
        category: Category to add the prompt to
        prompt: Test prompt to add
    """
    data = load_test_data("whatif_test_cases.json")

    if category not in data["test_categories"]:
        raise KeyError(f"Category '{category}' not found")

    category_data = data["test_categories"][category]

    if "test_prompts" not in category_data:
        category_data["test_prompts"] = []

    category_data["test_prompts"].append(prompt)
    save_test_data("whatif_test_cases.json", data)


# Convenience functions for testing frameworks


def pytest_group_filters_parametrize():
    """Get parametrize arguments for pytest group filters tests.

    Returns:
        Tuple of (argnames, argvalues) for pytest.mark.parametrize
    """
    test_cases = load_group_filters_test_cases()

    argnames = "prompt,expected,description"
    argvalues = [
        (case["prompt"], case["expected"], case["description"])
        for case in test_cases
    ]

    return argnames, argvalues


def pytest_whatif_category_parametrize(category: str):
    """Get parametrize arguments for pytest what-if category tests.

    Args:
        category: Name of the test category

    Returns:
        Tuple of (argnames, argvalues) for pytest.mark.parametrize
    """
    prompts = get_whatif_prompts_by_category(category)

    argnames = "prompt"
    argvalues = [(prompt,) for prompt in prompts]

    return argnames, argvalues


if __name__ == "__main__":
    # Quick validation when run directly
    print("Validating test data files...")
    results = validate_test_data_files()

    for filename, valid in results.items():
        status = "✅ Valid" if valid else "❌ Invalid"
        print(f"{filename}: {status}")

    if all(results.values()):
        print("\n🎉 All test data files are valid!")

        # Show summary
        group_filters_count = len(load_group_filters_test_cases())
        whatif_categories = list(load_whatif_test_cases().keys())

        print("\nSummary:")
        print(f"  Group filters test cases: {group_filters_count}")
        print(f"  What-if test categories: {len(whatif_categories)}")
        print(f"  Total what-if prompts: {len(get_all_whatif_prompts())}")
    else:
        print(
            "\n❌ Some test data files have issues. Please fix them before using."
        )
