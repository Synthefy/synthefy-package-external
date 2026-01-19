import os
import sys

import pandas as pd
import pytest

from synthefy_pkg.loaders.simple_loader import SimpleLoader


@pytest.fixture
def loader():
    return SimpleLoader("")


@pytest.fixture
def all_filter_filenames():
    return {
        "FILTER_A": {"file1.csv", "file2.csv"},
        "FILTER_B": {"file2.csv", "file3.csv"},
        "FILTER_C": {"file3.csv", "file4.csv"},
        "FILTER_D": {"file4.csv", "file5.csv"},
        "FILTER_E": {"file5.csv", "file6.csv"},
    }


def test_simple_filter(loader, all_filter_filenames):
    """Test single filter"""
    result = loader._compute_filter_logic("FILTER_A", all_filter_filenames)
    assert result == {"file1.csv", "file2.csv"}


def test_or_operation(loader, all_filter_filenames):
    """Test OR operation between filters"""
    result = loader._compute_filter_logic(
        "FILTER_A|FILTER_B", all_filter_filenames
    )
    assert result == {"file1.csv", "file2.csv", "file3.csv"}


def test_and_operation(loader, all_filter_filenames):
    """Test AND operation between filters"""
    result = loader._compute_filter_logic(
        "FILTER_A&FILTER_B", all_filter_filenames
    )
    assert result == {"file2.csv"}


def test_nested_operations(loader, all_filter_filenames):
    """Test nested operations with parentheses"""
    result = loader._compute_filter_logic(
        "(FILTER_A|FILTER_B)&FILTER_C", all_filter_filenames
    )
    assert result == {"file3.csv"}


def test_not_operation(loader, all_filter_filenames):
    """Test NOT operation"""
    result = loader._compute_filter_logic("~FILTER_A", all_filter_filenames)
    assert result == {"file3.csv", "file4.csv", "file5.csv", "file6.csv"}


def test_complex_expression(loader, all_filter_filenames):
    """Test complex expression with multiple operations"""
    result = loader._compute_filter_logic(
        "(FILTER_A|FILTER_B)&~(FILTER_C|FILTER_D)", all_filter_filenames
    )
    assert result == {"file1.csv", "file2.csv"}


def test_nested_paren_expression(loader, all_filter_filenames):
    """Test complex expression with nested parentheses"""
    result = loader._compute_filter_logic(
        "(FILTER_A|FILTER_B)&~((FILTER_C|~FILTER_D)&FILTER_E)",
        all_filter_filenames,
    )
    assert result == {"file1.csv", "file2.csv", "file3.csv"}


def test_empty_filter(loader, all_filter_filenames):
    """Test empty filter string"""
    with pytest.raises(Exception):  # Adjust exception type as needed
        loader._compute_filter_logic("", all_filter_filenames)


def test_invalid_filter_name(loader, all_filter_filenames):
    """Test with non-existent filter name"""
    with pytest.raises(KeyError):
        loader._compute_filter_logic("NONEXISTENT_FILTER", all_filter_filenames)


def test_malformed_expression(loader, all_filter_filenames):
    """Test malformed expression"""
    with pytest.raises(Exception):  # Adjust exception type as needed
        loader._compute_filter_logic("(FILTER_A|FILTER_B", all_filter_filenames)


@pytest.fixture
def simple_loader():
    return SimpleLoader("")


# unit test for apply_post_filters, get_filtered_filenames
@pytest.fixture
def sample_dfs():
    dfs = []
    names = []
    metas = []

    # Generate 10 different DataFrames with varying characteristics
    for i in range(10):
        rows = (i + 1) * 50  # Varying lengths: 50, 100, 150, ..., 500
        cols = (i % 3) + 2  # Varying columns: 2, 3, 4 cycling

        df = pd.DataFrame({f"Col_{j}": range(rows) for j in range(cols)})

        meta = {
            "title": f"Series_{i}",
            "frequency": "Monthly" if i % 2 == 0 else "Weekly",
            "category": "Economic" if i < 5 else "Financial",
            "description": f"Test Series {i}",
            "source": "FRED"
            if i % 3 == 0
            else ("BLS" if i % 3 == 1 else "Census"),
            "start_date": f"2020-01-{str(i + 1).zfill(2)}",
            "end_date": f"2024-01-{str(i + 1).zfill(2)}",
            "size": rows * cols,
            "length": rows,
            "num_columns": cols,
            "final_columns": [
                {
                    "name": f"Col_{j}",
                    "type": "float",
                    "title": f"Column {j}",
                    "column_id": f"Col_{j}",
                    "is_metadata": "no",
                    "description": f"Description of Column {j}",
                }
                for j in range(cols)
            ],
        }

        dfs.append(df)
        names.append(f"test_series_{i}")
        metas.append(meta)

    return dfs, names, metas


def test_apply_post_filters_multiple_series(simple_loader, sample_dfs):
    dfs, names, metas = sample_dfs

    # Test length filter that should only keep longer series
    post_filters = [{"length_geq": "200"}]

    filtered_dfs = []
    filtered_names = []
    filtered_metas = []

    for df, name, meta in zip(dfs, names, metas):
        result_dfs, result_names, result_metas = (
            simple_loader._apply_post_filters(
                df=df, name=name, meta=meta, post_filters=post_filters
            )
        )
        filtered_dfs.extend(result_dfs)
        filtered_names.extend(result_names)
        filtered_metas.extend(result_metas)

    # Should only keep series with length >= 200 (indices 3-9)
    assert len(filtered_dfs) == 7
    assert all(len(df) >= 200 for df in filtered_dfs)


def test_apply_post_filters_frequency_and_size(simple_loader, sample_dfs):
    dfs, names, metas = sample_dfs

    post_filters = [{"frequency": "Monthly"}, {"size_geq": "300"}]

    filtered_dfs = []
    filtered_names = []
    filtered_metas = []

    for df, name, meta in zip(dfs, names, metas):
        result_dfs, result_names, result_metas = (
            simple_loader._apply_post_filters(
                df=df, name=name, meta=meta, post_filters=post_filters
            )
        )
        filtered_dfs.extend(result_dfs)
        filtered_names.extend(result_names)
        filtered_metas.extend(result_metas)

    assert all(meta["frequency"] == "Monthly" for meta in filtered_metas)
    assert all(meta["size"] >= 300 for meta in filtered_metas)


def test_apply_post_filters_source_and_category(simple_loader, sample_dfs):
    dfs, names, metas = sample_dfs

    post_filters = [{"length_geq": "200"}, {"size_leq": "1000"}]

    filtered_dfs = []
    filtered_names = []
    filtered_metas = []

    for df, name, meta in zip(dfs, names, metas):
        result_dfs, result_names, result_metas = (
            simple_loader._apply_post_filters(
                df=df, name=name, meta=meta, post_filters=post_filters
            )
        )
        filtered_dfs.extend(result_dfs)
        filtered_names.extend(result_names)
        filtered_metas.extend(result_metas)

    assert all(
        meta["length"] >= 200 and meta["size"] <= 1000
        for meta in filtered_metas
    )


def test_apply_post_filters_complex_combination(simple_loader, sample_dfs):
    dfs, names, metas = sample_dfs

    post_filters = [
        {"length_geq": "200"},
        {"size_leq": "1000"},
    ]
    complex_filter = "length_geq_200&size_leq_1000"

    filtered_dfs = []
    filtered_names = []
    filtered_metas = []

    for df, name, meta in zip(dfs, names, metas):
        result_dfs, result_names, result_metas = (
            simple_loader._apply_post_filters(
                df=df,
                name=name,
                meta=meta,
                post_filters=post_filters,
                post_complex_filter=complex_filter,
            )
        )
        filtered_dfs.extend(result_dfs)
        filtered_names.extend(result_names)
        filtered_metas.extend(result_metas)

    assert all(len(df) >= 200 for df in filtered_dfs)
    assert all(len(df) * len(df.columns) <= 1000 for df in filtered_dfs)


def test_run_filters_size_length(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "num_columns": 10,
        "frequency": "Monthly",
        "title": "Test Series",
        "final_columns": [
            {"name": "A", "type": "float", "title": "Column A"},
            {"name": "B", "type": "float", "title": "Column B"},
        ],
    }

    # Test size filters
    filters = [{"size_geq": "500"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "size_geq_500" in result
    assert filename in result["size_geq_500"]

    # Test length filters
    filters = [{"length_leq": "150"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "length_leq_150" in result
    assert filename in result["length_leq_150"]

    # Test failing size filter
    filters = [{"size_geq": "2000"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "size_geq_2000" in result
    assert len(result["size_geq_2000"]) == 0


def test_run_filters_column_related(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "num_columns": 3,
        "frequency": "Monthly",
        "final_columns": [
            {
                "name": "Price",
                "type": "float",
                "title": "Price Column",
                "description": "Price of the item",
                "column_id": "Price",
                "is_metadata": "no",
            },
            {
                "name": "Volume",
                "type": "float",
                "title": "Trading Volume",
                "description": "Volume of the item",
                "column_id": "Volume",
                "is_metadata": "no",
            },
            {
                "name": "Date",
                "type": "datetime",
                "title": "Date",
                "description": "Date of the item",
                "column_id": "Date",
                "is_metadata": "yes",
            },
        ],
    }

    # Test num_columns filter
    filters = [{"num_columns": "3"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "num_columns_3" in result
    assert filename in result["num_columns_3"]

    # Test column keywords
    filters = [{"column_keywords": "Price"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "column_keywords_Price" in result
    assert filename in result["column_keywords_Price"]

    # Test multiple column keywords
    filters = [{"column_keywords": ["Price", "Volume"]}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "column_keywords_Price_Volume" in result
    assert filename in result["column_keywords_Price_Volume"]


def test_run_filters_frequency(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "num_columns": 3,
        "frequency": "Monthly",
        "final_columns": [],
    }

    # Test exact frequency match
    filters = [{"frequency": "Monthly"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "frequency_Monthly" in result
    assert filename in result["frequency_Monthly"]

    # Test frequency mismatch
    filters = [{"frequency": "Daily"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "frequency_Daily" in result
    assert len(result["frequency_Daily"]) == 0


def test_run_filters_multiple_conditions(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "num_columns": 3,
        "frequency": "Monthly",
        "title": "Economic Indicators",
        "category": "Economics",
        "final_columns": [
            {"name": "GDP", "type": "float", "title": "GDP Value"}
        ],
    }

    # Test multiple different types of filters
    filters = [
        {"size_geq": "500"},
        {"frequency": "Monthly"},
        {"category": "Economics"},
    ]

    result = simple_loader._run_filters(filename, metadata, filters)
    assert "size_geq_500" in result
    assert "frequency_Monthly" in result
    assert "category_Economics" in result
    assert filename in result["size_geq_500"]
    assert filename in result["frequency_Monthly"]
    assert filename in result["category_Economics"]


def test_run_filters_case_insensitive(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "title": "Economic Indicators",
        "category": "Economics",
    }

    # Test case-insensitive matching
    filters = [{"category": "economics"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "category_economics" in result
    assert filename in result["category_economics"]


def test_run_filters_final_columns_subcategory(simple_loader):
    filename = "test_series"
    metadata = {
        "size": 1000,
        "length": 100,
        "final_columns": [
            {"name": "GDP", "type": "float", "title": "Gross Domestic Product"}
        ],
    }

    # Test final_columns subcategory filtering
    filters = [{"final_columns_title": "Gross"}]
    result = simple_loader._run_filters(filename, metadata, filters)
    assert "final_columns_title_Gross" in result
    assert filename in result["final_columns_title_Gross"]


def test_run_filters_empty_filters(simple_loader):
    filename = "test_series"
    metadata = {"size": 1000, "length": 100}

    filters = []
    result = simple_loader._run_filters(filename, metadata, filters)
    assert isinstance(result, dict)
    assert len(result) == 0


def test_run_filters_invalid_metadata_key(simple_loader):
    filename = "test_series"
    metadata = {"size": 1000, "length": 100}

    # Test with non-existent metadata key
    filters = [{"non_existent_key": "value"}]
    with pytest.raises(KeyError):
        simple_loader._run_filters(filename, metadata, filters)


def test_stringify_filter_string_value(simple_loader):
    # Test with simple string value
    key = "frequency"
    value = "Monthly"
    result = simple_loader._stringify_filter(key, value)
    assert result == "frequency_Monthly"

    # Test with numeric string value
    key = "length_geq"
    value = "100"
    result = simple_loader._stringify_filter(key, value)
    assert result == "length_geq_100"

    # Test with special characters
    key = "category"
    value = "Economic & Financial"
    result = simple_loader._stringify_filter(key, value)
    assert result == "category_Economic & Financial"


def test_stringify_filter_list_value(simple_loader):
    # Test with simple list of strings
    key = "column_keywords"
    value = ["Price", "Volume"]
    result = simple_loader._stringify_filter(key, value)
    assert result == "column_keywords_Price_Volume"

    # Test with single item list
    key = "source"
    value = ["FRED"]
    result = simple_loader._stringify_filter(key, value)
    assert result == "source_FRED"

    # Test with empty list
    key = "tags"
    value = []
    result = simple_loader._stringify_filter(key, value)
    assert result == "tags_"

    # Test with list containing special characters
    key = "categories"
    value = ["Economic & Trade", "Financial Markets"]
    result = simple_loader._stringify_filter(key, value)
    assert result == "categories_Economic & Trade_Financial Markets"


def test_stringify_filter_edge_cases(simple_loader):
    # Test with empty string key
    key = ""
    value = "test"
    result = simple_loader._stringify_filter(key, value)
    assert result == "_test"

    # Test with empty string value
    key = "test"
    value = ""
    result = simple_loader._stringify_filter(key, value)
    assert result == "test_"

    # Test basic OR operation
    filter_names = [
        {"file1.txt", "file2.txt"},
        {"file2.txt", "file3.txt"},
        {"file4.txt"},
    ]
    result = simple_loader._apply_OR_filter(filter_names)
    assert result == {"file1.txt", "file2.txt", "file3.txt", "file4.txt"}


def test_apply_OR_filter_empty(simple_loader):
    # Test with empty input
    assert simple_loader._apply_OR_filter([]) == set()

    # Test with list containing empty sets
    filter_names = [set(), set(), set()]
    assert simple_loader._apply_OR_filter(filter_names) == set()


def test_apply_OR_filter_single(simple_loader):
    # Test with single set
    filter_names = [{"file1.txt", "file2.txt"}]
    result = simple_loader._apply_OR_filter(filter_names)
    assert result == {"file1.txt", "file2.txt"}


def test_apply_OR_filter_duplicates(simple_loader):
    # Test with duplicate values across sets
    filter_names = [
        {"file1.txt", "file2.txt"},
        {"file2.txt", "file2.txt"},
        {"file2.txt", "file3.txt"},
    ]
    result = simple_loader._apply_OR_filter(filter_names)
    assert result == {"file1.txt", "file2.txt", "file3.txt"}


def test_apply_AND_filter_basic(simple_loader):
    # Test basic AND operation
    filter_names = [
        {"file1.txt", "file2.txt", "file3.txt"},
        {"file2.txt", "file3.txt", "file4.txt"},
        {"file2.txt", "file3.txt", "file5.txt"},
    ]
    result = simple_loader._apply_AND_filter(filter_names)
    assert result == {"file2.txt", "file3.txt"}


def test_apply_AND_filter_empty(simple_loader):
    # Test with empty input
    assert simple_loader._apply_AND_filter([]) == set()

    # Test with list containing empty sets
    filter_names = [set(), set(), set()]
    assert simple_loader._apply_AND_filter(filter_names) == set()


def test_apply_AND_filter_single(simple_loader):
    # Test with single set
    filter_names = [{"file1.txt", "file2.txt"}]
    result = simple_loader._apply_AND_filter(filter_names)
    assert result == {"file1.txt", "file2.txt"}


def test_apply_AND_filter_no_overlap(simple_loader):
    # Test with no common elements
    filter_names = [
        {"file1.txt", "file2.txt"},
        {"file3.txt", "file4.txt"},
        {"file5.txt", "file6.txt"},
    ]
    result = simple_loader._apply_AND_filter(filter_names)
    assert result == set()


def test_apply_AND_filter_partial_overlap(simple_loader):
    # Test with partial overlap
    filter_names = [
        {"file1.txt", "file2.txt", "file3.txt"},
        {"file2.txt", "file3.txt", "file4.txt"},
        {"file3.txt", "file4.txt", "file5.txt"},
    ]
    result = simple_loader._apply_AND_filter(filter_names)
    assert result == {"file3.txt"}


def test_apply_AND_filter_complete_overlap(simple_loader):
    # Test with complete overlap
    filter_names = [
        {"file1.txt", "file2.txt"},
        {"file1.txt", "file2.txt"},
        {"file1.txt", "file2.txt"},
    ]
    result = simple_loader._apply_AND_filter(filter_names)
    assert result == {"file1.txt", "file2.txt"}


def test_filters_with_special_characters(simple_loader):
    # Test with filenames containing special characters
    or_filter_names = [
        {"file-1.txt", "file_2.txt"},
        {"file@3.txt", "file#4.txt"},
    ]
    or_result = simple_loader._apply_OR_filter(or_filter_names)
    assert or_result == {"file-1.txt", "file_2.txt", "file@3.txt", "file#4.txt"}

    and_filter_names = [
        {"file-1.txt", "file_2.txt"},
        {"file-1.txt", "file@3.txt"},
        {"file-1.txt", "file#4.txt"},
    ]
    and_result = simple_loader._apply_AND_filter(and_filter_names)
    assert and_result == {"file-1.txt"}
