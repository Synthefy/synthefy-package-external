"""
LLM Service for data modifications using OpenAI GPT-4.

The LLM generates Python code to modify the data, which is then executed.
"""

import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger
from models import DataFrameModel, RequiredColumns
from openai import OpenAI

# System prompt for the LLM
SYSTEM_PROMPT = """You are a data modification assistant for scenario simulation. You help users modify time series data by generating Python code.

You will receive:
1. A pandas DataFrame with time series and metadata columns
2. Information about which columns are timeseries vs metadata
3. A user request describing what modification they want

Your task:
1. Understand the user's modification request
2. Generate Python code that modifies the DataFrame `df` in place
3. Explain what the code does

CRITICAL RULES FOR SCENARIO SIMULATION:

STEP-BY-STEP PROCESS (ALWAYS FOLLOW THIS ORDER):

STEP 1: ZERO OUT ALL DATA AFTER MODIFICATION DATE
   - Identify the modification date/time (e.g., day 96, timestamp N)
   - ZERO OUT ALL metadata columns (discrete and continuous, except 'age') for rows AFTER the modification date
   - CRITICAL: DO NOT modify or zero out TIMESERIES columns - they must remain unchanged
   - The 'age' column should NEVER be zeroed out - it must remain constant
   - Use: df.loc[N:, 'column_name'] = 0 for metadata columns only (NOT timeseries columns)

STEP 2: COMPUTE STATISTICS FROM FIRST N DAYS
   - For each METADATA column mentioned in the user's query, calculate statistics from the FIRST N days (before modification)
   - Calculate: mean, std (standard deviation) from df.loc[:N-1, 'column_name']
   - These statistics will be used for the modification and noise
   - NOTE: Only modify metadata columns (discrete and continuous), NOT timeseries columns

STEP 3: APPLY MODIFICATIONS WITH NOISE
   - Apply the requested modification to METADATA columns mentioned in the query
   - CRITICAL: NEVER modify timeseries columns - they must remain exactly as they are
   - Use the mean from first N days as the base value
   - Use the std from first N days for generating noise
   - Add noise using: np.random.normal(0, first_n_std, M) where M is the number of days after modification

EXAMPLES:

Example 1: "Increase steps by 40% after day 96"
```python
# Step 1: Zero out all metadata columns after day 96 (except age and timeseries columns)
N = 96
# IMPORTANT: Get timeseries columns from the column types provided - DO NOT modify these
timeseries_cols = ['average_hrv', 'lowest_heart_rate', 'age_cva_diff']  # Use actual timeseries columns from context
metadata_cols = [col for col in df.columns if col != 'age' and col not in timeseries_cols]
for col in metadata_cols:
    df.loc[N:, col] = 0
# NOTE: Timeseries columns are NOT zeroed out - they remain unchanged

# Step 2: Compute stats from first 96 days for 'steps' (metadata column)
first_n_mean = df.loc[:N-1, 'steps'].mean()
first_n_std = df.loc[:N-1, 'steps'].std()
M = len(df) - N  # Number of days after modification

# Step 3: Apply 40% increase with noise (only to metadata column 'steps')
increase_factor = 1.4
df.loc[N:, 'steps'] = first_n_mean * increase_factor + np.random.normal(0, first_n_std, M)
# CRITICAL: Timeseries columns remain completely unchanged - do not modify them
```

Example 2: "Increase steps by 40% and active_calories by 30% after day 96"
```python
# Step 1: Zero out all metadata columns after day 96 (except age and timeseries columns)
N = 96
# IMPORTANT: Get timeseries columns from the column types provided - DO NOT modify these
timeseries_cols = ['average_hrv', 'lowest_heart_rate', 'age_cva_diff']  # Use actual timeseries columns from context
metadata_cols = [col for col in df.columns if col != 'age' and col not in timeseries_cols]
for col in metadata_cols:
    df.loc[N:, col] = 0
# NOTE: Timeseries columns are NOT zeroed out - they remain unchanged

# Step 2 & 3: For 'steps' (metadata column)
first_n_mean_steps = df.loc[:N-1, 'steps'].mean()
first_n_std_steps = df.loc[:N-1, 'steps'].std()
M = len(df) - N
df.loc[N:, 'steps'] = first_n_mean_steps * 1.4 + np.random.normal(0, first_n_std_steps, M)

# Step 2 & 3: For 'active_calories' (metadata column)
first_n_mean_cal = df.loc[:N-1, 'active_calories'].mean()
first_n_std_cal = df.loc[:N-1, 'active_calories'].std()
df.loc[N:, 'active_calories'] = first_n_mean_cal * 1.3 + np.random.normal(0, first_n_std_cal, M)
# CRITICAL: Timeseries columns remain completely unchanged - do not modify them
```

GENERAL RULES:
   - The DataFrame is available as `df` (pandas DataFrame)
   - Modify `df` in place or reassign columns
   - Only use pandas, numpy (as np), and standard library
   - DO NOT import anything - pandas and numpy are pre-imported
   - Preserve DataFrame structure (same columns, same number of rows)
   - Use df.loc for row-based modifications
   - Always use 0-based indexing (day 96 = index 96)
   - CRITICAL: NEVER modify timeseries columns - they must remain exactly as they are
   - ONLY modify metadata columns (discrete and continuous, except 'age')

OUTPUT FORMAT:
Return your response in this exact format:

```python
# Step 1: Zero out all metadata columns after modification date (except age and timeseries columns)
N = 96  # modification day
# Get timeseries columns from the column types information - DO NOT modify these
timeseries_cols = {timeseries_cols_str}  # These must remain unchanged - DO NOT modify
metadata_cols = [col for col in df.columns if col != 'age' and col not in timeseries_cols]
for col in metadata_cols:
    df.loc[N:, col] = 0

# Step 2 & 3: For each mentioned METADATA column, compute stats and apply modification
# [Your modification code here following the pattern from examples]
# REMEMBER: Never modify timeseries columns - they must remain unchanged
```

EXPLANATION:
A brief explanation of what the code does, including which metadata columns were zeroed out and which were modified. Note that timeseries columns were NOT modified.
"""


def build_user_prompt(
    df: pd.DataFrame,
    required_columns: RequiredColumns,
    user_query: str,
) -> str:
    """Build the user prompt with data context.

    Args:
        df: The DataFrame to modify
        required_columns: Information about column types
        user_query: The user's modification request

    Returns:
        Formatted prompt string
    """
    # Get data summary
    summary = df.describe().to_string()
    sample = df.head(5).to_string()

    # Get all column names for zeroing out
    all_columns = list(df.columns)

    # Format timeseries columns for the prompt
    timeseries_cols_str = str(required_columns.timeseries)

    prompt = f"""## DataFrame Information

**Shape:** {df.shape[0]} rows × {df.shape[1]} columns

**ALL COLUMNS:** {all_columns}

**Column Types:**
- Timeseries columns (DO NOT MODIFY - these must remain unchanged): {required_columns.timeseries}
- Discrete metadata columns (can modify): {required_columns.discrete}
- Continuous metadata columns (can modify): {required_columns.continuous}

**CRITICAL:** Timeseries columns {timeseries_cols_str} must NEVER be modified or zeroed out. Only modify metadata columns (discrete and continuous, except 'age').

**Data Sample (first 5 rows):**
```
{sample}
```

**Statistics:**
```
{summary}
```

## User Request
{user_query}

## IMPORTANT REMINDERS FOR SCENARIO SIMULATION:
1. STEP 1: ALWAYS zero out ALL metadata columns (discrete and continuous, except 'age') for rows AFTER the modification date
2. CRITICAL: DO NOT modify or zero out TIMESERIES columns - they must remain exactly as they are
3. STEP 2: Compute mean and std from the FIRST N days for each METADATA column mentioned in the query
4. STEP 3: Apply modifications using the mean from first N days, and use the std from first N days for noise
5. The 'age' column should NEVER be modified or zeroed out - it stays constant
6. Timeseries columns should NEVER be modified - they stay unchanged
7. Always follow the 3-step process: Zero out metadata → Compute stats → Apply with noise (metadata only)

Please generate Python code to perform this modification on the DataFrame `df`.
"""
    return prompt


def extract_code_and_explanation(response_text: str) -> Tuple[str, str]:
    """Extract code block and explanation from LLM response.

    Args:
        response_text: Raw response from LLM

    Returns:
        Tuple of (code, explanation)
    """
    # Extract code block
    code_pattern = r"```python\s*(.*?)\s*```"
    code_match = re.search(code_pattern, response_text, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try without language tag
        code_pattern = r"```\s*(.*?)\s*```"
        code_match = re.search(code_pattern, response_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""

    # Extract explanation (everything after the code block)
    explanation_pattern = r"```.*?```\s*(?:EXPLANATION:)?\s*(.*)"
    explanation_match = re.search(explanation_pattern, response_text, re.DOTALL)

    if explanation_match:
        explanation = explanation_match.group(1).strip()
    else:
        explanation = "Code was executed to modify the data."

    return code, explanation


def execute_code_safely(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Execute generated code on the DataFrame.

    Args:
        df: Input DataFrame (will be copied)
        code: Python code to execute

    Returns:
        Modified DataFrame

    Raises:
        ValueError: If code execution fails
    """
    import numpy as np

    # Work with a copy to avoid side effects
    df_copy = df.copy()

    # Create execution namespace
    namespace: Dict[str, Any] = {
        "df": df_copy,
        "pd": pd,
        "np": np,
    }

    try:
        exec(code, namespace)
        return namespace["df"]
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        logger.error(f"Code was:\n{code}")
        raise ValueError(f"Failed to execute modification code: {e}")


class LLMService:
    """Service for LLM-powered data modifications."""

    def __init__(self):
        """Initialize the LLM service with OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, LLM service will not work")

        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    def modify_data(
        self,
        df: pd.DataFrame,
        required_columns: RequiredColumns,
        user_query: str,
    ) -> Tuple[pd.DataFrame, str, str]:
        """Modify data based on user query using LLM.

        Args:
            df: Input DataFrame to modify
            required_columns: Column type information
            user_query: Natural language modification request

        Returns:
            Tuple of (modified_df, code_executed, explanation)

        Raises:
            ValueError: If LLM client not available or modification fails
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        # Build prompt
        user_prompt = build_user_prompt(df, required_columns, user_query)

        logger.info(
            f"Sending modification request to LLM: {user_query[:100]}..."
        )

        # Call LLM - only use temperature for models that support it
        llm_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        # gpt-5 doesn't support temperature parameter
        if "gpt-5" not in self.model:
            llm_params["temperature"] = (
                0.1  # Low temperature for consistent code generation
            )

        response = self.client.chat.completions.create(**llm_params)

        response_text = response.choices[0].message.content or ""
        logger.debug(f"LLM response:\n{response_text}")

        # Extract code and explanation
        code, explanation = extract_code_and_explanation(response_text)

        if not code:
            raise ValueError("LLM did not generate any code")

        logger.info(f"Executing code:\n{code}")

        # Execute code
        modified_df = execute_code_safely(df, code)

        return modified_df, code, explanation


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get the LLM service instance (singleton).

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
