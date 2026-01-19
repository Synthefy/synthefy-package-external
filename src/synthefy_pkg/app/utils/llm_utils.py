from typing import List

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from synthefy_pkg.app.data_models import (
    MetaData,
    OneContinuousMetaData,
    OneDiscreteMetaData,
    TimeStamps,
)

llm = AzureChatOpenAI(
        azure_endpoint="https://synthefy-synthesis.openai.azure.com/",
        api_key="98c35eba36fa48d2b61bbdf15b887311", # type: ignore
        model="gpt-4o",
        azure_deployment="SynthefyGPT4o",
        api_version="2024-02-15-preview",
    )

COMPILE = True


class MetaDataToParse(BaseModel):
    continuous_conditions: List[OneContinuousMetaData] = Field(
        default_factory=list, description="List of continuous metadata conditions"
    )
    discrete_conditions: List[OneDiscreteMetaData] = Field(
        default_factory=list, description="List of discrete metadata conditions"
    )
    timestamps: List[TimeStamps] = Field(
        default_factory=list, description="List of timestamp conditions"
    )
    num_examples: int = Field(
        default=1, description="Number of examples to synthesize or forecast"
    )


def validate_metadata(
    parsed_metadata: MetaDataToParse,
    continuous_columns: List[str],
    discrete_columns: List[str],
) -> MetaDataToParse:
    # Validate and filter discrete conditions
    parsed_metadata.discrete_conditions = [
        condition
        for condition in parsed_metadata.discrete_conditions
        if condition.name in discrete_columns
    ]

    # Validate and filter continuous conditions
    parsed_metadata.continuous_conditions = [
        condition
        for condition in parsed_metadata.continuous_conditions
        if condition.name in continuous_columns
    ]

    # Log any removed conditions
    removed_discrete = [
        c.name
        for c in parsed_metadata.discrete_conditions
        if c.name not in discrete_columns
    ]
    removed_continuous = [
        c.name
        for c in parsed_metadata.continuous_conditions
        if c.name not in continuous_columns
    ]
    if removed_discrete or removed_continuous:
        logger.warning(
            f"Removed invalid conditions: Discrete: {removed_discrete}, Continuous: {removed_continuous}"
        )
    return parsed_metadata


def extract_metadata_from_query(
    query: str,
    timeseries_columns: List[str],
    continuous_columns: List[str],
    discrete_columns: List[str],
    timestamps_col: List[str],
) -> MetaDataToParse:
    """
    Extracts metadata from a user query and returns it as a MetaData object.

    Args:
        query (str): The user's command to set metadata conditions.
        timeseries_columns (List[str]): List of available timeseries columns.
        continuous_columns (List[str]): List of available continuous columns.
        discrete_columns (List[str]): List of available discrete columns.
        timestamps_col (List[str]): List of available timestamp columns.

    Returns:
        MetaData: The extracted metadata conditions.
    """
    output_parser = PydanticOutputParser(pydantic_object=MetaDataToParse)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a specialized assistant designed to extract and set metadata conditions based on explicit user instructions.
                The user will provide commands to define metadata attributes for time series data, including both attribute names and their exact values.

                The metadata fields are:
                - **Timeseries Columns:** {timeseries_columns}
                - **Continuous Columns:** {continuous_columns}
                - **Discrete Columns:** {discrete_columns}
                - **Timestamps Columns:** {timestamps_col}

                Your task is to parse the user's command to set metadata conditions and return them in a JSON format that exactly matches the `MetaData` class structure.

                **Guidelines:**
                - **You can ONLY set the given columns. There may be typos in the columns names from the query. It's your job to match the columns correctly.**
                - **Output Format:** Only return a single JSON object without any additional text.
                - **Field Names:** Use the exact field names as defined in the `MetaData` class (`discrete_conditions`, `continuous_conditions`, `timestamps`).
                - **Structure:**
                  - `continuous_conditions`: List of objects with `name` (str) and `values` (List[float]).
                  - `discrete_conditions`: List of objects with `name` (str) and `values` (List[Union[str, int, float]]).
                  - `timestamps`: List of objects with `name` (str) and `values` (List[Any]). These should be parsable as timestamps in python.
                  - `num_examples`: int

                - **JSON Validity:** Ensure the JSON is valid and properly formatted.

                You will return a JSON object that matches the following Pydantic model:
                <Format Instructions>
                {format_instructions}
                </Format Instructions>


                - **No Extra Information:** Do not include explanations, comments, or any text outside the JSON object.
                """,
            ),
            (
                "human",
                """
                <Examples>
                **Input:**
                The metadata fields are:
                - **Timeseries Columns:** ['timeseries_1', 'timeseries_2', 'timeseries_3']
                - **Continuous Columns:** ['temperature', 'pressure', 'humidity']
                - **Discrete Columns:** ['status', 'error_code', 'priority']
                - **Timestamps Columns:** ['timestamp']

                query: Show me a forecast for timeseries_1 with device_statuses with values ["active", "inactive"], error_codes with values [100, 101, 102], and priority_levels with values ["high", "medium", "low"], and temperature = 22.5, 23.0, 21.8.
                **Output:**
                {{
                    "continuous_conditions": [
                        {{
                            "name": "temperature",
                            "values": [22.5, 23.0, 21.8]
                        }}
                    ],
                    "discrete_conditions": [
                        {{
                            "name": "device_status",
                            "values": ["active", "inactive"]
                        }},
                        {{
                            "name": "error_code",
                            "values": [100, 101, 102]
                        }},
                        {{
                            "name": "priority_levels",
                            "values": ["high", "medium", "low"]
                        }}
                    ],
                    "timestamps": [],
                    "num_examples": 1
                }}

                **Input:**
                The metadata fields are:
                - **Timeseries Columns:** ['packet_loss']
                - **Continuous Columns:** ['temperature', 'pressure', 'humidity']
                - **Discrete Columns:** ['status', 'error_code', 'priority']
                - **Timestamps Columns:** ['timestamp']

                query: Synthesize 5 examples of packet_loss for device id 3, device type 2, with temperature values [22.5, 23.0, 21.8] and status values ["ok", "fail"].
                **Output:**
                {{
                    "continuous_conditions": [
                        {{
                            "name": "temperature",
                            "values": [22.5, 23.0, 21.8]
                        }}
                    ],
                    "discrete_conditions": [
                        {{
                            "name": "status",
                            "values": ["ok", "fail"]
                        }}
                    ],
                    "timestamps": [],
                    "num_examples": 5
                }}

                </Examples>
                Input:
                The metadata fields are:
                - **Timeseries Columns:** {timeseries_columns}
                - **Continuous Columns:** {continuous_columns}
                - **Discrete Columns:** {discrete_columns}
                - **Timestamps Columns:** {timestamps_col}

                query: {query}
                Output:
                """,
            ),
        ]
    )

    chain = prompt_template | llm | output_parser
    try:
        ret = chain.invoke(
            {
                "timeseries_columns": timeseries_columns,
                "continuous_columns": continuous_columns,
                "discrete_columns": discrete_columns,
                "timestamps_col": timestamps_col,
                "format_instructions": output_parser.get_format_instructions(),
                "query": query,
            }
        )
        logger.info(f"Parsed metadata: {ret}")
        ret = validate_metadata(ret, continuous_columns, discrete_columns)

        logger.info(f"Metadata extracted: {ret}")
        return ret

    except Exception as e:
        logger.error(f"Error parsing metadata: {e} - Using default metadata from UI")
        return MetaDataToParse()
