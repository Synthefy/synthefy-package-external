import json
import os
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from loguru import logger
from openai import OpenAI

from synthefy_pkg.app.data_models import (
    LLMExplanationRequest,
    LLMExplanationResponse,
)

COMPILE = False
router = APIRouter(tags=["Explain Time Series"])

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    client = None


@router.post("/api/explain", response_model=LLMExplanationResponse)
async def explain_time_series(
    request: LLMExplanationRequest,
) -> LLMExplanationResponse:
    if client is None:
        raise HTTPException(
            status_code=500, detail="LLM client not initialized"
        )

    logger.info("Received request for explain time series.")
    # Compose a simple prompt using request data
    values = request.timeseries_to_summarize[0].values

    other_timeseries = {}
    if request.other_timeseries is not None:
        for ts in request.other_timeseries:
            other_timeseries[ts.name] = ts.values

    filter_string = "This data is a time series for the "
    if request.filters_applied:
        for filter_name, filter_values in request.filters_applied.items():
            filter_string += f"{filter_name} with values: {filter_values}."

    if request.timestamps and request.timestamps.values:
        # Timestamps provided: zip them
        timestamp_value_pairs = list(zip(request.timestamps.values, values))
        tuple_str = ", ".join(
            [f"({ts}: {val})" for ts, val in timestamp_value_pairs]
        )
    else:
        # No timestamps: just enumerate the values
        tuple_str = ", ".join(
            [f"(index {i}: {val})" for i, val in enumerate(values)]
        )

    other_timeseries_str = ""
    for ts_name, ts_values in other_timeseries.items():
        other_ts_value_str = ", ".join(f"{val}" for val in ts_values)
        other_timeseries_str += f"{ts_name}: {other_ts_value_str}\n"

    prompt = (
        f"You are a knowledgeable assistant that explains the patterns observed in the time series data through web search. "
        f"Explain the main summary of the web search in one sentence."
        f"{filter_string}"
        f'The time series data is for "{request.timeseries_to_summarize[0].name}" and the values are: {tuple_str}. '
        f"Other relevant time series data are: {other_timeseries_str}"
        f"Additional context: {request.text or ''}"
    )
    logger.info(f"Calling OpenAI web search API with prompt: {prompt}")

    # Call OpenAI with web search capability
    response = client.responses.create(
        model="gpt-4.1", tools=[{"type": "web_search_preview"}], input=prompt
    )

    # Extract explanation from the OpenAI response
    # The response format will include tool outputs and the assistant's message
    # We need to extract the message content
    explanation = ""
    if response.output[0].status == "completed":
        for item in response.output[1].content:  # type: ignore
            explanation += item.text  # type: ignore

    status = "success" if explanation else "error"
    return LLMExplanationResponse(explanation=explanation, status=status)


def extract_perc_change(request):
    # Implement logic to extract percent change from timeseries
    return 5.2  # Example


def extract_rate_of_change(request):
    # Implement logic
    return 1.1


def extract_peak_and_troughs(request):
    return {"peak": 100, "trough": 20}


def extract_business_context(request):
    return {"industry": "retail", "region": "US"}


def extract_datetime_context(request):
    return {"start": "2023-01-01", "end": "2023-03-31", "quarter": "Q1"}


def extract_ts_correlation(request):
    return {"correlated_metric": "sales", "correlation": 0.8}
