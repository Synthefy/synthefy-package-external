from typing import List

TOOL_INTENTS = [
    "FORECAST",
    "SEARCH_METADATA",
    "CHANGE_PARAMETERS",
    "VISUALIZE",
    "EXPLAIN",
    "ANOMALY_DETECTION",
    "UNKNOWN",
]


def classify_intents(user_input: str) -> List[str]:
    """
    Classify the user input into one or more tool intents.
    This is a simple keyword-based classifier for demonstration.
    """
    input_lower = user_input.lower()
    intents = []
    if any(word in input_lower for word in ["forecast", "predict", "future"]):
        intents.append("FORECAST")
    if any(
        word in input_lower
        for word in ["search", "metadata", "find dataset", "find data"]
    ):
        intents.append("SEARCH_METADATA")
    if any(
        word in input_lower
        for word in ["parameter", "change setting", "set parameter", "adjust"]
    ):
        intents.append("CHANGE_PARAMETERS")
    if any(
        word in input_lower
        for word in ["visualize", "plot", "show graph", "chart"]
    ):
        intents.append("VISUALIZE")
    if any(
        word in input_lower
        for word in ["explain", "why", "how does", "interpret"]
    ):
        intents.append("EXPLAIN")
    if any(
        word in input_lower
        for word in [
            "anomaly",
            "detect outlier",
            "outlier",
            "anomaly detection",
        ]
    ):
        intents.append("ANOMALY_DETECTION")
    if not intents:
        intents.append("UNKNOWN")
    return intents
