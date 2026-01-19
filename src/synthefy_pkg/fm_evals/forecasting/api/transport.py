from typing import Any, Dict, List

import numpy as np

from synthefy.data_models import (
    ForecastV2Response,
    SingleEvalSamplePayload,
    SingleSampleForecastPayload,
)
from synthefy_pkg.fm_evals.formats.eval_batch_format import (
    EvalBatchFormat,
    SingleEvalSample,
)
from synthefy_pkg.fm_evals.formats.forecast_output_format import (
    ForecastOutputFormat,
    SingleSampleForecast,
)

COMPILE = True


def eval_batch_to_request_payload(
    batch: EvalBatchFormat,
) -> Dict[str, Any]:
    samples_payload: List[List[Dict[str, Any]]] = []
    for b in range(batch.batch_size):
        row_payload: List[Dict[str, Any]] = []
        for nc in range(batch.num_correlates):
            s = batch[b, nc]
            row_payload.append(
                {
                    "sample_id": (
                        s.sample_id.tolist()
                        if hasattr(s.sample_id, "tolist")
                        else s.sample_id
                    ),
                    "history_timestamps": [
                        str(t) for t in s.history_timestamps
                    ],
                    "history_values": s.history_values.astype(float).tolist(),
                    "target_timestamps": [str(t) for t in s.target_timestamps],
                    "target_values": s.target_values.astype(float).tolist(),
                    "forecast": bool(s.forecast),
                    "metadata": bool(s.metadata),
                    "leak_target": bool(s.leak_target),
                    "column_name": s.column_name,
                }
            )
        samples_payload.append(row_payload)
    return {"samples": samples_payload}


def response_to_forecast_output(
    response_json: Dict[str, Any],
) -> ForecastOutputFormat:
    forecasts = response_json.get("forecasts", [])
    if not forecasts:
        raise ValueError("Mitra response contained no forecasts")

    nested_forecasts: List[List[SingleSampleForecast]] = []
    for row in forecasts:
        out_row: List[SingleSampleForecast] = []
        for f in row:
            sample_id = f.get("sample_id")
            ts = np.array(f.get("timestamps", []), dtype="datetime64[ns]")
            vals = np.asarray(f.get("values", []), dtype=np.float64)
            model_name = f.get("model_name", "mitra")
            out_row.append(
                SingleSampleForecast(
                    sample_id=sample_id,
                    timestamps=ts,
                    values=vals,
                    model_name=model_name,
                )
            )
        nested_forecasts.append(out_row)

    return ForecastOutputFormat(nested_forecasts)


def payload_to_eval_batch(
    samples_payload: List[List[SingleEvalSamplePayload]],
) -> EvalBatchFormat:
    """
    Convert nested payload of SingleEvalSamplePayload into EvalBatchFormat.
    Timestamps are expected as ISO 8601 strings and converted to numpy datetime64[ns].
    """
    samples: List[List[SingleEvalSample]] = []
    for row in samples_payload:
        row_samples: List[SingleEvalSample] = []
        for s in row:
            history_ts = np.array(s.history_timestamps, dtype="datetime64[ns]")
            target_ts = np.array(s.target_timestamps, dtype="datetime64[ns]")
            history_vals = np.asarray(s.history_values, dtype=np.float64)
            target_vals = np.asarray(s.target_values, dtype=np.float64)

            row_samples.append(
                SingleEvalSample(
                    sample_id=s.sample_id,
                    history_timestamps=history_ts,
                    history_values=history_vals,
                    target_timestamps=target_ts,
                    target_values=target_vals,
                    forecast=s.forecast,
                    metadata=s.metadata,
                    leak_target=s.leak_target,
                    column_name=s.column_name,
                )
            )
        samples.append(row_samples)

    return EvalBatchFormat(samples)


def forecasts_to_response(
    forecasts: ForecastOutputFormat,
    model_name: str = "",
) -> ForecastV2Response:
    """Convert internal forecast objects into API response model."""
    response_rows: List[List[SingleSampleForecastPayload]] = []
    for row in forecasts.forecasts:
        out_row: List[SingleSampleForecastPayload] = []
        for f in row:
            out_row.append(
                SingleSampleForecastPayload(
                    sample_id=f.sample_id,
                    timestamps=[str(t) for t in f.timestamps],
                    values=f.values.tolist(),
                    model_name=model_name,
                )
            )
        response_rows.append(out_row)
    return ForecastV2Response(forecasts=response_rows)
