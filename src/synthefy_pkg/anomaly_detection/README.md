# Anomaly Detection Configuration Guide

This document explains the configuration parameters used for running the anomaly detection script (`synthefy_anomaly_detector_v2.py`).

## Configuration Structure

The anomaly detection configuration is defined under the `anomaly_detection` key in the preprocessing config file. Here's a detailed breakdown of each parameter:

### Frequency Configuration
```json
"frequency": {
    "type": "H",
    "windows": {
        "volatility": 24,
        "pattern": 168,
        "min_points": 50,
        "baseline_multiplier": 7
    }
}
```
- `type`: Time series frequency identifier
  - Supported values: 
    - `"H"`: Hourly
    - `"D"`: Daily
    - `"T"` or `"min"`: Minutely
    - `"W"`: Weekly
    - `"M"`: Monthly
    - `"15T"`: 15-minutely
    - `"30T"`: 30-minutely
- `windows`: Window sizes for different calculations (can be auto-detected depending on the provided frequency)
  - `volatility`: Window size for volatility calculations (e.g., 24 hours)
  - `pattern`: Window size for pattern analysis (e.g., 168 hours = 1 week)
  - `min_points`: Minimum required data points for analysis
  - `baseline_multiplier`: Multiplier for baseline window (e.g., 7x volatility window)

### Threshold Configuration
```json
"sd_thresholds": {
    "peak": 2.5,
    "scattered": 2.5,
    "out_of_pattern": 2.5
}
```
- `peak`: Z-score threshold for peak anomalies
- `scattered`: Threshold for volatility-based anomalies
- `out_of_pattern`: Threshold for pattern deviation anomalies

### Concurrent Anomalies Configuration
```json
"concurrent_anomalies": {
    "enabled": false,
    "time_window": "1H",
    "min_kpis_for_concurrent_anomaly": 2,
    "combine_types": true
}
```
Concurrent anomalies are anomalies that occur across multiple KPIs within a specified time window. This feature helps identify system-wide issues that affect multiple metrics simultaneously.

- `enabled`: Enable/disable concurrent anomaly detection
- `time_window`: Time window to look for concurrent anomalies (e.g., "1H" for 1 hour). Any anomalies occurring within this window across different KPIs will be grouped together
- `min_kpis_for_concurrent_anomaly`: Minimum number of KPIs that must show anomalies within the time window for it to be considered a concurrent anomaly. For example, if set to 2, at least 2 different KPIs must show anomalies within the specified time window
- `combine_types`: Whether to consider different anomaly types together when looking for concurrent anomalies
  - If `true`: Will combine peak, scattered, and out-of-pattern anomalies
  - If `false`: Will only look for concurrent anomalies of the same type

Example scenario:
- If CPU usage shows a peak anomaly at 10:00 AM
- And Memory usage shows a scattered anomaly at 10:45 AM
- With `time_window: "1H"` and `combine_types: true`, these would be detected as concurrent anomalies
- This could indicate a system-wide issue affecting multiple metrics

## Required Data Columns

The script expects the following columns in your input data:

1. Timestamp column (specified in `timestamps_col`)
2. Group label columns (specified in `group_labels_cols`) - optional
3. Time series columns (specified in `timeseries_cols`)

## Example Configuration
### Example
Here's an example configuration:

```json
{
    "anomaly_detection": {
        "frequency": {
            "type": "H",
            "windows": {
                "volatility": 24,
                "pattern": 168,
                "min_points": 50,
                "baseline_multiplier": 7
            }
        },
        "sd_thresholds": {
            "peak": 2.5,
            "scattered": 2.5,
            "out_of_pattern": 2.5
        },
    }
}
```
### Minimal example
Here's a minimal example configuration:

```json
{
    "anomaly_detection": {
        "frequency": {
            "type": "H"
        }
    }
}
```
## Running the Script

To run the anomaly detection script:

```bash
python synthefy_anomaly_detector_v2.py --config path/to/config.json
```

Optional: Use previously saved results:
```bash
python synthefy_anomaly_detector_v2.py --config path/to/config.json --results path/to/results.json
```

## Output

The script generates:
1. JSON file containing detected anomalies
2. Visualization plots for top anomalies by type
3. Logs detailing the detection process

The outputs are saved in the directory specified by `SYNTHEFY_DATASETS_BASE` environment variable.
