from typing import List
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
from scipy import ndimage


def encode_discrete_metadata(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Encode categorical variables to integer positions."""
    for col in columns:
        df[col] = pd.factorize(df[col])[0]
    return df


def list_s3_files(s3_path: str, file_extension: str | None = None) -> List[str]:
    """
    List files in an S3 bucket/prefix with pagination support for >1000 objects.

    Args:
        s3_path: S3 path in format 's3://bucket-name/prefix/'
        file_extension: Optional file extension to filter by (e.g., '.csv', '.parquet')

    Returns:
        List of S3 URLs for the files found
    """
    # Parse S3 URL
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    # List objects in S3 with pagination
    s3_client = boto3.client("s3", region_name="us-east-1")

    files = []
    continuation_token = None

    while True:
        # Prepare request parameters
        list_kwargs = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,  # Maximum allowed per request
        }

        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        # Make the request
        response = s3_client.list_objects_v2(**list_kwargs)

        # Process the response
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]  # type: ignore
                if file_extension is None or key.endswith(file_extension):
                    files.append(f"s3://{bucket}/{key}")

        # Check if there are more objects to fetch
        if not response.get("IsTruncated", False):
            break

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    # Check if any files were found
    if not files:
        if file_extension:
            raise FileNotFoundError(
                f"No {file_extension} files found in {s3_path}"
            )
        else:
            raise FileNotFoundError(f"No files found in {s3_path}")

    return sorted(files)


def add_noise(
    df,
    columns=None,  # None means all numeric columns
    dist="gaussian",  # we can choose from gaussian, laplace or uniform distribution
    level=0.1,  # for relative: the ratio of the std of each column; absolute: the absolute value
    mode="relative",  # choose adding noise in 'relative' or 'absolute' ways
    p=1.0,  # chances of adding noise to a value
    clip=None,  # formart: [min, max]clip the value after adding noise (might be important for some cases)
    random_state=None,  # random seed used
):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for c in columns:
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(copy=True)
        scale = (np.nanstd(x) * level) if mode == "relative" else float(level)

        if dist == "gaussian":
            noise = rng.normal(0.0, scale, size=x.shape)
            mask = rng.random(x.shape) < p
            x[mask] += noise[mask]
        elif dist == "laplace":
            noise = rng.laplace(0.0, scale, size=x.shape)
            mask = rng.random(x.shape) < p
            x[mask] += noise[mask]
        elif dist == "uniform":
            noise = rng.uniform(-scale, scale, size=x.shape)
            mask = rng.random(x.shape) < p
            x[mask] += noise[mask]
        else:
            raise ValueError("Unknown dist: {}".format(dist))

        if clip is not None:
            x = np.clip(x, clip[0], clip[1])
        out[c] = x
    return out


def resample_df(
    df,
    time_col=None,  # time column name
    freq=None,  # frequency string (e.g., '500ms','2S','5T', following the notations in pandas)
    agg="mean",  # agregation （string or dict）
    upsample_interpolate="linear",  # interpolation method (string)
    step=None,  # step sampling: take one row every k rows
    keep_non_numeric="first",  # aggregation strategy for non-numeric columns if any, choose from'first','last' and 'drop'
):
    if step is not None:
        if step <= 0:
            raise ValueError("step must be positive.")
        return df.iloc[::step].reset_index(drop=True)

    if time_col is None or freq is None:
        raise ValueError(
            "Provide (time_col & freq) for time resampling, or use step for stride sampling."
        )

    if time_col not in df.columns:
        raise KeyError("time_col '{}' not in columns.".format(time_col))

    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.set_index(time_col).sort_index()

    num = tmp.select_dtypes(include=[float, int])
    nonnum = tmp.drop(columns=num.columns, errors="ignore")

    resampled = num.resample(freq)
    if isinstance(agg, str) and agg.lower() in {
        "mean",
        "sum",
        "median",
        "min",
        "max",
        "first",
        "last",
    }:
        down = getattr(resampled, agg)()
    else:
        down = resampled.aggregate(agg)

    if upsample_interpolate is not None:
        down = down.interpolate(
            method=upsample_interpolate, limit_direction="both"
        )

    if keep_non_numeric == "first":
        nonnum_rs = nonnum.resample(freq).first()
    elif keep_non_numeric == "last":
        nonnum_rs = nonnum.resample(freq).last()
    else:
        nonnum_rs = pd.DataFrame(index=down.index)

    out = pd.concat([down, nonnum_rs], axis=1)
    return out.reset_index().rename(columns={"index": time_col})


def lowpass_filter(
    df,
    columns=None,
    method="moving_average",  # choose from 'moving_average', 'ema', and 'gaussian'
    window=5,  # if using int: window size; if str: time window (e.g., '5T')
    time_col=None,  # required for time-based rolling window
    ema_alpha=None,  # if None, use window to infer alpha=2/(window+1)
    sample_rate_hz=None,  # if None, use time_col to infer sample rate
    sigma=None,  # standard deviation for Gaussian filter (if None, use window/6)
):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    out = df.copy()

    if method == "moving_average":
        if isinstance(window, str):
            if time_col is None:
                raise ValueError(
                    "time_col required for time-based rolling window."
                )
            tmp = out.copy()
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
            tmp = tmp.set_index(time_col).sort_index()
            tmp[columns] = tmp[columns].rolling(window, min_periods=1).mean()
            return tmp.reset_index()
        else:
            out[columns] = (
                out[columns].rolling(window=int(window), min_periods=1).mean()
            )
            return out

    elif method == "ema":
        if ema_alpha is None:
            if isinstance(window, str):
                raise ValueError(
                    "EMA needs numeric window if ema_alpha is None."
                )
            ema_alpha = 2.0 / (int(window) + 1.0)
        out[columns] = (
            out[columns].ewm(alpha=float(ema_alpha), adjust=False).mean()
        )
        return out

    elif method == "gaussian":
        if isinstance(window, str):
            raise ValueError("Gaussian filter needs numeric window size.")
        if sigma is None:
            sigma = int(window) / 6.0  # Rule of thumb: sigma = window_size / 6

        # Apply Gaussian filter to each column
        for col in columns:
            data = out[col].values
            # Handle NaN values by forward filling
            # if not numerical, skip
            if np.issubdtype(data.dtype, str):
                data = data.astype(float)
            if not np.issubdtype(data.dtype, np.number):
                continue

            mask = ~np.isnan(data)
            if np.any(mask):
                # Forward fill NaN values for filtering
                data_filled = pd.Series(data).ffill().bfill().values
                # Apply Gaussian filter
                filtered_data = ndimage.gaussian_filter1d(
                    data_filled, sigma=sigma
                )
                # Restore NaN values where they were originally
                filtered_data[~mask] = np.nan
                out[col] = filtered_data
        return out

    else:
        raise ValueError("unknown method '{}'".format(method))
