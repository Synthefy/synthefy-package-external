import math
import multiprocessing as mp
import os
import time
import warnings
from multiprocessing import Manager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import linalg, signal
from tqdm import tqdm

from synthefy_pkg.preprocessing.relational.time_series_comparison_utils import (
    compare_time_series,
)


def reduce_dimensions(
    embeddings, method="pca", n_components=2, reducer=None, **kwargs
):
    """
    Perform dimensionality reduction on embeddings.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: 'pca', 'tsne', or 'umap'
        n_components: number of dimensions to reduce to

    Returns:
        reduced embeddings of shape (n_samples, n_components)
    """
    if embeddings.shape[1] < n_components:
        return embeddings
    if reducer is None:
        if len(method) == 0:
            return embeddings[:, :n_components]
        if method.lower() == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components, **kwargs)
        elif method.lower() == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=n_components, **kwargs)
        elif method.lower() == "umap":
            from umap import UMAP  # type: ignore
            # UMAP = None
            # raise NotImplementedError("UMAP is not installed")

            reducer = UMAP(**kwargs)

        else:
            return embeddings[:, :n_components]

    result = reducer.fit_transform(embeddings)
    # for the type checker
    if sp.issparse(result):
        assert hasattr(result, "toarray"), (
            "UMAP embedding is not a valid sparse matrix"
        )
        if isinstance(result, sp.csr_matrix):
            result = result.toarray()
        else:
            result = result
    assert isinstance(result, np.ndarray), "result is not a numpy array"
    return result[:, :n_components]


def cut_timeseries_by_date(timeseries, timestamps, cutoff_date):
    """
    Cut timeseries data up to a specific date.

    Args:
        timeseries: DataFrame or array of time series values
        timestamps: DatetimeIndex or array of timestamps
        cutoff_date: timestamp cutoff (inclusive)

    Returns:
        filtered timeseries
    """
    date_mask = timestamps <= pd.Timestamp(cutoff_date)
    return timeseries[date_mask]


warnings.filterwarnings(
    "ignore", message="Non-stationary starting autoregressive parameters"
)
warnings.filterwarnings(
    "ignore", message="Non-invertible starting MA parameters"
)


def timeseries_to_fixed_vector(
    timeseries, methods=[], embed_dims=[], window_size=None
):
    """
    Convert variable-length time series to fixed-length vector.

    Args:
        timeseries: numpy array of time series data
        method: 'statistical', 'sampling', 'wavelet', etc.
        window_size: size for window-based methods

    Returns:
        fixed-length vector representation
    """
    embeddings = []
    assert len(methods) == len(embed_dims), (
        "methods and embed_dims must have the same length"
    )
    for method, embed_dim in zip(methods, embed_dims):
        if method == "statistical":
            # Statistical features
            features = []
            # Basic statistics
            features.extend(
                [
                    np.mean(timeseries),
                    np.std(timeseries),
                    np.min(timeseries),
                    np.max(timeseries),
                    np.median(timeseries),
                    np.percentile(timeseries, 25),
                    np.percentile(timeseries, 75),
                ]
            )

            # Trend features
            if len(timeseries) > 1:
                # Linear trend coefficient
                x = np.arange(len(timeseries))
                A = np.vstack([x, np.ones(len(x))]).T
                slope, _ = np.linalg.lstsq(A, timeseries, rcond=None)[0]
                features.append(slope)

                # Autocorrelation at different lags
                for lag in [1, 7, 30]:
                    if len(timeseries) > lag:
                        autocorr = np.corrcoef(
                            timeseries[:-lag], timeseries[lag:]
                        )[0, 1]
                        features.append(autocorr)
                    else:
                        features.append(0)

            embeddings.append(np.array(features))

        elif method == "sampling":
            # Uniform sampling to fixed length
            target_len = embed_dim  # Define your fixed length
            indices = np.linspace(0, len(timeseries) - 1, target_len, dtype=int)
            embeddings.append(timeseries[indices])

        elif method == "fourier":
            # Fast Fourier Transform features
            fft_features = np.abs(np.fft.rfft(timeseries))
            # Take first N coefficients or resample
            target_len = embed_dim  # Define your fixed length
            if len(fft_features) > target_len:
                indices = np.linspace(
                    0, len(fft_features) - 1, target_len, dtype=int
                )
                embeddings.append(fft_features[indices])
            else:
                embeddings.append(
                    np.pad(fft_features, (0, target_len - len(fft_features)))
                )
        elif method == "wavelet":
            from scipy.fftpack import dct  # DCT as wavelet alternative

            # Using scipy.signal's CWT as a substitute for wavelet transforms
            # Set parameters for the transform
            target_len = embed_dim  # Define fixed length for output

            # Pad input if needed to reduce edge effects
            orig_len = len(timeseries)
            padded = np.pad(timeseries, (16, 16), mode="symmetric")

            # Define scales for the continuous wavelet transform
            num_scales = min(64, orig_len // 2)
            scales = np.arange(1, num_scales + 1)

            # Perform continuous wavelet transform using Ricker (Mexican hat) wavelet
            # Note: CWT is computationally intensive for long series
            coeffs = signal.cwt(padded, signal.ricker, scales)

            # Take central portion to avoid edge effects
            coeffs = coeffs[:, 16 : 16 + orig_len]

            # Average or sample along the time axis to get fixed length
            # Option 1: Average across time windows
            if orig_len > target_len:
                window_size = orig_len // target_len
                features = np.array(
                    [
                        np.mean(
                            coeffs[:, i * window_size : (i + 1) * window_size],
                            axis=1,
                        )
                        for i in range(target_len)
                    ]
                ).flatten()
            else:
                # For short series, use the DCT as an alternative
                # DCT is related to wavelets but more efficient
                dct_coeffs = dct(timeseries, type=2)
                features = np.abs(dct_coeffs[:target_len])
                if len(features) < target_len:
                    features = np.pad(features, (0, target_len - len(features)))

            # Ensure fixed length output
            if len(features) > target_len:
                features = features[:target_len]

            embeddings.append(features)

        elif method == "ssa":
            # Singular Spectral Analysis (SSA)
            target_len = embed_dim  # Define fixed length for output

            # Check for NaN values in the time series
            if np.isnan(timeseries).any():
                # Handle NaN values: either fill them or return default features
                timeseries = np.nan_to_num(
                    timeseries, nan=0.0
                )  # Replace NaNs with zeros TODO: use a more proper way

            # Set embedding dimension (window length)
            if window_size is None:
                window_size = min(len(timeseries) // 2, 20)  # Default value

            # Step 1: Embedding - Create trajectory matrix
            K = len(timeseries) - window_size + 1
            trajectory_matrix = np.zeros((window_size, K))
            for i in range(K):
                trajectory_matrix[:, i] = timeseries[i : i + window_size]

            # Step 2: SVD of trajectory matrix
            U, sigma, Vt = linalg.svd(trajectory_matrix, full_matrices=False)

            # Step 3: Extract components
            # Use singular values and first few left/right singular vectors as features
            features = []

            # Add singular values (importance of components)
            features.extend(sigma[: min(10, len(sigma))])

            # Add elements from principal components (U matrix)
            for i in range(min(5, U.shape[1])):
                features.extend(U[: min(5, U.shape[0]), i])

            # Ensure fixed length
            if len(features) > target_len:
                features = features[:target_len]
            else:
                features = np.pad(features, (0, target_len - len(features)))

            embeddings.append(np.array(features))

        elif method == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller

            # ARIMA parameter fitting
            target_len = embed_dim  # Define fixed length for output

            # Check if series is too short for meaningful ARIMA
            if len(timeseries) < 10:
                embeddings.append(
                    np.zeros(target_len)
                )  # Return zeros for very short series
                continue
            elif np.allclose(timeseries, timeseries[0], rtol=1e-5, atol=1e-8):
                # For constant series, set differencing to 0 and other parameters to 0
                features = np.zeros(target_len)
                features[0] = 0  # d parameter = 0 for constant series
                embeddings.append(features)
                continue

            # Try to fit ARIMA model with automatic parameter selection
            features = []

            # Check for stationarity (determines d parameter)
            adf_result = adfuller(timeseries)
            p_value = adf_result[1]
            # Determine d based on stationarity test
            d = 0 if p_value < 0.05 else 1
            features.append(d)  # Add differencing parameter

            # Try different p, q values (1,1), (2,0), (0,1), (2,2)
            best_aic = np.inf
            best_model = None
            best_order = None

            for p in range(3):
                for q in range(3):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(timeseries, order=(p, d, q))
                            model_fit = model.fit()
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_model = model_fit
                                best_order = (p, d, q)
                    except Exception:
                        # Skip this model configuration if fitting fails
                        continue
            if best_model is not None:
                # Add model order
                assert isinstance(best_order, tuple), (
                    "best_order is not a tuple"
                )
                features.extend(best_order)

                # Add model coefficients
                if hasattr(best_model, "arparams"):
                    features.extend(
                        best_model.arparams[: min(5, len(best_model.arparams))]
                    )
                if hasattr(best_model, "maparams"):
                    features.extend(
                        best_model.maparams[: min(5, len(best_model.maparams))]
                    )

                # Add AIC and log-likelihood
                features.append(best_model.aic)
                features.append(best_model.llf)
            else:
                # Fallback if no model fits
                features.extend([0, d, 0])  # p, d, q
                features.extend(
                    [0] * (target_len - len(features))
                )  # Placeholder for coefficients and stats
            # # Fallback for any error
            # features = [0] * 15

            # Ensure fixed length
            if len(features) > target_len:
                features = features[:target_len]
            else:
                features = np.pad(features, (0, target_len - len(features)))

            embeddings.append(np.array(features))

        else:
            continue
    if len(embeddings) == 0:
        return np.zeros((1,))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.nan_to_num(
        embeddings, nan=0.0
    )  # TODO: find a more elegant way to catch nans
    embeddings[np.isinf(embeddings)] = 0.0
    return embeddings


# def compute_series_similarity(series_list, indices, method="euclidean"):
#     """
#     Compute the pairwise similarities between the series in the series list
#     although these are symmetric, doesn't skip symmetric pairs
#     """
#     similarity_matrix = np.zeros((len(indices), len(series_list)))
#     for i in tqdm(range(len(indices))):
#         for j in tqdm(range(len(series_list))):
#             idx = indices[i]
#             similarity_matrix[i, j] = compare_time_series(
#                 series_list[idx], series_list[j], method=method
#             )
#     return similarity_matrix


def _init_worker(shared_dict):
    """Initialize worker process with shared data"""
    global _worker_shared_data
    _worker_shared_data = shared_dict


def _process_matrix_block(block_data):
    """Process a block of the similarity matrix in a single process"""
    from tqdm.auto import tqdm

    global _worker_shared_data

    block_id, row_indices, col_indices = block_data
    results = []
    total = len(row_indices) * len(col_indices)

    # Get shared data
    series_list = _worker_shared_data["series_list"]
    indices = _worker_shared_data["indices"]
    method = _worker_shared_data["method"]

    # Create a progress bar for this block
    with tqdm(
        total=total,
        desc=f"Block {block_id + 1}",
        leave=False,
        position=block_id % 10,
    ) as pbar:
        # Process all cell comparisons in this block
        for i_local in row_indices:
            idx = indices[i_local]  # Get the actual series index

            for j in col_indices:
                try:
                    similarity = compare_time_series(
                        series_list[idx], series_list[j], method=method
                    )
                    results.append((i_local, j, similarity))
                except Exception:
                    results.append((i_local, j, 0.0))
                pbar.update(1)

    return results


def compute_series_similarity(
    series_list, indices, method="euclidean", n_jobs=32, inner_processes=4
):
    """
    Compute pairwise similarities between time series using flat parallelism with optimized startup.

    Args:
        series_list: List of time series
        indices: Indices to compute similarities for
        method: Method to use for comparison
        n_jobs: Number of row parallel processes
        inner_processes: Number of column parallel processes

    Returns:
        similarity_matrix: Matrix of pairwise similarities
    """

    if n_jobs <= 0:
        similarity_matrix = np.zeros((len(indices), len(series_list)))
        for i in tqdm(range(len(indices))):
            for j in tqdm(range(len(series_list))):
                idx = indices[i]
                similarity_matrix[i, j] = compare_time_series(
                    series_list[idx], series_list[j], method=method
                )
        return similarity_matrix

    # Determine total number of processes to create
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 300
    total_processes = min(n_jobs * inner_processes, cpu_count - 1, 128)

    # Calculate dimensions
    n_rows = len(indices)
    n_cols = len(series_list)
    total_comparisons = n_rows * n_cols

    # Determine optimal block size
    blocks_per_dim = math.ceil(math.sqrt(total_processes))
    row_blocks = min(blocks_per_dim, n_rows)
    col_blocks = min(total_processes // row_blocks, n_cols)

    # Calculate actual block sizes
    row_block_size = math.ceil(n_rows / row_blocks)
    col_block_size = math.ceil(n_cols / col_blocks)

    # Create blocks (only pass indices, not actual data)
    blocks = []
    block_id = 0

    for r_start in range(0, n_rows, row_block_size):
        r_end = min(r_start + row_block_size, n_rows)
        row_range = list(range(r_start, r_end))

        for c_start in range(0, n_cols, col_block_size):
            c_end = min(c_start + col_block_size, n_cols)
            col_range = list(range(c_start, c_end))

            blocks.append((block_id, row_range, col_range))
            block_id += 1

    start_time = time.time()
    actual_blocks = len(blocks)
    # Determine total number of processes to create
    total_processes = min(n_jobs * inner_processes, actual_blocks, 128)

    print(f"Splitting {n_rows}×{n_cols} matrix into {actual_blocks} blocks")
    print(f"Using {total_processes} parallel processes")
    print(f"Total comparisons: {total_comparisons}")

    # Initialize result matrix
    similarity_matrix = np.zeros((n_rows, n_cols))

    # Create overall progress bar
    main_pbar = tqdm(total=total_comparisons, desc="Overall progress")

    # Create shared data dictionary using Manager
    # This avoids copying large data to each process
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict["series_list"] = series_list
    shared_dict["indices"] = indices
    shared_dict["method"] = method

    # Create process pool with initialization
    with mp.get_context("spawn").Pool(
        processes=total_processes,
        initializer=_init_worker,
        initargs=(shared_dict,),
    ) as pool:
        try:
            # Submit all blocks for processing
            # Use starmap_async to start all processes quickly
            results = []
            results_async = pool.map_async(
                _process_matrix_block,
                blocks,
                callback=lambda x: results.extend(x),
            )

            # Poll until complete, updating progress
            while not results_async.ready():
                time.sleep(0.1)

                # Check if any new results added
                current_count = sum(len(batch) for batch in results)
                if current_count > main_pbar.n:
                    main_pbar.update(current_count - main_pbar.n)

                    # Report statistics
                    if current_count % max(1, total_comparisons // 20) < 100:
                        elapsed = time.time() - start_time
                        speed = current_count / elapsed if elapsed > 0 else 0
                        remaining = total_comparisons - current_count
                        est_time = (
                            remaining / speed if speed > 0 else float("inf")
                        )

                        print(
                            f"\nProgress: {current_count}/{total_comparisons} "
                            f"({current_count / total_comparisons:.1%})"
                        )
                        print(
                            f"Speed: {speed:.2f} comp/sec, Est. remaining: {est_time / 60:.1f} min"
                        )

            # Get all results
            all_results = results_async.get()

            # Flatten results and update matrix
            for batch_results in all_results:
                for i, j, sim in batch_results:
                    similarity_matrix[i, j] = sim

            # Ensure progress bar is complete
            main_pbar.update(total_comparisons - main_pbar.n)

        except KeyboardInterrupt:
            print("\nInterrupted. Shutting down...")
            pool.terminate()
            main_pbar.close()
            raise
        except Exception as e:
            print(f"\nError: {str(e)}")
            pool.terminate()
            main_pbar.close()
            raise
        finally:
            main_pbar.close()

    # Final stats
    total_time = time.time() - start_time
    print(
        f"\nCompleted in {total_time:.2f}s ({total_comparisons / total_time:.2f} comp/sec)"
    )

    return similarity_matrix
