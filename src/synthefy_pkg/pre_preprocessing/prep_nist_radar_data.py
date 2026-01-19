"""
Run from the directory that has the 'SimulatedRadarWaveforms' folder
"""

import glob
import os

import h5py
import numpy as np
import pandas as pd
from loguru import logger

logger.add("process.log", level="INFO")

COMPILE = False


def main():
    # Collect file paths
    file_paths = [
        f
        for f in glob.glob("./SimulatedRadarWaveforms" + "/**/*", recursive=True)
        if os.path.isfile(f)
    ]
    wave_file_paths = [i for i in file_paths if i.endswith(".mat")]
    csv_file_paths = [i for i in file_paths if i.endswith(".csv")]

    df = pd.DataFrame()

    # Process each wave mat file
    for file_path in wave_file_paths:
        try:
            file_name_root = os.path.basename(file_path).replace(".mat", "")
            csv_file_name = (
                file_name_root.replace("subset", "waveformTableSubset") + ".csv"
            )
            csv_file_path = next(
                (i for i in csv_file_paths if csv_file_name in i), None
            )

            if not csv_file_path:
                logger.warning(f"CSV file for {file_name_root} not found. Skipping.")
                continue

            with h5py.File(file_path, "r") as f:
                radar = np.array(
                    f.get(file_name_root.replace("subset", "waveformSubset"))
                )
                status_data = np.array(
                    f.get(file_name_root.replace("subset", "radarStatusSubset")),
                    dtype=bool,
                ).flatten()

            if status_data.sum() == 0:
                logger.warning(f"All samples are noise for the file: {file_name_root}")
                continue

            full_shape = radar.shape
            radar = radar[status_data]

            window_size = radar.shape[1]
            logger.info(f"{file_name_root}")
            logger.info(f"Data size after noise removal: {full_shape} -> {radar.shape}")

            flattened_array = radar.ravel()

            # Create a DataFrame for the waveform data
            wave_df = pd.DataFrame(
                {"real": flattened_array["real"], "imag": flattened_array["imag"]}
            )

            # Read the corresponding PDW CSV file
            pdw = pd.read_csv(csv_file_path)
            pdw = pdw.iloc[status_data, :]
            pdw = pdw.loc[np.repeat(pdw.index, window_size)].reset_index(drop=True)

            # Concatenate the waveform data and the PDW data
            appender_df = pd.concat([wave_df, pdw], axis=1)
            df = pd.concat([df, appender_df], axis=0)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Saving to parquet: with size {df.shape}")
    df.to_parquet("nist_waveforms_with_pdw.parquet", engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
