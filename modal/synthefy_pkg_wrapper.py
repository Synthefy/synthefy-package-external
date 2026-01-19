"""
Modal wrapper for running synthefy_pkg scripts on cloud GPUs.

Usage:
    uv run modal run modal/wrapper.py

This script sets up a Modal image with:
    - CUDA 12.2 runtime
    - Python 3.11
    - All synthefy_pkg dependencies
    - The synthefy_pkg source code

Functions can then run on Modal's GPU infrastructure.

Data Storage:
    Uses a Modal Volume (synthefy-data) mounted at /data for persistent storage.
    - SYNTHEFY_DATASETS_BASE points to /data
    - Upload raw data with: modal run modal/wrapper.py::upload_data
    - Download outputs with: modal run modal/wrapper.py::download_data
"""

import os
from pathlib import Path

import modal


def get_task_name_from_config(config: str) -> str:
    """
    Generate a descriptive task name from the config filename.

    Args:
        config: Config file path or name

    Returns:
        str: Task name like "oura-preprocess" or "ppg-synthesis"
    """
    # Extract config name without extension
    config_name = os.path.basename(config)
    if "." in config_name:
        config_name = config_name.rsplit(".", 1)[0]

    # Extract dataset/task identifier
    # e.g., "config_oura_preprocessing.json" -> "oura-preprocess"
    # e.g., "config_ppg_synthesis.yaml" -> "ppg-synthesis"
    parts = config_name.replace("config_", "").split("_")

    if len(parts) >= 2:
        dataset = parts[0]
        task = parts[1] if len(parts) > 1 else "task"
        return f"{dataset}-{task}"
    elif len(parts) == 1:
        return parts[0]
    else:
        return "task"


# Define the Modal app with a default name
# The app name can be overridden using Modal's --name CLI option:
#   modal run --name synthefy-oura-preprocess modal/wrapper.py::run_preprocess --config config_oura_preprocessing.json
app = modal.App("synthefy")  # type: ignore

# Get the repo root (parent of this file's directory)
REPO_ROOT = Path(__file__).parent.parent

# Modal Volume for persistent data storage (datasets, preprocessed data, models)
data_volume = modal.Volume.from_name("synthefy-data", create_if_missing=True)  # type: ignore
DATA_MOUNT_PATH = "/data"

# Build the image with CUDA, Python 3.11, and synthefy_pkg dependencies
# Layer order is important for caching:
#   1. System deps (rarely change) - CACHED
#   2. Lock file + dependency install (change when deps change) - CACHED
#   3. Source code + package install (changes frequently) - separate layer
image = (
    # CUDA base image with Python 3.11
    modal.Image.from_registry(  # type: ignore
        "nvidia/cuda:12.2.2-runtime-ubuntu22.04",
        add_python="3.11",
    )
    # System dependencies for packages that need compilation
    # Set non-interactive mode for package installation
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install(
        "build-essential",
        "git",
        "curl",
        "wget",
        "libxml2-dev",
        "libxslt-dev",
        "libfontconfig1-dev",
        "libfreetype6-dev",
        "wkhtmltopdf",  # For PDF report generation
    )
    # Install uv for fast dependency management
    .run_commands("pip install uv")
    # --- CACHED LAYER: Dependencies ---
    # Copy pyproject.toml and uv.lock for dependency resolution
    .add_local_file(
        local_path=REPO_ROOT / "pyproject.toml",
        remote_path="/app/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        local_path=REPO_ROOT / "uv.lock",
        remote_path="/app/uv.lock",
        copy=True,
    )
    # Install ONLY dependencies (not the package itself)
    # This compiles deps from pyproject.toml and installs them
    # Cached until pyproject.toml or uv.lock changes
    .run_commands(
        "cd /app && uv export --frozen --no-hashes > /tmp/requirements.txt && "
        "uv pip install --system -r /tmp/requirements.txt"
    )
    # --- SOURCE CODE LAYER ---
    # Copy source code (changes frequently, but deps are cached above)
    .add_local_dir(
        local_path=REPO_ROOT / "src" / "synthefy_pkg",
        remote_path="/app/src/synthefy_pkg",
        copy=True,
    )
    # Copy examples folder (contains configs and scripts)
    .add_local_dir(
        local_path=REPO_ROOT / "examples",
        remote_path="/app/examples",
        copy=True,
    )
    # Install the package in editable mode (fast, deps already installed)
    .run_commands("cd /app && uv pip install --system --no-deps -e .")
    # Set environment variables for data paths
    .env(
        {
            "SYNTHEFY_DATASETS_BASE": DATA_MOUNT_PATH,
            "SYNTHEFY_PACKAGE_BASE": "/app",
            "LICENSE_KEY": "",
        }
    )
)


@app.function(
    image=image,
    gpu="H100:2",  # Options: T4, L4, A10G, A100, H100 (use "H100:2" for 2 GPUs)
    timeout=3600,  # 1 hour timeout
    volumes={DATA_MOUNT_PATH: data_volume},
)
def check_gpu() -> dict:
    """
    Simple GPU check to verify the setup works.

    Returns:
        dict: GPU availability information.
    """
    import torch

    gpu_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if gpu_available else 0
    device_names = (
        [torch.cuda.get_device_name(i) for i in range(device_count)]
        if gpu_available
        else []
    )

    result = {
        "gpu_available": gpu_available,
        "device_count": device_count,
        "device_names": device_names,
        "cuda_version": torch.version.cuda if gpu_available else "N/A",  # type: ignore
        "torch_version": torch.__version__,
    }

    print("=" * 50)
    print("GPU Availability Check (from Modal)")
    print("=" * 50)
    print(f"  GPU available: {gpu_available}")
    print(f"  Device count: {device_count}")
    print(f"  CUDA version: {result['cuda_version']}")
    print(f"  PyTorch version: {result['torch_version']}")
    if device_names:
        print("  GPU devices:")
        for i, name in enumerate(device_names):
            print(f"    GPU {i}: {name}")
    print("=" * 50)

    if gpu_available:
        print(f"✅ GPU is available! Found {device_count} device(s).")
    else:
        print("❌ No GPU available.")

    return result


@app.function(
    image=image,
    # No GPU requested (preprocessing does not require strong GPU)
    memory=344064,  # 336 GB RAM for large preprocessing jobs (Modal max: 128-344064 MiB)
    timeout=86400,  # 24 hour timeout for long preprocessing jobs (max allowed by Modal)
    volumes={DATA_MOUNT_PATH: data_volume},
)
def run_preprocess(config: str, skip_embeddings: bool = False) -> str:
    """
    Run data preprocessing on a GPU instance.

    Args:
        config: Path to the preprocessing config file.
                Can be absolute or relative to /app/examples/configs/preprocessing_configs/
        skip_embeddings: If True, skip the embedding step.

    Returns:
        str: The stdout output from the preprocessing script.

    Usage:
        modal run modal/wrapper.py::run_preprocess --config config_air_quality_preprocessing.json
    """
    import os
    import subprocess
    import sys

    # Resolve config path
    if config.startswith("/"):
        config_path = config
    elif os.path.exists(
        f"/app/examples/configs/preprocessing_configs/{config}"
    ):
        config_path = f"/app/examples/configs/preprocessing_configs/{config}"
    elif os.path.exists(f"/app/examples/{config}"):
        config_path = f"/app/examples/{config}"
    else:
        config_path = f"/app/{config}"

    script_path = "/app/examples/preprocess_data.py"

    cmd = [sys.executable, script_path, "--config", config_path]
    if skip_embeddings:
        cmd.append("--skip_timeseries_embeddings")

    print(f"Running preprocessing: {' '.join(cmd)}")
    print(f"Config path: {config_path}")
    print(f"Data volume mounted at: {DATA_MOUNT_PATH}")
    print("=" * 80)

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        cwd="/app",
        env={
            **dict(os.environ),
            "PYTHONPATH": "/app/src",
        },
    )

    # Stream output in real-time
    stdout_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            stdout_lines.append(line)

    # Wait for process to complete
    returncode = process.wait()
    stdout = "\n".join(stdout_lines)

    print("=" * 80)
    print(f"Preprocessing completed with return code: {returncode}")

    if returncode != 0:
        raise RuntimeError(
            f"Preprocessing failed with code {returncode}\n{stdout}"
        )

    # Verify that output files were created before committing
    # Extract dataset name from config filename
    import json

    with open(config_path, "r") as f:
        config_data = json.load(f)
    filename = config_data.get("filename", "")
    dataset_name = filename.split("/")[0] if "/" in filename else "unknown"
    output_dir = os.path.join(DATA_MOUNT_PATH, dataset_name)

    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        npy_files = [f for f in files if f.endswith(".npy")]
        print(f"Found {len(npy_files)} .npy files in {output_dir}")
        if npy_files:
            print("Sample files:", npy_files[:5])
        else:
            print(f"WARNING: No .npy files found in {output_dir}")
            print(f"All files in directory: {files[:10]}")
    else:
        print(f"WARNING: Output directory {output_dir} does not exist!")

    # Commit volume changes to persist preprocessed data
    data_volume.commit()
    print("Volume committed successfully")

    return stdout


@app.function(
    image=image,
    gpu="H100:8",  # 8 H100 GPUs
    timeout=86400,  # 24 hour timeout for long training jobs (max allowed by Modal)
    volumes={DATA_MOUNT_PATH: data_volume},
)
def run_synthesize(config: str, checkpoint: str | None = None) -> str:
    """
    Run synthesis model training on a GPU instance.

    Args:
        config: Path to the synthesis config file.
                Can be absolute or relative to /app/examples/configs/synthesis_configs/
        checkpoint: Optional path to a model checkpoint to resume training.

    Returns:
        str: The stdout output from the synthesis script.

    Usage:
        modal run modal/wrapper.py::run_synthesize --config config_air_quality_synthesis.yaml
    """
    import os
    import subprocess
    import sys

    # Resolve config path
    if config.startswith("/"):
        config_path = config
    elif os.path.exists(f"/app/examples/configs/synthesis_configs/{config}"):
        config_path = f"/app/examples/configs/synthesis_configs/{config}"
    elif os.path.exists(f"/app/examples/{config}"):
        config_path = f"/app/examples/{config}"
    else:
        config_path = f"/app/{config}"

    script_path = "/app/examples/synthesize.py"

    cmd = [sys.executable, script_path, "--config", config_path]
    if checkpoint:
        cmd.extend(["--model_checkpoint_path", checkpoint])

    print(f"Running synthesis training: {' '.join(cmd)}")
    print(f"Config path: {config_path}")
    print(f"Data volume mounted at: {DATA_MOUNT_PATH}")
    print("=" * 80)

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        cwd="/app",
        env={
            **dict(os.environ),
            "PYTHONPATH": "/app/src",
        },
    )

    # Stream output in real-time
    stdout_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            stdout_lines.append(line)

    # Wait for process to complete
    returncode = process.wait()
    stdout = "\n".join(stdout_lines)

    print("=" * 80)
    print(f"Synthesis training completed with return code: {returncode}")

    if returncode != 0:
        raise RuntimeError(
            f"Synthesis training failed with code {returncode}\n{stdout}"
        )

    # Commit volume changes to persist model outputs
    data_volume.commit()
    print("Volume committed successfully")

    return stdout


@app.function(
    image=image,
    gpu="H100:8",  # 8 H100 GPUs
    timeout=86400,  # 24 hour timeout for synthetic data generation (max allowed by Modal)
    volumes={DATA_MOUNT_PATH: data_volume},
    name="generate_synthetic_data",
)
def run_generate_synthetic_data(
    config: str,
    model_checkpoint_path: str,
    splits: str = "test,val,train",  # Comma-separated string like "test,val,train"
    metadata_path: str | None = None,
    preprocess_config_path: str | None = None,
    output_dir: str | None = None,
    plot_fourier: bool = False,
    downsample_factor: int | None = None,
    run_postprocessing: bool = True,
    output_filename_prefix: str | None = None,
) -> str:
    """
    Generate synthetic data using a trained model.

    Args:
        config: Path to the synthesis config file.
                Can be absolute or relative to /app/examples/configs/synthesis_configs/
        model_checkpoint_path: Path to the model checkpoint file.
        splits: Comma-separated list of splits to generate (e.g., "test,val,train").
        metadata_path: Optional path to metadata JSON/parquet file.
        preprocess_config_path: Optional path to preprocessing config.
        output_dir: Optional output directory override.
        plot_fourier: If True, plot Fourier transforms.
        downsample_factor: Optional downsampling factor for postprocessing.
        run_postprocessing: If True, run postprocessing after generation.
        output_filename_prefix: Optional prefix for output filenames.

    Returns:
        str: The stdout output from the generation script.

    Usage:
        modal run modal/synthefy_pkg_wrapper.py::run_generate_synthetic_data \\
            --config config_oura_synthesis.yaml \\
            --model-checkpoint-path /data/models/oura_synthesis.ckpt \\
            --splits "test,val,train"
    """
    import os
    import subprocess
    import sys

    # Resolve config path
    if config.startswith("/"):
        config_path = config
    elif os.path.exists(f"/app/examples/configs/synthesis_configs/{config}"):
        config_path = f"/app/examples/configs/synthesis_configs/{config}"
    elif os.path.exists(f"/app/examples/{config}"):
        config_path = f"/app/examples/{config}"
    else:
        config_path = f"/app/{config}"

    # Resolve model checkpoint path (if relative, assume it's in the volume)
    if not model_checkpoint_path.startswith("/"):
        model_checkpoint_path = os.path.join(
            DATA_MOUNT_PATH, model_checkpoint_path
        )
    elif not model_checkpoint_path.startswith(DATA_MOUNT_PATH):
        # If absolute but not in volume, check if it exists in volume
        basename = os.path.basename(model_checkpoint_path)
        volume_model_path = os.path.join(DATA_MOUNT_PATH, "models", basename)
        if os.path.exists(volume_model_path):
            model_checkpoint_path = volume_model_path

    script_path = "/app/examples/generate_synthetic_data.py"

    cmd = [
        sys.executable,
        script_path,
        "--config",
        config_path,
        "--model_checkpoint_path",
        model_checkpoint_path,
    ]

    # Parse splits and add to command
    if splits:
        splits_list = [s.strip() for s in splits.split(",")]
        cmd.extend(["--splits"] + splits_list)

    if metadata_path:
        cmd.extend(["--metadata_path", metadata_path])
    if preprocess_config_path:
        cmd.extend(["--preprocess_config_path", preprocess_config_path])
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    if plot_fourier:
        cmd.append("--plot_fourier")
    if downsample_factor is not None:
        cmd.extend(["--downsample_factor", str(downsample_factor)])
    if not run_postprocessing:
        cmd.append(
            "--no_run_postprocessing"
        )  # Script uses --no_run_postprocessing flag
    if output_filename_prefix:
        cmd.extend(["--output_filename_prefix", output_filename_prefix])

    print(f"Running synthetic data generation: {' '.join(cmd)}")
    print(f"Config path: {config_path}")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Data volume mounted at: {DATA_MOUNT_PATH}")
    print("=" * 80)

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        cwd="/app",
        env={
            **dict(os.environ),
            "PYTHONPATH": "/app/src",
        },
    )

    # Stream output in real-time
    stdout_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            stdout_lines.append(line)

    # Wait for process to complete
    returncode = process.wait()
    stdout = "\n".join(stdout_lines)

    print("=" * 80)
    print(f"Synthetic data generation completed with return code: {returncode}")

    if returncode != 0:
        raise RuntimeError(
            f"Synthetic data generation failed with code {returncode}\n{stdout}"
        )

    # Commit volume changes to persist generated data
    data_volume.commit()
    print("Volume committed successfully")

    return stdout


@app.function(
    image=image,
    timeout=86400,  # 24 hour timeout for postprocessing reports (max allowed by Modal)
    volumes={DATA_MOUNT_PATH: data_volume},
    name="postprocessing_run",
)
def run_postprocessing_report(
    config: str,
    run_name: str | None = None,
    model_name: str | None = None,
    splits: str = None,  # Comma-separated string like "train,val,test" or None
    exclude_appendix: bool = False,
) -> str:
    """
    Run postprocessing report generation on a Modal instance.

    Args:
        config: Path to the synthesis config file.
                Can be absolute or relative to /app/examples/configs/synthesis_configs/
        run_name: Optional override for the run_name in the config file.
        model_name: Optional model name to use in the report.
        splits: Optional list of split types to include (e.g., ["train", "val", "test"]).
                Defaults to ["test"] if not provided.
        exclude_appendix: If True, exclude appendix sections from the report.

    Returns:
        str: The stdout output from the postprocessing report script.

    Usage:
        modal run modal/synthefy_pkg_wrapper.py::run_postprocessing_report \\
            --config config_oura_synthesis.yaml
    """
    import os
    import subprocess
    import sys

    # Resolve config path
    if config.startswith("/"):
        config_path = config
    elif os.path.exists(f"/app/examples/configs/synthesis_configs/{config}"):
        config_path = f"/app/examples/configs/synthesis_configs/{config}"
    elif os.path.exists(f"/app/examples/{config}"):
        config_path = f"/app/examples/{config}"
    else:
        config_path = f"/app/{config}"

    script_path = "/app/examples/postprocessing_report.py"

    cmd = [sys.executable, script_path, "--config", config_path]
    if run_name:
        cmd.extend(["--run_name", run_name])
    if model_name:
        cmd.extend(["--model_name", model_name])
    if splits is not None:
        # Parse comma-separated string into list
        splits_list = [s.strip() for s in splits.split(",")]
        cmd.extend(["--splits"] + splits_list)
    if exclude_appendix:
        cmd.append("--exclude_appendix")

    print(f"Running postprocessing report: {' '.join(cmd)}")
    print(f"Config path: {config_path}")
    print(f"Data volume mounted at: {DATA_MOUNT_PATH}")
    print("=" * 80)

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        cwd="/app",
        env={
            **dict(os.environ),
            "PYTHONPATH": "/app/src",
        },
    )

    # Stream output in real-time
    stdout_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            stdout_lines.append(line)

    # Wait for process to complete
    returncode = process.wait()
    stdout = "\n".join(stdout_lines)

    print("=" * 80)
    print(f"Postprocessing report completed with return code: {returncode}")

    if returncode != 0:
        raise RuntimeError(
            f"Postprocessing report failed with code {returncode}\n{stdout}"
        )

    # Commit volume changes to persist report outputs
    data_volume.commit()
    print("Volume committed successfully")

    return stdout


@app.function(image=image, volumes={DATA_MOUNT_PATH: data_volume})
def list_volume_files(path: str = "") -> list[str]:
    """
    List files in the data volume.

    Args:
        path: Path relative to the volume root. Empty string lists root.

    Returns:
        list[str]: List of files and directories.

    Usage:
        modal run modal/wrapper.py::list_volume_files --path air_quality
    """
    import os

    full_path = os.path.join(DATA_MOUNT_PATH, path)

    if not os.path.exists(full_path):
        print(f"Path does not exist: {full_path}")
        return []

    if os.path.isfile(full_path):
        print(f"File: {full_path}")
        return [full_path]

    files = []
    for root, dirs, filenames in os.walk(full_path):
        rel_root = os.path.relpath(root, DATA_MOUNT_PATH)
        for d in dirs:
            dir_path = os.path.join(rel_root, d) if rel_root != "." else d
            files.append(f"{dir_path}/")
        for f in filenames:
            file_path = os.path.join(rel_root, f) if rel_root != "." else f
            files.append(file_path)

    print(f"Files in volume at '{path or '/'}': {len(files)} items")
    for f in sorted(files)[:50]:  # Show first 50
        print(f"  {f}")
    if len(files) > 50:
        print(f"  ... and {len(files) - 50} more")

    return sorted(files)


@app.function(
    image=image,
    volumes={DATA_MOUNT_PATH: data_volume},
    name="copy_volume_directory",
)
def copy_volume_directory(source_path: str, dest_path: str) -> str:
    """
    Copy a directory within the Modal volume.

    Args:
        source_path: Source directory path (relative to /data)
        dest_path: Destination directory path (relative to /data)

    Returns:
        str: Status message.

    Usage:
        modal run modal/synthefy_pkg_wrapper.py::copy_volume_directory \\
            --source-path training_logs/oura/Time_Series_Diffusion_Training/synthesis_ppg \\
            --dest-path training_logs/oura/Time_Series_Diffusion_Training/synthesis_ppg_old
    """
    import os
    import shutil

    full_source = os.path.join(DATA_MOUNT_PATH, source_path)
    full_dest = os.path.join(DATA_MOUNT_PATH, dest_path)

    if not os.path.exists(full_source):
        raise FileNotFoundError(f"Source path does not exist: {full_source}")

    # Remove destination if it already exists
    if os.path.exists(full_dest):
        print(f"Destination path already exists, removing: {full_dest}")
        shutil.rmtree(full_dest)

    # Create parent directory if needed
    os.makedirs(os.path.dirname(full_dest), exist_ok=True)

    print(f"Copying {full_source} to {full_dest}...")
    shutil.copytree(full_source, full_dest)
    print("✅ Successfully copied directory")

    # Commit volume changes
    data_volume.commit()
    print("Volume committed successfully")

    return f"Copied {source_path} to {dest_path}"


@app.function(
    image=image, volumes={DATA_MOUNT_PATH: data_volume}, name="copy_volume_file"
)
def copy_volume_file(source_path: str, dest_path: str) -> str:
    """
    Copy a single file within the Modal volume.

    Args:
        source_path: Source file path (relative to /data)
        dest_path: Destination file path (relative to /data)

    Returns:
        str: Status message.

    Usage:
        modal run modal/synthefy_pkg_wrapper.py::copy_volume_file \\
            --source-path oura_subset/daily_ts_100k_with_custom_split.parquet \\
            --dest-path oura_superset/daily_ts_100k_with_custom_split.parquet
    """
    import os
    import shutil

    full_source = os.path.join(DATA_MOUNT_PATH, source_path)
    full_dest = os.path.join(DATA_MOUNT_PATH, dest_path)

    if not os.path.exists(full_source):
        raise FileNotFoundError(f"Source file does not exist: {full_source}")

    if not os.path.isfile(full_source):
        raise ValueError(f"Source path is not a file: {full_source}")

    # Create parent directory if needed
    dest_dir = os.path.dirname(full_dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Remove destination if it already exists
    if os.path.exists(full_dest):
        print(f"Destination file already exists, removing: {full_dest}")
        os.remove(full_dest)

    print(f"Copying {full_source} to {full_dest}...")
    shutil.copy2(full_source, full_dest)

    # Verify copy
    if os.path.exists(full_dest):
        source_size = os.path.getsize(full_source)
        dest_size = os.path.getsize(full_dest)
        print(f"✅ Successfully copied file ({source_size:,} bytes)")
        if source_size != dest_size:
            raise RuntimeError(
                f"File size mismatch: source={source_size}, dest={dest_size}"
            )
    else:
        raise RuntimeError(
            f"Copy appeared to succeed but destination file is missing: {full_dest}"
        )

    # Commit volume changes
    data_volume.commit()
    print("Volume committed successfully")

    return f"Copied {source_path} to {dest_path}"


@app.function(image=image, volumes={DATA_MOUNT_PATH: data_volume})
def upload_data(local_path: str, remote_path: str) -> str:
    """
    Upload data to the Modal volume.

    Note: This function is called remotely but the actual file transfer
    happens via Modal's volume API. Use the local entrypoint instead.

    Args:
        local_path: Path to the local file/directory to upload.
        remote_path: Destination path in the volume (relative to /data).

    Returns:
        str: Status message.
    """
    import os

    full_remote_path = os.path.join(DATA_MOUNT_PATH, remote_path)
    print(f"Upload destination: {full_remote_path}")
    print(
        "Note: Use 'modal volume put synthefy-data <local> <remote>' for uploads"
    )
    return f"Remote path would be: {full_remote_path}"


@app.local_entrypoint()
def upload_local(local_path: str, remote_path: str) -> None:
    """
    Upload a local file or directory to the Modal volume.

    Usage:
        modal run modal/wrapper.py::upload_local --local-path ./data.parquet --remote-path air_quality/data.parquet
    """
    import os
    import subprocess

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path not found: {local_path}")

    print(f"Uploading {local_path} to synthefy-data:{remote_path}")

    # Use modal volume put command
    cmd = ["modal", "volume", "put", "synthefy-data", local_path, remote_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Upload failed: {result.stderr}")

    print(f"Successfully uploaded to synthefy-data:{remote_path}")


@app.local_entrypoint()
def download_local(remote_path: str, local_path: str) -> None:
    """
    Download a file or directory from the Modal volume to local.

    Usage:
        modal run modal/wrapper.py::download_local --remote-path air_quality --local-path ./output
    """
    import os
    import subprocess

    print(f"Downloading synthefy-data:{remote_path} to {local_path}")

    # Ensure local directory exists
    local_dir = (
        os.path.dirname(local_path)
        if "." in os.path.basename(local_path)
        else local_path
    )
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    # Use modal volume get command
    cmd = ["modal", "volume", "get", "synthefy-data", remote_path, local_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")

    print(f"Successfully downloaded to {local_path}")


@app.function(
    image=image,
    volumes={DATA_MOUNT_PATH: data_volume},
    secrets=[modal.Secret.from_name("aws-secret-anvil")],  # type: ignore
    timeout=3600,  # 1 hour timeout (max 86400s for Modal)
)
def download_from_s3_to_volume(s3_url: str, volume_path: str) -> str:
    """
    Download a file from S3 and save it to the Modal volume.

    Args:
        s3_url: S3 URL in format 's3://bucket-name/key/path'
        volume_path: Destination path in the volume (relative to /data)

    Returns:
        str: Status message with the full path where the file was saved.

    Usage:
        modal run modal/wrapper.py::download_from_s3_to_volume \\
            --s3-url s3://oura-production-rops-external-partner-landing/anvil/daily_ts_10k.parquet \\
            --volume-path oura/daily_ts_full.parquet
    """
    import os
    from urllib.parse import urlparse

    import boto3
    from botocore.exceptions import ClientError

    # Parse S3 URL
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(
            f"Invalid S3 URL scheme: {s3_url}. Must start with 's3://'"
        )

    bucket = parsed.netloc
    s3_key = parsed.path.lstrip("/")

    if not bucket or not s3_key:
        raise ValueError(f"Invalid S3 URL: {s3_url}. Missing bucket or key.")

    # Construct full destination path in volume
    full_dest_path = os.path.join(DATA_MOUNT_PATH, volume_path)

    # Create destination directory if it doesn't exist
    dest_dir = os.path.dirname(full_dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading from S3: s3://{bucket}/{s3_key}")
    print(f"Destination: {full_dest_path}")

    # Create S3 client using credentials from Modal secret
    # Modal secrets expose environment variables, so boto3 will automatically
    # use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from the secret
    from botocore.config import Config

    # Use eu-central-1 region for this bucket
    region = "eu-central-1"
    print(f"Using AWS region: {region}")

    s3_config = Config(
        signature_version="s3v4",
        region_name=region,
    )
    s3_client = boto3.client("s3", config=s3_config)

    try:
        # Download file from S3
        s3_client.download_file(bucket, s3_key, full_dest_path)

        # Verify download
        if (
            os.path.exists(full_dest_path)
            and os.path.getsize(full_dest_path) > 0
        ):
            file_size = os.path.getsize(full_dest_path)
            print(
                f"✅ Successfully downloaded {file_size:,} bytes to {full_dest_path}"
            )
        else:
            raise RuntimeError(
                f"Download appeared to succeed but file is missing or empty: {full_dest_path}"
            )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))
        raise RuntimeError(
            f"Failed to download from S3: {error_code} - {error_msg}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error downloading from S3: {str(e)}"
        ) from e

    # Commit volume changes to persist the file
    data_volume.commit()
    print("Volume committed successfully")

    return f"File saved to {full_dest_path} ({os.path.getsize(full_dest_path):,} bytes)"


@app.local_entrypoint()
def main() -> None:
    """
    Default entrypoint: run GPU check to verify setup.

    Usage:
        uv run modal run modal/wrapper.py
    """
    print("Testing Modal + synthefy_pkg setup...")
    print("Dispatching GPU check to Modal cloud...")
    print()

    _ = check_gpu.remote()

    print()
    print("=" * 50)
    print("Setup verification complete!")
    print("=" * 50)
    print()
    print("Available commands:")
    print()
    print("  # Upload data to volume:")
    print("  modal run modal/wrapper.py::upload_local \\")
    print("    --local-path ./data.parquet \\")
    print("    --remote-path air_quality/data.parquet")
    print()
    print("  # Download from S3 to volume:")
    print("  modal run modal/wrapper.py::download_from_s3_to_volume \\")
    print("    --s3-url s3://bucket/key.parquet \\")
    print("    --volume-path oura/daily_ts_full.parquet")
    print()
    print("  # Run preprocessing (with task-specific app name):")
    print(
        "  modal run --name synthefy-oura-preprocess modal/wrapper.py::run_preprocess \\"
    )
    print("    --config config_oura_preprocessing.json")
    print()
    print("  # Run synthesis training (with task-specific app name):")
    print(
        "  modal run --name synthefy-ppg-synthesis modal/wrapper.py::run_synthesize \\"
    )
    print("    --config config_ppg_synthesis.yaml")
    print()
    print("  # List volume files:")
    print("  modal run modal/wrapper.py::list_volume_files --path air_quality")
    print()
    print("  # Download results:")
    print("  modal run modal/wrapper.py::download_local \\")
    print("    --remote-path air_quality --local-path ./output")
