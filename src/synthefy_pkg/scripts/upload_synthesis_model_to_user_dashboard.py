import argparse
import os
import subprocess

from loguru import logger

from synthefy_pkg.utils.config_utils import load_yaml_config

SYNTHEFY_DATASETS_BASE = os.getenv("SYNTHEFY_DATASETS_BASE")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Upload synthesis model artifacts to user's S3 bucket."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Local experiment name",
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Local run name"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Local Dataset Name"
    )
    parser.add_argument(
        "--preprocessing_config_file",
        type=str,
        required=True,
        help="Config used to preprocess the dataset",
    )
    parser.add_argument(
        "--synthesis_config_file",
        type=str,
        required=True,
        help="Config used to train the synthesis model",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dev_or_prod",
        type=str,
        required=True,
        help="Whether to upload to dev or prod",
    )
    parser.add_argument(
        "--s3_dataset_name",
        type=str,
        required=True,
        help="The user's dataset name under which to upload this model",
    )
    parser.add_argument(
        "--s3_model_name",
        type=str,
        required=True,
        help="The model name to show to the user",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
        help="The user_id associated with the user's account",
    )
    parser.add_argument(
        "--override_on_s3",
        default=True,
        action="store_false",
        help="Whether to override the model on S3 if it already exists",
    )
    return parser.parse_args()


def s3_cp_upload(local_path, s3_uri, override_on_s3=True, recursive=False):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Source path does not exist: {local_path}")

    # Check if resource already exists on S3
    check_command = ["aws", "s3", "ls", s3_uri]
    check_process = subprocess.run(
        check_command, capture_output=True, text=True
    )

    resource_exists = check_process.returncode == 0
    if resource_exists and not recursive:
        if not override_on_s3:
            raise FileExistsError(
                f"Resource already exists at {s3_uri} and override_on_s3 is False"
            )
        logger.debug(
            f"Resource exists at {s3_uri}, will override since override_on_s3 is True"
        )

    command = ["aws", "s3", "cp", local_path, s3_uri]
    if recursive:
        command.append("--recursive")

    logger.debug(f"Running: {' '.join(command)}")

    # Use Popen to stream output live
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if process.stdout is None or process.stderr is None:
        raise subprocess.CalledProcessError(1, command)

    # Stream the output line by line
    for line in process.stdout:
        print(line, end="")  # Print each line as it is received

    # Wait for the process to complete and get the return code
    process.wait()

    if process.returncode != 0:
        logger.error("Error uploading to S3:")
        for line in process.stderr:
            logger.error(line.strip())
        raise subprocess.CalledProcessError(process.returncode, command)

    logger.debug("Upload completed successfully.")


def join_uri(*parts):
    return "/".join(part.strip("/") for part in parts[:-1]) + "/" + parts[-1]


def validate_arguments(args):
    # Validate synthesis config file exists
    if not os.path.exists(args.synthesis_config_file):
        logger.error(
            f"Synthesis config file {args.synthesis_config_file} does not exist."
        )
        raise FileNotFoundError(
            f"Synthesis config file {args.synthesis_config_file} does not exist."
        )

    # Load synthesis config
    config = load_yaml_config(args.synthesis_config_file)

    # Validate experiment name
    if args.experiment_name != config.get("execution_config", {}).get(
        "experiment_name"
    ):
        logger.error("Experiment name does not match the config.")
        raise ValueError("Experiment name does not match the config.")

    # Validate run name
    if args.run_name != config.get("execution_config", {}).get("run_name"):
        logger.error("Run name does not match the config.")
        raise ValueError("Run name does not match the config.")

    # Validate dataset name
    if not SYNTHEFY_DATASETS_BASE:
        logger.error("Environment variable SYNTHEFY_DATASETS_BASE is not set.")
        raise EnvironmentError(
            "Environment variable SYNTHEFY_DATASETS_BASE is not set."
        )

    dataset_path = os.path.join(SYNTHEFY_DATASETS_BASE, args.dataset_name)
    if not os.path.isdir(dataset_path):
        logger.error(f"Dataset directory {dataset_path} does not exist.")
        raise NotADirectoryError(
            f"Dataset directory {dataset_path} does not exist."
        )

    # Ensure that a model checkpoint exists in the correct location
    model_checkpoint_path = os.path.join(
        dataset_path, args.model_checkpoint_path
    )
    if not os.path.isfile(model_checkpoint_path):
        logger.error(
            f"Model checkpoint {model_checkpoint_path} does not exist."
        )
        raise FileNotFoundError(
            f"Model checkpoint {model_checkpoint_path} does not exist."
        )

    # Validate dev or prod
    if args.dev_or_prod not in ["dev", "prod"]:
        logger.error("Dev or prod must be either dev or prod.")
        raise ValueError("Dev or prod must be either dev or prod.")

    logger.info("All local validation checks passed :)")


def upload_dataset_to_s3(args, base_s3_uri):
    dataset_path = os.path.join(str(SYNTHEFY_DATASETS_BASE), args.dataset_name)
    s3_uri = join_uri(base_s3_uri, args.s3_dataset_name)
    s3_cp_upload(
        dataset_path, s3_uri, override_on_s3=args.override_on_s3, recursive=True
    )
    logger.info(f"Uploaded dataset to {s3_uri}")


def upload_configs_to_s3(args, base_s3_uri):
    config_path = args.synthesis_config_file
    s3_uri = join_uri(
        base_s3_uri,
        args.s3_dataset_name,
        f"config_{args.s3_dataset_name}_synthesis.yaml",
    )
    s3_cp_upload(config_path, s3_uri, override_on_s3=args.override_on_s3)
    logger.info(f"Uploaded synthesis config to {s3_uri}")

    config_path = args.preprocessing_config_file
    s3_uri = join_uri(
        base_s3_uri,
        args.s3_dataset_name,
        f"config_{args.s3_dataset_name}_preprocessing.json",
    )
    s3_cp_upload(config_path, s3_uri, override_on_s3=args.override_on_s3)
    logger.info(f"Uploaded preprocessing config to {s3_uri}")


def upload_model_checkpoint_to_s3(args, base_s3_uri):
    model_checkpoint_path = args.model_checkpoint_path
    s3_uri = join_uri(
        base_s3_uri,
        "training_logs",
        args.s3_dataset_name,
        args.s3_model_name,
        "output",
        "model",
        "model.ckpt",
    )
    s3_cp_upload(
        model_checkpoint_path, s3_uri, override_on_s3=args.override_on_s3
    )
    logger.info(f"Uploaded model checkpoint to {s3_uri}")


def main():
    args = parse_arguments()
    validate_arguments(args)
    base_s3_uri = f"s3://synthefy-{args.dev_or_prod}-logs/{args.user_id}"

    upload_dataset_to_s3(args, base_s3_uri)
    upload_configs_to_s3(args, base_s3_uri)
    upload_model_checkpoint_to_s3(args, base_s3_uri)


if __name__ == "__main__":
    main()

"""
Example usage:
python3 upload_synthesis_model_to_user_dashboard.py \
    --experiment_name "PPG_Training" \
    --run_name "ppg_synthesis_run_1" \
    --dataset_name "ppg" \
    --preprocessing_config_file path_to_preprocessing.json \
    --synthesis_config_file path_to_synthesis_config.yaml \
    --model_checkpoint_path path_to_model_checkpoint.ckpt \
    --dev_or_prod "dev" \
    --s3_dataset_name "ppg_dataset" \
    --s3_model_name "ppg_synthesis_model" \
    --user_id "123-456-789 (not the correct format)" \
"""
