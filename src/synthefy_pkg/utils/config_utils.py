import os
import re
from typing import Any, Dict

import yaml
from loguru import logger


class EnvLoader(yaml.SafeLoader):
    """
    A custom YAML loader that automatically expands environment variables in
    scalar strings. Any scalar containing a '$' will be implicitly tagged with
    '!env' and then processed by the custom constructor.
    """

    pass


# Register an implicit resolver: any scalar with a '$' is tagged as '!env'
EnvLoader.add_implicit_resolver("!env", re.compile(r".*\$.*"), None)


def env_var_constructor(loader: yaml.Loader, node: yaml.ScalarNode) -> str:
    """
    YAML constructor for the !env tag. Expands environment variables in the
    provided scalar string using os.path.expandvars.

    Args:
        loader (yaml.Loader): The YAML loader instance.
        node (yaml.ScalarNode): The YAML scalar node containing the string.

    Returns:
        str: The scalar value with any environment variables expanded.
    """
    value = loader.construct_scalar(node)
    expanded = os.path.expandvars(value)  # type: ignore
    # If there are still '$' signs, one or more env variables may be missing.
    if "$" in expanded:
        match = re.search(r"\$\{?([^}^{]+)\}?", expanded)
        if match:
            missing_var = match.group(1)
            if missing_var != "dataset_name":
                logger.warning(
                    f"Environment variable '{missing_var}' not found in '{value}'"
                )
    return expanded


# Associate the custom constructor with the !env tag in our custom loader.
EnvLoader.add_constructor("!env", env_var_constructor)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file using the custom EnvLoader, which handles
    environment variable substitution.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration as a Python dictionary with environment
                        variables expanded.

    Raises:
        RuntimeError: If the file cannot be loaded.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=EnvLoader)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}")
