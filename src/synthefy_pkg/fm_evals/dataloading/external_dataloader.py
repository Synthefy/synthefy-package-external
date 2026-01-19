import importlib
import sys
from pathlib import Path
from typing import Any

from synthefy_pkg.configs.execution_configurations import Configuration
from synthefy_pkg.fm_evals.dataloading.base_eval_dataloader import (
    BaseEvalDataloader,
)


class ExternalDataloader:
    """
    A dataloader that dynamically imports and uses external dataloader classes.

    Args:
        config: Configuration object containing the external dataloader specification
        external_dataloader_spec: String in format "path::class_name" where:
            - path: Path to the Python file containing the dataloader class
            - class_name: Name of the dataloader class to instantiate
    """

    def __init__(self, config: Configuration, external_dataloader_spec: str):
        # Store config for potential use by external dataloader
        self.config = config

        if "::" not in external_dataloader_spec:
            raise ValueError(
                "external_dataloader_spec must be in format 'path::class_name'"
            )

        self.path, self.class_name = external_dataloader_spec.split("::", 1)
        self.path = self.path.strip()
        self.class_name = self.class_name.strip()

        if not self.path or not self.class_name:
            raise ValueError(
                "Both path and class_name must be non-empty in 'path::class_name'"
            )

        # Load the external dataloader
        self.external_dataloader = self._load_external_dataloader()

    def _load_external_dataloader(self) -> Any:
        """
        Dynamically import and instantiate the external dataloader class.

        Returns:
            An instance of the external dataloader class

        Raises:
            ImportError: If the module or class cannot be imported
            ValueError: If the class doesn't have an __iter__ method
        """
        try:
            # Convert path to absolute path if it's relative
            path_obj = Path(self.path)
            if not path_obj.is_absolute():
                # Assume relative to current working directory
                path_obj = Path.cwd() / path_obj

            # Add the directory containing the file to Python path
            module_dir = str(path_obj.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            # Import the module
            module_name = path_obj.stem  # filename without extension
            module = importlib.import_module(module_name)

            # Get the class
            if not hasattr(module, self.class_name):
                raise ImportError(
                    f"Class '{self.class_name}' not found in module '{module_name}'"
                )

            dataloader_class = getattr(module, self.class_name)

            # Instantiate the class
            # Pass config if the class accepts it, otherwise call without arguments
            try:
                instance = dataloader_class(self.config)
            except TypeError:
                # If the class doesn't accept config, try without arguments
                instance = dataloader_class()

            # Verify it has an __iter__ method
            if not hasattr(instance, "__iter__"):
                raise ValueError(
                    f"Class '{self.class_name}' must have an '__iter__' method"
                )

            return instance

        except ImportError as e:
            raise ImportError(
                f"Failed to import external dataloader from '{self.path}': {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate external dataloader '{self.class_name}' from '{self.path}': {str(e)}"
            ) from e

    def __len__(self):
        """Return the length of the external dataloader."""
        # Use hasattr to check if the external dataloader has __len__
        if hasattr(self.external_dataloader, "__len__"):
            return len(self.external_dataloader)
        else:
            # If the external dataloader doesn't have __len__, return 0
            return 0

    def __iter__(self):
        """Iterate over the external dataloader."""
        # Use hasattr to check if the external dataloader is iterable
        if hasattr(self.external_dataloader, "__iter__"):
            for item in self.external_dataloader:
                yield item
        else:
            # If the external dataloader is not iterable, yield nothing
            return

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the external dataloader.
        This allows access to any additional methods or attributes the external dataloader might have.
        """
        return getattr(self.external_dataloader, name)
