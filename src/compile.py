"""Compiles necessary files for obfuscation."""

import importlib.util
import os
import sys
import traceback
from pprint import pformat

from Cython.Build import cythonize
from loguru import logger
from setuptools import Extension, setup


def get_extension_mapping(package_dir="synthefy_pkg"):
    extension_mapping = dict()
    erroneous_files = dict()

    # Walk through the directory
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]

                logger.debug(
                    f"Attempting to import {module_name} from {file_path}"
                )

                try:
                    # Load the module without executing non-declaration code
                    logger.debug(f"Loading {module_name} from {file_path}")
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path
                    )

                    # Check if spec and loader are valid
                    if spec is None or spec.loader is None:
                        raise ImportError(
                            f"Could not create spec for {file_path}"
                        )

                    module = importlib.util.module_from_spec(spec)

                    # Register the module in sys.modules before executing
                    sys.modules[module_name] = module

                    try:
                        spec.loader.exec_module(module)
                    finally:
                        # Clean up sys.modules if we added it
                        if module_name in sys.modules:
                            del sys.modules[module_name]

                    # Check if COMPILE is present and True
                    if (
                        hasattr(module, "COMPILE")
                        and getattr(module, "COMPILE") is True
                    ):
                        relative_path = os.path.relpath(file_path, package_dir)
                        module_path = relative_path.replace("/", ".").replace(
                            ".py", ""
                        )
                        # Handle different package directory structures
                        if package_dir.startswith("synthefy-"):
                            # For server packages, use the package name from the directory
                            package_name = package_dir.split("/")[-1]
                            extension_mapping[
                                package_name + "." + module_path
                            ] = package_dir + "/" + relative_path
                        else:
                            # For synthefy_pkg, use the original logic
                            extension_mapping[
                                package_dir + "." + module_path
                            ] = package_dir + "/" + relative_path
                    else:
                        logger.debug(
                            f"Skipping {file_path} as COMPILE is not True"
                        )

                except Exception as e:
                    # Handle any exception that occurs during import
                    print(f"Error importing {file_path}. Deleting the file.")
                    # Delete the file
                    os.remove(file_path)
                    erroneous_files[file_path] = (
                        str(e) + "\nFull traceback:\n" + traceback.format_exc()
                    )

    if len(erroneous_files):
        print("\n" + "=" * 80)
        print("❌ COMPILATION ERRORS SUMMARY")
        print("=" * 80)
        print(f"Total files with errors: {len(erroneous_files)}")
        print(f"Files successfully processed: {len(extension_mapping)}")
        print()

        print("📋 ERROR DETAILS:")
        print("-" * 40)

        for i, (file_path, error_details) in enumerate(
            erroneous_files.items(), 1
        ):
            print(f"\n{i:2d}. File: {file_path}")
            print("    Error: ", end="")

            # Extract the main error message (first line)
            error_lines = error_details.split("\n")
            main_error = error_lines[0] if error_lines else "Unknown error"
            print(main_error)

            # Show a simplified traceback if it's an import error
            if "ModuleNotFoundError" in main_error:
                print("    Type: Import dependency issue")
                print("    Solution: Check if required modules are available")
            elif "AssertionError" in main_error:
                print("    Type: Assertion failed during import")
                print(
                    "    Solution: Check environment variables or configuration"
                )
            elif "SyntaxError" in main_error:
                print("    Type: Python syntax error")
                print("    Solution: Fix syntax issues in the file")
            else:
                print("    Type: General import/execution error")
                print("    Solution: Review the file for issues")

        print("\n" + "=" * 80)
        print("💡 TROUBLESHOOTING TIPS:")
        print("   • Ensure all required dependencies are installed")
        print("   • Check that environment variables are properly set")
        print("   • Verify that files have correct Python syntax")
        print(
            "   • Some files may be skipped if they don't have COMPILE = True"
        )
        print("=" * 80)

        # Continue with successfully loaded files instead of raising an exception
        print(
            f"\n⚠️  Continuing compilation with {len(extension_mapping)} successfully loaded files..."
        )
        print(
            f"   (Skipping {len(erroneous_files)} files that failed to load)\n"
        )
    else:
        logger.debug(
            f"No errors found in {package_dir}; will compile {len(extension_mapping)} files"
        )

    return extension_mapping


def convert_to_pyx(extensions):
    """Converts all Python files to the .pyx format for Cython."""
    for module, file_path in extensions.items():
        if file_path.endswith(".py"):
            new_file_path = file_path[:-2] + "pyx"
            os.rename(file_path, new_file_path)
            # os.copyfile(file_path, new_file_path)
            extensions[module] = new_file_path

        else:
            raise ValueError(f"File is not a Python file: {file_path}")

    return extensions


def compile(extensions):
    """Compiles all the .pyx files."""
    extensions_to_build = []
    for module, file_path in extensions.items():
        logger.info(f"Adding module {module}")
        # Define the extension
        ext = Extension(name=module, sources=[file_path])
        # cyt = cythonize(ext, compiler_directives={"language_level": "3"})
        extensions_to_build.append(ext)

    logger.info(f"Compiling all models in {extensions_to_build}")
    setup(
        name="cython_build",
        ext_modules=cythonize(
            extensions_to_build, compiler_directives={"language_level": "3"}
        ),
        script_args=[
            "build_ext",
            "--inplace",
        ],  # This builds the .so files in-place
    )


def cleanup(extension):
    """Delete all the intermediate .c files generated."""
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile Python files with COMPILE=True flag"
    )
    parser.add_argument(
        "package_dir",
        nargs="?",
        default="synthefy_pkg",
        help="Package directory to compile (default: synthefy_pkg)",
    )
    args = parser.parse_args()

    logger.info(f"Getting all files to compile in {args.package_dir}...")
    extensions = get_extension_mapping(args.package_dir)
    logger.debug(f"Extensions to compile: {extensions}")

    if not extensions:
        logger.warning(
            f"No files with COMPILE=True found in {args.package_dir}"
        )
        sys.exit(0)

    logger.info("Converting all python files to .pyx for Cython...")
    extensions = convert_to_pyx(extensions)

    logger.info("Compiling all .pyx files individually...")
    compile(extensions)
