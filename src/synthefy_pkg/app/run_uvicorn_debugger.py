import os

import uvicorn
from dotenv import load_dotenv

# This script acts as a reliable entry point for the VS Code debugger.
# It ensures that environment variables from .env.local are loaded *before*
# uvicorn is started.

if __name__ == "__main__":
    # Load the .env.local file from the project's root directory.
    # The VS Code debugger will launch this script from the workspaceFolder root.
    root_dir = os.getenv("ROOT_DIR")
    if not root_dir:
        print("ROOT_DIR is not set, using default path")
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
    dotenv_path = os.path.join(root_dir, ".env.local")
    load_dotenv(dotenv_path=dotenv_path)

    # Fetch the port from the now-loaded environment variables.
    # Default to 8000 if LOCAL_PORT is not found in the .env.local file.
    port = int(os.getenv("LOCAL_PORT", "8000"))

    # NOTE: The app path is hardcoded here.
    # This is generally fine for a specific project's debug configuration.
    app_path = "src.synthefy_pkg.app.dev_server:app"

    print(
        f"--- Starting Uvicorn for {app_path} on port {port} with reloading ---"
    )

    # Programmatically start uvicorn. This is equivalent to the CLI command.
    # Using reload=True works correctly with the VS Code debugger.
    uvicorn.run(app_path, host="127.0.0.1", port=port, reload=True)
