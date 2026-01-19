import os

# Run this from the root using
# make run_dev_server

def get_application():
    config_path: str = os.environ.get("SYNTHEFY_CONFIG_PATH", "src/synthefy_pkg/app/services/configs/api_config_general_dev.yaml")
    from synthefy_pkg.app.main import create_app
    return create_app(config_path=config_path)

app = get_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dev_server:app", host="0.0.0.0", port=8004, reload=False)
