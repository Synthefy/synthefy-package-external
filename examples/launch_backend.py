import argparse
import os

import uvicorn
from fastapi import FastAPI

from synthefy_pkg.app.routers import (
    forecast,
    foundation_models,
    pretrained_anomaly,
    pretrained_anomaly_v2,
    synthesis,
)


def create_app(config_path: str) -> FastAPI:
    os.environ["SYNTHEFY_CONFIG_PATH"] = config_path

    app = FastAPI()

    app.include_router(synthesis.router)
    app.include_router(forecast.router)
    app.include_router(pretrained_anomaly.router)
    app.include_router(pretrained_anomaly_v2.router)
    app.include_router(foundation_models.router)

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Synthefy Platform API"}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    app = create_app(args.config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
