import argparse
import importlib
import os
import sys
import uuid

import aioboto3
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add sys.path with src folder.
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

# Add the following imports for API usage tracking
from synthefy_pkg.app.middleware.api_usage_middleware import APIUsageMiddleware
from synthefy_pkg.app.middleware.logging_middleware import LoggingMiddleware

# Add metrics manager
from synthefy_pkg.app.middleware.metrics_manager import (
    MetricsMiddleware,
)

# Router imports are now handled dynamically via importlib based on SYNTHEFY_ROUTER env var

SYNTHEFY_PACKAGE_BASE = str(os.getenv("SYNTHEFY_PACKAGE_BASE"))
DEFAULT_CORRELATION_ID = "N/A"
assert load_dotenv(os.path.join(SYNTHEFY_PACKAGE_BASE, "examples/configs/.env"))

COMPILE = False


# Define a custom formatter function
def custom_formatter(record):
    correlation_id = record["extra"].get("correlation_id", "N/A")
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"correlation_id={correlation_id} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )


logger.remove()  # Remove the default handler
logger.add(sys.stdout, format=custom_formatter, level="DEBUG", catch=True)

# Bind the logger with a default 'correlation_id' so local runs + tests work
logger = logger.bind(correlation_id=DEFAULT_CORRELATION_ID)


def create_app(config_path: str) -> FastAPI:
    os.environ["SYNTHEFY_CONFIG_PATH"] = config_path

    app = FastAPI(
        title="Synthefy Platform API",
        description="""
        ## Synthefy Platform API
        
        The Synthefy Platform provides comprehensive APIs for time series analysis, forecasting, 
        synthesis, and anomaly detection powered by advanced machine learning models.
        
        ### Key Features:
        - **Time Series Synthesis**: Generate synthetic time series data
        - **Forecasting**: Predict future values with confidence intervals
        - **Anomaly Detection**: Identify anomalous patterns in your data
        - **Search & Retrieval**: Advanced data discovery capabilities
        - **Training**: Custom model training and fine-tuning
        
        ### Authentication
        Most endpoints require API key authentication. Contact your administrator to obtain an API key.
        
        ### Rate Limiting
        API calls are rate-limited to ensure fair usage. See individual endpoint documentation for specific limits.
        
        ### Support
        For technical support, please contact our team or visit our documentation portal.
        """,
        version="1.0.0",
        terms_of_service="https://synthefy.com/terms",
        contact={
            "name": "Synthefy Support",
            "url": "https://synthefy.com/support",
            "email": "support@synthefy.com",
        },
        license_info={
            "name": "Commercial License",
            "url": "https://synthefy.com/license",
        },
        servers=[
            {
                "url": "https://api.synthefy.com",
                "description": "Production server",
            },
            {
                "url": "https://staging-api.synthefy.com",
                "description": "Staging server",
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server",
            },
        ],
        docs_url="/docs",
        redoc_url="/documents",
        openapi_url="/openapi.json",
    )

    # Add Logging Middleware as the FIRST middleware
    app.add_middleware(LoggingMiddleware)

    # Create aioboto3 session at startup
    app.state.aioboto3_session = aioboto3.Session()

    # Conditionally import and set Celery app only when USE_CELERY is true
    if os.getenv("USE_CELERY", "false").lower() == "true":
        from synthefy_pkg.app.celery_app import celery_app

        logger.info("Using Celery app")
        app.state.celery_app = celery_app
    else:
        app.state.celery_app = None
        logger.info("Not using Celery app")

    # Always enable API Usage Tracking Middleware - after LoggingMiddleware
    app.add_middleware(APIUsageMiddleware)

    # Add CORSMiddleware (after API Usage middleware so it's processed first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(MetricsMiddleware, enable_metrics=True)

    # Enhanced Redoc documentation with template from static folder
    @app.get("/documents", include_in_schema=False)
    async def custom_redoc_html():
        import os

        from fastapi.responses import HTMLResponse

        # Read the template file
        template_path = os.path.join(
            os.path.dirname(__file__), "static", "redoc-template.html"
        )

        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Replace template variables
        html_content = template_content.replace(
            "{{title}}",
            str(app.title),
        )
        html_content = html_content.replace(
            "{{openapi_url}}",
            str(app.openapi_url),
        )

        return HTMLResponse(html_content)

    @app.middleware("http")
    async def add_correlation_id_middleware(request: Request, call_next):
        # Extract or generate the X-Request-ID
        correlation_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Add the Correlation-ID to Loguru's context
        with logger.contextualize(correlation_id=correlation_id):
            # Continue processing the request
            response = await call_next(request)
            response.headers["X-Request-ID"] = correlation_id
            return response

    # Get the router name(s) from environment variable
    router_names = os.environ.get("SYNTHEFY_ROUTER", None)

    # List of all available routers
    all_routers = [
        "access_model_api",
        "cpu_load_test",
        "data_retrieval",
        "forecast",
        "foundation_models",
        "metadata_for_foundation_models",
        "post_preprocess",
        "postprocess",
        "preprocess",
        "pretrained_anomaly",
        "pretrained_anomaly_v2",
        "health",
        "search",
        "setup_ui",
        "summarize",
        "synthesis",
        "synthetic_data_agent",
        "task_management",
        "train",
        "user_api_keys",
        "view",
        "zero_shot_anomaly",
        "explain",
    ]

    if router_names:
        # Support comma-separated list of routers
        router_names = [r.strip() for r in router_names.split(",") if r.strip()]
        for router_name in router_names:
            try:
                logger.debug(f"Attempting to import router: {router_name}")
                router_module = importlib.import_module(
                    f"synthefy_pkg.app.routers.{router_name}"
                )
                app.include_router(router_module.router)
                logger.info(f"Included router: {router_name}")
            except Exception as e:
                logger.error(
                    f"Failed to load router '{router_name}': {type(e).__name__}: {str(e)}"
                )
                import traceback

                logger.error(
                    f"Traceback for router '{router_name}':\n{traceback.format_exc()}"
                )
    else:
        # Import and include all routers when SYNTHEFY_ROUTER is undefined
        logger.info("SYNTHEFY_ROUTER not defined. Including all routers.")
        for router_name in all_routers:
            try:
                logger.debug(f"Attempting to import router: {router_name}")
                # Import the router module
                router_module = importlib.import_module(
                    f"synthefy_pkg.app.routers.{router_name}"
                )

                # Include the router
                app.include_router(router_module.router)

                logger.info(f"Included router: {router_name}")
            except Exception as e:
                # Log detailed error for any router that fails to load
                logger.error(
                    f"Failed to load router '{router_name}': {type(e).__name__}: {str(e)}"
                )
                # In multiprocessing, continue with other routers instead of crashing
                import traceback

                logger.error(
                    f"Traceback for router '{router_name}':\n{traceback.format_exc()}"
                )

    # Static router inclusions have been removed in favor of dynamic imports

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Synthefy Platform API"}

    # Developer Portal
    @app.get("/developer", include_in_schema=False)
    async def developer_portal():
        import os

        from fastapi.responses import FileResponse

        portal_path = os.path.join(
            os.path.dirname(__file__), "static", "developer-portal.html"
        )
        return FileResponse(portal_path, media_type="text/html")

    return app


# Module-level app instance for uvicorn workers
# This will be created when the module is imported if SYNTHEFY_CONFIG_PATH is set
def _create_app_from_env():
    """Create app from environment variable for uvicorn workers."""
    config_path = os.environ.get("SYNTHEFY_CONFIG_PATH")
    if config_path:
        return create_app(config_path)
    return None


# Module-level app instance - will be None until SYNTHEFY_CONFIG_PATH is set
app = _create_app_from_env()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    args = parser.parse_args()

    logger.info(f"Starting FastAPI with config: {args.config}")

    # Set config path in environment for workers
    os.environ["SYNTHEFY_CONFIG_PATH"] = args.config

    if args.workers > 1:
        # Use Gunicorn with UvicornWorker for multiple workers with timeout control
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "gunicorn",
            "synthefy_pkg.app.main:app",
            "-w",
            str(args.workers),
            "-k",
            "uvicorn.workers.UvicornWorker",
            "--bind",
            "0.0.0.0:8000",
            "--timeout",
            "300",  # 60 second timeout for worker startup
        ]
        subprocess.run(cmd)
    else:
        # Use uvicorn directly for single worker
        # Create app here since SYNTHEFY_CONFIG_PATH is now set
        single_worker_app = create_app(args.config)
        uvicorn.run(single_worker_app, host="0.0.0.0", port=8000)
