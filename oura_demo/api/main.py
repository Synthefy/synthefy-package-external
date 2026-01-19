"""
Oura Demo API - Main FastAPI application.

A lightweight demo API for time series synthesis with:
- Dynamic config loading by dataset_name
- File upload with column validation
- LLM-powered data modifications (GPT-4)
- Synthesis model inference
- Static file serving for the React UI (in Docker)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from routers.config import router as config_router
from routers.inference import router as inference_router
from routers.llm import router as llm_router
from routers.upload import router as upload_router

# Path to the built React UI (set via environment variable in Docker)
UI_DIST_PATH = Path(
    os.getenv(
        "UI_DIST_PATH",
        str(Path(__file__).parent.parent / "ui" / "dist"),
    )
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Oura Demo API...")
    logger.info(
        f"SYNTHEFY_PACKAGE_BASE: {os.getenv('SYNTHEFY_PACKAGE_BASE', 'not set')}"
    )
    logger.info(
        f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}"
    )
    logger.info(f"UI_DIST_PATH: {UI_DIST_PATH}")
    logger.info(f"UI available: {UI_DIST_PATH.exists()}")

    yield

    # Shutdown
    logger.info("Shutting down Oura Demo API...")


app = FastAPI(
    title="Oura Demo API",
    description="""
    A lightweight demo API for time series synthesis.

    ## Features

    - **Config Loading**: Dynamic config loading by dataset name (oura, oura_subset, ppg)
    - **File Upload**: Upload parquet/CSV files with column validation
    - **LLM Modifications**: Modify data using natural language (GPT-4)
    - **Synthesis**: Generate synthetic time series using trained models

    ## Workflow

    1. Select a dataset (`GET /api/config/{dataset_name}`)
    2. Upload your data (`POST /api/upload/{dataset_name}`)
    3. (Optional) Modify data via LLM (`POST /api/llm/modify`)
    4. Run synthesis (`POST /api/synthesize`)
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)
app.include_router(upload_router)
app.include_router(llm_router)
app.include_router(inference_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Static file serving for the React UI
# This must be added AFTER the API routers to ensure API routes take precedence
if UI_DIST_PATH.exists():
    logger.info(f"Mounting static files from {UI_DIST_PATH}")

    # Mount the assets directory (JS, CSS, images)
    assets_path = UI_DIST_PATH / "assets"
    if assets_path.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_path)),
            name="static_assets",
        )

    @app.get("/")
    async def serve_index():
        """Serve the React app's index.html at root."""
        index_path = UI_DIST_PATH / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"error": "UI not found", "path": str(index_path)}

    # Catch-all route for SPA client-side routing
    # This handles routes like /dataset/oura that React Router manages
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes (SPA routing)."""
        # Don't interfere with API routes, docs, or static assets
        if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "assets/")):
            # These are handled by FastAPI's own routes or static file mounts
            return {"error": "Route not found"}

        # Try to serve the file directly if it exists (e.g., favicon.ico)
        file_path = UI_DIST_PATH / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # Otherwise, serve index.html for SPA routing
        index_path = UI_DIST_PATH / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))

        return {"error": "UI not found"}

else:
    logger.warning(f"UI dist not found at {UI_DIST_PATH}, API-only mode")

    @app.get("/")
    async def root():
        """Root endpoint with API info (no UI available)."""
        return {
            "name": "Oura Demo API",
            "version": "0.1.0",
            "docs": "/docs",
            "ui_available": False,
            "endpoints": {
                "config": "/api/config/{dataset_name}",
                "upload": "/api/upload/{dataset_name}",
                "llm_modify": "/api/llm/modify",
                "synthesize": "/api/synthesize",
            },
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
