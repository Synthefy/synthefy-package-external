# Oura Demo

A lightweight demo application for time series synthesis with:
- React UI for data visualization and interaction
- FastAPI backend for file processing, LLM modifications, and synthesis inference

## Docker (Recommended)

The easiest way to run the Oura Demo is using Docker. The Docker image includes the compiled `synthefy_pkg` and a pre-built React UI.

### Pull the Pre-built Docker Image

If a pre-built image is available in a registry:

```bash
docker pull synthefy/external/oura-demo:latest
docker tag synthefy/external/oura-demo:latest oura-demo
```

### Build the Docker Image (Alternative)

If you prefer to build locally, from the `synthefy-package` root directory:

**For your host platform (recommended):**
```bash
docker build -f Dockerfile.oura -t oura-demo .
```

**For a specific platform (if you need to build for a different architecture):**
```bash
# Build for linux/amd64 (Intel/AMD 64-bit)
docker build --platform linux/amd64 -f Dockerfile.oura -t oura-demo .

# Build for linux/arm64 (Apple Silicon, ARM-based systems)
docker build --platform linux/arm64 -f Dockerfile.oura -t oura-demo .
```

**Note:** If you encounter a platform mismatch error (e.g., "exec format error"), rebuild the image for your host platform or use the `--platform` flag when running (see below).

### Run the UI Demo (Docker)

The UI demo includes both the React frontend and FastAPI backend in a single container:

**With a license key (production):**
```bash
docker run -p 8001:8001 \
  -e LICENSE_KEY="your_jwt_license_key" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo
```

**For development (bypasses license check):**
```bash
docker run -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo
```

**If you encounter a platform mismatch error**, specify the platform explicitly:
```bash
# For linux/amd64 hosts (Intel/AMD 64-bit)
docker run --platform linux/amd64 -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo

# For linux/arm64 hosts (Apple Silicon, ARM-based systems)
docker run --platform linux/arm64 -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo
```

**Run in background:**
```bash
docker run -d --name oura-demo -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo
```

**Run in background with platform specification (if needed):**
```bash
docker run -d --name oura-demo --platform linux/amd64 -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo
```

**Access the UI Demo:**

Once running, open your browser to:
- **UI**: http://localhost:8001
- **API Docs (Swagger)**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

**Stop the UI Demo Container:**
```bash
docker stop oura-demo
docker rm oura-demo  # Remove container after stopping
```

### Run the Backend Only (Docker)

To run just the FastAPI backend without the UI (useful for API-only access or production deployments):

**Development mode:**
```bash
docker run -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo \
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**With platform specification (if needed):**
```bash
docker run --platform linux/amd64 -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo \
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**Run in background:**
```bash
docker run -d --name oura-backend -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  oura-demo \
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8001
```

**Access the Backend API:**

- **API Docs (Swagger)**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health
- **API Endpoints**: http://localhost:8001/api/*

**Stop the Backend Container:**
```bash
docker stop oura-backend
docker rm oura-backend
```

### Running Real Synthesis (with trained models)

By default, the demo uses **mock synthesis** which returns modified versions of the original data. To run **real synthesis** with trained models, you need to mount your model checkpoints and datasets:

```bash
docker run -d --name oura-demo -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e OPENAI_API_KEY="sk-..." \
  -v ~/data:/root/data \
  -v ~/datasets:/home/synthefy/datasets \
  oura-demo
```

**Required model paths** (inside the container):
| Dataset | Model Path |
|---------|-----------|
| `ppg` | `/root/data/training_logs/ppg/Time_Series_Diffusion_Training/synthesis_ppg/checkpoints/best_model.ckpt` |
| `oura` | `/root/data/training_logs/oura/Time_Series_Diffusion_Training/synthesis_oura_flexible/checkpoints/best_model.ckpt` |
| `oura_subset` | `/root/data/training_logs/oura_subset/Time_Series_Diffusion_Training/synthesis_oura_subset_flexible/checkpoints/best_model.ckpt` |

**Override model path** via environment variable:
```bash
docker run -p 8001:8001 \
  -e LICENSE_KEY="" \
  -e SYNTHESIS_MODEL_PATH=/path/to/your/model.ckpt \
  -v /local/path/to/model.ckpt:/path/to/your/model.ckpt \
  oura-demo
```

**Copy files into a running container:**
```bash
docker cp ~/data/training_logs oura-demo:/root/data/training_logs
```

---

## Running the Main Inference Server (Docker)

The `oura-demo` Docker image includes the full compiled `synthefy_pkg`, so you can also run the main Synthefy Inference Server for production synthesis, forecasting, and anomaly detection APIs.

### Start the Inference Server

```bash
# Development mode
docker run -it --rm -p 8000:8000 \
  -v ~/data:/home/synthefy/datasets \
  -e LICENSE_KEY="" \
  oura-demo \
  python3 /home/synthefy/synthefy-package/examples/launch_backend.py \
    --config /home/synthefy/synthefy-package/examples/configs/api_configs/api_config_ppg.yaml
```

### Available API Configs

| Config | Dataset |
|--------|---------|
| `api_config_ppg.yaml` | PPG (photoplethysmography) |
| `api_config_air_quality.yaml` | Air quality |

### Test the Server

```bash
# Health check
curl http://localhost:8000/
# {"message":"Synthefy Platform API"}

# View API docs
open http://localhost:8000/docs
```

### Full Server with All Routers

For access to all API routers (training, search, etc.):

```bash
docker run -d --name synthefy-api -p 8000:8000 \
  -v ~/data:/home/synthefy/datasets \
  -e LICENSE_KEY="" \
  oura-demo \
  python3 /home/synthefy/synthefy-package/src/synthefy_pkg/app/main.py \
    --config /home/synthefy/synthefy-package/examples/configs/api_configs/api_config_ppg.yaml \
    --workers 1
```

### Selective Routers

Start with only specific endpoints:

```bash
docker run -d --name synthefy-api -p 8000:8000 \
  -v ~/data:/home/synthefy/datasets \
  -e LICENSE_KEY="" \
  -e SYNTHEFY_ROUTER="synthesis,forecast,health" \
  oura-demo \
  python3 /home/synthefy/synthefy-package/src/synthefy_pkg/app/main.py \
    --config /home/synthefy/synthefy-package/examples/configs/api_configs/api_config_ppg.yaml
```

### Python Client Example

```python
import httpx
import pandas as pd

# Load your data
df = pd.read_parquet("your_data.parquet")

# Use first window_size rows (e.g., 256 for PPG)
input_data = df.iloc[:256].to_dict()

# Make request
client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)
response = client.post("/api/synthesis/ppg", json=input_data)

# Get synthetic data
synthetic_df = pd.DataFrame(response.json())
print(synthetic_df.head())
```

### Stop the Inference Server

```bash
docker stop synthefy-api && docker rm synthefy-api
```

For more details, see [examples/INFERENCE_SERVER_README.md](../examples/INFERENCE_SERVER_README.md).

---

## Local Development Setup

If you prefer to run locally without Docker, follow the instructions below.

## Architecture

```
oura_demo/
├── api/                         # FastAPI backend
│   ├── main.py                  # FastAPI app entry point
│   ├── models.py                # Pydantic data models
│   ├── routers/
│   │   ├── config.py            # Config loading endpoints
│   │   ├── upload.py            # File upload + validation
│   │   ├── llm.py               # LLM modification endpoints (GPT-4)
│   │   └── inference.py         # Synthesis endpoints
│   ├── services/
│   │   ├── config_loader.py     # Dynamic config loader
│   │   ├── llm_service.py       # OpenAI GPT-4 integration
│   │   └── demo_synthesis_service.py  # Synthesis wrapper
│   └── pyproject.toml
├── ui/                          # React frontend (Vite + TypeScript + Tailwind)
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── DatasetSelector.tsx
│   │   │   ├── FileUploader.tsx
│   │   │   ├── TimeSeriesChart.tsx
│   │   │   ├── DataTable.tsx
│   │   │   ├── LLMChat.tsx
│   │   │   └── SynthesisPanel.tsx
│   │   ├── types/               # TypeScript types
│   │   ├── utils/               # API client
│   │   └── App.tsx              # Main app component
│   └── package.json
└── README.md
```

## Supported Datasets

| Dataset | Window Size | Timeseries Columns | Num Channels |
|---------|-------------|-------------------|--------------|
| `oura` | 96 | average_hrv, lowest_heart_rate, age_cva_diff, highest_temperature, stressed_duration, latency | 6 |
| `oura_subset` | 192 | average_hrv, lowest_heart_rate, age_cva_diff | 3 |
| `ppg` | 256 | BVP | 1 |

## Setup Instructions

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.10+ and `uv` installed
2. **Node.js**: Ensure you have Node.js and npm installed
   - **macOS (with Homebrew)**: `brew install node`
   - **Linux**: Use your distribution's package manager (e.g., `sudo apt install nodejs npm` on Ubuntu)
   - **Windows**: Download from [nodejs.org](https://nodejs.org/)
   - Verify installation: `node --version` and `npm --version`
3. **Environment Variables**: Create a `.env` file in the `oura_demo` directory

### Step 1: Configure Environment Variables

Create a `.env` file in the `oura_demo` directory:

```bash
cd oura_demo
cp .env.example .env
```

Edit `.env` and update the paths and API keys as needed:

```bash
SYNTHEFY_DATASETS_BASE=/Users/raimi.shah/data
SYNTHEFY_PACKAGE_BASE=/Users/raimi.shah/synthefy-package/
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY="AIzaSyC_..."
LICENSE_KEY=""
```

**Note**: Update `SYNTHEFY_DATASETS_BASE` and `SYNTHEFY_PACKAGE_BASE` to match your local paths.

### Step 2: Source Environment Variables

Before running the backend, source the environment variables:

```bash
# From the oura_demo directory
set -a && source .env && set +a
```

Alternatively, if you have a `.env.local` file in the root `synthefy-package` directory:

```bash
set -a && source /path/to/synthefy-package/.env.local && set +a
```

### Step 3: Install Backend Dependencies

```bash
cd oura_demo/api
uv sync
```

This will install all required Python dependencies using `uv`.

### Step 4: Install Frontend Dependencies

```bash
cd oura_demo/ui
npm install
```

This will install all required Node.js dependencies.

## Running the Application

### Start the Backend Server

From the `oura_demo/api` directory:

```bash
# Option 1: Using uvicorn directly
uv run uvicorn main:app --reload --port 8001

# Option 2: Using the main module
uv run python main.py
```

The backend will be available at:
- API: http://localhost:8001
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Start the Frontend Development Server

From the `oura_demo/ui` directory:

```bash
npm run dev
```

The UI will be available at http://localhost:5173

**Note**: The Vite dev server is configured to proxy `/api` requests to the backend at `localhost:8001`, so make sure the backend is running first.

## Quick Start (All in One)

If you want to run both services, use separate terminal windows:

**Terminal 1 - Backend:**
```bash
cd oura_demo
set -a && source .env && set +a
cd api
uv run uvicorn main:app --reload --port 8001
```

**Terminal 2 - Frontend:**
```bash
cd oura_demo/ui
npm run dev
```

Then open http://localhost:5173 in your browser.

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## API Endpoints

### Config

- `GET /api/config/datasets` - List available datasets
- `GET /api/config/{dataset_name}` - Get config info for a dataset

### Upload

- `POST /api/upload/{dataset_name}` - Upload parquet/CSV file with validation

### LLM Modification

- `POST /api/llm/modify` - Modify data using natural language (GPT-4)

### Synthesis

- `POST /api/synthesize` - Generate synthetic time series

## Example Usage

### 1. Get Config
```bash
curl http://localhost:8001/api/config/oura
```

### 2. Upload File
```bash
curl -X POST "http://localhost:8001/api/upload/oura" \
  -F "file=@your_data.parquet"
```

### 3. Modify Data with LLM
```bash
curl -X POST "http://localhost:8001/api/llm/modify" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "oura",
    "data": {"columns": {"average_hrv": [50, 52, 48, ...], ...}},
    "user_query": "Add 10% noise to average_hrv"
  }'
```

### 4. Run Synthesis
```bash
curl -X POST "http://localhost:8001/api/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "oura",
    "data": {"columns": {...}}
  }'
```

## Data Format

The API uses a column-oriented DataFrame format:

```json
{
  "columns": {
    "average_hrv": [50.0, 52.0, 48.0, ...],
    "lowest_heart_rate": [55, 54, 56, ...],
    "gender_male": [1, 1, 1, ...]
  }
}
```

This maps to a pandas DataFrame where each key is a column name and values are lists of the same length.
