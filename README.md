# Example Inference
python inference_example.py     --task forecast    --dataset oura_subset     --model-type flexible     --num-samples 3     --forecast-length 90     --seed 42     --output results.parquet


# Example UI
 /home/raimi/synthefy-package-external && UI_DIST_PATH="/home/raimi/synthefy-package-external/oura_demo/ui/dist" SYNTHEFY_PACKAGE_BASE=/home/raimi/synthefy-package-external SYNTHEFY_DATASETS_BASE=~/data PYTHONPATH="/home/raimi/synthefy-package-external/src:/home/raimi/synthefy-package-external/oura_demo/api:$PYTHONPATH" uv run uvicorn main:app --reload --port 8001 --app-dir /home/raimi/synthefy-package-external/oura_demo/api
