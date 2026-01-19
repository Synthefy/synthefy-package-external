# Ollama Scripts Pipeline

This directory contains a set of scripts for processing and generating metadata descriptions using Ollama LLM. The pipeline consists of several scripts that work together to process large batches of metadata files efficiently.

## Scripts Overview

### 1. `prepare_combined_prompts.py`

Prepares prompts for Ollama processing by combining metadata information.

**Usage:**

```bash
python prepare_combined_prompts.py --data_dir <path_to_data> [--include-expand] [--batch-size <size>]
```

**Features:**

- Processes metadata files in batches
- Generates two types of prompts:
  - Extract prompts: For extracting structured information
  - Expand prompts: For generating comprehensive descriptions
- Supports configurable batch sizes
- Creates JSONL files containing prompts for batch processing

### 2. `process_batch_prompts.py`

Processes the prepared prompts using multiple Ollama instances across GPUs.

**Usage:**

```bash
python process_batch_prompts.py --config <config_file> --input-dir <input_directory> [--concurrency <num>]
```

**Features:**

- Distributed processing across multiple GPUs
- Adaptive concurrency control
- Resource monitoring and management
- Supports multiple Ollama instances
- Requires a TOML configuration file for GPU and model settings

### 3. `update_metadata_descriptions.py`

Updates metadata files with the generated descriptions from Ollama responses.

**Usage:**

```bash
python update_metadata_descriptions.py --base-dir <base_directory>
```

**Features:**

- Processes Ollama responses and updates metadata files
- Preserves original descriptions
- Adds extracted and expanded descriptions
- Includes logging for tracking progress

### 4. `ollama-batch-servers.sh`

Shell script for managing multiple Ollama server instances with optimized resource allocation.

**Usage:**

```bash
./ollama-batch-servers.sh <instances_per_gpu> <num_gpus>
# Example: ./ollama-batch-servers.sh 2 4  # 2 instances per GPU, using 4 GPUs
```

**Features:**

- Distributes Ollama instances across multiple GPUs
- Optimizes CPU core allocation per instance
- Configures memory limits and NUMA bindings
- Supports GPU memory management
- Provides detailed logging for each instance
- Automatically calculates optimal batch sizes
- Sets up environment variables for performance

**Configuration:**

- Adjustable parameters for:
  - CPU cores per instance
  - GPU layers
  - Batch sizes
  - Memory allocation
  - Load timeout
  - Keep-alive duration
  - Parallel processing

**Requirements:**

- NVIDIA GPUs (optimized for A100-SXM4-80GB)
- `numactl` (optional, for NUMA optimization)
- `cgroups` (optional, for memory limits)
- Ollama binary at `/usr/local/bin/ollama`

## Pipeline Flow

0. **Initial Setup**
   - Stop any existing Ollama services:

     ```bash
     sudo systemctl stop ollama
     sudo systemctl disable ollama
     sudo pkill -f 'ollama serve'
     ```

   - Create `config.toml` with your GPU configuration:

     ```toml
     # config.toml
     model = "llama3.2"
     system_message = "Your system message here"
     
     [ollama_instances]
     # 2 instances per GPU across 8 GPUs (A100-SXM4-80GB)
     "localhost:11434" = 0
     "localhost:11435" = 0
     "localhost:11436" = 1
     "localhost:11437" = 1
     "localhost:11438" = 2
     "localhost:11439" = 2
     "localhost:11440" = 3
     "localhost:11441" = 3
     "localhost:11442" = 4
     "localhost:11443" = 4
     "localhost:11444" = 5
     "localhost:11445" = 5
     "localhost:11446" = 6
     "localhost:11447" = 6
     "localhost:11448" = 7
     "localhost:11449" = 7
     ```

   - Start Ollama servers (2 instances per GPU, 8 GPUs):

     ```bash
     chmod +x ollama-batch-servers.sh
     ./ollama-batch-servers.sh 2 8
     ```

   - Verify servers are running:

     ```bash
     ps aux | grep 'ollama serve'
     ```

1. **Preparation Phase**

   ```bash
   # Generate prompts with batch size 1000, including expand prompts
   python prepare_combined_prompts.py \
     --data_dir /path/to/your/data \
     --include-expand \
     --batch-size 1000
   ```

2. **Processing Phase**

   ```bash
   # Process prompts with auto-configured concurrency
   python process_batch_prompts.py \
     --config config.toml \
     --input-dir /path/to/your/data
   
   # Or specify concurrency manually (e.g., 32 concurrent requests)
   python process_batch_prompts.py \
     --config config.toml \
     --input-dir /path/to/your/data \
     --concurrency 32
   ```

3. **Update Phase**

   ```bash
   # Update metadata files with generated descriptions
   python update_metadata_descriptions.py \
     --base-dir /path/to/your/data
   ```

**Note:** Replace `/path/to/your/data` with your actual data directory path. The example configuration above is optimized for 8x A100-SXM4-80GB GPUs with 80GB memory each. Adjust the number of instances and GPU assignments based on your hardware configuration.

## Configuration

### TOML Configuration Example

```toml
# config.toml
model = "llama3.2"
system_message = "Your system message here"

[ollama_instances]
"localhost:11434" = 0
"localhost:11435" = 1
"localhost:11436" = 2
```

## Requirements

- Python 3.7+
- Ollama instances running on specified ports
- GPU(s) for processing
- Required Python packages:
  - asyncio
  - psutil
  - toml
  - loguru
  - ollama-python

## Best Practices

1. Monitor GPU usage during processing
2. Adjust concurrency based on available resources
3. Use appropriate batch sizes for your data
4. Keep system message consistent across runs
5. Regularly check logs for any issues
6. When using `ollama-batch-servers.sh`:
   - Stop any existing Ollama services first
   - Monitor GPU memory usage
   - Check server logs in `ollama-server-logs` directory
   - Adjust instances per GPU based on memory requirements
   - Consider NUMA topology for optimal performance

## Logging

All scripts include logging functionality:

- Process progress and statistics
- Error tracking
- Performance metrics
- GPU utilization

Logs are saved in the specified base directory with rotation enabled.
