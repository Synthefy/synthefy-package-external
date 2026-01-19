#!/usr/bin/env python3

import argparse
import asyncio
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import psutil
import toml
from ollama import AsyncClient

# Global rate limiters per GPU and performance tracking
gpu_rate_limiters = {}
gpu_semaphores = {}
gpu_stats = {}

# Create a thread pool for I/O-bound tasks
CPU_WORKERS = max(1, multiprocessing.cpu_count() // 2)
thread_pool = ThreadPoolExecutor(max_workers=CPU_WORKERS)


def log_message(message: str):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}", flush=True)


def process_response(response_text: str) -> str:
    """Process response text in a separate CPU thread."""
    return response_text.strip()


def update_gpu_stats(gpu_index: int, duration: float):
    """Update GPU performance statistics."""
    if gpu_index not in gpu_stats:
        gpu_stats[gpu_index] = {
            "total_time": 0.0,
            "tasks_completed": 0,
            "avg_duration": 0.0,
        }

    stats = gpu_stats[gpu_index]
    stats["total_time"] += duration
    stats["tasks_completed"] += 1
    stats["avg_duration"] = stats["total_time"] / stats["tasks_completed"]


async def save_response_async(
    output_dir: Path,
    batch_num: int,
    series_idx: int,
    prompt_type: str,
    prompt: str,
    response_text: str,
):
    """Asynchronously saves the response to a JSON file."""

    def _save():
        # Create the series-specific directory path using the same pattern as metadata files
        series_dir = output_dir / f"{output_dir.name}_{series_idx}"
        series_dir.mkdir(parents=True, exist_ok=True)

        # Create a consistent filename using prompt_type
        filename = f"{output_dir.name}_{series_idx}_{prompt_type}_response.json"
        file_path = series_dir / filename

        # Create the output dictionary
        output_data = {
            "series_idx": series_idx,
            "prompt_type": prompt_type,
            "prompt": prompt,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
        }

        # Write the response to the JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
            json_file.flush()

    # Run the file I/O in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_pool, _save)


async def chat(
    output_dir: Path,
    batch_num: int,
    series_idx: int,
    prompt_type: str,
    system_msg: dict,
    message: dict,
    host: str,
    gpu_index: int,
    model: str,
):
    try:
        # Start timing
        start_time = datetime.now()

        # Make the API request using the specified host and model
        response = await AsyncClient(host=f"http://{host}").chat(
            model=model,
            messages=[system_msg, message],
            options={
                "temperature": 0.0,  # Most deterministic
                "top_p": 1.0,  # No nucleus sampling
                "num_predict": 4096,  # Keep high token limit
                "stop": [
                    "\n\n"
                ],  # Stop at double newline to prevent hallucination
                "repeat_penalty": 1.1,  # Slight penalty for repetition
            },
        )

        # Stop timing
        end_time = datetime.now()

        # Extract the response content
        response_text = response["message"]["content"]
        prompt_text = message["content"]

        # Process response in a separate CPU thread
        loop = asyncio.get_event_loop()
        processed_response = await loop.run_in_executor(
            thread_pool, process_response, response_text
        )

        # Asynchronously save the response
        await save_response_async(
            output_dir,
            batch_num,
            series_idx,
            prompt_type,
            prompt_text,
            processed_response,
        )

        # Calculate duration and word count
        duration = (end_time - start_time).total_seconds()
        word_count = len(processed_response.split())

        # Calculate words per second (WPS)
        wps = word_count / duration if duration > 0 else 0

        # Update GPU statistics
        update_gpu_stats(gpu_index, duration)

        # Log the GPU index, word count, duration, and words per second
        log_message(
            f"Host: {host}, GPU: {gpu_index}, Batch: {batch_num}, Series: {series_idx}, Type: {prompt_type}, Words: {word_count}, Duration: {duration:.2f}s, WPS: {wps:.2f}"
        )

        return duration, prompt_type

    except Exception as e:
        log_message(f"Error on host {host} for series {series_idx}: {e}")
        return 0, prompt_type


async def worker(
    output_dir: Path,
    host: str,
    gpu_index: int,
    model: str,
    task_queue: asyncio.Queue,
    global_semaphore: asyncio.Semaphore,
):
    """Worker function to process tasks using the specified host."""
    # Get or create GPU-specific rate limiter with fixed high concurrency for A100s
    if gpu_index not in gpu_rate_limiters:
        gpu_rate_limiters[gpu_index] = {
            "extract": asyncio.Semaphore(64),  # High concurrency for A100 80GB
            "expand": asyncio.Semaphore(48),  # Slightly lower for expand tasks
        }
        gpu_semaphores[gpu_index] = asyncio.Semaphore(
            32
        )  # Per-GPU concurrent tasks

    while True:
        try:
            # Get task from queue
            task = await task_queue.get()
            if task is None:  # Poison pill
                break

            batch_num, series_idx, prompt_type, system_msg, message = task

            # Check system resources and pause if CPU/memory is too high
            resources = psutil.virtual_memory()
            if resources.percent > 95:
                log_message(
                    f"Memory usage high ({resources.percent}%) - pausing for 5 seconds"
                )
                await asyncio.sleep(5)
                continue

            # Use appropriate rate limiter based on task type
            rate_limiter = gpu_rate_limiters[gpu_index][prompt_type]

            # Apply both GPU semaphore and type-specific rate limiting
            async with global_semaphore:
                async with gpu_semaphores[gpu_index]:
                    async with rate_limiter:
                        duration, task_type = await chat(
                            output_dir,
                            batch_num,
                            series_idx,
                            prompt_type,
                            system_msg,
                            message,
                            host,
                            gpu_index,
                            model,
                        )

                        # Update stats without affecting concurrency
                        if duration > 0:
                            update_gpu_stats(gpu_index, duration)
                            # Minimal delay to prevent overwhelming
                            await asyncio.sleep(0.01)

        except Exception as e:
            log_message(f"Worker error on host {host}: {e}")
            await asyncio.sleep(1)  # Brief pause on error
        finally:
            task_queue.task_done()


async def process_batch_file(
    batch_file: Path,
    output_dir: Path,
    config: Dict,
    max_concurrency: int,
    gpus: Dict[str, int],
):
    """Process a single batch file of prompts."""
    model = config.get("model", "llama3.2")
    system_msg = {"role": "system", "content": config.get("system_message", "")}

    # Extract batch number from filename
    try:
        batch_num = int(batch_file.stem.split("_")[-1])
    except (ValueError, IndexError):
        batch_num = 0

    # Load prompts from JSONL file
    prompts = []
    with open(batch_file, "r") as file:
        for line in file:
            prompts.append(json.loads(line))

    # Create global semaphore for overall concurrency
    global_semaphore = asyncio.Semaphore(max_concurrency)

    # Create an async queue and populate it with prompts
    task_queue = asyncio.Queue()

    # Group prompts by type for better distribution
    extract_prompts = []
    expand_prompts = []

    for prompt in prompts:
        series_idx = prompt.get("series_idx")
        prompt_type = prompt.get("prompt_type")
        if series_idx is not None and prompt_type:
            message = {"role": "user", "content": prompt["content"]}
            if prompt_type == "extract":
                extract_prompts.append((series_idx, message))
            else:
                expand_prompts.append((series_idx, message))

    # Sort GPUs by their indices to ensure consistent assignment
    sorted_gpus = sorted(gpus.items(), key=lambda x: int(x[1]))

    # Distribute tasks evenly across GPUs
    extract_per_gpu = len(extract_prompts) // len(sorted_gpus)
    expand_per_gpu = len(expand_prompts) // len(sorted_gpus)

    for i, (host, gpu_index) in enumerate(sorted_gpus):
        # Calculate slice indices for this GPU
        extract_start = i * extract_per_gpu
        extract_end = (
            extract_start + extract_per_gpu
            if i < len(sorted_gpus) - 1
            else len(extract_prompts)
        )
        expand_start = i * expand_per_gpu
        expand_end = (
            expand_start + expand_per_gpu
            if i < len(sorted_gpus) - 1
            else len(expand_prompts)
        )

        # Add extract tasks for this GPU
        for idx, message in extract_prompts[extract_start:extract_end]:
            await task_queue.put(
                [batch_num, idx, "extract", system_msg, message]
            )

        # Add expand tasks for this GPU
        for idx, message in expand_prompts[expand_start:expand_end]:
            await task_queue.put(
                [batch_num, idx, "expand", system_msg, message]
            )

    # Add poison pills for workers
    for _ in range(len(gpus)):
        await task_queue.put(None)

    log_message(
        f"Processing batch {batch_num}: {task_queue.qsize()} tasks across {len(gpus)} Ollama instances"
    )

    # Create a list of worker tasks, one for each Ollama instance
    tasks = []
    for host, gpu_index in gpus.items():
        tasks.append(
            worker(
                output_dir, host, gpu_index, model, task_queue, global_semaphore
            )
        )

    # Process tasks
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        log_message(f"Error processing batch {batch_num}: {e}")


def validate_config(config: Dict) -> None:
    """Validate the configuration file."""
    if "ollama_instances" not in config:
        raise ValueError("Config must contain 'ollama_instances' section")

    if not isinstance(config["ollama_instances"], dict):
        raise ValueError("'ollama_instances' must be a dictionary")

    if not config["ollama_instances"]:
        raise ValueError("'ollama_instances' cannot be empty")

    # Validate each host:port and GPU index
    for host, gpu_idx in config["ollama_instances"].items():
        # Check host:port format
        try:
            host_part, port_part = host.split(":")
            if not host_part or not port_part.isdigit():
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid host:port format: {host}")

        # Check GPU index is non-negative integer
        if not isinstance(gpu_idx, int) or gpu_idx < 0:
            raise ValueError(
                f"GPU index must be non-negative integer, got: {gpu_idx}"
            )

    # Validate model name exists
    if "model" not in config:
        log_message("No model specified in config, using default: llama3.2")
        config["model"] = "llama3.2"

    # Validate system message exists
    if "system_message" not in config:
        log_message("No system message specified in config, using empty string")
        config["system_message"] = ""


async def main(args):
    # Load and validate configuration
    config = toml.load(args.config)
    validate_config(config)
    gpus = config["ollama_instances"]

    # Log GPU configuration
    log_message(
        f"Using {len(gpus)} Ollama instances across {len(set(gpus.values()))} unique GPUs"
    )
    for host, gpu_idx in sorted(gpus.items(), key=lambda x: (x[1], x[0])):
        log_message(f"  GPU {gpu_idx}: {host}")

    # Get input directory and use it as the base directory for outputs
    input_dir = Path(args.input_dir)
    output_dir = (
        input_dir  # Save responses in the same base directory as metadata files
    )

    # Calculate default concurrency if not specified
    if args.concurrency <= 0:
        cpu_count = os.cpu_count()
        cores_for_concurrency = cpu_count if cpu_count is not None else 4
        max_concurrency = min(len(gpus) * 4, max(1, cores_for_concurrency // 2))
        log_message(
            f"Setting concurrency to {max_concurrency} based on system resources"
        )
    else:
        max_concurrency = args.concurrency

    # Get all batch files
    batch_files = sorted(input_dir.glob("*_combined_prompts_batch_*.jsonl"))
    if not batch_files:
        log_message(f"No batch files found in {input_dir}")
        return

    log_message(f"Found {len(batch_files)} batch files to process")

    # Process each batch file
    for batch_file in batch_files:
        await process_batch_file(
            batch_file, output_dir, config, max_concurrency, gpus
        )
        log_message(f"Completed processing {batch_file.name}")

    # Log final GPU statistics
    log_message("\nFinal GPU Statistics:")
    for gpu_index, stats in gpu_stats.items():
        avg_duration = stats["avg_duration"]
        tasks_completed = stats["tasks_completed"]
        log_message(
            f"GPU {gpu_index}: Completed {tasks_completed} tasks, Average duration: {avg_duration:.2f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process batched prompts using Ollama"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration TOML file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing batch prompt files and metadata",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Maximum number of concurrent requests (default: auto)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
        log_message("All batches processed successfully")
    except KeyboardInterrupt:
        log_message("Process interrupted by user")
    except Exception as e:
        log_message(f"Error in main process: {e}")
    finally:
        thread_pool.shutdown(wait=True)
