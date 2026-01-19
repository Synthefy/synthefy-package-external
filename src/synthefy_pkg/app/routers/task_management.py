"""
Task Management API Router
Provides endpoints for canceling tasks, monitoring workers, and scaling operations.
"""

import os
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

from celery import current_app
from celery.result import AsyncResult
from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel

from synthefy_pkg.app.celery_app import celery_app

router = APIRouter(prefix="/api/task-management", tags=["Task Management"])


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    traceback: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class WorkerInfo(BaseModel):
    hostname: str
    status: str
    active_tasks: int
    processed_tasks: int
    load_avg: Optional[List[float]] = None


class CancelTaskRequest(BaseModel):
    task_id: str
    terminate: bool = False  # If True, forcefully terminate the task


class ScaleWorkersRequest(BaseModel):
    target_workers: int  # Desired number of workers
    worker_type: str = "default"  # Queue type


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get detailed status of a specific task"""
    try:
        result = AsyncResult(task_id, app=celery_app)

        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status,
            result=result.result if result.ready() else None,
            traceback=result.traceback if result.failed() else None,
        )

        # Get progress information if available
        if result.status == "PROGRESS" and hasattr(result, "info"):
            response.progress = result.info

        return response

    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving task status: {str(e)}"
        )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, request: CancelTaskRequest):
    """Cancel a running or pending task"""
    try:
        result = AsyncResult(task_id, app=celery_app)

        if result.status in ["PENDING", "RETRY", "PROGRESS"]:
            # Revoke the task
            celery_app.control.revoke(task_id, terminate=request.terminate)

            # If terminate is True, send SIGTERM to the worker process
            if request.terminate:
                # Get active tasks to find which worker is running this task
                inspect = celery_app.control.inspect()
                active_tasks = inspect.active()

                if active_tasks:
                    for worker_name, tasks in active_tasks.items():
                        for task in tasks:
                            if task["id"] == task_id:
                                logger.info(
                                    f"Forcefully terminating task {task_id} on worker {worker_name}"
                                )
                                # Note: In production, you'd want more sophisticated worker management
                                break

            return {
                "message": f"Task {task_id} cancellation requested",
                "terminated": request.terminate,
            }
        else:
            return {
                "message": f"Task {task_id} is in status {result.status} and cannot be cancelled"
            }

    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error cancelling task: {str(e)}"
        )


@router.get("/workers", response_model=List[WorkerInfo])
async def get_workers_info():
    """Get information about all active workers"""
    try:
        inspect = celery_app.control.inspect()

        # Get various worker stats
        stats = inspect.stats() or {}
        active_tasks = inspect.active() or {}

        workers_info = []

        for worker_name in stats.keys():
            worker_stats = stats[worker_name]
            worker_active_tasks = len(active_tasks.get(worker_name, []))

            worker_info = WorkerInfo(
                hostname=worker_name,
                status="online",
                active_tasks=worker_active_tasks,
                processed_tasks=worker_stats.get("total", {}).get(
                    "task.received", 0
                ),
                load_avg=worker_stats.get("rusage", {}).get("utime", None),
            )
            workers_info.append(worker_info)

        return workers_info

    except Exception as e:
        logger.error(f"Error getting workers info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving workers info: {str(e)}"
        )


@router.post("/workers/restart")
async def restart_workers(background_tasks: BackgroundTasks):
    """Restart all Celery workers (graceful restart)"""
    try:
        # Send warm shutdown signal to all workers
        celery_app.control.broadcast("shutdown", reply=True)

        # Add background task to restart workers after shutdown
        background_tasks.add_task(restart_workers_after_delay)

        return {
            "message": "Worker restart initiated. Workers will restart gracefully."
        }

    except Exception as e:
        logger.error(f"Error restarting workers: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error restarting workers: {str(e)}"
        )


async def restart_workers_after_delay():
    """Background task to restart workers after graceful shutdown"""
    try:
        # Wait for workers to shut down gracefully
        time.sleep(10)

        # In Kubernetes, the pod should automatically restart due to the restart policy
        # For local development, you might need different logic
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            logger.info(
                "Running in Kubernetes - pod will restart automatically"
            )
        else:
            # Local development restart logic
            logger.info("Restarting Celery workers locally")
            # This would need to be customized based on your local setup

    except Exception as e:
        logger.error(f"Error in background worker restart: {str(e)}")


@router.post("/workers/scale")
async def scale_workers(request: ScaleWorkersRequest):
    """Scale the number of workers (Kubernetes only)"""
    try:
        # This endpoint is primarily for Kubernetes environments
        if not os.getenv("KUBERNETES_SERVICE_HOST"):
            raise HTTPException(
                status_code=400,
                detail="Worker scaling is only supported in Kubernetes environments",
            )

        # In a full implementation, this would interface with Kubernetes API
        # to scale the worker deployment

        # For now, return guidance
        return {
            "message": f"To scale workers to {request.target_workers}, update the Kubernetes deployment:",
            "kubectl_command": f"kubectl scale deployment synthefy-workers --replicas={request.target_workers} -n synthefy",
            "note": "This endpoint would ideally integrate with Kubernetes API for automatic scaling",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scaling workers: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error scaling workers: {str(e)}"
        )


@router.get("/tasks/active")
async def get_active_tasks():
    """Get all currently active (running) tasks"""
    try:
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active() or {}

        all_active_tasks = []
        for worker_name, tasks in active_tasks.items():
            for task in tasks:
                task_info = {
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "worker": worker_name,
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {}),
                    "time_start": task.get("time_start"),
                }
                all_active_tasks.append(task_info)

        return {
            "active_tasks": all_active_tasks,
            "total_active": len(all_active_tasks),
        }

    except Exception as e:
        logger.error(f"Error getting active tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving active tasks: {str(e)}"
        )


@router.get("/queues/info")
async def get_queue_info():
    """Get information about task queues (enhanced for KEDA monitoring)"""
    try:
        import redis

        # Get Celery queue info
        inspect = celery_app.control.inspect()
        reserved_tasks = inspect.reserved() or {}
        scheduled_tasks = inspect.scheduled() or {}

        # Get direct Redis queue lengths (important for KEDA)
        redis_client = redis.Redis.from_url("redis://synthefy-redis:6379/0")

        redis_queue_info = {}
        try:
            # Check default queue that KEDA monitors
            redis_queue_info["default"] = redis_client.llen("default")
        except Exception as redis_error:
            logger.warning(f"Could not get Redis queue lengths: {redis_error}")
            redis_queue_info = {"error": "Redis connection failed"}

        # Worker-based queue info (Celery perspective)
        worker_queue_info = {}
        for worker_name in reserved_tasks.keys():
            worker_queue_info[worker_name] = {
                "reserved_tasks": len(reserved_tasks[worker_name]),
                "scheduled_tasks": len(scheduled_tasks.get(worker_name, [])),
            }

        return {
            "redis_queues": redis_queue_info,  # Direct Redis queue lengths (what KEDA sees)
            "worker_queues": worker_queue_info,  # Per-worker queue info
            "total_queued": sum(
                info["reserved_tasks"] + info["scheduled_tasks"]
                for info in worker_queue_info.values()
            ),
            "scaling_info": {
                "keda_triggers": {
                    "default": "Scales when > 2 tasks",
                },
                "current_trigger_status": {
                    "default": int(redis_queue_info.get("default", 0)) > 2,
                },
            },
        }

    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving queue info: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for the task management system"""
    try:
        # Check if we can connect to Celery
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if stats:
            return {
                "status": "healthy",
                "active_workers": len(stats),
                "celery_connection": "ok",
            }
        else:
            return {
                "status": "degraded",
                "active_workers": 0,
                "celery_connection": "no workers found",
            }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "celery_connection": "failed",
        }
