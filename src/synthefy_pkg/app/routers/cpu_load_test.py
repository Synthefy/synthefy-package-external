import asyncio
import math
import random
import time
from typing import Dict, List

import numpy as np
from fastapi import APIRouter, Body
from loguru import logger
from pydantic import BaseModel, Field

COMPILE = False
router = APIRouter(tags=["CPU Load Test"])


class CPUIntensiveRequest(BaseModel):
    """Request model for CPU-intensive operations."""

    matrix_size: int = Field(
        default=1000, description="Size of matrices to multiply"
    )
    iterations: int = Field(
        default=100, description="Number of iterations to run"
    )
    complexity: str = Field(
        default="medium",
        description="Complexity level: low, medium, high, extreme",
    )
    use_parallel: bool = Field(
        default=True, description="Whether to use parallel processing"
    )


class CPUIntensiveResponse(BaseModel):
    """Response model for CPU-intensive operations."""

    result: float
    execution_time: float
    cpu_usage_estimate: float
    matrix_size: int
    iterations: int
    complexity: str
    message: str


def _safe_limit_result(value: float, max_value: float = 1e15) -> float:
    """
    Safely limit a result value to prevent JSON serialization issues.

    Args:
        value: The value to limit
        max_value: Maximum allowed value (default: 1e15)

    Returns:
        Limited value that is safe for JSON serialization
    """
    if not np.isfinite(value):
        return 0.0

    if abs(value) > max_value:
        # Return a large but safe value with the same sign
        return max_value if value > 0 else -max_value

    return value


def _matrix_multiplication_cpu_intensive(size: int, iterations: int) -> float:
    """Perform CPU-intensive matrix multiplication."""
    result = 0.0

    for _ in range(iterations):
        # Create large random matrices
        matrix_a = np.random.rand(size, size)
        matrix_b = np.random.rand(size, size)

        # Perform matrix multiplication (CPU-intensive)
        matrix_c = np.dot(matrix_a, matrix_b)

        # Additional CPU-intensive operations with safe limits
        temp_result = np.sum(matrix_c) + np.trace(matrix_c)

        # Safe determinant calculation
        try:
            det = np.linalg.det(matrix_c)
            if np.isfinite(det):
                temp_result += det
        except Exception:
            pass

        # More CPU work: eigenvalue computation with safe limits
        try:
            eigenvalues = np.linalg.eigvals(matrix_c)
            if np.all(np.isfinite(eigenvalues)):
                temp_result += np.sum(eigenvalues)
        except Exception:
            pass

        result = _safe_limit_result(result + temp_result)

    return result


def _prime_factorization_cpu_intensive(iterations: int) -> float:
    """Perform CPU-intensive prime factorization."""
    result = 0.0

    for _ in range(iterations):
        # Generate large numbers for factorization
        large_number = random.randint(1000000, 9999999)

        # Simple but CPU-intensive factorization
        factors = []
        n = large_number
        i = 2

        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 1

        if n > 1:
            factors.append(n)

        result = _safe_limit_result(result + sum(factors))

    return result


def _mathematical_computations_cpu_intensive(iterations: int) -> float:
    """Perform CPU-intensive mathematical computations."""
    result = 0.0

    for _ in range(iterations):
        # Trigonometric computations
        for i in range(1000):
            angle = i * math.pi / 1000
            temp = math.sin(angle) + math.cos(angle)
            # Avoid tan() which can produce very large values
            if (
                abs(math.cos(angle)) > 1e-10
            ):  # Avoid division by very small numbers
                temp += math.tan(angle)
            result = _safe_limit_result(result + temp)

        # Logarithmic computations
        for i in range(1, 1001):
            temp = math.log(i) + math.log10(i) + math.log2(i)
            result = _safe_limit_result(result + temp)

        # Exponential computations with safe limits
        for i in range(100):
            # Limit exponential growth to prevent overflow
            exp_term = math.exp(
                min(i / 10, 20)
            )  # Cap at exp(20) to prevent overflow
            pow_term = math.pow(min(i, 100), 2)  # Cap power to prevent overflow
            sqrt_term = math.sqrt(i)
            temp = exp_term + pow_term + sqrt_term
            result = _safe_limit_result(result + temp)

    return result


def _extreme_cpu_intensive(iterations: int) -> float:
    """Perform extremely CPU-intensive operations."""
    result = 0.0

    for _ in range(iterations):
        # Large matrix operations
        size = 500
        matrix_a = np.random.rand(size, size)
        matrix_b = np.random.rand(size, size)
        matrix_c = np.random.rand(size, size)

        # Multiple matrix multiplications
        temp1 = np.dot(matrix_a, matrix_b)
        temp2 = np.dot(temp1, matrix_c)
        temp3 = np.dot(matrix_c, matrix_a)

        # SVD decomposition (very CPU-intensive) with safe limits
        try:
            u, s, vh = np.linalg.svd(temp2)
            if (
                np.all(np.isfinite(s))
                and np.all(np.isfinite(u))
                and np.all(np.isfinite(vh))
            ):
                temp_result = np.sum(s) + np.sum(u) + np.sum(vh)
                result = _safe_limit_result(result + temp_result)
        except Exception:
            pass

        # QR decomposition with safe limits
        try:
            q, r = np.linalg.qr(temp3)
            if np.all(np.isfinite(q)) and np.all(np.isfinite(r)):
                temp_result = np.sum(q) + np.sum(r)
                result = _safe_limit_result(result + temp_result)
        except Exception:
            pass

        # Cholesky decomposition with safe limits
        try:
            positive_definite = np.dot(temp1, temp1.T) + np.eye(size)
            cholesky_result = np.linalg.cholesky(positive_definite)
            if np.all(np.isfinite(cholesky_result)):
                temp_result = np.sum(cholesky_result)
                result = _safe_limit_result(result + temp_result)
        except Exception:
            pass

    return result


@router.post(
    "/api/cpu-load-test",
    response_model=CPUIntensiveResponse,
    include_in_schema=False,
)
async def cpu_intensive_operation(
    request: CPUIntensiveRequest = Body(...),
) -> CPUIntensiveResponse:
    """
    Perform CPU-intensive operations to test system performance and scaling.

    This endpoint is designed to max out CPU usage for load testing purposes.
    """
    start_time = time.time()

    logger.info(
        f"Starting CPU-intensive operation: {request.complexity} complexity, "
        f"{request.iterations} iterations, matrix_size={request.matrix_size}"
    )

    try:
        if request.complexity == "low":
            result = _matrix_multiplication_cpu_intensive(
                request.matrix_size // 4, request.iterations // 4
            )
            message = "Low complexity CPU test completed"
        elif request.complexity == "medium":
            result = _matrix_multiplication_cpu_intensive(
                request.matrix_size, request.iterations
            )
            result = _safe_limit_result(
                result
                + _prime_factorization_cpu_intensive(request.iterations // 2)
            )
            message = "Medium complexity CPU test completed"
        elif request.complexity == "high":
            result = _matrix_multiplication_cpu_intensive(
                request.matrix_size, request.iterations
            )
            result = _safe_limit_result(
                result + _prime_factorization_cpu_intensive(request.iterations)
            )
            result = _safe_limit_result(
                result
                + _mathematical_computations_cpu_intensive(
                    request.iterations // 2
                )
            )
            message = "High complexity CPU test completed"
        elif request.complexity == "extreme":
            result = _extreme_cpu_intensive(request.iterations)
            result = _safe_limit_result(
                result
                + _mathematical_computations_cpu_intensive(request.iterations)
            )
            result = _safe_limit_result(
                result + _prime_factorization_cpu_intensive(request.iterations)
            )
            message = "Extreme complexity CPU test completed"
        else:
            raise ValueError(f"Invalid complexity level: {request.complexity}")

        # Final safety check on the result
        result = _safe_limit_result(result)

        execution_time = time.time() - start_time

        # Estimate CPU usage based on execution time and complexity
        cpu_usage_estimate = min(95.0, (execution_time / 10.0) * 100)

        logger.info(
            f"CPU-intensive operation completed in {execution_time:.2f}s, "
            f"estimated CPU usage: {cpu_usage_estimate:.1f}%"
        )

        return CPUIntensiveResponse(
            result=result,
            execution_time=execution_time,
            cpu_usage_estimate=cpu_usage_estimate,
            matrix_size=request.matrix_size,
            iterations=request.iterations,
            complexity=request.complexity,
            message=message,
        )

    except Exception as e:
        logger.error(f"CPU-intensive operation failed: {str(e)}")
        raise Exception(f"CPU-intensive operation failed: {str(e)}")


@router.post(
    "/api/cpu-load-test/parallel",
    response_model=CPUIntensiveResponse,
    include_in_schema=False,
)
async def cpu_intensive_parallel_operation(
    request: CPUIntensiveRequest = Body(...),
) -> CPUIntensiveResponse:
    """
    Perform CPU-intensive operations using parallel processing to max out all CPU cores.
    """
    start_time = time.time()

    logger.info(
        f"Starting parallel CPU-intensive operation: {request.complexity} complexity, "
        f"{request.iterations} iterations, matrix_size={request.matrix_size}"
    )

    try:
        # Create multiple tasks to run in parallel
        tasks = []

        if request.complexity == "low":
            num_tasks = 2
        elif request.complexity == "medium":
            num_tasks = 4
        elif request.complexity == "high":
            num_tasks = 8
        elif request.complexity == "extreme":
            num_tasks = 16
        else:
            raise ValueError(f"Invalid complexity level: {request.complexity}")

        # Create parallel tasks
        for i in range(num_tasks):
            task = asyncio.create_task(
                asyncio.to_thread(
                    _matrix_multiplication_cpu_intensive,
                    request.matrix_size // num_tasks,
                    request.iterations // num_tasks,
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        result = _safe_limit_result(sum(results))

        execution_time = time.time() - start_time

        # Higher CPU usage estimate for parallel operations
        cpu_usage_estimate = min(98.0, (execution_time / 5.0) * 100)

        logger.info(
            f"Parallel CPU-intensive operation completed in {execution_time:.2f}s, "
            f"estimated CPU usage: {cpu_usage_estimate:.1f}%"
        )

        return CPUIntensiveResponse(
            result=result,
            execution_time=execution_time,
            cpu_usage_estimate=cpu_usage_estimate,
            matrix_size=request.matrix_size,
            iterations=request.iterations,
            complexity=request.complexity,
            message=f"Parallel {request.complexity} complexity CPU test completed with {num_tasks} tasks",
        )

    except Exception as e:
        logger.error(f"Parallel CPU-intensive operation failed: {str(e)}")
        raise Exception(f"Parallel CPU-intensive operation failed: {str(e)}")


@router.get(
    "/api/cpu-load-test/health",
    include_in_schema=False,
)
async def cpu_load_test_health() -> Dict[str, str]:
    """Health check endpoint for CPU load test router."""
    return {
        "status": "healthy",
        "message": "CPU load test endpoints are available",
    }
