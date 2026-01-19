"""
Memory monitoring utility for tracking and limiting memory usage during preprocessing.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil
from loguru import logger

# --- Constants ---
# Unit conversion
BYTES_PER_MB: int = 1024 * 1024
BYTES_PER_FLOAT32: int = 4
CHECK_STEP: int = 5000
DEFAULT_ALLOCATION_SAFETY_FACTOR: float = 1.5

DEFAULT_GROUP_CHECK_FREQUENCY: int = 10  # Check every N groups
LARGE_NUMBER_OF_WINDOW_THRESHOLD: int = (
    10000  # Windows threshold for more frequent checks
)
COARSE_CHECK_STEP: int = 10000  # Step when dataset is large


@dataclass
class MemoryThresholds:
    """Memory thresholds configuration"""

    # Process memory limits (in MB)
    process_memory_mb: float
    # System memory usage limits (as percentage)
    system_memory_percent: float
    # Check interval (in seconds)
    check_interval: float


class MemoryLimitError(Exception):
    """Custom exception for memory limit exceeded"""

    pass


class MemoryMonitor:
    """
    Memory monitoring utility that tracks process and system memory usage
    and raises errors when thresholds are exceeded.
    """

    def __init__(self, thresholds: MemoryThresholds):
        self.thresholds = thresholds
        self.process = psutil.Process(os.getpid())
        self.last_check_time = 0.0
        self.peak_memory_mb = 0.0
        self.monitoring_enabled = True
        # Overhead metrics
        self._metrics = {
            "check_calls": 0,
            "check_time_s": 0.0,  # time spent inside check_memory_limits
            "ensure_calls": 0,
            "ensure_time_s": 0.0,  # time spent inside ensure_allocation_possible
            # get_current_memory_usage that are called directly (outside check/ensure)
            "misc_get_usage_calls": 0,
            "misc_get_usage_time_s": 0.0,
        }

    @staticmethod
    def should_check(now_ts: float, last_ts: float, interval_s: float) -> bool:
        """Time-based gate for memory checks to avoid excessive polling."""
        return (now_ts - last_ts) >= interval_s

    def _collect_current_memory_usage(self) -> Dict[str, float]:
        """Internal helper that collects memory stats without recording metrics."""
        process_memory = self.process.memory_info()
        process_memory_mb = process_memory.rss / BYTES_PER_MB

        system_memory = psutil.virtual_memory()
        system_memory_percent = system_memory.percent
        system_available_mb = system_memory.available / BYTES_PER_MB

        self.peak_memory_mb = max(self.peak_memory_mb, process_memory_mb)

        return {
            "process_memory_mb": process_memory_mb,
            "system_memory_percent": system_memory_percent,
            "system_available_mb": system_available_mb,
            "peak_memory_mb": self.peak_memory_mb,
        }

    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics (records metrics for direct calls)."""
        start_ts = time.perf_counter()
        stats = self._collect_current_memory_usage()
        self._metrics["misc_get_usage_calls"] += 1
        self._metrics["misc_get_usage_time_s"] += time.perf_counter() - start_ts
        return stats

    def check_memory_limits(
        self, operation_name: str = "Unknown"
    ) -> Dict[str, float]:
        """
        Check if memory usage is within limits.
        Raises MemoryError if limits are exceeded.
        """
        if not self.monitoring_enabled:
            return {}

        check_start = time.perf_counter()
        current_time = time.time()

        # Only check if enough time has passed since last check
        if current_time - self.last_check_time < self.thresholds.check_interval:
            return {}

        self.last_check_time = current_time

        try:
            # Use internal collector to avoid double-counting misc get_usage metrics
            memory_stats = self._collect_current_memory_usage()

            # Log current memory usage (lazy formatting to avoid string assembly cost when disabled)
            logger.debug(
                "Memory check during {}: Process: {:.1f}MB, System: {:.1f}% (Available: {:.1f}MB), Peak: {:.1f}MB",
                operation_name,
                memory_stats["process_memory_mb"],
                memory_stats["system_memory_percent"],
                memory_stats["system_available_mb"],
                memory_stats["peak_memory_mb"],
            )

            # Check process memory limit
            if (
                memory_stats["process_memory_mb"]
                > self.thresholds.process_memory_mb
            ):
                error_msg = (
                    f"Process memory limit exceeded during {operation_name}: "
                    f"{memory_stats['process_memory_mb']:.1f}MB > "
                    f"{self.thresholds.process_memory_mb:.1f}MB limit"
                )
                logger.error(error_msg)
                raise MemoryLimitError(error_msg)

            # Check system memory limit
            if (
                memory_stats["system_memory_percent"]
                > self.thresholds.system_memory_percent
            ):
                error_msg = (
                    f"System memory limit exceeded during {operation_name}: "
                    f"{memory_stats['system_memory_percent']:.1f}% > "
                    f"{self.thresholds.system_memory_percent:.1f}% limit"
                )
                logger.error(error_msg)
                raise MemoryLimitError(error_msg)

            return memory_stats

        except MemoryLimitError:
            # Re-raise MemoryError - don't catch it here
            raise
        except psutil.NoSuchProcess:
            logger.warning("Process no longer exists during memory check")
            return {}
        except Exception as e:
            logger.warning(f"Error during memory check: {e}")
            return {}
        finally:
            self._metrics["check_calls"] += 1
            self._metrics["check_time_s"] += time.perf_counter() - check_start

    def ensure_allocation_possible(
        self,
        required_mb: float,
        operation_name: str = "Unknown",
        safety_factor: float = 1.0,
    ) -> None:
        """Best-effort early check before large allocations.

        Raises MemoryLimitError if projected usage is likely to exceed configured limits.
        """
        ensure_start = time.perf_counter()
        try:
            # Use internal collector to avoid double-counting misc get_usage metrics
            memory_stats = self._collect_current_memory_usage()

            projected_process_mb = (
                memory_stats["process_memory_mb"] + required_mb * safety_factor
            )
            if projected_process_mb > self.thresholds.process_memory_mb:
                raise MemoryLimitError(
                    (
                        f"Projected process memory would exceed limit during {operation_name}: "
                        f"{projected_process_mb:.1f}MB > {self.thresholds.process_memory_mb:.1f}MB"
                    )
                )

            # System availability check
            vm = psutil.virtual_memory()
            total_mb = vm.total / BYTES_PER_MB
            used_mb = total_mb - (vm.available / BYTES_PER_MB)
            projected_used_percent = (
                (used_mb + required_mb * safety_factor) / total_mb
            ) * 100.0
            if projected_used_percent > self.thresholds.system_memory_percent:
                raise MemoryLimitError(
                    (
                        f"Projected system memory usage would exceed limit during {operation_name}: "
                        f"{projected_used_percent:.1f}% > {self.thresholds.system_memory_percent:.1f}%"
                    )
                )
        except MemoryLimitError:
            raise
        except Exception as e:
            # Be conservative: if we can't reliably check, log and continue without blocking
            logger.debug(
                f"Allocation feasibility check skipped due to error: {e}"
            )
        finally:
            self._metrics["ensure_calls"] += 1
            self._metrics["ensure_time_s"] += time.perf_counter() - ensure_start

    def log_memory_summary(self, operation_name: str = "Operation"):
        """Log a summary of memory usage"""
        try:
            memory_stats = self.get_current_memory_usage()
            logger.info(
                f"Memory summary for {operation_name}: "
                f"Current: {memory_stats['process_memory_mb']:.1f}MB, "
                f"Peak: {memory_stats['peak_memory_mb']:.1f}MB, "
                f"System usage: {memory_stats['system_memory_percent']:.1f}%"
            )
        except Exception as e:
            logger.warning(f"Error logging memory summary: {e}")

    def reset_metrics(self) -> None:
        """Reset recorded overhead metrics."""
        self._metrics.update(
            {
                "check_calls": 0,
                "check_time_s": 0.0,
                "ensure_calls": 0,
                "ensure_time_s": 0.0,
                "misc_get_usage_calls": 0,
                "misc_get_usage_time_s": 0.0,
            }
        )

    def log_overhead_summary(
        self, total_runtime_s: Optional[float] = None
    ) -> None:
        """Log a concise summary of memory-monitoring overhead.

        Args:
            total_runtime_s: Optional total runtime to compute percentage overhead.
        """
        try:
            check_ms = self._metrics["check_time_s"] * 1000.0
            ensure_ms = self._metrics["ensure_time_s"] * 1000.0
            misc_ms = self._metrics["misc_get_usage_time_s"] * 1000.0
            total_ms = check_ms + ensure_ms + misc_ms
            pct = (
                (total_ms / (total_runtime_s * 1000.0)) * 100.0
                if total_runtime_s and total_runtime_s > 0
                else None
            )

            pct_text = f" ({pct:.2f}% of runtime)" if pct is not None else ""
            logger.info(
                "Memory monitoring overhead: total={:.1f}ms{} | checks={} ({:.1f}ms) | ensures={} ({:.1f}ms) | direct_get_usage={} ({:.1f}ms)",
                total_ms,
                pct_text,
                self._metrics["check_calls"],
                check_ms,
                self._metrics["ensure_calls"],
                ensure_ms,
                self._metrics["misc_get_usage_calls"],
                misc_ms,
            )
        except Exception as e:
            logger.debug(f"Failed to log overhead summary: {e}")

    def disable_monitoring(self):
        """Disable memory monitoring"""
        self.monitoring_enabled = False
        logger.info("Memory monitoring disabled")

    def enable_monitoring(self):
        """Enable memory monitoring"""
        self.monitoring_enabled = True
        logger.info("Memory monitoring enabled")

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory usage during an operation"""
        logger.info(f"Starting memory monitoring for: {operation_name}")
        start_memory = self.get_current_memory_usage()

        try:
            yield self
        finally:
            end_memory = self.get_current_memory_usage()
            memory_delta = (
                end_memory["process_memory_mb"]
                - start_memory["process_memory_mb"]
            )

            logger.info(
                f"Memory monitoring completed for {operation_name}: "
                f"Delta: {memory_delta:+.1f}MB, "
                f"Peak: {end_memory['peak_memory_mb']:.1f}MB"
            )


def generate_memory_error_guidance(
    config: Dict[str, Any], error_message: str
) -> str:
    """
    Generate user-friendly guidance for memory errors based on preprocessing config.

    Args:
        config: Preprocessing configuration dictionary
        error_message: Original memory error message

    Returns:
        User-friendly error message with specific recommendations
    """
    # Extract current values
    window_size = config["window_size"]
    current_stride = config["stride"]

    # Recommendations in non-technical language
    recommendations = []

    # Suggest creating less data by increasing stride
    suggested_stride = current_stride * 2
    recommendations.append(
        f"Increase stride from {current_stride} to {suggested_stride} to create less data during preprocessing"
    )

    # Suggest creating smaller windows
    suggested_window_size = max(1, window_size // 2)
    recommendations.append(
        f"Reduce window_size from {window_size} to {suggested_window_size} to create smaller windows"
    )

    base_message = "Memory limit exceeded while preparing your data."
    issue_type = "Your job used more memory than allowed, which can happen with large files or detailed settings."
    recommendations_text = "\n".join([f"  • {rec}" for rec in recommendations])

    return (
        f"{base_message}\n\n"
        f"🔍 What happened: {issue_type}\n\n"
        f"💡 How to fix:\n{recommendations_text}\n\n"
        f"🛠️ Technical details: {error_message}"
    )
