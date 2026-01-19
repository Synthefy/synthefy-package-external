import os

import lightning as L
from loguru import logger
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


class MLFlowSystemMonitorCallback(L.Callback):
    """
    Callback to monitor system metrics using MLflow SystemMetricsMonitor.

    environment variable MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING is ignored by MLFlowLogger, this is a workaround
    https://github.com/Lightning-AI/pytorch-lightning/issues/20563
    """

    def __init__(
        self,
        config,
        tracking_uri: str = "",
        run_id: str | None = None,
        experiment_name: str = "",
    ):
        """
        Initialize the callback.

        Args:
            config: Configuration object containing MLflow settings
            tracking_uri: MLflow tracking URI
            run_id: Run ID for logging to MLflow
            experiment_name: Name of the MLflow experiment
        """
        super().__init__()
        self.config = config
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.system_monitor = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start system metrics monitoring at the beginning of training."""
        # Only run on the main process (rank 0)
        if not trainer.is_global_zero:
            return

        # Skip if MLflow is not enabled
        if not self.config.use_mlflow:
            logger.info("MLflow system monitoring skipped - MLflow not enabled")
            return

        try:
            # Set tracking URI and experiment if provided
            if self.experiment_name and self.tracking_uri:
                import mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)

            self.system_monitor = SystemMetricsMonitor(
                run_id=self.run_id,
            )
            self.system_monitor.start()
            logger.info("Started MLflow system metrics monitoring")

        except Exception as e:
            logger.exception(f"Failed to start MLflow system metrics monitoring: {e}")

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop system metrics monitoring at the end of training."""
        # Only run on the main process (rank 0)
        if not trainer.is_global_zero:
            return

        if self.system_monitor is not None:
            try:
                self.system_monitor.finish()
                logger.info("Finished MLflow system metrics monitoring")
            except Exception as e:
                logger.exception(f"Failed to finish MLflow system metrics monitoring: {e}")
