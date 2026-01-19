import os

from lightning import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_only
from loguru import logger


class LogFileArtifactCallback(Callback):
    """Callback to log log files as MLflow artifacts every n steps."""

    def __init__(
        self,
        config,
        log_every_n_steps: int = 100,
        tracking_uri: str = "",
        run_id: str | None = None,
        experiment_name: str = "",
    ):
        """
        Initialize the callback.

        Args:
            config: Configuration object containing MLflow settings and log directory
            log_every_n_steps: How often to log artifacts (default: every 100 steps)
            tracking_uri: MLflow tracking URI
            run_id: Run ID for logging to MLflow
            experiment_name: Name of the MLflow experiment
        """
        super().__init__()
        self.config = config
        self.log_every_n_steps = log_every_n_steps
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.experiment_name = experiment_name

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log log files as artifacts every n training steps."""
        # Skip if not on the right step interval, but log in the first step
        if trainer.global_step > 1 and trainer.global_step % self.log_every_n_steps != 0:
            return

        # Skip if MLflow is not enabled
        if not self.config.use_mlflow:
            return

        try:
            import mlflow

            # Get the log directory
            log_dir = self.config.get_log_dir()

            # Define the log files to log
            log_files = [
                f"run_logs_{self.config.training_config.logger_level}.log",
                "run_logs_TRACE.log",
            ]

            # Log each log file if it exists
            for log_file in log_files:
                log_file_path = os.path.join(log_dir, log_file)
                if os.path.exists(log_file_path):
                    # Set tracking URI and experiment if provided
                    if self.experiment_name and self.tracking_uri:
                        mlflow.set_tracking_uri(self.tracking_uri)
                        mlflow.set_experiment(self.experiment_name)
                    mlflow.log_artifact(
                        log_file_path,
                        "logs",
                        run_id=self.run_id,
                    )
                    logger.trace(
                        f"Logged {log_file} as MLflow artifact for step {trainer.global_step}"
                    )
                else:
                    logger.warning(
                        f"Log file {log_file} not found at {log_file_path}, could not log to MLFlow"
                    )

        except Exception as e:
            logger.exception(
                f"Failed to log log files as MLflow artifacts for step {trainer.global_step}: {e}"
            )
