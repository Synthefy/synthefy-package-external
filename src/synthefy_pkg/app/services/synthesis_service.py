import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

from synthefy_pkg.app.config import SynthesisSettings
from synthefy_pkg.app.data_models import (
    MetaData,
    OneContinuousMetaData,
    OneDiscreteMetaData,
    OneTimeSeries,
    SynthefyRequest,
    SynthefyResponse,
    SynthesisRequest,
    SynthesisResponse,
    TimeStamps,
    WindowsSelectionOptions,
)
from synthefy_pkg.app.utils.api_utils import (
    array_to_timeseries,
    convert_list_to_isoformat,
    create_synthefy_response_from_other_types,
    generate_ts_summary,
    metadata_to_dataframe,
)
from synthefy_pkg.experiments.synthesis_experiment import SynthesisExperiment
from synthefy_pkg.preprocessing.preprocess import DataPreprocessor
from synthefy_pkg.utils.scaling_utils import (
    load_continuous_scalers,
    load_discrete_encoders,
    load_timeseries_scalers,
    transform_using_scaler,
)
from synthefy_pkg.utils.synthesis_utils import add_missing_timeseries_columns

COMPILE = True


class SynthesisService:
    dataset_name: str
    settings: SynthesisSettings
    experiment: SynthesisExperiment
    channel_names: List[str] = []  # default [] only for unit test
    window_size: Optional[int] = None
    preprocess_config: Dict[str, Any]
    # TODO fill in full type
    encoders: Optional[Dict[str, Any]] = None
    timeseries_scalers: Optional[Dict[str, Any]] = None

    def __init__(self, dataset_name: str, settings: SynthesisSettings):
        logger.info(f"Initializing SynthesisService with settings: {settings}")
        self.dataset_name = dataset_name
        self.settings = settings
        # open the model
        self.experiment = SynthesisExperiment(
            self.settings.synthesis_config_path
        )

        with open(self.settings.preprocess_config_path, "r") as f:
            preprocess_config = yaml.safe_load(f)
        self.preprocess_config = preprocess_config

        self.channel_names = preprocess_config.get("timeseries", {}).get(
            "cols", []
        )
        if len(self.channel_names) == 0:
            raise ValueError("No channel names found in the preprocess config")
        self.window_size = preprocess_config.get("window_size")
        if self.window_size is None:
            raise ValueError("window_size not found in the preprocess config")
        self.continuous_col_names = json.load(
            open(
                os.path.join(
                    self.settings.dataset_path,
                    self.dataset_name,
                    "continuous_windows_columns.json",
                )
            )
        )
        # get the encoders/scalers
        self.saved_scalers = {
            "timeseries": load_timeseries_scalers(self.dataset_name),
            "continuous": load_continuous_scalers(self.dataset_name),
        }
        self.encoders = load_discrete_encoders(self.dataset_name)

    def update_experiment_with_constraints(
        self, request: SynthefyRequest
    ) -> None:
        """
        Update the experiment with the constraints from the request
        inputs:
            request: SynthefyRequest object
        description:
            Convert synthesis constraints from request into a dictionary format and update experiment
        """
        if not request.synthesis_constraints:
            self.experiment.configuration.dataset_config.use_constraints = False
            return

        constraints_dict = defaultdict(dict)
        for constraint in request.synthesis_constraints.constraints:
            constraints_dict[constraint.channel_name][
                constraint.constraint_name
            ] = constraint.constraint_value

        self.experiment.configuration.dataset_config.use_constraints = True
        self.experiment.configuration.dataset_config.projection_during_synthesis = request.synthesis_constraints.projection_during_synthesis
        self.experiment.configuration.dataset_config.extract_equality_constraints_from_windows = False
        self.experiment.configuration.dataset_config.user_provided_constraints = constraints_dict

    async def get_time_series_synthesis(
        self, request: SynthefyRequest, streaming: bool = False
    ) -> SynthefyResponse:
        """
        inputs:
            request: SynthefyRequest object indicating user's request for synthesis
        outputs:
            synthesis_response: SynthefyResponse object
        description:
        """
        try:
            self.update_experiment_with_constraints(request)
            if (
                request.selected_windows.window_type
                != WindowsSelectionOptions.CURRENT_VIEW_WINDOWS
            ):
                raise ValueError(
                    "You can only synthesize based on the current view windows."
                )

            # for each selected window, run the synthesis
            # TODO - run all at once - remove SynthesisRequest/Response fully
            synthesis_responses: List[SynthesisResponse] = []
            for window_idx in request.selected_windows.window_indices:
                for _ in range(request.n_synthesis_windows):
                    synthesis_request = SynthesisRequest(
                        input_timeseries=request.windows[
                            window_idx
                        ].timeseries_data,
                        metadata=request.windows[window_idx].metadata,
                        timestamps=request.windows[window_idx].timestamps,
                        text=request.text,
                        n_synthesis_windows=request.n_synthesis_windows,
                    )
                    (
                        x_axis_values,
                        continuous_conditions,
                        discrete_conditions,
                        original_discrete_windows,
                    ) = self._preprocess_request(synthesis_request)
                    logger.info("Preprocessed SynthesisRequest")
                    timeseries_preds = self.run_synthetic_data_generation(
                        continuous_conditions, discrete_conditions
                    )
                    logger.info("Generated Synthetic Data")
                    timeseries_preds = transform_using_scaler(
                        windows=timeseries_preds,
                        timeseries_or_continuous="timeseries",
                        original_discrete_windows=original_discrete_windows,
                        dataset_name=self.dataset_name,
                        inverse_transform=True,
                    )
                    logger.info("Unscaled Generated Synthetic Data")
                    synthesis_response = (
                        await self.convert_to_synthesis_response(
                            x_axis_values,
                            timeseries_preds,
                            synthesis_request.input_timeseries,
                            synthesis_request.metadata.discrete_conditions,
                            synthesis_request.metadata.continuous_conditions,
                            synthesis_request.text,
                        )
                    )
                    logger.info("Converted to SynthesisResponse")
                    synthesis_responses.append(synthesis_response)

            return await create_synthefy_response_from_other_types(
                self.dataset_name,
                synthesis_responses,
                streaming=streaming,
                text=request.text,
            )
        finally:
            # Always cleanup after synthesis
            if hasattr(self, "experiment") and hasattr(
                self.experiment, "cleanup"
            ):
                self.experiment.cleanup()

    def _preprocess_request(
        self, request: SynthesisRequest
    ) -> Tuple[TimeStamps, np.ndarray, np.ndarray, np.ndarray]:
        """
        inputs:
            request: SynthesisRequest object
        outputs:
            x_axis_values: TimeStamps object
            continuous_conditions: np.ndarray (num_windows, window_size, num_continuous_features)
            discrete_conditions: np.ndarray (num_windows, window_size, num_discrete_features)
            original_discrete: np.ndarray (num_windows, window_size, num_discrete_features)
        description:
            It raises an exception if the format is incorrect or necessary fields are missing.
            It creates the metadata and time in the desired format for the model.
        """
        df = metadata_to_dataframe(
            metadata=request.metadata,
            timestamps=(
                request.timestamps
                if len(self.preprocess_config.get("timestamps_col", [])) > 0
                else None
            ),
        )
        if self.preprocess_config.get("add_lag_data", False):
            df = add_missing_timeseries_columns(
                df, self.settings.preprocess_config_path
            )
        # no metadata case
        if len(df) == 0:
            if self.window_size is None:
                raise ValueError(
                    "window_size cannot be None when processing empty dataframe"
                )
            continuous_conditions = np.zeros((1, self.window_size, 0))
            discrete_conditions = np.zeros((1, self.window_size, 0))
            original_discrete = np.array([1, self.window_size, 0])
        else:
            preprocessor = DataPreprocessor(
                self.settings.preprocess_config_path
            )
            if not preprocessor.use_label_col_as_discrete_metadata:
                preprocessor.group_labels_cols = []
            # We need only continuous and discrete windows and the data is already
            # grouped, so we can set these 2 empty
            if not self.preprocess_config.get("add_lag_data", False):
                preprocessor.timeseries_cols = []
                preprocessor.timeseries_scalers_info = {}
            preprocessor.process_data(
                df,
                saved_scalers=self.saved_scalers,
                saved_encoders=self.encoders or {},
                save_files_on=False,
            )
            continuous_conditions, discrete_conditions, original_discrete = (
                preprocessor.windows_data_dict["continuous"]["windows"],
                preprocessor.windows_data_dict["discrete"]["windows"],
                preprocessor.windows_data_dict["original_discrete"]["windows"],
            )

        x_axis_values = TimeStamps(
            name=request.timestamps.name if request.timestamps else "index",
            values=(
                request.timestamps.values
                if request.timestamps
                else list(range(self.window_size or 0))
            ),
        )

        # convert x_axis_values to isoformat for JSON serialization
        if request.timestamps:
            x_axis_values.values = (
                convert_list_to_isoformat(x_axis_values.values)
                if not isinstance(request.timestamps.values[0], int)
                and not isinstance(request.timestamps.values[0], float)
                else [int(i) for i in x_axis_values.values]
            )
        return (
            x_axis_values,
            continuous_conditions,
            discrete_conditions,
            original_discrete,
        )

    def run_synthetic_data_generation(
        self, continuous_conditions: np.ndarray, discrete_conditions: np.ndarray
    ):
        """
        Run the synthetic data generation on the metadata
        inputs:
            continuous_conditions: np.ndarray
            discrete_conditions: np.ndarray
        outputs:
            timeseries_preds - np.ndarray (1, num_channels, window_size)
        """
        # location of the trained model
        timeseries_preds = self.experiment.generate_one_synthetic_window(
            model_checkpoint_path=self.settings.synthesis_model_path,
            continuous_conditions=continuous_conditions,
            discrete_conditions=discrete_conditions,
        )
        return timeseries_preds

    async def convert_to_synthesis_response(
        self,
        x_axis_values: TimeStamps,
        timeseries_preds: np.ndarray,
        input_timeseries: List[OneTimeSeries],
        discrete_conditions: List[OneDiscreteMetaData],
        continuous_conditions: List[OneContinuousMetaData],
        text: Optional[str] = "",
    ) -> SynthesisResponse:
        """
        Convert the timeseries prediction to the SynthesisResponse format
        inputs:
            x_axis_values: TimeStamps object
            timeseries_preds: (1, num_channels, window_size) np.ndarray
            discrete_conditions: List[OneDiscreteMetaData] - from the request
            continuous_conditions: List[OneContinuousMetaData] - from the request
        outputs:
            SynthesisResponse object
        """
        timeseries_data = array_to_timeseries(
            timeseries_preds[0].T,
            channel_names=[
                channel_name + "_synthetic"
                for channel_name in self.channel_names
            ],
        )
        if self.settings.show_gt_synthesis_timeseries:
            timeseries_data += input_timeseries

        response = await generate_ts_summary(
            query=text,
            org_ts=[ts.dict() for ts in input_timeseries],
            out_ts=[ts.dict() for ts in timeseries_data],
        )

        return SynthesisResponse(
            x_axis=x_axis_values,
            timeseries_data=timeseries_data,
            metadata=MetaData(
                discrete_conditions=discrete_conditions,
                continuous_conditions=continuous_conditions,
            ),
            text=response,
        )
