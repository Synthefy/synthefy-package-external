from synthefy_pkg.app.data_models import ZeroShotAnomalyRequest, ZeroShotAnomalyResponse

COMPILE = True


class ZeroShotAnomalyService:
    def __init__(self):
        pass

    def zero_shot_anomaly_detection(
        self, request: ZeroShotAnomalyRequest
    ) -> ZeroShotAnomalyResponse:
        """
        inputs:
            request: ZeroShotAnomalyRequest object with data for anomaly detection
        outputs:
            zero_shot_anomaly_response: ZeroShotAnomalyResponse object
        description:
            This function runs zero-shot anomaly detection on the given data
        """
        request = self._preprocess_request(request)  # error handling
        return self._zero_shot_anomaly_detection(request)

    def _preprocess_request(
        self, request: ZeroShotAnomalyRequest
    ) -> ZeroShotAnomalyRequest:
        """
        inputs:
            request: ZeroShotAnomalyRequest object
        outputs:
            zero_shot_anomaly_request: ZeroShotAnomalyRequest object
        description:
            This function converts the request dictionary to a ZeroShotAnomalyRequest object.
            It raises an exception if the format is incorrect or necessary fields are missing.
            It also preprocesses the request to make it easier to use for the model.
        """
        # TODO

    def _zero_shot_anomaly_detection(self, request: ZeroShotAnomalyRequest):
        """
        Run the search on the metadata
        inputs: request - ZeroShotAnomalyRequest object
        outputs: zero_shot_anomaly_response - ZeroShotAnomalyResponse object
        """
        # TODO
