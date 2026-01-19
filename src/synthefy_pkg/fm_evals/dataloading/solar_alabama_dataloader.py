import random

from synthefy_pkg.fm_evals.dataloading.traffic_pems_dataloader import (
    TrafficPEMSDataloader,
)


class SolarAlabamaDataloader(TrafficPEMSDataloader):
    """Solar Alabama dataloader with identical functionality to TrafficPEMSDataloader."""

    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering
        super().__init__(random_ordering=random_ordering)
        self.data_location = "s3://synthefy-fm-eval-datasets/solar_alabama/"
        self.csv_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.csv_files)
