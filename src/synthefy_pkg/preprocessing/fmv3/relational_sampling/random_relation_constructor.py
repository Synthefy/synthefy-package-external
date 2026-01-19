import numpy as np
from loguru import logger

from synthefy_pkg.preprocessing.fmv3.relational_sampling.base_relation_constructor import (
    BaseRelationConstructor,
)


class RandomRelationConstructor(BaseRelationConstructor):
    """
    Random relation constructor.

    Assigns each sample to a random class, and allows
    any class to sample correlates from any other class.
    """

    def __init__(self, data_dir: str, num_classes: int):
        super().__init__(data_dir, "random")
        self.num_classes = num_classes

    def construct_relations(self) -> None:
        total_samples = self.get_total_samples()

        classes = np.random.randint(0, self.num_classes, total_samples)
        relational_matrix = (
            np.ones((self.num_classes, self.num_classes)) / self.num_classes
        )

        self.save_numpy_files(classes, relational_matrix)
        logger.info(
            f"Saved {self.num_classes} classes and relational matrix to {self.relational_sampling_dir}"
        )
