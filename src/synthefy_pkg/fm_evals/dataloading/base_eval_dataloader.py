from abc import ABC, abstractmethod

from synthefy_pkg.configs.execution_configurations import Configuration


# This is not actually a pytorch dataloader.
# It's an iterable that returns batches of data.
# Generally speaking, other dataloaders can subclass this as well.
class BaseEvalDataloader(ABC):
    @abstractmethod
    def __init__(self, random_ordering: bool = False):
        self.random_ordering = random_ordering

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
