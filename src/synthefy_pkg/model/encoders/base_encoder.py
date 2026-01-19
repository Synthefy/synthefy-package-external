COMPILE = True


class BaseEncoder:
    """Base class for all encoders."""

    def __init__(self, config) -> None:
        self.config = config

    def encode(self, x):
        raise NotImplementedError
