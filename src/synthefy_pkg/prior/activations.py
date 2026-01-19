from __future__ import annotations

import copy
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class StdScaleLayer(nn.Module):
    """Standard scaling layer that normalizes input features.

    Computes mean and standard deviation on the first batch and uses these
    statistics to normalize subsequent inputs using (x - mean) / std.
    The statistics are computed along dimension 0.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fit the info on the first batch
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6

        return (x - self.mean) / self.std


class SignActivation(nn.Module):
    """Sign function as an activation layer.

    Returns 1.0 for inputs >= 0, and -1.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x >= 0.0).float() - 1.0


class RandomCategoricalStepActivation(nn.Module):
    """categorical step activation layer.

    Returns a step function of the input with discrete levels.
    Uses balanced cutoffs to ensure relatively uniform distribution of step assignments.
    """

    def __init__(self, max_n_levels: int = 10, random_hash=False):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.random_hash = random_hash
        n_levels = np.random.randint(2, max_n_levels + 1)
        self.n_levels = n_levels  # Store as instance variable
        # Create levels from 0 to n_levels-1
        self.levels = torch.arange(n_levels, dtype=torch.float32)

        # Create cutoffs that balance the rate at which different steps are assigned
        # Using quantiles of a standard normal distribution to create balanced cutoffs
        # This ensures each level has roughly equal probability of being selected
        # Direct computation using inverse error function (erfinv) for efficiency
        quantiles = torch.linspace(0, 1, n_levels + 1)
        # Add perturbation to make distribution less uniform
        # Perturbation is proportional to the spacing between quantiles
        spacing = 1.0 / n_levels
        perturbation = (
            torch.randn(n_levels + 1) * spacing * 0.3
        )  # 30% of spacing as noise
        # Keep endpoints fixed at 0 and 1
        perturbation[0] = 0.0
        perturbation[-1] = 0.0
        quantiles = torch.clamp(quantiles + perturbation, 0, 1)
        # Convert quantiles to z-scores using inverse normal CDF
        # For normal distribution: z = sqrt(2) * erfinv(2*p - 1)
        self.cutoffs = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(
            2.0 * quantiles - 1.0
        )
        # Register cutoffs as buffer so they move to the correct device
        self.register_buffer("cutoffs_buffer", self.cutoffs)

        if self.random_hash:
            self.hash_values = torch.rand((self.n_levels,)) * 6 - 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize the input to have mean 0 and std 1
        x_std = (x - x.mean()) / (x.std() + 1e-6)

        # Find which cutoff bin each input falls into
        # Use torch.bucketize to efficiently assign inputs to levels
        level_indices = torch.bucketize(x_std, self.cutoffs_buffer) - 1
        # Clamp to valid range [0, n_levels-1]
        level_indices = torch.clamp(level_indices, 0, self.n_levels - 1)

        # Return the corresponding levels
        if self.random_hash:
            return self.hash_values[level_indices]
        else:
            return level_indices


class Heaviside(nn.Module):
    """Heaviside function as an activation layer.

    Returns 1.0 for inputs >= 0, and 0.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0.0).float()


class RBFActivation(nn.Module):
    """Radial Basis Function (RBF) activation layer.

    Implements the Gaussian RBF: f(x) = exp(-x^2)
    Useful for localized feature representations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


class RandomFreqSineActivation(nn.Module):
    """Random frequency sine activation with fixed random scale and bias.

    Applies sine activation with randomly initialized (but fixed) frequency scaling and phase shift:
    f(x) = sin(scale * standardize(x) + bias)

    The scale and bias parameters are initialized randomly but remain constant during training
    (requires_grad=False).

    Args:
        min_scale (float): Minimum value for random frequency scaling (default: 0.1)
        max_scale (float): Maximum value for random frequency scaling (default: 100)
    """

    def __init__(self, min_scale=0.1, max_scale=100):
        super().__init__()
        log_min_scale = np.log(min_scale)
        log_max_scale = np.log(max_scale)
        self.scale = nn.Parameter(
            torch.exp(
                log_min_scale + (log_max_scale - log_min_scale) * torch.rand(1)
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(2 * np.pi * torch.rand(1), requires_grad=False)
        self.stdscaler = StdScaleLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.scale * self.stdscaler(x) + self.bias)


class RandomFunctionActivation(nn.Module):
    """Random Fourier feature based activation function.

    Generates a random periodic function by combining multiple sine waves with
    different frequencies, phases and weights. The input is first standardized.

    Args:
        n_frequencies (int): Number of frequency components to use (default: 256)
    """

    def __init__(self, n_frequencies: int = 256, frequency_min: float = 1.0):
        super().__init__()

        self.freqs = nn.Parameter(
            (n_frequencies * torch.rand(n_frequencies)) + frequency_min,
            requires_grad=False,  # TODO: implies frequencies selected between 0 and num_frequencies hertz
            # n_frequencies * torch.min(torch.abs(torch.randn(n_frequencies)) / (n_frequencies * 10), torch.ones(n_frequencies)), requires_grad=False # TODO: now frequencies with longer periods are biased
        )
        self.bias = nn.Parameter(
            2 * np.pi * torch.rand(n_frequencies), requires_grad=False
        )
        self.stdscaler = StdScaleLayer()

        decay_exponent = -np.exp(np.random.uniform(np.log(1.8), np.log(2.4)))
        with torch.no_grad():
            freq_factors = self.freqs**decay_exponent
            freq_factors = freq_factors / (freq_factors**2).sum().sqrt()
        # Sort frequencies in-place and get sorted indices
        sorted_freqs, sorted_indices = torch.sort(self.freqs)
        self.freqs.data = sorted_freqs
        self.l2_weights = nn.Parameter(
            freq_factors.sort(descending=True)[0]
            * (torch.rand(n_frequencies) - 0.5)
            * 2,
            requires_grad=False,
        )
        print(self.freqs, freq_factors.sort()[0], self.l2_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stdscaler(x)
        x = torch.sin(self.freqs * x[..., None] + self.bias)
        x = (self.l2_weights * x).sum(dim=-1)
        return x


class FunctionActivation(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class RandomScaleLayer(nn.Module):
    """Random scaling layer with optional per-feature parameters.

    Applies random scaling and bias: f(x) = scale * (x + bias)

    Args:
        individual (bool, optional): If True, uses different parameters for each
            input feature. Defaults to False.
    """

    def __init__(self, individual: bool = False):
        super().__init__()
        self.individual = individual
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        n_out = x.shape[-1] if self.individual else 1
        # self.scale = torch.exp(
        #     np.log(1.0) + 2 * torch.randn(1, n_out, device=x.device)
        # )
        self.scale = torch.exp(2 * torch.randn(1, n_out, device=x.device))
        # use uniform on [0, 1] since we round to integers anyway
        self.bias = torch.randn(1, n_out, device=x.device)
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialize(x)

        # print("scale and bias", self.scale, self.bias)

        return self.scale * (x + self.bias)


class ExpActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / (torch.std(x) * 2)
        retv = torch.exp(x)
        retv = torch.clamp(retv, max=90)
        return retv


class SqrtAbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(x))


class UnitIntervalIndicator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.abs(x) <= 1.0).float()


class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / (torch.std(x))
        return x**2


class AbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class IdentityFactory:
    def __init__(self, act_class: nn.Module, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(
            self.act_class(),
        )


class StdRandomScaleFactory:
    def __init__(self, act_class, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(
            StdScaleLayer(),
            RandomScaleLayer(individual=self.individual),
            self.act_class(),
        )


class RandomChoiceActivation(nn.Module):
    """Randomly selects and instantiates one activation function from a list.

    Args:
        act_list: List of activation function constructors to choose from.

    Attributes:
        act: The randomly selected activation function instance
    """

    def __init__(self, act_list: List[nn.Module]):
        super().__init__()
        selected_index = np.random.randint(len(act_list))
        self.act = act_list[selected_index]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xout = self.act(x)
        # print(get_activation_name(self.act), x[:5,0], xout[:5,0])
        return xout


class RandomChoiceFactory:
    """Factory class for creating RandomChoice activations"""

    def __init__(self, act_classes):
        self.act_classes = act_classes

    def __call__(self):
        return RandomChoiceActivation(self.act_classes)


def get_activations(
    random: bool = True, scale: bool = True, diverse: bool = True
):
    """Generate a list of activation functions with various configurations.

    This function creates a list of activation functions by combining simple activations
    with optional random functions, scaling, and diversity options.

    Args:
        random: If True, adds RandomFunctionActivation to the list and samples it multiple
            times to increase probability of selection. Defaults to True.

        scale: If True, wraps activations with StdRandomScaleFactory to add standardization
            and random scaling. Defaults to True.

        diverse: If True, adds RandomChoiceFactory instances to allow different activation
            functions in each layer. Defaults to True.
    """
    # Start with a set of simple activations
    simple_activations = [
        nn.Tanh,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.SELU,
        nn.SiLU,
        nn.ReLU,
        nn.Softplus,
        nn.ReLU6,
        nn.Hardtanh,
        SignActivation,
        RBFActivation,
        ExpActivation,
        SqrtAbsActivation,
        UnitIntervalIndicator,
        SineActivation,
        SquareActivation,
        AbsActivation,
        # RandomFunctionActivation,
    ]
    activations = simple_activations
    # if random:
    #     # Add random activation and sample it more often
    #     activations += [RandomFunctionActivation] * 10

    if scale:
        # Create scaled versions using StdRandomScaleFactory
        activations = [StdRandomScaleFactory(act) for act in activations]

    if diverse:
        # Add possibility to have different activation functions in each layer
        base_activations = copy.deepcopy(activations)
        activations += [RandomChoiceFactory(base_activations)] * len(
            base_activations
        )

    # activations = [RandomChoiceFactory(activations)]

    return activations


def get_activation_name(activation: nn.Module):
    if isinstance(activation, nn.Tanh):
        return "tanh"
    elif isinstance(activation, nn.LeakyReLU):
        return "leaky_relu"
    elif isinstance(activation, nn.ELU):
        return "elu"
    elif isinstance(activation, nn.Identity):
        return "identity"
    elif isinstance(activation, nn.SELU):
        return "selu"
    elif isinstance(activation, nn.SiLU):
        return "silu"
    elif isinstance(activation, nn.ReLU):
        return "relu"
    elif isinstance(activation, nn.Softplus):
        return "softplus"
    elif isinstance(activation, nn.ReLU6):
        return "relu6"
    elif isinstance(activation, nn.Hardtanh):
        return "hardtanh"
    elif isinstance(activation, SignActivation):
        return "sign"
    elif isinstance(activation, RBFActivation):
        return "rbf"
    elif isinstance(activation, ExpActivation):
        return "exp"
    elif isinstance(activation, SqrtAbsActivation):
        return "sqrt_abs"
    elif isinstance(activation, UnitIntervalIndicator):
        return "unit_interval_indicator"
    elif isinstance(activation, SineActivation):
        return "sine"
    elif isinstance(activation, SquareActivation):
        return "square"
    elif isinstance(activation, AbsActivation):
        return "abs"
    elif isinstance(activation, RandomFunctionActivation):
        freqs = "_".join([str(i) for i in activation.freqs.tolist()])[:5]
        bias = "_".join([str(i) for i in activation.bias.tolist()])[:5]
        l2_weights = "_".join([str(i) for i in activation.l2_weights.tolist()])[
            :5
        ]
        return f"random_function_f{freqs}_b{bias}_w{l2_weights}"
    elif isinstance(activation, StdScaleLayer):
        return "std_scale"
    elif isinstance(activation, RandomScaleLayer):
        return "random_scale"
    elif isinstance(activation, RandomChoiceActivation):
        return get_activation_name(activation.act)
    else:
        if isinstance(activation, nn.Sequential):
            return get_activation_name(activation[-1])
        raise ValueError(f"Unknown activation function: {activation}")


def get_activations_by_name(
    name: str,
    random: bool = False,
    scale: bool = False,
    diverse: bool = False,
    diverse_names: list[str] = [],
):
    """Generate a list of activation functions with various configurations.

    This function creates a list of activation functions by combining simple activations
    with optional random functions, scaling, and diversity options.

    Args:
        random: If True, adds RandomFunctionActivation to the list and samples it multiple
            times to increase probability of selection. Defaults to True.

        scale: If True, wraps activations with StdRandomScaleFactory to add standardization
            and random scaling. Defaults to True.

        diverse: If True, adds RandomChoiceFactory instances to allow different activation
            functions in each layer. Defaults to True.
    """
    # Start with a set of simple activations
    activations = {
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "identity": nn.Identity,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "relu6": nn.ReLU6,
        "hardtanh": nn.Hardtanh,
        "sign": SignActivation,
        "rbf": RBFActivation,
        "exp": ExpActivation,
        "sqrt_abs": SqrtAbsActivation,
        "unit_interval_indicator": UnitIntervalIndicator,
        "sine": SineActivation,
        "square": SquareActivation,
        "abs": AbsActivation,
    }
    if random:
        activations["random"] = RandomFunctionActivation
    if diverse:
        activations["diverse"] = RandomChoiceFactory(
            [activations[d] for d in diverse_names]
        )
    if scale:
        return StdRandomScaleFactory(activations[name])
    return IdentityFactory(activations[name])
