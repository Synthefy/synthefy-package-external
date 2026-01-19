"""
Parallel neural network modules

This module provides parallel neural network components for training multiple
networks simultaneously. It includes parallel dense layers, layer normalization,
and utility functions for building parallel MLPs.
"""

import math
import numbers
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

"""
Base neural network modules
"""


class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "bnorm":
        return [nn.BatchNorm1d(dim, affine=False)]
    if name == "norm":
        return [Norm()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


def weight_init(m):
    """
    Initialize weights for neural network modules.

    Applies orthogonal initialization to linear layers and parallel dense layers,
    and calls reset_parameters() for modules that have it.

    Args:
        m: A PyTorch module to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = float(nn.init.calculate_gain("relu"))
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain: float = 1):
    """
    Initialize a parallel tensor with orthogonal weights.

    Applies orthogonal initialization to each parallel slice of the tensor
    independently, ensuring that each parallel network has orthogonal weights.

    Args:
        tensor: A 3D tensor of shape (n_parallel, in_features, out_features)
        gain: Scaling factor for the orthogonal initialization

    Returns:
        The initialized tensor

    Raises:
        ValueError: If tensor has fewer than 3 dimensions
    """
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=int(gain))
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


class DenseParallel(nn.Module):
    """
    A parallel dense layer that processes multiple inputs simultaneously.

    This layer can operate in two modes:
    1. Single mode (n_parallel=1): Behaves like a standard nn.Linear layer
    2. Parallel mode (n_parallel>1): Processes n_parallel independent linear transformations

    In parallel mode, the layer applies n_parallel different weight matrices
    to the same input, producing n_parallel outputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        """
        Initialize a parallel dense layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            n_parallel: Number of parallel networks (1 for single mode)
            bias: Whether to include bias terms
            device: Device to place the layer on
            dtype: Data type for the layer parameters
            reset_params: Whether to initialize parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(out_features, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (n_parallel, in_features, out_features), **factory_kwargs
                )
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        """
        Load weights from a list of nn.Linear modules.

        This is useful for converting a list of standard linear layers
        into a single parallel layer.

        Args:
            module_list: List of nn.Linear modules with length n_parallel
        """
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        """
        Initialize the layer parameters using Kaiming uniform initialization.
        """
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Forward pass through the parallel dense layer.

        Args:
            input: Input tensor of shape (batch_size, in_features) in single mode
                   or (n_parallel, batch_size, in_features) in parallel mode

        Returns:
            Output tensor of shape (batch_size, out_features) in single mode
            or (n_parallel, batch_size, out_features) in parallel mode
        """
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.n_parallel,
            self.bias is not None,
        )


class ParallelLayerNorm(nn.Module):
    """
    A parallel layer normalization module.

    This module applies layer normalization across multiple parallel networks.
    It can operate in single mode (n_parallel=1) or parallel mode (n_parallel>1).
    """

    def __init__(
        self,
        normalized_shape,
        n_parallel,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Initialize a parallel layer normalization module.

        Args:
            normalized_shape: Shape of the features to normalize
            n_parallel: Number of parallel networks
            eps: Small constant for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(
                    torch.empty(
                        *[int(x) for x in self.normalized_shape],
                        **factory_kwargs,
                    )
                )
                self.bias = nn.Parameter(
                    torch.empty(
                        *[int(x) for x in self.normalized_shape],
                        **factory_kwargs,
                    )
                )
            else:
                self.weight = nn.Parameter(
                    torch.empty(
                        n_parallel,
                        1,
                        *[int(x) for x in self.normalized_shape],
                        **factory_kwargs,
                    )
                )
                self.bias = nn.Parameter(
                    torch.empty(
                        n_parallel,
                        1,
                        *[int(x) for x in self.normalized_shape],
                        **factory_kwargs,
                    )
                )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the affine parameters (weight=1, bias=0).
        """
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        """
        Load weights from a list of nn.LayerNorm modules.

        Args:
            module_list: List of nn.LayerNorm modules with length n_parallel
        """
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        """
        Forward pass through the parallel layer normalization.

        Args:
            input: Input tensor to normalize

        Returns:
            Normalized tensor with optional affine transformation applied
        """
        norm_input = F.layer_norm(
            input, [int(x) for x in self.normalized_shape], None, None, self.eps
        )
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


def _parallel_nl(name: str, dim: int, n_parallel: int) -> tp.List[nn.Module]:
    """
    Create parallel non-linearity layers.

    Args:
        name: Name of the non-linearity ('irelu', 'relu', 'ntanh', 'layernorm', 'tanh')
        dim: Dimension for layer normalization (if applicable)
        n_parallel: Number of parallel networks

    Returns:
        List of parallel non-linearity modules

    Raises:
        ValueError: If the non-linearity name is not supported
    """
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [
            ParallelLayerNorm(normalized_shape=[dim], n_parallel=n_parallel),
            nn.Tanh(),
        ]
    if name == "layernorm":
        return [ParallelLayerNorm([dim], n_parallel=n_parallel)]
    if name == "tanh":
        return [nn.Tanh()]
    raise ValueError(f"Unknown non-linearity {name}")


def parallel_mlp(
    *layers: tp.Sequence[tp.Union[int, str]], n_parallel: int = 2
) -> nn.Sequential:
    """
    Build a parallel multi-layer perceptron (MLP).

    This function creates a sequence of parallel layers where each layer
    processes n_parallel independent networks simultaneously.

    Args:
        *layers: Alternating sequence of integers (layer dimensions) and strings (non-linearities)
                 The first element must be an integer specifying the input dimension
        n_parallel: Number of parallel networks to create

    Returns:
        A sequential module containing the parallel MLP

    Example:
        # Creates a parallel MLP with input_dim=64, hidden_dim=128, output_dim=32
        # and ReLU activation, with 2 parallel networks
        mlp = parallel_mlp(64, "relu", 128, "relu", 32, n_parallel=2)
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(
                _parallel_nl(layer, prev_dim, n_parallel=n_parallel)
            )
        else:
            assert isinstance(layer, int)
            sequence.append(
                DenseParallel(prev_dim, layer, n_parallel=n_parallel)
            )
            prev_dim = layer
    return nn.Sequential(*sequence)
