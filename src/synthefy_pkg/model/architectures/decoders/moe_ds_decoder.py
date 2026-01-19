from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = kwargs.get("dim", 7168)
        self.topk = kwargs.get("n_activated_experts", 1)
        self.n_groups = kwargs.get("n_expert_groups", 1)
        self.topk_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        self.weight = nn.Parameter(
            torch.empty(
                kwargs.get("n_routed_experts", 1), kwargs.get("dim", 7168)
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(kwargs.get("n_routed_experts", 1)))
            if self.dim == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = torch.nn.functional.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(
                x.size(0), self.n_groups, dtype=torch.bool
            ).scatter_(1, indices, False)
            scores = scores.masked_fill_(
                mask.unsqueeze(-1), float("-inf")
            ).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def update_moe_bias(
    bias_update_rate: float,
    biases: Optional[torch.nn.Parameter],
    assignments: torch.Tensor,
    experts_list: List[int],
):
    """
    Update the biases of the MoE module.
    using the function:
    updated_bias = bias + bias_update_rate * sign(num_assigned - average_assignment)

    Args:
        bias_update_rate (float): The rate at which to update the biases.
        biases (torch.Tensor): The biases of the MoE module.
        assignments (torch.Tensor): The assignments of the MoE module.
        num_experts (int): The number of experts in the MoE module.

    """
    if biases is None:
        return
    average_assignment = torch.prod(torch.tensor(assignments.shape)) / len(
        experts_list
    )
    assignment_per_expert = torch.tensor(
        [(assignments == i).int().sum() for i in experts_list]
    )
    bias_update = bias_update_rate * torch.sign(
        torch.tensor(assignment_per_expert) - average_assignment
    )
    return biases + bias_update


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.
    This only handles Deepseek style MoE usage TODO: will eventually be replaced by MixedMoE, here for reference

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = kwargs.get("dim", 128)
        self.world_size = kwargs.get("world_size", 1)
        self.world_size = (
            dist.get_world_size() if self.world_size == 1 else self.world_size
        )
        assert (
            kwargs.get("n_routed_experts", 1) % kwargs.get("world_size", 1) == 0
        ), (
            f"Number of experts must be divisible by world size (world_size={kwargs.get('world_size', 1)})"
        )
        self.n_routed_experts = kwargs.get("n_routed_experts", 1)
        self.n_local_experts = kwargs.get("n_routed_experts", 1) // kwargs.get(
            "world_size", 1
        )
        self.n_activated_experts = kwargs.get("n_activated_experts", 1)
        self.experts_start_idx = kwargs.get("rank", 0) * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(**kwargs)
        self.experts = nn.ModuleList(
            [
                Expert(kwargs.get("dim", 128), kwargs.get("moe_inter_dim", 128))
                if self.experts_start_idx <= i < self.experts_end_idx
                else None  # type: ignore
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = torch.nn.Sequential(
            torch.nn.Linear(
                kwargs.get("dim", 128),
                kwargs.get("n_shared_experts", 1)
                * kwargs.get("moe_inter_dim", 128),
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                kwargs.get("n_shared_experts", 1)
                * kwargs.get("moe_inter_dim", 128),
                kwargs.get("dim", 128),
            ),
        )
        self.bias_update_rate = kwargs.get("bias_update_rate", 0.01)
        self.training = True

    def train(self, mode: bool = True):
        """
        Set the training mode for the MoE module.
        """
        super().train(mode)
        self.training = mode
        return self

    def eval(self):
        """
        Set the evaluation mode for the MoE module.
        """
        super().eval()
        self.training = False
        return self

    def forward(
        self, embeddings: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            embeddings (torch.Tensor): Embeddings tensor. shape: (batch_size, seq_len, num_correlates, dim)
            x (torch.Tensor): Input tensor. shape: (batch_size, seq_len, num_correlates, 1)

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = embeddings.size()
        x = embeddings.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if self.world_size > 1:
            dist.all_reduce(y)

        # use gradient free bias updates after every forward pass
        if self.training:
            assert isinstance(self.gate.bias, torch.nn.Parameter), (
                "Bias is not initialized"
            )
            update_moe_bias(
                self.bias_update_rate,
                self.gate.bias,
                indices,
                list(range(self.experts_start_idx, self.experts_end_idx)),
            )
        return (y + z).view(shape)


class MixedMoE(nn.Module):
    """
    An MoE module designed to take in both the original inputs and the ouputs of the multivariate embeddings.
    Combines experts that are trained on the original inputs and the outputs of the multivariate embeddings.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(
        self, input_expert_modules, embed_expert_modules, **kwargs: Any
    ):
        """
        Initializes the Mixed MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
            input_expert_modules (list): List of input expert modules.
        """
        super().__init__()
        self.input_expert_modules = input_expert_modules
        self.embed_expert_modules = embed_expert_modules
        self.num_initialized_experts = len(input_expert_modules) + len(
            embed_expert_modules
        )

        self.dim = kwargs.get("dim", 128)
        self.world_size = kwargs.get("world_size", 1)
        self.world_size = (
            dist.get_world_size() if self.world_size == 1 else self.world_size
        )
        assert (
            kwargs.get("n_routed_experts", 1) % kwargs.get("world_size", 1) == 0
        ), (
            f"Number of experts must be divisible by world size (world_size={kwargs.get('world_size', 1)})"
        )

        # the n_local_experts respresents the
        self.n_routed_experts = kwargs.get("n_routed_experts", 1)
        self.n_local_experts = kwargs.get("n_routed_experts", 1) // kwargs.get(
            "world_size", 1
        )
        self.n_activated_experts = kwargs.get("n_activated_experts", 1)
        self.n_input_experts = len(input_expert_modules)
        self.experts_start_idx = kwargs.get("rank", 0) * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(**kwargs)
        self.experts = nn.ModuleList(
            self.input_expert_modules
            + self.embed_expert_modules
            + [
                Expert(kwargs.get("dim", 128), kwargs.get("moe_inter_dim", 128))
                if self.experts_start_idx <= i < self.experts_end_idx
                else None  # type: ignore
                for i in range(
                    self.n_routed_experts - self.num_initialized_experts
                )
            ]
        )
        self.shared_experts = torch.nn.Sequential(
            torch.nn.Linear(
                kwargs.get("dim", 128),
                kwargs.get("n_shared_experts", 1)
                * kwargs.get("moe_inter_dim", 128),
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                kwargs.get("n_shared_experts", 1)
                * kwargs.get("moe_inter_dim", 128),
                kwargs.get("dim", 128),
            ),
        )
        self.bias_update_rate = kwargs.get("bias_update_rate", 0.01)
        self.training = True

    def train(self, mode: bool = True):
        """
        Set the training mode for the MoE module.
        """
        super().train(mode)
        self.training = mode
        return self

    def eval(self):
        """
        Set the evaluation mode for the MoE module.
        """
        super().eval()
        self.training = False
        return self

    def forward(
        self, embeddings: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = embeddings.size()
        embeddings = embeddings.view(-1, self.dim)
        weights, indices = self.gate(embeddings)
        y = torch.zeros_like(embeddings)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()

        # right now, always run the pre-initialized experts
        expert_indices = list(range(self.num_initialized_experts)) + list(
            range(self.experts_start_idx, self.experts_end_idx)
        )

        for i in expert_indices:
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            net_inputs = x[idx] if i < self.n_input_experts else embeddings[idx]
            y[idx] += expert(net_inputs) * weights[idx, top, None]
        z = self.shared_experts(embeddings)
        if self.world_size > 1:
            dist.all_reduce(y)

        # use gradient free bias updates after every forward pass
        if self.training:
            assert isinstance(self.gate.bias, torch.nn.Parameter), (
                "Bias is not initialized"
            )
            update_moe_bias(
                self.bias_update_rate,
                self.gate.bias,
                indices,
                list(range(self.experts_start_idx, self.experts_end_idx)),
            )

        return (y + z).view(shape)
