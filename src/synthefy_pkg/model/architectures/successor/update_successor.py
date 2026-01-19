import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthefy_pkg.model.architectures.successor.parallel_modules import (
    mlp,
    parallel_mlp,
)


class BackwardMap(nn.Module):
    def __init__(
        self,
        goal_dim,
        z_dim,
        hidden_dim,
        hidden_layers: int = 2,
        norm=True,
        batch_norm=False,
    ) -> None:
        super().__init__()
        seq = [goal_dim, hidden_dim, "ntanh"]
        for _ in range(hidden_layers - 1):
            seq += [hidden_dim, "relu"]
        seq += [z_dim]
        if norm:
            seq += ["norm"]
        if batch_norm:
            seq += ["bnorm"]
        self.net = mlp(*seq)

    def forward(self, x):
        return self.net(x)


def build_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [input_dim, hidden_dim, "ntanh"]
    for _ in range(hidden_layers - 2):
        seq += [hidden_dim, "relu"]
    seq += [hidden_dim // 2, "relu"]
    if num_parallel == 1:
        return mlp(*seq)
    return parallel_mlp(*seq, n_parallel=num_parallel)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        hidden_dim,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
        num_parallel: int = 2,
        output_dim=None,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        self.embed_z = build_embedding(
            obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel
        )
        self.embed_sa = build_embedding(
            obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel
        )

        seq = (
            [hidden_dim]
            + [hidden_dim, "relu"] * hidden_layers
            + [output_dim if output_dim else z_dim]
        )
        if num_parallel == 1:
            self.Fs = mlp(*seq)
        else:
            self.Fs = parallel_mlp(*seq, n_parallel=num_parallel)

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        action: tp.Optional[torch.Tensor] = None,
    ):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            if action is not None:
                action = action.expand(self.num_parallel, -1, -1)
        # print(obs.shape)
        # print(z.shape)
        # print(torch.cat([obs, z], dim=-1).shape)
        z_embedding = self.embed_z(
            torch.cat([obs, z], dim=-1)
        )  # num_parallel x bs x h_dim // 2
        if action is not None:
            sa_embedding = self.embed_sa(
                torch.cat([obs, action], dim=-1)
            )  # num_parallel x bs x h_dim // 2
        else:
            sa_embedding = self.embed_sa(obs)  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SamplingSeedActor(nn.Module):
    def __init__(self, action_dim, z_dim, batch_size):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.powers = (
            torch.tensor([2**i for i in range(self.z_dim)][::-1])
            .to("cuda")
            .repeat(batch_size, 1)
        )
        self.max_seed = 2**z_dim + 20000
        self.seed_to_action = []

        for i in range(self.max_seed):
            torch.random.manual_seed(i)
            action = (torch.rand(size=(self.action_dim,)) * 2 - 1).unsqueeze(0)
            self.seed_to_action.append(action)
        self.seed_to_action = np.array(self.seed_to_action).squeeze()

    def forward(self, obs_hash, z):
        actions = []
        seed_long = (z * self.powers).sum(1)
        final_seed = (seed_long + obs_hash.reshape(-1)) % self.max_seed
        actions = self.seed_to_action[final_seed.cpu().numpy().astype(np.int32)]
        return torch.FloatTensor(actions).to("cuda")


def get_targets_uncertainty(preds, dim=0, num_parallel_scaling=2):
    preds_mean = preds.mean(dim=dim)
    preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
    preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
    preds_diffs = torch.abs(
        preds_uns - preds_uns2
    )  # n_parallel x n_parallel x ...
    preds_unc = (
        preds_diffs.sum(
            dim=(dim, dim + 1),
        )
        / num_parallel_scaling
    )
    return preds_mean, preds_unc


# @torch.compile(**COMPILE_OPTIONS)
def _update_psm(
    obs: torch.Tensor,
    counterfactual_embedding: torch.Tensor,
    discount: torch.Tensor,
    next_obs: torch.Tensor,
    future_embedding: torch.Tensor,
    dynamics_embedding: torch.Tensor,
    fb_pessimism_penalty: float,
    stddev: float,
    stddev_clip: float,
    ortho_coef: float,
    clip_grad_norm: float,
    num_parallel: int,
    target_counterfactual_model: nn.Module,
    target_forward_model: nn.Module,
    target_backward_model: nn.Module,
    forward_model: nn.Module,
    backward_model: nn.Module,
) -> list[torch.Tensor]:
    # compute target successor measure

    # get the off diagonal values (TODO: may need to move out of this function for efficiency)
    off_diag = 1 - torch.eye(obs.shape[0], obs.shape[0], device=obs.device)
    off_diag_sum = off_diag.sum()

    with torch.no_grad():
        next_counterfactual = target_counterfactual_model(
            next_obs, dynamics_embedding
        )
        target_forward_linear, _ = target_forward_model(
            next_obs, dynamics_embedding, next_counterfactual
        )  # num_parallel x batch x z_dim
        target_backward_linear = target_backward_model(
            future_embedding
        )  # batch x z_dim
        target_Ms = torch.matmul(
            target_forward_linear, target_backward_linear.T
        )  # num_parallel x batch x batch
        target_M_mean, target_M_unc = get_targets_uncertainty(
            target_Ms, num_parallel_scaling=num_parallel
        )  # batch x batch
        target_M = (
            target_M_mean - fb_pessimism_penalty * target_M_unc
        )  # batch x batch
        target_M = target_M.unsqueeze(0).expand(num_parallel, -1, -1)

    # compute FB loss
    Fs, ws = forward_model(
        obs, dynamics_embedding, counterfactual_embedding
    )  # num_parallel x batch x z_dim
    B = backward_model(future_embedding)  # batch x z_dim
    Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

    diff = Ms - discount * target_M  # num_parallel x batch x batch
    fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
    fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
    fb_loss = fb_offdiag + fb_diag

    # compute orthonormality loss for backward embedding
    Cov = torch.matmul(B, B.T)
    orth_loss_diag = -Cov.diag().mean()
    orth_loss_offdiag = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag_sum
    orth_loss = orth_loss_offdiag + orth_loss_diag
    fb_loss += 1 * orth_loss

    # compute orthonormality loss for w
    Cov_w0 = torch.matmul(ws[0], ws[0].T)
    Cov_w1 = torch.matmul(ws[1], ws[1].T)
    orth_loss_diag_w = -Cov_w0.diag().mean() - Cov_w1.diag().mean()
    orth_loss_offdiag_w = (
        0.5 * (Cov_w0 * off_diag).pow(2).sum() / off_diag_sum
        + 0.5 * (Cov_w1 * off_diag).pow(2).sum() / off_diag_sum
    )
    orth_loss_w = orth_loss_offdiag_w + orth_loss_diag_w
    fb_loss += 1 * orth_loss_w

    # # optimize FB
    # optim_f_psm.zero_grad(set_to_none=True)
    # optim_b.zero_grad(set_to_none=True)

    # fb_loss.backward()
    # if clip_grad_norm > 0:
    #     torch.nn.utils.clip_grad_norm_(self.forward.parameters(), clip_grad_norm)
    #     torch.nn.utils.clip_grad_norm_(self.backward.parameters(), clip_grad_norm)
    # self.optim_f_psm.step()
    # self.optim_b.step()

    return [
        target_M,
        Ms,
        Fs,
        B,
        dynamics_embedding,
        fb_loss,
        fb_diag,
        fb_offdiag,
        orth_loss,
        orth_loss_diag,
        orth_loss_offdiag,
    ]
