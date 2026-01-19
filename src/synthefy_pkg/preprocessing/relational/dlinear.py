#!/usr/bin/env python3
# dlinear_train.py
# Self-contained DLinear model + simple dataloader + training loop
# Usage (synthetic demo):
#   python dlinear_train.py --seq_len 96 --pred_len 24 --epochs 10
# Usage (CSV with columns as channels; header row expected):
#   python dlinear_train.py --data_csv path/to/data.csv --seq_len 96 --pred_len 24 --epochs 20

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------
# Utilities
# ---------------------------


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    return batch.to(device)


# ---------------------------
# Series decomposition (moving average)
# ---------------------------


class SeriesDecomp(nn.Module):
    """
    Decompose a series into trend (moving average) and seasonal (residual).
    Input/Output shape: (B, C, L)
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        # Build a fixed 1D average pooling via conv1d with constant weights
        weight = torch.ones(1, 1, kernel_size) / kernel_size
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, L)
        padding = (self.kernel_size - 1) // 2
        # Pad at both ends with replication to keep length
        x_padded = F.pad(x, (padding, padding), mode="replicate")
        # Depthwise conv: apply same kernel per channel
        trend = F.conv1d(
            x_padded, self.weight.expand(x.size(1), -1, -1), groups=x.size(1)
        )
        seasonal = x - trend
        return seasonal, trend


def append_activation(layers: list[nn.Module], activation: str = "relu"):
    """
    Appends an activation function to a list of layers.
    supports: relu, none
    """
    if activation == "relu":
        layers.append(nn.ReLU())
    return layers


def generate_layers(
    num_layers: int,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    activation: str = "relu",
):
    if num_layers > 1:
        layers = []
        layers.append(nn.Linear(seq_len, hidden_size))
        layers = append_activation(layers, activation)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers = append_activation(layers, activation)
        layers.append(nn.Linear(hidden_size, pred_len))
        network = nn.Sequential(*layers)
        return network
    return nn.Linear(seq_len, pred_len)


# ---------------------------
# DLinear model
# ---------------------------


class DLinear(nn.Module):
    """
    DLinear model from "Are Transformers Effective for Time Series Forecasting?"
    - Decomposition into seasonal & trend components via moving average
    - Linear mapping from seq_len -> pred_len, either shared or per-channel
    Expect input shape (B, L, C); outputs (B, pred_len, C)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        num_layers: int = 1,
        hidden_size: int = 128,
        activation: str = "relu",
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation

        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            # Per-channel linear layers

            self.seasonal_layers = nn.ModuleList(
                [
                    generate_layers(
                        num_layers, seq_len, pred_len, hidden_size, activation
                    )
                    for _ in range(channels)
                ]
            )
            self.trend_layers = nn.ModuleList(
                [
                    generate_layers(
                        num_layers, seq_len, pred_len, hidden_size, activation
                    )
                    for _ in range(channels)
                ]
            )
        else:
            # Shared across channels (applied channel-wise by broadcasting)
            self.seasonal_layer = generate_layers(
                num_layers, seq_len, pred_len, hidden_size, activation
            )
            self.trend_layer = generate_layers(
                num_layers, seq_len, pred_len, hidden_size, activation
            )

        # Initialize like in many linear baselines
        self.reset_parameters()

    def reset_parameters(self):
        def _init_linear(m: nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if self.individual:
            for m in list(self.seasonal_layers) + list(self.trend_layers):
                if isinstance(m, nn.Linear):
                    _init_linear(m)
                elif isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Linear):
                            _init_linear(layer)
        else:
            if self.num_layers > 1:
                assert isinstance(self.seasonal_layer, nn.Sequential), (
                    "seasonal_layer should be a nn.Sequential"
                )
                assert isinstance(self.trend_layer, nn.Sequential), (
                    "trend_layer should be a nn.Sequential"
                )
                for layer in self.seasonal_layer:
                    if isinstance(layer, nn.Linear):
                        _init_linear(layer)
                for layer in self.trend_layer:
                    if isinstance(layer, nn.Linear):
                        _init_linear(layer)
            else:
                assert isinstance(self.seasonal_layer, nn.Linear), (
                    "seasonal_layer should be a nn.Linear"
                )
                assert isinstance(self.trend_layer, nn.Linear), (
                    "trend_layer should be a nn.Linear"
                )
                _init_linear(self.seasonal_layer)
                _init_linear(self.trend_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = x.permute(0, 2, 1)  # -> (B, C, L)
        seasonal, trend = self.decomp(x)  # both (B, C, L)

        if self.individual:
            outs = []
            for ch in range(self.channels):
                s = self.seasonal_layers[ch](
                    seasonal[:, ch, :]
                )  # (B, pred_len)
                t = self.trend_layers[ch](trend[:, ch, :])  # (B, pred_len)
                outs.append(s + t)
            out = torch.stack(outs, dim=2)  # (B, pred_len, C)
        else:
            # Apply shared linear per channel by reshaping
            B, C, L = seasonal.shape
            s = self.seasonal_layer(
                seasonal.reshape(B * C, L)
            )  # (B*C, pred_len)
            t = self.trend_layer(trend.reshape(B * C, L))  # (B*C, pred_len)
            out = (
                (s + t).reshape(B, C, self.pred_len).permute(0, 2, 1)
            )  # (B, pred_len, C)
        return out


# ---------------------------
# Data: simple sliding-window dataset
# ---------------------------


class WindowedTimeSeries(Dataset):
    """
    Builds (input_window -> forecast_window) pairs from a multivariate series.
    - data: np.ndarray of shape (T, C)
    - returns x: (seq_len, C), y: (pred_len, C)
    Optional standardization (fit on full data or externally).
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ):
        assert data.ndim == 2, "data should be (T, C)"
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.T, self.C = self.data.shape

        if mean is None or std is None:
            self.mean = self.data.mean(axis=0, keepdims=True)
            self.std = self.data.std(axis=0, keepdims=True)
        else:
            self.mean = mean
            self.std = std

        self.std = np.where(
            self.std < eps, 1.0, self.std
        )  # avoid divide-by-zero
        self.norm = (self.data - self.mean) / self.std

        self.num_samples = self.T - (seq_len + pred_len) + 1
        self.num_samples = max(self.num_samples, 0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.norm[idx : idx + self.seq_len]  # (L, C)
        y = self.norm[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        # return tensors
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------
# Training / Evaluation
# ---------------------------


@dataclass
class TrainConfig:
    seq_len: int = 96
    pred_len: int = 24
    kernel_size: int = 25
    individual: bool = False
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_split: float = 0.2
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataloaders(
    data: np.ndarray, cfg: TrainConfig
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = WindowedTimeSeries(data, cfg.seq_len, cfg.pred_len)
    n_val = int(len(dataset) * cfg.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(123)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, dataset.C


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, grad_clip: float
):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = (
            to_device(xb, device),
            to_device(yb, device),
        )  # (B, L, C), (B, pred, C)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_total, mae_total, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = to_device(xb, device), to_device(yb, device)
        # Ensure we have tensors, not lists
        if isinstance(xb, (list, tuple)):
            xb = torch.stack(xb) if len(xb) > 1 else xb[0].unsqueeze(0)  # type: ignore
        if isinstance(yb, (list, tuple)):
            yb = torch.stack(yb) if len(yb) > 1 else yb[0].unsqueeze(0)  # type: ignore
        preds = model(xb)
        mse = F.mse_loss(preds, yb, reduction="sum").item()
        mae = F.l1_loss(preds, yb, reduction="sum").item()
        mse_total += mse
        mae_total += mae
        n += yb.numel() if hasattr(yb, "numel") else len(yb)
    return mse_total / n, mae_total / n


# ---------------------------
# Data helpers (CSV or synthetic)
# ---------------------------


def load_csv(path: str) -> np.ndarray:
    # Lightweight CSV loader (no pandas dependency). Assumes header row, comma-separated.
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    # skip header
    rows = [list(map(float, line.split(","))) for line in lines[1:]]
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def make_synthetic(T: int = 5000, C: int = 4, seed: int = 123) -> np.ndarray:
    """
    Multivariate sine + time-varying trend + mild noise for demo/training.
    Returns (T, C) float32 array.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    data = []
    for c in range(C):
        freq = 2 * math.pi / (48 + 8 * c)  # distinct seasonalities
        seasonal = np.sin(freq * t) + 0.5 * np.sin(0.5 * freq * t + 0.3)
        trend = 0.001 * (t - T / 2) + 0.2 * np.sin(
            2 * math.pi * t / (T / 3 + 10 * c)
        )
        noise = 0.1 * rng.standard_normal(T).astype(np.float32)
        series = seasonal + trend + noise + c  # shift per channel
        data.append(series)
    arr = np.stack(data, axis=1).astype(np.float32)  # (T, C)
    return arr


# ---------------------------
# Main
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DLinear minimal training script"
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="Path to CSV with columns as channels; header required",
    )
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument(
        "--individual", action="store_true", help="Per-channel linear layers"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="dlinear.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    # Data
    if args.data_csv is None:
        print("No --data_csv provided; generating synthetic data...")
        data = make_synthetic(T=6000, C=4, seed=123)
    else:
        print(f"Loading CSV from {args.data_csv}")
        data = load_csv(args.data_csv)

    cfg = TrainConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        kernel_size=args.kernel_size,
        individual=args.individual,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    train_loader, val_loader, channels = make_dataloaders(data, cfg)

    # Model
    model = DLinear(
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        channels=channels,
        kernel_size=cfg.kernel_size,
        individual=cfg.individual,
    ).to(cfg.device)

    # Optimizer/Loss
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    # Training
    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, cfg.device, cfg.grad_clip
        )
        val_mse, val_mae = evaluate(model, val_loader, cfg.device)
        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | val MSE: {val_mse:.6f} | val MAE: {val_mae:.6f}"
        )

        # Track best
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "seq_len": cfg.seq_len,
                        "pred_len": cfg.pred_len,
                        "channels": channels,
                        "kernel_size": cfg.kernel_size,
                        "individual": cfg.individual,
                    },
                },
                args.save_path,
            )

    print(
        f"Done. Best val MSE: {best_val:.6f}. Model saved to: {args.save_path}"
    )


if __name__ == "__main__":
    main()
