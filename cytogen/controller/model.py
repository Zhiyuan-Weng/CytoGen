"""Lightweight MLP regression for layout-conditioned segmentation failure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FailureModelConfig:
    hidden_units: int = 64
    batch_size: int = 1024
    epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    huber_delta: float = 0.1
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.hidden_units <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("model dimensions and training lengths must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")


class FailureRegressor(nn.Module):
    """Two-hidden-layer failure function h_t(z) from the paper."""

    def __init__(self, hidden_units: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.network(descriptors).squeeze(-1)


@dataclass
class FittedFailureModel:
    model: FailureRegressor
    descriptor_mean: np.ndarray
    descriptor_scale: np.ndarray
    losses: list[float]
    config: FailureModelConfig

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        descriptors = np.asarray(descriptors, dtype=np.float32)
        normalized = (descriptors - self.descriptor_mean) / self.descriptor_scale
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(
                torch.as_tensor(normalized, dtype=torch.float32, device=device)
            )
        return prediction.detach().cpu().numpy().astype(float)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "descriptor_mean": self.descriptor_mean,
                "descriptor_scale": self.descriptor_scale,
                "config": asdict(self.config),
                "final_loss": self.losses[-1],
            },
            Path(path),
        )


def _training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def fit_failure_model(
    descriptors: np.ndarray,
    targets: np.ndarray,
    config: FailureModelConfig,
) -> FittedFailureModel:
    """Fit h_t with instance-level Huber regression."""
    config.validate()
    descriptors = np.asarray(descriptors, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != 3:
        raise ValueError(f"Expected Nx3 descriptors, got {descriptors.shape}")
    if targets.shape != (len(descriptors),):
        raise ValueError("Failure targets must contain one value per descriptor")
    if not len(descriptors) or not np.all(np.isfinite(descriptors)):
        raise ValueError("Descriptors must be non-empty and finite")
    if not np.all(np.isfinite(targets)) or np.any((targets < 0) | (targets > 1)):
        raise ValueError("Failure targets must be finite values in [0, 1]")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _training_device(config.device)
    descriptor_mean = descriptors.mean(axis=0)
    descriptor_scale = descriptors.std(axis=0)
    descriptor_scale = np.maximum(descriptor_scale, 1e-6)
    normalized = (descriptors - descriptor_mean) / descriptor_scale
    dataset = TensorDataset(
        torch.as_tensor(normalized, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=loader_generator,
    )
    model = FailureRegressor(config.hidden_units).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.HuberLoss(delta=config.huber_delta)
    losses = []
    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_descriptors, batch_targets in loader:
            batch_descriptors = batch_descriptors.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_descriptors)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_descriptors)
            total_items += len(batch_descriptors)
        losses.append(total_loss / max(total_items, 1))
    return FittedFailureModel(
        model=model,
        descriptor_mean=descriptor_mean,
        descriptor_scale=descriptor_scale,
        losses=losses,
        config=config,
    )
