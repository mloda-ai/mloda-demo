"""Small MLP used as the explainable classifier in the demo.

Hooks cleanly into zennit: forward(query_tensor) -> logits, all layers standard
nn.Linear + ReLU so LRP composites attach without adapters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS


class CreditRiskMLP(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_COLUMNS)) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out: torch.Tensor = self.fc3(x)
        return out


def train_mlp(
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> CreditRiskMLP:
    torch.manual_seed(seed)
    model = CreditRiskMLP(n_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    for _epoch in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    model.eval()
    return model
