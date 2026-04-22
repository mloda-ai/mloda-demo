"""Spike: train a small MLP on German Credit and verify Zennit LRP attribution.

Replaces the TabPFN spike (TabPFN 7.x requires an online license token we do not
want in the demo critical path). A 3-layer MLP trains in ~10 seconds on 1000 rows
and hooks cleanly into zennit with no autograd gymnastics.

Runs end-to-end:
    1. Load demo_data/german_credit.csv
    2. Encode categoricals to integers
    3. Train an MLP (20 -> 32 -> 16 -> 2) for 200 epochs
    4. Run zennit EpsilonPlus LRP on a single query row
    5. Print predicted class + top-5 feature attributions

Usage:
    python scripts/spike_mlp_zennit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus


def load_and_encode() -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_path = Path(__file__).resolve().parent.parent / "demo_data" / "german_credit.csv"
    df = pd.read_csv(csv_path)
    y = (df["class"] == "good").astype(int).to_numpy()
    features = df.drop(columns=["class"]).copy()

    cat_cols = features.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        features[col] = features[col].astype("category").cat.codes

    features = (features - features.mean()) / (features.std() + 1e-8)
    X = features.to_numpy(dtype=np.float32)
    return X, y, list(features.columns)


class CreditRiskMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 200) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            preds = logits.argmax(dim=1)
            acc = float((preds == y).float().mean())
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.3f}")


def main() -> None:
    X, y, feature_names = load_and_encode()
    print(f"Loaded German Credit: X={X.shape} y={y.shape}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X_train, y_train = X[idx[:800]], y[idx[:800]]
    X_test = X[idx[800:801]]

    model = CreditRiskMLP(n_features=X.shape[1])
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    print("Training MLP...")
    train(model, X_t, y_t)

    model.eval()
    q = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)

    with Gradient(model=model, composite=EpsilonPlus()):
        logits = model(q)
        target_class = int(logits.argmax(dim=1).item())
        logits[:, target_class].sum().backward()

    relevance = q.grad.detach().numpy()
    print(f"\nPredicted class: {target_class}")
    print(f"Attribution shape: {relevance.shape}")
    print(f"Attribution sum: {relevance.sum():+.4f}")
    print(f"Attribution abs-max: {np.abs(relevance).max():.4f}")
    top = np.argsort(-np.abs(relevance[0]))[:5]
    print("Top-5 features by |attribution|:")
    for rank, j in enumerate(top, 1):
        print(f"  {rank}. {feature_names[j]:<25s}  {relevance[0, j]:+.4f}")


if __name__ == "__main__":
    main()
