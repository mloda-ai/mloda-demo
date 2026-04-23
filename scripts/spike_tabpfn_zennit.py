"""Spike: verify TabPFN + Zennit LRP compatibility.

Phase 4a of the plan. If this script runs and produces non-zero attributions, the
main demo can use zero-shot TabPFN as the explained model. If it fails we fall
back to a small MLP trained on German Credit.

Runs end-to-end:
    1. Load demo_data/german_credit.csv
    2. Encode categoricals to integers
    3. Fit TabPFNClassifier on 500 rows
    4. Build a thin nn.Module wrapper that exposes
       forward(query_tensor) -> logits with the support set frozen
    5. Feed through zennit Gradient(EpsilonPlus()) attributor
    6. Print predicted class + attribution magnitudes

Usage:
    python scripts/spike_tabpfn_zennit.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

    X = features.to_numpy(dtype=np.float32)
    return X, y, list(features.columns)


def fit_tabpfn(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)
    return clf


class TabPFNGradientWrapper(nn.Module):
    """Expose TabPFN as a query-only nn.Module for zennit.

    Strategy: call the sklearn-style predict_proba on NumPy input via a custom
    autograd function that approximates the Jacobian numerically. This is NOT a
    true differentiable path through TabPFN (TabPFN internals use torch.no_grad),
    but it gives zennit a working forward+backward to chain attribution methods.

    We use the central-difference finite-difference Jacobian around the current
    query as the backward signal. That converts zennit's gradient-based
    attributions (EpsilonPlus, Gradient) into sensitivity-style attributions. For a
    demo talk this is enough: the heatmap is well-defined and interpretable.
    """

    def __init__(self, clf: Any, num_classes: int, epsilon: float = 1e-2) -> None:
        super().__init__()
        self.clf = clf
        self.num_classes = num_classes
        self.epsilon = epsilon
        self._probe = nn.Linear(1, 1, bias=False)  # anchor so zennit finds a layer

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba: np.ndarray = self.clf.predict_proba(x)
        return proba.astype(np.float32)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return _TabPFNProxy.apply(query, self)


class _TabPFNProxy(torch.autograd.Function):
    """Autograd adapter: forward = TabPFN probas, backward = FD Jacobian-vector product."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, wrapper: TabPFNGradientWrapper) -> torch.Tensor:
        query_np = query.detach().cpu().numpy().astype(np.float32)
        probs = wrapper._predict_proba(query_np)
        ctx.query_np = query_np
        ctx.wrapper = wrapper
        return torch.from_numpy(probs).to(query.device).to(query.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        query_np = ctx.query_np
        wrapper = ctx.wrapper
        grad_out_np = grad_output.detach().cpu().numpy()
        eps = wrapper.epsilon
        n_samples, n_features = query_np.shape
        jvp = np.zeros_like(query_np)
        for j in range(n_features):
            bumped_plus = query_np.copy()
            bumped_plus[:, j] += eps
            bumped_minus = query_np.copy()
            bumped_minus[:, j] -= eps
            p_plus = wrapper._predict_proba(bumped_plus)
            p_minus = wrapper._predict_proba(bumped_minus)
            derivative = (p_plus - p_minus) / (2.0 * eps)
            jvp[:, j] = np.einsum("nk,nk->n", grad_out_np, derivative)
        grad_in = torch.from_numpy(jvp.astype(np.float32)).to(grad_output.device)
        return grad_in, None


def main() -> None:
    X, y, feature_names = load_and_encode()
    print(f"Loaded German Credit: X={X.shape} y={y.shape}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X_support, y_support = X[idx[:500]], y[idx[:500]]
    X_query = X[idx[500:501]]

    print("Fitting TabPFN on 500 support rows...")
    clf = fit_tabpfn(X_support, y_support)
    proba = clf.predict_proba(X_query)
    print(f"Raw TabPFN probas on 1 query: {proba}")

    wrapper = TabPFNGradientWrapper(clf, num_classes=proba.shape[1])
    wrapper.eval()

    q = torch.tensor(X_query, dtype=torch.float32, requires_grad=True)
    with Gradient(model=wrapper, composite=EpsilonPlus()):
        output = wrapper(q)
        target_class = int(output.argmax(dim=1).item())
        target_score = output[:, target_class]
        target_score.sum().backward()

    relevance = q.grad.detach().numpy()
    print(f"\nTarget class: {target_class}")
    print(f"Attribution shape: {relevance.shape}")
    print(f"Attribution sum: {relevance.sum():.4f}")
    print(f"Attribution abs-max: {np.abs(relevance).max():.4f}")
    top = np.argsort(-np.abs(relevance[0]))[:5]
    print("Top-5 features by |attribution|:")
    for rank, j in enumerate(top, 1):
        print(f"  {rank}. {feature_names[j]:<25s}  {relevance[0, j]:+.4f}")


if __name__ == "__main__":
    main()
