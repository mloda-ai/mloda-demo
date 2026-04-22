"""Artifact: persist the trained MLP + its fitted encoder.

Using a plain pickle file next to demo_data so the first training run writes it
and subsequent runs (including the marimo notebook reloading) load instantly.
"""

from __future__ import annotations

import pickle  # nosec B403
from dataclasses import dataclass
from pathlib import Path

from mloda_demo.feature_groups.classifier.encoder import EncoderState
from mloda_demo.feature_groups.classifier.mlp import CreditRiskMLP
from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

ARTIFACT_PATH = DEMO_DATA_DIR / "credit_risk_classifier.pkl"
MODEL_STATE_PATH = DEMO_DATA_DIR / "credit_risk_mlp.pt"


@dataclass
class ClassifierArtifact:
    model: CreditRiskMLP
    encoder: EncoderState


def save_artifact(artifact: ClassifierArtifact, path: Path = ARTIFACT_PATH) -> None:
    payload = {
        "encoder": artifact.encoder,
        "model_state_dict": artifact.model.state_dict(),
        "n_features": artifact.model.fc1.in_features,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_artifact(path: Path = ARTIFACT_PATH) -> ClassifierArtifact | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)  # nosec B301
    model = CreditRiskMLP(n_features=payload["n_features"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return ClassifierArtifact(model=model, encoder=payload["encoder"])


def save_model_for_zennit(model: CreditRiskMLP, path: Path = MODEL_STATE_PATH) -> Path:
    """Persist the nn.Module as a torch-loadable file for the Zennit FG's _load_model."""

    import torch

    torch.save(model, path)
    return path
