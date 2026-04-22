"""CreditRiskClassifierFG: MLP trained on UCI German Credit; predicts per customer.

Emits a single feature `credit_risk` whose value is the predicted class index
(0 = bad, 1 = good) for each customer assembled by the upstream mloda pipeline.

On first call the FG trains an MLP on `demo_data/german_credit.csv`, saves both
the model state and the categorical encoder as a pickle artifact next to the
data, and also writes a torch-loadable `credit_risk_mlp.pt` for the downstream
Zennit attribution FG to pick up via its `_load_model(model_path)` hook.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type

import pandas as pd
import torch

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.feature_groups.classifier.artifact import (
    ARTIFACT_PATH,
    MODEL_STATE_PATH,
    ClassifierArtifact,
    load_artifact,
    save_artifact,
    save_model_for_zennit,
)
from mloda_demo.feature_groups.classifier.encoder import (
    FEATURE_COLUMNS,
    encode_training_frame,
    fit_encoder,
)
from mloda_demo.feature_groups.classifier.mlp import train_mlp
from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR


class CreditRiskClassifierFG(FeatureGroup):
    """Emit the predicted credit-risk class per customer."""

    compute_framework: Type[ComputeFramework] = PandasDataFrame

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, _data_access_collection: Any = None) -> bool:
        return str(feature_name) == "credit_risk"

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [Index(("customer_id",))]

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed(col) for col in ("customer_id", *FEATURE_COLUMNS)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        artifact = _ensure_artifact()
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        X = artifact.encoder.encode(df[FEATURE_COLUMNS])
        with torch.no_grad():
            logits = artifact.model(torch.tensor(X, dtype=torch.float32))
            preds = logits.argmax(dim=1).numpy()
        return {"credit_risk": preds.tolist()}

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {cls.compute_framework}


def _ensure_artifact() -> ClassifierArtifact:
    cached = load_artifact()
    if cached is not None:
        return cached

    training_df = pd.read_csv(DEMO_DATA_DIR / "german_credit.csv")
    encoder = fit_encoder(training_df)
    X, y = encode_training_frame(training_df, encoder)
    model = train_mlp(X, y)

    artifact = ClassifierArtifact(model=model, encoder=encoder)
    save_artifact(artifact)
    save_model_for_zennit(model)
    return artifact


__all__ = ["CreditRiskClassifierFG", "ARTIFACT_PATH", "MODEL_STATE_PATH"]
