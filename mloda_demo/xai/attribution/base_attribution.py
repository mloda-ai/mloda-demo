# Adapted from https://github.com/mloda-ai/mloda-fraunhofer-xai
# (fraunhofer_xai/feature_groups/attribution/base_attribution.py)

from __future__ import annotations

import hashlib
from abc import abstractmethod
from typing import Any, List, Optional, Set, Type

import numpy as np
import pandas as pd

from mloda.provider import BaseArtifact, DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.xai.attribution.model_artifact import ModelArtifact


class AttributionFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Abstract base for all XAI attribution feature groups.

    Feature naming convention: {input_features}__attribution
    Subclasses implement the actual attribution computation for their
    specific framework (zennit/PyTorch, iNNvestigate/TF, LXT/Transformers).

    Options:
        model_path: Path to the serialized model file.
        xai_method: Attribution method name (e.g. "LRP", "DeepLift", "GradCAM").
        target_class: Target class index for attribution (default: predicted class).
    """

    MODEL_PATH = "model_path"
    XAI_METHOD = "xai_method"
    TARGET_CLASS = "target_class"

    PREFIX_PATTERN = r".*__attribution$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES: Optional[int] = None

    PROPERTY_MAPPING = {
        MODEL_PATH: {
            "explanation": "Path to the serialized model file",
            DefaultOptionKeys.context: True,
        },
        XAI_METHOD: {
            "explanation": "Attribution method name (e.g. LRP, DeepLift, GradCAM)",
            DefaultOptionKeys.context: True,
        },
        TARGET_CLASS: {
            "explanation": "Target class index for attribution (default: predicted class)",
            DefaultOptionKeys.context: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features for attribution (comma-separated)",
            DefaultOptionKeys.context: True,
        },
    }

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return ModelArtifact

    @staticmethod
    def _artifact_key(output_name: str, xai_method: str = "LRP") -> str:
        key_input = f"{output_name}:{xai_method}"
        name_hash = hashlib.md5(key_input.encode(), usedforsecurity=False).hexdigest()[:12]
        return f"attribution_{name_hash}"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source_features = cls._extract_source_features(feature)
            model_path = feature.options.get(cls.MODEL_PATH)
            if model_path is None:
                raise ValueError(f"model_path option is required for {feature.name}")

            xai_method_raw = feature.options.get(cls.XAI_METHOD)
            xai_method = str(xai_method_raw) if xai_method_raw is not None else "LRP"

            target_class_raw = feature.options.get(cls.TARGET_CLASS)
            target_class = int(target_class_raw) if target_class_raw is not None else None

            data = cls._run_attribution(
                data,
                features,
                str(feature.name),
                source_features,
                str(model_path),
                xai_method,
                target_class,
            )

        return data

    @classmethod
    @abstractmethod
    def _run_attribution(
        cls,
        data: Any,
        features: FeatureSet,
        output_name: str,
        source_features: List[str],
        model_path: str,
        xai_method: str,
        target_class: Optional[int],
    ) -> Any: ...


class AttributionPandasFeatureGroup(AttributionFeatureGroup):
    """Pandas compute framework implementation for attribution.

    Loads or caches the model via ModelArtifact, runs attribution on each row,
    and stores the resulting attribution map as a serialized numpy array per row.
    """

    @classmethod
    def compute_framework_rule(cls) -> Set[type[Any]]:
        return {PandasDataFrame}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        df: pd.DataFrame = data
        for feature in features.features:
            source_features = cls._extract_source_features(feature)
            missing = [f for f in source_features if f not in df.columns]
            if missing:
                raise ValueError(f"Source features not found in data: {missing}")

    @classmethod
    def _load_or_cache_model(
        cls,
        features: FeatureSet,
        artifact_key: str,
        model_path: str,
    ) -> Any:
        cached = ModelArtifact.load_model(features, artifact_key)
        if cached is not None:
            return cached["model"]

        model = cls._load_model(model_path)
        ModelArtifact.save_model(features, artifact_key, {"model": model, "model_path": model_path})
        return model

    @classmethod
    @abstractmethod
    def _load_model(cls, model_path: str) -> Any:
        """Load a model from the given path. Implemented by framework-specific subclasses."""
        ...

    @classmethod
    @abstractmethod
    def _compute_attributions(
        cls,
        model: Any,
        input_data: np.ndarray[Any, Any],
        xai_method: str,
        target_class: Optional[int],
    ) -> np.ndarray[Any, Any]:
        """Compute attributions for the given input batch. Returns array of same shape as input."""
        ...

    @classmethod
    def _run_attribution(
        cls,
        data: Any,
        features: FeatureSet,
        output_name: str,
        source_features: List[str],
        model_path: str,
        xai_method: str,
        target_class: Optional[int],
    ) -> Any:
        df: pd.DataFrame = data
        artifact_key = cls._artifact_key(output_name, xai_method)

        model = cls._load_or_cache_model(features, artifact_key, model_path)
        input_data = df[source_features].values

        attributions = cls._compute_attributions(model, input_data, xai_method, target_class)

        df[output_name] = [row.tolist() for row in attributions]
        return df
