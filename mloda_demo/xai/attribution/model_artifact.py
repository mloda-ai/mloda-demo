from __future__ import annotations

from typing import Any, Dict, Optional

from mloda.provider import BaseArtifact, FeatureSet


class ModelArtifact(BaseArtifact):
    """Artifact for persisting loaded models used by attribution plugins.

    Stores a model along with metadata so that attribution methods can reuse
    a previously loaded model across feature calculations within the same
    mloda run.
    """

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        return artifact

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        options = cls.get_singular_option_from_options(features)
        if options is None or features.name_of_one_feature is None:
            return None
        return options.get(str(features.name_of_one_feature))

    @classmethod
    def load_model(cls, features: FeatureSet, artifact_key: str) -> Optional[Dict[str, Any]]:
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            if artifacts and isinstance(artifacts, dict) and artifact_key in artifacts:
                return artifacts[artifact_key]  # type: ignore[no-any-return]
        return None

    @classmethod
    def save_model(cls, features: FeatureSet, artifact_key: str, artifact_data: Dict[str, Any]) -> None:
        if features.artifact_to_save:
            if not isinstance(features.save_artifact, dict):
                features.save_artifact = {}
            features.save_artifact[artifact_key] = artifact_data
