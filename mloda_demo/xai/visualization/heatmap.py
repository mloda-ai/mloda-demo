# Adapted from https://github.com/mloda-ai/mloda-fraunhofer-xai
# (fraunhofer_xai/feature_groups/visualization/heatmap.py)

from __future__ import annotations

from typing import Any, Set

import numpy as np
import pandas as pd

from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options


class HeatmapFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Renders a heatmap from a multi-value column (e.g. attribution scores).

    Produces a base64-encoded PNG stored in the output column.
    Uses configuration-based matching: pass the source column via Options(in_features=...).

    Example:
        Feature("heatmap", Options({"in_features": "feat_a&feat_b__attribution", ...}))
    """

    FEATURE_NAME = "heatmap"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    COLORMAP = "colormap"
    TITLE = "title"

    PROPERTY_MAPPING = {
        COLORMAP: {
            "explanation": "Matplotlib colormap name (default: RdBu_r)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "RdBu_r",
        },
        TITLE: {
            "explanation": "Title displayed above the heatmap",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "Heatmap",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source column containing per-row attribution lists",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        _data_access_collection: Any = None,
    ) -> bool:
        _name = str(feature_name)
        if not _name.startswith(cls.FEATURE_NAME):
            return False
        return super().match_feature_group_criteria(feature_name, options, _data_access_collection)


class HeatmapPandasFeatureGroup(HeatmapFeatureGroup):
    """Pandas implementation: renders heatmap to base64 PNG."""

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
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df: pd.DataFrame = data

        for feature in features.features:
            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]

            matrix = np.array(df[source_col].tolist())

            colormap = str(feature.options.get(cls.COLORMAP) or "RdBu_r")
            title = str(feature.options.get(cls.TITLE) or "Heatmap")

            # Derive display names from source column: "a&b&c__attribution" -> ["a", "b", "c"]
            base_name = source_col.rsplit("__", 1)[0] if "__" in source_col else source_col
            display_names = base_name.split("&") if "&" in base_name else [base_name]

            fig, ax = plt.subplots(figsize=(10, 6))
            abs_max = np.max(np.abs(matrix))
            im = ax.imshow(matrix, aspect="auto", cmap=colormap, vmin=-abs_max, vmax=abs_max)
            ax.set_xlabel("Features")
            ax.set_ylabel("Samples")
            ax.set_xticks(range(len(display_names)))
            ax.set_xticklabels(display_names)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="Relevance")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            df[str(feature.name)] = img_b64

        return df
