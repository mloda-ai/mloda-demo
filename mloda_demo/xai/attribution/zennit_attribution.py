from __future__ import annotations

from typing import Any, Optional, Set

import numpy as np

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.xai.attribution.base_attribution import AttributionPandasFeatureGroup


SUPPORTED_METHODS = frozenset({"LRP", "EpsilonPlus", "EpsilonAlpha2Beta1"})


class ZennitAttributionFeatureGroup(AttributionPandasFeatureGroup):
    """Zennit-based attribution for PyTorch models via Layer-wise Relevance Propagation.

    Supports composites selectable via xai_method:
        LRP / EpsilonPlus (default): EpsilonPlus composite.
        EpsilonAlpha2Beta1: Alpha2-Beta1 decomposition rule.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        _data_access_collection: Any = None,
    ) -> bool:
        method = options.get("xai_method")
        if method is not None and str(method) not in SUPPORTED_METHODS:
            return False
        return super().match_feature_group_criteria(feature_name, options, _data_access_collection)

    @classmethod
    def compute_framework_rule(cls) -> Set[type[Any]]:
        return {PandasDataFrame}

    @classmethod
    def _load_model(cls, model_path: str) -> Any:
        import torch

        model = torch.load(model_path, map_location="cpu", weights_only=False)  # nosec B614
        model.eval()
        return model

    @classmethod
    def _compute_attributions(
        cls,
        model: Any,
        input_data: np.ndarray[Any, Any],
        xai_method: str,
        target_class: Optional[int],
    ) -> np.ndarray[Any, Any]:
        import torch
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonAlpha2Beta1, EpsilonPlus

        tensor_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

        if xai_method == "EpsilonAlpha2Beta1":
            composite = EpsilonAlpha2Beta1()
        else:
            composite = EpsilonPlus()
        attributor = Gradient(model=model, composite=composite)

        with attributor:
            output = model(tensor_input)
            if target_class is not None:
                target = output[:, target_class]
            else:
                target = output.max(dim=1).values
            target.sum().backward()

        grad = tensor_input.grad
        if grad is None:
            raise RuntimeError("Zennit attributor did not populate input gradient")
        relevance: np.ndarray[Any, Any] = grad.detach().numpy()
        return relevance
