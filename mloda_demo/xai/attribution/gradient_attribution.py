from __future__ import annotations

from typing import Any, Optional, Set

import numpy as np

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.xai.attribution.base_attribution import AttributionPandasFeatureGroup


SUPPORTED_METHODS = frozenset({"Gradient", "GradientInput"})


class GradientAttributionFeatureGroup(AttributionPandasFeatureGroup):
    """Pure PyTorch gradient-based attribution (no external XAI library).

    Supports:
        Gradient: vanilla backpropagation gradients.
        GradientInput: gradient multiplied by input (Gradient x Input).
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        _data_access_collection: Any = None,
    ) -> bool:
        method = options.get("xai_method")
        if method is None or str(method) not in SUPPORTED_METHODS:
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

        tensor_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

        output = model(tensor_input)
        if target_class is not None:
            target = output[:, target_class]
        else:
            target = output.max(dim=1).values
        target.sum().backward()

        grad_tensor = tensor_input.grad
        if grad_tensor is None:
            raise RuntimeError("Backward did not populate input gradient")
        grad: np.ndarray[Any, Any] = grad_tensor.detach().numpy()
        if xai_method == "GradientInput":
            result: np.ndarray[Any, Any] = grad * input_data
            return result
        return grad
