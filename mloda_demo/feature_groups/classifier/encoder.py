"""Categorical encoding shared between German Credit (training) and customer rows (inference).

We fit the encoder once on German Credit and store it in the artifact so the same
categorical-to-int mapping is used for both training rows and query rows from the
mixed-source mloda pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

FEATURE_COLUMNS = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
]


@dataclass
class EncoderState:
    """Persistable encoder state: categorical value→int maps + numeric mean/std."""

    categorical_maps: dict[str, dict[str, int]]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]

    def encode(self, df: pd.DataFrame) -> np.ndarray[Any, Any]:
        rows = []
        for _, row in df.iterrows():
            encoded: list[float] = []
            for col in FEATURE_COLUMNS:
                raw = row[col]
                if col in self.categorical_maps:
                    code_map = self.categorical_maps[col]
                    encoded.append(float(code_map.get(str(raw), -1)))
                else:
                    mean = self.numeric_mean[col]
                    std = self.numeric_std[col] or 1.0
                    encoded.append((float(raw) - mean) / std)
            rows.append(encoded)
        return np.asarray(rows, dtype=np.float32)


def fit_encoder(df: pd.DataFrame) -> EncoderState:
    categorical_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLUMNS:
        uniques = sorted(df[col].astype(str).unique())
        categorical_maps[col] = {val: idx for idx, val in enumerate(uniques)}

    numeric_mean = {col: float(df[col].mean()) for col in NUMERIC_COLUMNS}
    numeric_std = {col: float(df[col].std()) for col in NUMERIC_COLUMNS}

    return EncoderState(
        categorical_maps=categorical_maps,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
    )


def encode_training_frame(df: pd.DataFrame, state: EncoderState) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    X = state.encode(df[FEATURE_COLUMNS])
    y = (df["class"] == "good").astype(int).to_numpy()
    return X, y
