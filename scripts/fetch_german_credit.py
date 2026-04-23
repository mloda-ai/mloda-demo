"""Fetch UCI German Credit dataset (OpenML id 31) and cache as CSV.

Run from the repo root:
    python scripts/fetch_german_credit.py

Writes `demo_data/german_credit.csv` with 1000 rows, 21 columns (20 features + class).
This is the TabPFN support set used as background examples at inference time.
"""

from __future__ import annotations

from pathlib import Path

import openml


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "demo_data" / "german_credit.csv"

    dataset = openml.datasets.get_dataset(
        31, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    df, _y, _categorical, _names = dataset.get_data(dataset_format="dataframe")

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
