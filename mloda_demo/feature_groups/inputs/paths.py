"""Resolve the demo_data directory once for all input FeatureGroups.

demo_data lives at <repo-root>/demo_data; this module is at
<repo-root>/mloda_demo/feature_groups/inputs/paths.py, so we walk up three parents.
"""

from __future__ import annotations

from pathlib import Path

DEMO_DATA_DIR = Path(__file__).resolve().parents[3] / "demo_data"
