[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-demo

Live demo built for the **applydata Berlin 2026** talk *"Building Deterministic Context Layers for AI Agents"*.

Mixed-source credit-risk pipeline: JSON + synthetic Excel + synthetic Markdown → one row per customer → an MLP classifier trained on UCI German Credit → Fraunhofer Zennit LRP attribution → method swap (EpsilonPlus ↔ Gradient). All orchestrated by mloda FeatureGroups. Packaged as a reactive marimo notebook.

> The point of the demo: **the LLM / agent on top is non-deterministic. The context layer below it is deterministic.** Same mixed-source inputs always produce the same predictions and the same explanations.

## Quick reproduce

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras

python scripts/fetch_german_credit.py
python scripts/synthesize_docs.py

marimo edit demo/applydata_credit_xai.py
```

Run all checks with `tox`. Run integration tests explicitly with `pytest -m slow`.

## Structure

```
mloda_demo/
├── feature_groups/
│   ├── inputs/                   # 3 root FGs: applications.json, xlsx, markdown
│   ├── classifier/               # MLP + artifact + CreditRiskClassifierFG
│   └── merged/                   # reserved for future CustomerRow FG
├── xai/                          # vendored from mloda-fraunhofer-xai
│   ├── attribution/              # Zennit LRP + Gradient attribution FGs
│   └── visualization/            # heatmap renderer
demo_data/                        # pruned fake_data + synthetic docs + German Credit CSV
scripts/                          # fetch and synthesize helpers
demo/                             # marimo notebook
tests/                            # unit + integration
```

## Attribution

The `mloda_demo/xai/` tree is vendored (copy-pasted) from [mloda-fraunhofer-xai](https://github.com/mloda-ai/mloda-fraunhofer-xai). Each file carries a provenance header pointing at the upstream source.

## Related

- [mloda](https://github.com/mloda-ai/mloda) — core library
- [mloda-fraunhofer-xai](https://github.com/mloda-ai/mloda-fraunhofer-xai) — upstream XAI plugins
- [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template) — starting point for this repo
