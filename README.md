[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-demo

Live demos built on mloda, one per talk.

## Install

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras   # every demo plus dev tools
uv sync                # light install: no PyTorch, enough for the Physical AI demo
```

Run all checks with `tox`. Run integration tests with `pytest -m slow`.

## Demos

### Credit risk (applydata Berlin 2026)

Talk: *"Building Deterministic Context Layers for AI Agents"*. Needs the `credit-risk` extra (PyTorch, Zennit).

Mixed-source credit-risk pipeline: JSON + synthetic Excel + synthetic Markdown → one row per customer → an MLP classifier trained on UCI German Credit → Fraunhofer Zennit LRP attribution → method swap (EpsilonPlus ↔ Gradient). All orchestrated by mloda FeatureGroups. Executable as a deterministic CLI tool.

> The point of the demo: **the LLM / agent on top is non-deterministic. The context layer below it is deterministic.** Same mixed-source inputs always produce the same predictions and the same explanations.

```bash
which mloda-demo  # verify CLI is installed
mloda-demo discover
mloda-demo run --customer app-customer-c
mloda-demo predict --customer app-customer-c
mloda-demo explain --customer app-customer-c
```

Run `mloda-demo --help` for all commands.

### Physical AI (Berlin Physical AI, ML, and CV Meetup 2026)

Talk: *"One Pipeline, Any Device"*. One distance definition across logs, replay, and a new depth sensor, explained by an OpenTelemetry trace in a marimo notebook. In progress.

## Structure

```
mloda_demo/
├── feature_groups/
│   ├── inputs/                   # 3 root FGs: applications.json, xlsx, markdown
│   └── classifier/               # MLP + artifact + CreditRiskClassifierFG
├── xai/
│   ├── attribution/              # Zennit LRP + Gradient attribution FGs
│   └── visualization/            # heatmap renderer
demo_data/                        # customer data + trained artifacts
tests/                            # unit + integration CLI tests
```

## Open Source Libraries

- [mloda](https://github.com/mloda-ai/mloda): feature orchestration framework (Apache 2.0)
- [mloda-registry](https://github.com/mloda-ai/mloda-registry): plugins (including the OpenTelemetry extender), guides, and best practices (Apache 2.0)
- [marimo](https://marimo.io/): reactive Python notebooks (Apache 2.0)
- [OpenTelemetry](https://opentelemetry.io/): tracing API and SDK (Apache 2.0)
- [PyTorch](https://pytorch.org/): MLP model training and inference (BSD 3-Clause)
- [Zennit](https://github.com/chr5tphr/zennit): Layer-wise Relevance Propagation (LGPLv3+)
- [OpenML](https://www.openml.org/): German Credit dataset source
- [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template): starting point for this repo
