[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# mloda-demo

Live demos built on mloda, one per talk.

## Install

```bash
uv venv
source .venv/bin/activate
```

Then pick one:

- Everything, including dev tools: `uv sync --all-extras`. Run all checks with `tox`, integration tests with `pytest -m slow`.
- Credit risk only: `uv sync --extra credit-risk`.
- Physical AI only (no PyTorch): `uv sync`.

## Demos

### Credit risk (applydata Berlin 2026)

Talk: *"Building Deterministic Context Layers for AI Agents"*. Needs the `credit-risk` extra (PyTorch, Zennit).

Mixed-source credit-risk pipeline: JSON + synthetic Excel + synthetic Markdown → one row per customer → an MLP classifier trained on UCI German Credit → Fraunhofer Zennit LRP attribution → method swap (EpsilonPlus ↔ Gradient). All orchestrated by mloda FeatureGroups. Executable as a deterministic CLI tool.

> The point of the demo: **the LLM / agent on top is non-deterministic. The context layer below it is deterministic.** Same mixed-source inputs always produce the same predictions and the same explanations.

```bash
which mloda-demo  # verify CLI is installed
mloda-demo discover
mloda-demo run duration credit_amount checking_status --customer app-customer-c
mloda-demo predict --customer app-customer-c
mloda-demo explain --customer app-customer-c
```

Run `mloda-demo --help` for all commands.

### Physical AI (Berlin Physical AI, ML, and CV Meetup 2026)

Talk: *"One Pipeline, Any Device"*. One definition of the nearest obstacle ahead, two readers, and an OpenTelemetry trace that shows where a wrong number came from. No PyTorch needed.

A real robot: the TUM RGB-D benchmark's Pioneer with a Kinect on top (`freiburg2_pioneer_slam`) drives at a chair. Read through `DepthPng`, a generic 16-bit PNG reader that declares no scale, the conversion assumes millimetres and the robot puts the chair at 2.9 m. Read through `TumDepth`, which declares 5000 per metre, it stops at 0.58 m. Only the readers are device-specific; `DepthToMetres`, `NearestAhead` and `StopRule` are one shared definition, and, with the check on, a reader without a declared scale is refused before the first frame is loaded.

```bash
marimo edit notebooks/physical_ai.py
python scripts/physical_ai_slides.py   # redraws the pipeline pictures for the slides from real runs
```

The notebook, plots and talk slides (`slides/physical_ai.pdf`) use the mloda.ai colours with system fonts, and work offline.

Frames in `demo_data/physical_ai/pioneer_slam/` are from the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)): J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.

## Structure

```
mloda_demo/
├── feature_groups/
│   ├── inputs/                   # 3 root FGs: applications.json, xlsx, markdown
│   └── classifier/               # MLP + artifact + CreditRiskClassifierFG
├── xai/
│   ├── attribution/              # Zennit LRP + Gradient attribution FGs
│   └── visualization/            # heatmap renderer
├── physical_ai/                  # readers, shared definition, welded copies, runner, trace, plot, style
demo_data/                        # customer data + trained artifacts; physical_ai/ robot clip
notebooks/                        # marimo notebook and its CSS for the Physical AI talk
slides/                           # Physical AI talk slides (HTML source, pipeline pictures, PDF)
tests/                            # unit + integration tests
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
