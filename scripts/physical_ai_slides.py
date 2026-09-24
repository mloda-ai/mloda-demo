"""Draw the two pipeline pictures for the slides from real runs and their traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib

from mloda_demo.physical_ai import welded
from mloda_demo.physical_ai.plot import pipeline_graph
from mloda_demo.physical_ai.readers import DepthPng, TumDepth
from mloda_demo.physical_ai.runner import run
from mloda_demo.physical_ai.trace import chain

SLIDES = Path(__file__).resolve().parents[1] / "slides"
SVG = {"format": "svg", "transparent": True, "metadata": {"Date": None}}


def main(out_dir: Path = SLIDES) -> None:
    matplotlib.rcParams["svg.hashsalt"] = "mloda-demo"
    kinect = run(None, features=welded.KINECT_FEATURES, feature_groups=welded.KINECT_CHAIN, label="kinect")
    tum = run(None, features=welded.TUM_FEATURES, feature_groups=welded.TUM_CHAIN, label="tum")
    generic, declared = run(DepthPng), run(TumDepth)
    for result in (kinect, tum, generic, declared):
        if not result.passed:
            raise RuntimeError(f"{result.label}: {result.error}")
    if chain(kinect) == chain(tum) or chain(generic) == chain(declared):
        raise RuntimeError("the two runs of a picture must differ")
    pipeline_graph([chain(kinect), chain(tum)]).savefig(out_dir / "pipelines_welded.svg", **SVG)
    pipeline_graph([chain(generic), chain(declared)]).savefig(out_dir / "pipeline_shared.svg", **SVG)
    print("wrote", out_dir / "pipelines_welded.svg", "and", out_dir / "pipeline_shared.svg")


if __name__ == "__main__":
    main()
