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


def main() -> None:
    matplotlib.rcParams["svg.hashsalt"] = "mloda-demo"
    kinect = run(None, features=welded.KINECT_FEATURES, feature_groups=welded.KINECT_CHAIN, label="kinect")
    tum = run(None, features=welded.TUM_FEATURES, feature_groups=welded.TUM_CHAIN, label="tum")
    welded_picture = pipeline_graph([chain(kinect), chain(tum)])
    welded_picture.savefig(SLIDES / "pipelines_welded.svg", **SVG)
    shared_picture = pipeline_graph([chain(run(DepthPng)), chain(run(TumDepth))])
    shared_picture.savefig(SLIDES / "pipeline_shared.svg", **SVG)
    print("wrote", SLIDES / "pipelines_welded.svg", "and", SLIDES / "pipeline_shared.svg")


if __name__ == "__main__":
    main()
