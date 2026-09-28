"""One process, one definition: the call the talk repeats, and the picture drawn from its OpenLineage events."""

from __future__ import annotations

from collections.abc import Iterable

from mloda.provider import FeatureGroup

from mloda_demo.one_process import ml, robot, semantic, welded
from mloda_demo.one_process.lineage import Lineage, run

FEATURE_GROUPS: frozenset[type[FeatureGroup]] = frozenset(
    {
        welded.TumFrames,
        welded.TumMetres,
        welded.TumNearest,
        welded.TumStop,
        welded.RedwoodFrames,
        welded.RedwoodMetres,
        welded.RedwoodNearest,
        welded.RedwoodStop,
        robot.Tum,
        robot.Redwood,
        robot.Sim,
        robot.Metres,
        robot.Nearest,
        robot.Stop,
        ml.History,
        ml.Applicant,
        ml.MonthlyPayment,
        semantic.Finance,
        semantic.Sales,
        semantic.Revenue,
        semantic.PerCustomer,
    }
)


def mloda_run(features: Iterable[str], title: str | None = None) -> Lineage:
    """One mloda call over the requested chains; the result draws itself as the pipeline picture under the title."""
    return run(features, FEATURE_GROUPS, title)


__all__ = ["FEATURE_GROUPS", "Lineage", "mloda_run", "run"]
