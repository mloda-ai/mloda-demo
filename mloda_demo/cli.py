"""mloda-demo CLI: a 4-command terminal entry point for the credit-risk demo.

Subcommands:
    discover                  List loaded FeatureGroups and their feature names.
    run <feature>...          Fetch features via mloda.run_all and print as a table.
    predict [--customer X]    Run the classifier (auto-pulls upstream input FGs).
    explain [--customer X] [--method M]
                              Compute XAI attribution per customer.

Designed for the applydata 2026 talk: the deterministic substrate underneath
any agent on top. Same inputs, same outputs, every time.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import pandas as pd

# Importing the demo FGs registers them so mloda.run_all and
# get_feature_group_docs can see them. The demo uses explicit imports rather
# than entry-point discovery (PluginLoader.all() is a no-op here).
from mloda_demo.extenders.lineage_extender import LineageExtender
from mloda_demo.feature_groups.classifier.artifact import MODEL_STATE_PATH, load_artifact
from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import (  # noqa: F401
    CreditRiskClassifierFG,
    _ensure_artifact,
)
from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS
from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG
from mloda_demo.xai.attribution.gradient_attribution import GradientAttributionFeatureGroup  # noqa: F401
from mloda_demo.xai.attribution.zennit_attribution import ZennitAttributionFeatureGroup  # noqa: F401

from mloda.core.api.plugin_docs import get_feature_group_docs
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.filter.filter_type_enum import FilterType
from mloda.core.filter.global_filter import GlobalFilter
from mloda.user import Feature, Link
from mloda.user import mloda as mloda_api

# Lazy-load ApplicationsFG implementations to avoid __subclasses__ conflicts.
# Only the active implementation is imported, allowing JSON/SQLite backend swapping.
_applications_fg_cache: dict[str, type] = {}


def _get_applications_fg(data_source: str) -> type:
    """Lazy-load and cache the appropriate ApplicationsFG implementation."""
    if data_source not in _applications_fg_cache:
        if data_source == "sqlite":
            from mloda_demo.feature_groups.inputs.applications_sqlite_fg import ApplicationsSqliteFG

            _applications_fg_cache["sqlite"] = ApplicationsSqliteFG
        else:
            from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG

            _applications_fg_cache["json"] = ApplicationsFG
    return _applications_fg_cache[data_source]


# Live-edit target for the "swap one string" demo beat. Change this constant
# on stage and rerun `mloda-demo explain` to flip the attribution method.
DEFAULT_XAI_METHOD = "EpsilonPlus"

DEMO_DATA_DIR = Path(__file__).resolve().parents[1] / "demo_data"


def _get_demo_fgs(data_source: str = "json") -> tuple[type, ...]:
    """Get all demo FGs including the active applications implementation."""
    apps_fg = _get_applications_fg(data_source)
    return (
        apps_fg,
        FinancialsFG,
        QuestionnaireFG,
        CreditRiskClassifierFG,
        ZennitAttributionFeatureGroup,
        GradientAttributionFeatureGroup,
    )


# Lazy-loaded for discovery; use _get_demo_fgs() to access.
_DEMO_FGS: tuple[type, ...] | None = None


def _build_input_links(data_source: str) -> set[Link]:
    """Build input links using the appropriate ApplicationsFG implementation."""
    apps_fg = _get_applications_fg(data_source)
    return {
        Link.inner_on(apps_fg, FinancialsFG),
        Link.inner_on(apps_fg, QuestionnaireFG),
    }


# Default INPUT_LINKS for backward compatibility (json data source).
# Lazy-loaded on first access to avoid premature FG registration.
INPUT_LINKS: set[Link] | None = None


def _get_input_links() -> set[Link]:
    global INPUT_LINKS
    if INPUT_LINKS is None:
        INPUT_LINKS = _build_input_links("json")
    return INPUT_LINKS


def _feature_surface(fg_cls: Any) -> str:
    """Best-effort enumeration of the feature names a FG can serve.

    `supported_feature_names` is empty for FGs that declare their surface via
    `input_data()` (DataCreator) or via `match_feature_group_criteria`. Fall
    back to inspecting `input_data().feature_names` when present.
    """
    declared = set(fg_cls.feature_names_supported() or set())
    input_data = fg_cls.input_data()
    if input_data is not None and hasattr(input_data, "feature_names"):
        declared |= set(input_data.feature_names)
    return ", ".join(sorted(str(n) for n in declared)) if declared else "(matched dynamically)"


def cmd_discover(args: argparse.Namespace) -> int:
    docs = list(get_feature_group_docs(name=args.name))
    if not docs:
        print("No FeatureGroups loaded.", file=sys.stderr)
        return 1

    print(f"Data source: {args.data_source}  (applications backend)")
    print()

    docs_by_name = {d.name: d for d in docs}
    for fg_cls in _get_demo_fgs(args.data_source):
        name = fg_cls.__name__
        if args.name and args.name not in name:
            continue
        doc_obj = docs_by_name.get(name)
        print(name)
        print(f"  features: {_feature_surface(fg_cls)}")
        if doc_obj is not None and getattr(doc_obj, "description", None):
            first_line = str(doc_obj.description).strip().splitlines()[0]
            print(f"  doc:      {first_line}")
        print()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Ensure the active applications FG is registered with mloda before run_all.
    _get_applications_fg(args.data_source)

    if args.data_source == "sqlite":
        db_path = DEMO_DATA_DIR / "applications.db"
        if not db_path.exists():
            print("Error: applications.db not found. Run `mloda-demo migrate` first.", file=sys.stderr)
            return 1

    # customer_id is claimed by all three input FGs (index column). Request it
    # separately so mloda doesn't fail on multiple matches, and use it as the
    # join key across feature groups.
    feature_names = [n for n in args.features if n != "customer_id"]
    if not feature_names:
        print("Error: must request at least one feature (customer_id is automatic).", file=sys.stderr)
        return 1
    features: list[Feature | str] = [Feature.not_typed(name) for name in feature_names]

    kwargs: dict[str, Any] = {
        "features": features,
        "compute_frameworks": ["PandasDataFrame"],
        "links": _build_input_links(args.data_source),
    }

    if args.lineage:
        extender = LineageExtender()
        kwargs["function_extender"] = {extender}

    if args.filter:
        gf = GlobalFilter()
        for spec in args.filter:
            parts = spec.split(",", 2)
            if len(parts) != 3:
                print(f"Error: filter format is FEATURE,TYPE,VALUE (got {spec!r})", file=sys.stderr)
                return 1
            feat, ftype, val = parts
            try:
                if ftype == "range":
                    lo, hi = val.split(":")
                    gf.add_filter(feat, FilterType.RANGE, {"min": float(lo), "max": float(hi)})
                elif ftype == "equal":
                    gf.add_filter(feat, FilterType.EQUAL, {"value": val})
                else:
                    print(f"Error: unknown filter type {ftype!r} (range|equal)", file=sys.stderr)
                    return 1
            except ValueError as e:
                print(f"Error parsing filter {spec!r}: {e}", file=sys.stderr)
                return 1
        kwargs["global_filter"] = gf

    results = mloda_api.run_all(**kwargs)

    if args.lineage:
        print(extender.visualize())
        print()

    frames = [_to_dataframe(r).reset_index(drop=True) for r in results]
    merged = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    apps_fg = cast(Any, _get_applications_fg(args.data_source))
    apps_df: pd.DataFrame = apps_fg.calculate_feature(None, cast(Any, None))
    customer_ids = apps_df["customer_id"].reset_index(drop=True)
    merged.insert(0, "customer_id", customer_ids)

    if args.customer:
        merged = merged[merged["customer_id"] == args.customer]
        if merged.empty:
            print(f"No customer matching {args.customer!r}.", file=sys.stderr)
            return 1

    print(merged.to_string(index=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    # Ensure the active applications FG is registered with mloda before run_all.
    _get_applications_fg(args.data_source)

    if args.data_source == "sqlite":
        db_path = DEMO_DATA_DIR / "applications.db"
        if not db_path.exists():
            print("Error: applications.db not found. Run `mloda-demo migrate` first.", file=sys.stderr)
            return 1

    features: list[Feature | str] = [Feature.not_typed("credit_risk")]
    results = mloda_api.run_all(
        features=features,
        compute_frameworks=["PandasDataFrame"],
        links=_build_input_links(args.data_source),
    )
    if not results:
        print("mloda.run_all returned no results.", file=sys.stderr)
        return 1
    df = _to_dataframe(results[0]).reset_index(drop=True)
    apps_fg = cast(Any, _get_applications_fg(args.data_source))
    apps_df: pd.DataFrame = apps_fg.calculate_feature(None, cast(Any, None))
    customer_ids = apps_df["customer_id"].reset_index(drop=True)
    df.insert(0, "customer_id", customer_ids)
    if args.customer:
        df = df[df["customer_id"] == args.customer]
        if df.empty:
            print(f"No customer matching {args.customer!r}.", file=sys.stderr)
            return 1
    df = df.assign(decision=df["credit_risk"].map({1: "good", 0: "bad"}))
    print(df.to_string(index=False))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    # Ensure the active applications FG is registered with mloda before prepare.
    _get_applications_fg(args.data_source)

    features: list[Feature | str] = [Feature.not_typed(name) for name in args.features]
    session = mloda_api.prepare(features=features, compute_frameworks=["PandasDataFrame"])

    if session.engine is None:
        print("Error: could not build execution plan.", file=sys.stderr)
        return 1

    requested = ", ".join(args.features)
    print(f"Execution Plan  [{requested}]")
    print()

    step_num = 0
    fg_steps = []
    for step in session.engine.execution_planner:
        if isinstance(step, FeatureGroupStep):
            step_num += 1
            fg_name = step.feature_group.__name__
            feat_names = sorted(set(str(f.name) for f in step.features.features))
            feat_str = ", ".join(feat_names) if feat_names else ""
            fg_steps.append((step_num, fg_name, feat_str))
            print(f"  [{step_num}] {fg_name:<25}  ->  {feat_str}")

    print()
    input_fgs = [name for _, name, _ in fg_steps[:-1]]
    output_fg = fg_steps[-1][1] if fg_steps else ""
    if len(input_fgs) > 1:
        print(f"  Data flow: {', '.join(input_fgs)} joined → {output_fg}")
    print(f"  {step_num} steps, always computed in this order (deterministic).")
    print()
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    db_path = DEMO_DATA_DIR / "applications.db"
    json_path = DEMO_DATA_DIR / "applications.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.", file=sys.stderr)
        return 1

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS applications")
    cursor.execute("CREATE TABLE applications (id TEXT PRIMARY KEY, requested_amount INTEGER, purpose TEXT)")

    for record in data:
        cursor.execute(
            "INSERT INTO applications (id, requested_amount, purpose) VALUES (?, ?, ?)",
            (record.get("id"), record.get("requested_amount"), record.get("purpose")),
        )

    conn.commit()
    conn.close()

    print(f"Migrated {len(data)} records → {db_path}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    # Ensure the active applications FG is registered with mloda before use.
    _get_applications_fg(args.data_source)

    if args.data_source == "sqlite":
        db_path = DEMO_DATA_DIR / "applications.db"
        if not db_path.exists():
            print("Error: applications.db not found. Run `mloda-demo migrate` first.", file=sys.stderr)
            return 1

    _ensure_artifact()
    none_fs: Any = None
    apps_fg = cast(Any, _get_applications_fg(args.data_source))
    apps = apps_fg.calculate_feature(None, none_fs)
    fin = FinancialsFG.calculate_feature(None, none_fs)
    qa = QuestionnaireFG.calculate_feature(None, none_fs)
    merged = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")
    if args.customer:
        merged = merged[merged["customer_id"] == args.customer]
    if merged.empty:
        print(f"No customer matching {args.customer!r}.", file=sys.stderr)
        return 1

    artifact = load_artifact()
    if artifact is None:
        raise RuntimeError("Classifier artifact missing after _ensure_artifact()")
    encoded = artifact.encoder.encode(merged[FEATURE_COLUMNS])

    method = args.method
    if method in ("Gradient", "GradientInput"):
        attrib_cls: Any = GradientAttributionFeatureGroup
    else:
        attrib_cls = ZennitAttributionFeatureGroup
    model = attrib_cls._load_model(str(MODEL_STATE_PATH))
    relevance = attrib_cls._compute_attributions(model, encoded, method, None)

    customer_ids = merged["customer_id"].tolist()
    for i, customer_id in enumerate(customer_ids):
        row = relevance[i]
        ranked = sorted(zip(FEATURE_COLUMNS, row), key=lambda kv: abs(float(kv[1])), reverse=True)
        print(f"customer_id={customer_id}  method={method}")
        for name, value in ranked[:5]:
            print(f"  {name:>30s}  {float(value):+.4f}")
        print()
    return 0


def _to_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mloda-demo",
        description="Run the applydata 2026 credit-risk demo from the terminal.",
    )
    parser.add_argument(
        "--data-source",
        choices=["json", "sqlite"],
        default="json",
        help="Applications data backend: 'json' (default) or 'sqlite' (requires: mloda-demo migrate).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_disc = sub.add_parser("discover", help="List loaded FeatureGroups and their feature names.")
    p_disc.add_argument("--name", help="Filter FGs by partial name.")
    p_disc.set_defaults(func=cmd_discover)

    p_run = sub.add_parser("run", help="Fetch features via mloda.run_all and print the resulting table.")
    p_run.add_argument("features", nargs="+", help="Feature names to fetch.")
    p_run.add_argument("--customer", help="Filter to one customer_id.")
    p_run.add_argument("--lineage", action="store_true", help="Show data lineage (which FG computed what).")
    p_run.add_argument(
        "--filter",
        action="append",
        metavar="FEATURE,TYPE,VALUE",
        help="Apply row filter (repeatable): FEATURE,range,LO:HI or FEATURE,equal,VAL",
    )
    p_run.set_defaults(func=cmd_run)

    p_pred = sub.add_parser("predict", help="Run the credit-risk classifier (auto-pulls input FGs).")
    p_pred.add_argument("--customer", help="Filter to one customer_id.")
    p_pred.set_defaults(func=cmd_predict)

    p_exp = sub.add_parser("explain", help="Compute XAI attribution for one or all customers.")
    p_exp.add_argument("--customer", help="Filter to one customer_id.")
    p_exp.add_argument(
        "--method",
        default=DEFAULT_XAI_METHOD,
        help=(
            f"XAI method (default: {DEFAULT_XAI_METHOD}). "
            "Options: EpsilonPlus, EpsilonAlpha2Beta1, Gradient, GradientInput."
        ),
    )
    p_exp.set_defaults(func=cmd_explain)

    p_plan = sub.add_parser("plan", help="Show the execution plan (DAG) for a set of features without running.")
    p_plan.add_argument("features", nargs="+", help="Feature names to plan.")
    p_plan.set_defaults(func=cmd_plan)

    p_mig = sub.add_parser("migrate", help="Migrate applications data from JSON to SQLite.")
    p_mig.set_defaults(func=cmd_migrate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
