"""Generate synthetic financial-overview xlsx + questionnaire md per customer.

Run from the repo root:
    python scripts/synthesize_docs.py

Inputs:
    demo_data/applications.json  (customer ids + purpose)

Outputs:
    demo_data/financial_overview_<customer_id>.xlsx   (numeric German-Credit fields)
    demo_data/qa_<customer_id>.md                     (categorical German-Credit fields)

Each customer's random draws are seeded from their customer_id for reproducibility.
All schemas align directly to OpenML German Credit (id 31) so the merged row can be
fed to TabPFN with the downloaded CSV as the support set.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

CHECKING_STATUS = ["<0", "0<=X<200", ">=200", "no checking"]
CREDIT_HISTORY = [
    "no credits/all paid",
    "all paid",
    "existing paid",
    "delayed previously",
    "critical/other existing credit",
]
SAVINGS_STATUS = ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"]
EMPLOYMENT = ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
PERSONAL_STATUS = ["male div/sep", "female div/dep/mar", "male single", "male mar/wid"]
OTHER_PARTIES = ["none", "co applicant", "guarantor"]
PROPERTY_MAGNITUDE = ["real estate", "life insurance", "car", "no known property"]
OTHER_PAYMENT_PLANS = ["bank", "stores", "none"]
HOUSING = ["rent", "own", "for free"]
JOB = [
    "unemp/unskilled non res",
    "unskilled resident",
    "skilled",
    "high qualif/self emp/mgmt",
]
OWN_TELEPHONE = ["yes", "none"]
FOREIGN_WORKER = ["yes", "no"]


def seed_for(customer_id: str) -> int:
    digest = hashlib.md5(customer_id.encode(), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def synth_financials(rng: random.Random) -> dict[str, int]:
    return {
        "duration": rng.choice([6, 12, 18, 24, 36, 48]),
        "installment_commitment": rng.randint(1, 4),
        "residence_since": rng.randint(1, 4),
        "age": rng.randint(22, 68),
        "existing_credits": rng.randint(1, 4),
        "num_dependents": rng.randint(1, 2),
    }


def synth_questionnaire(rng: random.Random) -> dict[str, str]:
    return {
        "checking_status": rng.choice(CHECKING_STATUS),
        "credit_history": rng.choice(CREDIT_HISTORY),
        "savings_status": rng.choice(SAVINGS_STATUS),
        "employment": rng.choice(EMPLOYMENT),
        "personal_status": rng.choice(PERSONAL_STATUS),
        "other_parties": rng.choice(OTHER_PARTIES),
        "property_magnitude": rng.choice(PROPERTY_MAGNITUDE),
        "other_payment_plans": rng.choice(OTHER_PAYMENT_PLANS),
        "housing": rng.choice(HOUSING),
        "job": rng.choice(JOB),
        "own_telephone": rng.choice(OWN_TELEPHONE),
        "foreign_worker": rng.choice(FOREIGN_WORKER),
    }


def write_financial_xlsx(path: Path, customer_id: str, fields: dict[str, int]) -> None:
    rows = [{"field": k, "value": v} for k, v in fields.items()]
    df = pd.DataFrame(rows)
    df.attrs["customer_id"] = customer_id
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="financials", index=False)


def write_questionnaire_md(path: Path, customer_id: str, applicant_name: str, fields: dict[str, str]) -> None:
    lines = [
        f"# Customer Questionnaire: {applicant_name}",
        "",
        f"customer_id: {customer_id}",
        "",
        "## Banking Relationship",
        f"- checking_status: {fields['checking_status']}",
        f"- credit_history: {fields['credit_history']}",
        f"- savings_status: {fields['savings_status']}",
        "",
        "## Employment and Status",
        f"- employment: {fields['employment']}",
        f"- personal_status: {fields['personal_status']}",
        f"- job: {fields['job']}",
        "",
        "## Obligations and Collateral",
        f"- other_parties: {fields['other_parties']}",
        f"- property_magnitude: {fields['property_magnitude']}",
        f"- other_payment_plans: {fields['other_payment_plans']}",
        f"- housing: {fields['housing']}",
        "",
        "## Contact",
        f"- own_telephone: {fields['own_telephone']}",
        f"- foreign_worker: {fields['foreign_worker']}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    demo_data = repo_root / "demo_data"
    applications = json.loads((demo_data / "applications.json").read_text())

    for app in applications:
        cid = app["id"]
        rng = random.Random(seed_for(cid))  # nosec B311

        financials = synth_financials(rng)
        questionnaire = synth_questionnaire(rng)

        write_financial_xlsx(demo_data / f"financial_overview_{cid}.xlsx", cid, financials)
        write_questionnaire_md(demo_data / f"qa_{cid}.md", cid, app["applicant_name"], questionnaire)
        print(f"  {cid}: financials={financials} qa={questionnaire}")

    print(f"\nGenerated docs for {len(applications)} customers in {demo_data}")


if __name__ == "__main__":
    main()
