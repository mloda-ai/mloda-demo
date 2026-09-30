"""Build demo_data/retail/ledger.csv.gz from UCI Online Retail II.

Run from the repo root (openpyxl is in the credit-risk extra):
    uv run --extra credit-risk python scripts/fetch_retail.py [path/to/online_retail_II.xlsx]
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = Path(__file__).resolve().parent.parent / "demo_data" / "retail" / "ledger.csv.gz"
START, END = "2010-06-01", "2010-09-01"


def read_workbook(path: str | None) -> dict[str, pd.DataFrame]:
    if path is not None:
        return pd.read_excel(path, sheet_name=None)
    # URL is a fixed https address.
    with urllib.request.urlopen(URL, timeout=60) as response:  # nosec B310
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("online_retail_II.xlsx") as workbook:
        return pd.read_excel(workbook, sheet_name=None)


def main() -> None:
    raw = pd.concat(read_workbook(sys.argv[1] if len(sys.argv) > 1 else None).values(), ignore_index=True)
    raw = raw.dropna(subset=["Customer ID"]).drop_duplicates()  # the two yearly sheets overlap in December 2010
    invoice = raw["Invoice"].astype(str)
    raw = raw[~invoice.str.startswith("A")]  # bad-debt adjustments, not orders
    ledger = pd.DataFrame(
        {
            "invoice": raw["Invoice"].astype(str),
            "customer_id": raw["Customer ID"].astype("int64"),
            "invoice_date": raw["InvoiceDate"],
            "quantity": raw["Quantity"].astype("int64"),
            "price": raw["Price"].astype("float64"),
        }
    )
    ledger = ledger[(ledger["invoice_date"] >= START) & (ledger["invoice_date"] < END)]
    ledger = ledger.sort_values(["invoice_date", "invoice"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Wrote {len(ledger)} lines, {ledger['customer_id'].nunique()} customers to {OUT}")


if __name__ == "__main__":
    main()
