"""A tax office and its clients' books: the folders, the rule that admits a booking, the chart the office reads."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from mloda.community.feature_groups.experimental.german_ledger import AdmissibilityPolicyGroup
from mloda.community.feature_groups.experimental.german_ledger.policy import (
    DatevJournalFeatureGroup,
    JournalFeatureGroup,
    PeriodBound,
)
from mloda.community.feature_groups.experimental.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup
from mloda.provider import FeatureGroup
from mloda.user import DataAccessCollection, PluginCollector

DATA = Path(__file__).resolve().parents[2] / "demo_data" / "runtime_lineage"
CLIENT_A = DATA / "client_a"  # DATEV batch, states SKR04
CLIENT_B = DATA / "client_b"  # DATEV batch in SKR03, states no chart
CLIENT_B_CORRECTED = DATA / "client_b_corrected"  # the same bookings, states SKR03
CLIENT_C = DATA / "client_c"  # client A's books as a GDPdU dossier
BOOKS = DataAccessCollection(folders={str(folder) for folder in (CLIENT_A, CLIENT_B, CLIENT_B_CORRECTED, CLIENT_C)})


class Closing2025(AdmissibilityPolicyGroup):
    """Business year 2025: a booking counts when it is dated inside the year."""

    RULES = (PeriodBound(date(2025, 1, 1), date(2025, 12, 31)),)


class Skr03Accounts(SkrAccountFeatureGroup):
    """The office reading a client's books under SKR03."""

    CHART = "03"


_SHARED: frozenset[type[FeatureGroup]] = frozenset(
    {JournalFeatureGroup, DatevJournalFeatureGroup, Closing2025, SourcesFeatureGroup}
)
# A run enables one chart, never both.
READS_SKR04 = PluginCollector.enabled_feature_groups({*_SHARED, SkrAccountFeatureGroup})
READS_SKR03 = PluginCollector.enabled_feature_groups({*_SHARED, Skr03Accounts})
