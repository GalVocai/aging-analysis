"""Which transistors take part in the group averages, **per measurement day**.

This is user state, not analysis logic. Nothing here decides whether a
transistor is good -- it only records what the user decided, so the GUI can
edit it and the aggregation can respect it.

The key is ``(timepoint, esn, tnumber)``. Per-day matters physically: a
channel can read fine at baseline and misbehave at 6D, and excluding it
everywhere would throw away good baseline data. Excluding a whole device is
just excluding each of its transistors on that day.

``ALL_DAYS`` marks an exclusion that applies to every timepoint -- useful
for a channel that never worked, and the migration path for selections
saved before this was per-day.

Semantics, matching sensorlab's convention:

* Excluded transistors are **kept** in the tidy table, marked ``included=False``.
* They are **left out of the group mean / AVDEV / n** for that day only.
* Per-device plots can grey them out rather than dropping them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: This project's own user-state directory. Deliberately NOT ``~/.sensorlab``.
STATE_DIR = Path.home() / ".qfn_aging"
DEFAULT_SELECTION_PATH = STATE_DIR / "selection.json"
#: Small UI state (last data root, etc.) so the app reopens where you left it.
SETTINGS_PATH = STATE_DIR / "settings.json"

#: Timepoint value meaning "every day".
ALL_DAYS = "*"

_Key = tuple[str, str, int]


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    """UI preferences. Unreadable or missing file means 'no preferences'."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict, path: Path = SETTINGS_PATH) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass          # UI convenience only -- never block the app over it


@dataclass
class Selection:
    """Excluded ``(timepoint, esn, tnumber)`` triples, with a note each."""

    excluded: dict[_Key, str] = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _key(timepoint: str, esn: str, tnumber: int) -> _Key:
        return (str(timepoint), str(esn), int(tnumber))

    # -- queries ---------------------------------------------------------

    def is_included(self, timepoint: str, esn: str, tnumber: int) -> bool:
        """False if excluded for this day, or excluded for all days."""
        return not (self._key(timepoint, esn, tnumber) in self.excluded
                    or self._key(ALL_DAYS, esn, tnumber) in self.excluded)

    def reason(self, timepoint: str, esn: str, tnumber: int) -> str:
        return (self.excluded.get(self._key(timepoint, esn, tnumber))
                or self.excluded.get(self._key(ALL_DAYS, esn, tnumber), ""))

    def excluded_for(self, timepoint: str, esn: str) -> list[int]:
        """Transistors of a device excluded on this day (including all-days)."""
        esn = str(esn)
        out = {t for (tp, e, t) in self.excluded
               if e == esn and tp in (str(timepoint), ALL_DAYS)}
        return sorted(out)

    def timepoints(self) -> list[str]:
        return sorted({tp for tp, _, _ in self.excluded})

    def count_for(self, timepoint: str) -> int:
        """How many exclusions affect this day (day-specific + all-days)."""
        return sum(1 for tp, _, _ in self.excluded
                   if tp in (str(timepoint), ALL_DAYS))

    def __len__(self) -> int:
        return len(self.excluded)

    # -- edits -----------------------------------------------------------

    def exclude(self, timepoint: str, esn: str, tnumber: int, reason: str = "") -> None:
        self.excluded[self._key(timepoint, esn, tnumber)] = reason

    def include(self, timepoint: str, esn: str, tnumber: int) -> None:
        """Re-include for this day.

        An all-days exclusion is dropped too, otherwise re-ticking a row that
        was excluded globally would appear to do nothing.
        """
        self.excluded.pop(self._key(timepoint, esn, tnumber), None)
        self.excluded.pop(self._key(ALL_DAYS, esn, tnumber), None)

    def exclude_device(self, timepoint: str, esn: str, tnumbers: list[int],
                       reason: str = "") -> None:
        for t in tnumbers:
            self.exclude(timepoint, esn, t, reason)

    def include_device(self, timepoint: str, esn: str, tnumbers: list[int]) -> None:
        for t in tnumbers:
            self.include(timepoint, esn, t)

    def clear(self, timepoint: str | None = None) -> None:
        """Clear everything, or just one day's entries."""
        if timepoint is None:
            self.excluded.clear()
            return
        for key in [k for k in self.excluded if k[0] == str(timepoint)]:
            del self.excluded[key]

    # -- persistence -----------------------------------------------------

    def to_records(self) -> list[dict]:
        return [
            {"timepoint": tp, "esn": e, "tnumber": t, "reason": r}
            for (tp, e, t), r in sorted(self.excluded.items())
        ]

    def save(self, path: Path = DEFAULT_SELECTION_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 2, "excluded": self.to_records()},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_selection(path: Path = DEFAULT_SELECTION_PATH) -> Selection:
    """Load the saved selection.

    Records written before exclusions were per-day carry no ``timepoint``;
    they are read as :data:`ALL_DAYS` so behaviour is unchanged rather than
    silently dropping them. A missing or unreadable file means 'nothing
    excluded' -- never an error, so a corrupt file cannot block analysis.
    """
    path = Path(path)
    sel = Selection()
    if not path.is_file():
        return sel
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("excluded", []):
            sel.exclude(
                rec.get("timepoint", ALL_DAYS),      # legacy -> all days
                rec["esn"], rec["tnumber"], rec.get("reason", ""),
            )
    except Exception:
        return Selection()
    return sel


def apply_selection(df: pd.DataFrame, selection: Selection | None) -> pd.DataFrame:
    """Add ``included`` / ``exclude_reason`` columns to a tidy table.

    Matching is per row, so the same transistor can be in on one day and out
    on another. Rows are never dropped --
    :func:`qfn_aging.grouping.aggregate_groups` averages only the included
    ones; plotting can grey out the rest.
    """
    df = df.copy()
    if selection is None or not selection.excluded:
        df["included"] = True
        df["exclude_reason"] = ""
        return df

    keys = list(zip(df["timepoint"].astype(str), df["esn"].astype(str),
                    df["tnumber"].astype(int)))
    day_hit = [k in selection.excluded for k in keys]
    all_hit = [(ALL_DAYS, e, t) in selection.excluded for _, e, t in keys]

    df["included"] = [not (d or a) for d, a in zip(day_hit, all_hit)]
    df["exclude_reason"] = [
        selection.excluded.get(k) or selection.excluded.get((ALL_DAYS, k[1], k[2]), "")
        for k in keys
    ]
    return df
