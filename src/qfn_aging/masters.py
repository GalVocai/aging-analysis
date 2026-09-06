"""Read sensorlab's master workbooks back into one tidy table.

Layout produced by :mod:`qfn_aging.runner`::

    <analysis_root>/<timepoint>/<run_key>/<condition>/<MODE>/master_<MODE>.xlsx

A master workbook can hold **several sheets**, one per major of the same
kind -- e.g. the JG master of a hydrogen run contains ``JG_n03`` (air),
``JG_n05`` (under H2) and ``JG_n06`` (recovery). Those are different
physical measurements and must never be averaged together, so the sheet
name is carried through and becomes part of the grouping key.

Some masters also key rows by more than (ESN, TNumber): ZOZO emits one row
per settling segment and working-point state. Those sub-keys are folded
into a ``series`` string so that each (metric, series) pair is a distinct
curve rather than being silently averaged across segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

# Columns that identify a row rather than measure something.
_ID_COLS = {"ESN", "TNumber", "condition", "day"}
# Extra row keys that subdivide a metric into separate curves.
_SERIES_COLS = ("state_Vj", "segment", "Vbg", "part")


def _series_label(row: pd.Series, cols: list[str]) -> str:
    if not cols:
        return ""
    return "|".join(f"{c}={row[c]}" for c in cols)


def _tidy_sheet(df: pd.DataFrame, timepoint: str, run_key: str, mode: str,
                sheet: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    series_cols = [c for c in _SERIES_COLS if c in df.columns]
    id_cols = set(_ID_COLS) | set(series_cols)
    # flags/notes are diagnostics, not metrics; keep only numeric measurements
    metric_cols = [
        c for c in df.columns
        if c not in id_cols
        and not str(c).startswith("flags")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        return pd.DataFrame()

    out = df.melt(
        id_vars=[c for c in df.columns if c in id_cols],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    out["series"] = (
        out.apply(lambda r: _series_label(r, series_cols), axis=1) if series_cols else ""
    )
    out = out.rename(columns={"ESN": "esn", "TNumber": "tnumber"})
    out["esn"] = out["esn"].astype(str)
    out["timepoint"] = timepoint
    out["run_key"] = run_key
    out["mode"] = mode
    out["sheet"] = sheet

    keep = ["esn", "tnumber", "timepoint", "run_key", "mode", "sheet",
            "series", "metric", "value"]
    return out[[c for c in keep if c in out.columns]].dropna(subset=["value"])


def load_masters(analysis_root: str | Path,
                 modes: list[str] | None = None,
                 progress: Callable[[int, int, str], None] | None = None,
                 ) -> pd.DataFrame:
    """Load every master under ``analysis_root`` into one tidy frame.

    Returns columns: esn, tnumber, timepoint, run_key, mode, sheet, series,
    metric, value. ``condition`` is deliberately NOT taken from the files --
    it comes from this project's roster via
    :func:`qfn_aging.grouping.add_group_keys`, so the roster stays the single
    source of truth.

    ``progress`` is called as ``progress(done, total, label)`` before each
    workbook is read, so a GUI can show real progress rather than freezing.
    Reading ~130 workbooks takes several seconds.
    """
    analysis_root = Path(analysis_root)
    frames: list[pd.DataFrame] = []

    paths = [p for p in sorted(analysis_root.glob("*/*/*/*/master_*.xlsx"))
             if not p.name.startswith("~$")
             and not (modes and p.parent.name not in modes)]
    total = len(paths)

    for done, path in enumerate(paths):
        mode = path.parent.name
        if progress is not None:
            progress(done, total, f"{path.parents[2].name} / {path.parents[1].name} / {mode}")
        # <root>/<timepoint>/<run_key>/<condition>/<MODE>/master_<MODE>.xlsx
        timepoint = path.parents[3].name
        run_key = path.parents[2].name
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception:
            continue
        for sheet, df in sheets.items():
            # sensorlab's population diagnostics, not measurements
            if sheet.endswith("_groups"):
                continue
            tidy = _tidy_sheet(df, timepoint, run_key, mode, sheet)
            if not tidy.empty:
                frames.append(tidy)

    if progress is not None:
        progress(total, total, "done")

    if not frames:
        return pd.DataFrame(columns=["esn", "tnumber", "timepoint", "run_key",
                                     "mode", "sheet", "series", "metric", "value"])
    return pd.concat(frames, ignore_index=True)
