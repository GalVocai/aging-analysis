"""Group-average layer: aggregate per-device metric values into
(condition, category) group means, mirroring sensorlab's degradation.py
(mean +/- AVDEV) but keyed on this study's own axes -- see
[[project-qfn-aging]] grouping rule: group by (condition x category) only,
never by Type A/B.

Input shape: a tidy long-form table with one row per (esn, timepoint,
metric, value) -- the eventual output of a not-yet-built measurement
ingestion stage. This module is deliberately decoupled from ingestion so
it can be tested with synthetic data now and wired to real result files
later without changing this code.
"""

from __future__ import annotations

import pandas as pd

from .allocation import Allocation

# Rows are averaged only across devices/transistors. Everything that
# identifies *what was measured* stays in the key -- notably `sheet`, since
# one master can hold several majors of the same kind (JG_n03 air vs JG_n05
# under H2), and `series`, which carries sub-keys like ZOZO's segment.
GROUP_COLS: tuple[str, ...] = ("timepoint", "condition", "category",
                               "sheet", "series", "metric")


def add_group_keys(df: pd.DataFrame, alloc: Allocation) -> pd.DataFrame:
    """Join condition + category onto a tidy table keyed by an 'esn' column."""
    df = df.copy()
    df["esn"] = df["esn"].astype(str)
    df["condition"] = df["esn"].map(lambda e: alloc.device(e).condition)
    df["category"] = df["esn"].map(lambda e: alloc.device(e).category)
    return df


def _avdev(s: pd.Series) -> float:
    """Excel AVEDGE-equivalent: mean absolute deviation from the mean."""
    return (s - s.mean()).abs().mean()


def aggregate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group key (see :data:`GROUP_COLS`): mean, avdev, n.

    ``df`` must already carry 'condition' and 'category' columns (see
    :func:`add_group_keys`) plus 'timepoint', 'metric', 'value'. Missing
    optional key columns ('sheet', 'series') are filled with "" so a simple
    table without them still aggregates correctly.

    If an ``included`` column is present (see
    :func:`qfn_aging.selection.apply_selection`), only rows the user kept
    are averaged; excluded rows stay in the input for plotting but do not
    influence mean, avdev or n. A group whose transistors are all excluded
    disappears from the output rather than reporting a mean of nothing.

    ``n`` counts transistors, not devices -- every included transistor in
    the group contributes equally to the mean.
    """
    df = df.copy()
    for col in GROUP_COLS:
        if col not in df.columns:
            df[col] = ""
    if "included" in df.columns:
        df = df[df["included"].astype(bool)]
    if df.empty:
        return pd.DataFrame(columns=[*GROUP_COLS, "mean", "avdev", "n"])
    grouped = df.groupby(list(GROUP_COLS), dropna=False)["value"]
    return grouped.agg(mean="mean", avdev=_avdev, n="count").reset_index()
