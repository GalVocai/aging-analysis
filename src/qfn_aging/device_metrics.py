"""Focused per-transistor metrics for the Devices tab.

The analysis tidy table contains roughly 85 metrics in long form.  The
Devices tab needs only the small, user-selected subset below, so it is
pivoted once after loading and then reused for every tree and summary
refresh.  This keeps checkbox interactions independent of Excel I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .allocation import PARC_ONLY, PD_ONLY, PDPARC_H2, PDPARC_NO_H2
from .selection import Selection


@dataclass(frozen=True)
class DeviceMetricSpec:
    key: str
    label: str
    sheet: str
    metric: str
    series: str = ""
    scale: float = 1.0
    decimals: int = 1


# Keep this order: it is the left-to-right order in both Devices tables.
# JG_H2 stores voltage shifts in V; the user requested a common mV display.
DEVICE_METRICS: tuple[DeviceMetricSpec, ...] = (
    DeviceMetricSpec("dvth03_bg0", "ΔVth03 BG0 [mV]", "JG_n03", "dVth_bg0_mV"),
    DeviceMetricSpec("dvth03_bg4", "ΔVth03 BG4 [mV]", "JG_n03", "dVth_bg4_mV"),
    DeviceMetricSpec("dvth_h2_bg0", "ΔVth H2 BG0 [mV]", "JG_H2", "delta_Vth_H2",
                     "Vbg=0", 1000.0),
    DeviceMetricSpec("dvth_h2_bg4", "ΔVth H2 BG4 [mV]", "JG_H2", "delta_Vth_H2",
                     "Vbg=4", 1000.0),
    DeviceMetricSpec("dvth_rec_bg0", "ΔVth Rec BG0 [mV]", "JG_H2",
                     "delta_Vth_recovery", "Vbg=0", 1000.0),
    DeviceMetricSpec("dvth_rec_bg4", "ΔVth Rec BG4 [mV]", "JG_H2",
                     "delta_Vth_recovery", "Vbg=4", 1000.0),
    DeviceMetricSpec("max_resp_bg0", "Max Resp BG0 [%]", "JG_H2", "max_resp_H2_pct",
                     "Vbg=0"),
    DeviceMetricSpec("max_resp_bg4", "Max Resp BG4 [%]", "JG_H2", "max_resp_H2_pct",
                     "Vbg=4"),
    DeviceMetricSpec("delta_id", "ΔId [nA]", "SE_n04", "response_delta"),
    DeviceMetricSpec("response_time", "Response t90 [s]", "SE_n04", "t90_response_s"),
)

DEVICE_METRIC_KEYS: tuple[str, ...] = tuple(spec.key for spec in DEVICE_METRICS)
CATEGORY_ORDER: tuple[str, ...] = (PARC_ONLY, PD_ONLY, PDPARC_H2, PDPARC_NO_H2)
KEY_COLUMNS: tuple[str, ...] = ("timepoint", "esn", "tnumber")


def format_metric(value, spec: DeviceMetricSpec) -> str:
    """Compact table text; missing/non-numeric values are an em dash."""
    if pd.isna(value):
        return "—"
    return f"{float(value):.{spec.decimals}f}"


def build_device_metric_table(tidy: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (day, device, transistor), ten metric columns wide."""
    empty = pd.DataFrame(columns=[*KEY_COLUMNS, *DEVICE_METRIC_KEYS])
    if tidy.empty:
        return empty

    pieces: list[pd.DataFrame] = []
    series = tidy["series"].fillna("") if "series" in tidy else pd.Series("", index=tidy.index)
    for spec in DEVICE_METRICS:
        mask = ((tidy["sheet"] == spec.sheet)
                & (tidy["metric"] == spec.metric)
                & (series == spec.series))
        part = tidy.loc[mask, [*KEY_COLUMNS, "value"]].copy()
        if part.empty:
            continue
        part["value"] = pd.to_numeric(part["value"], errors="coerce") * spec.scale
        part["metric_key"] = spec.key
        pieces.append(part)

    if not pieces:
        return empty

    long = pd.concat(pieces, ignore_index=True).dropna(subset=["value"])
    # Duplicate source rows should not multiply a transistor in the UI.  Mean
    # is deterministic and matches the aggregation semantics used elsewhere.
    wide = (long.pivot_table(index=list(KEY_COLUMNS), columns="metric_key",
                             values="value", aggfunc="mean")
                 .reset_index())
    wide.columns.name = None
    for key in DEVICE_METRIC_KEYS:
        if key not in wide:
            wide[key] = float("nan")
    return wide[[*KEY_COLUMNS, *DEVICE_METRIC_KEYS]]


def category_summary(rows: pd.DataFrame, selection: Selection,
                     timepoint: str) -> pd.DataFrame:
    """Mean, AVDEV and n for each category/metric in one condition tab."""
    out_rows: list[dict] = []
    if rows.empty:
        return pd.DataFrame(columns=["category", "metric", "mean", "avdev", "n"])

    included = [selection.is_included(timepoint, str(r.esn), int(r.tnumber))
                for r in rows.itertuples()]
    kept = rows.loc[included]
    for category in CATEGORY_ORDER:
        group = kept[kept["category"] == category]
        for spec in DEVICE_METRICS:
            values = pd.to_numeric(group.get(spec.key, pd.Series(dtype=float)),
                                   errors="coerce").dropna()
            mean = values.mean() if not values.empty else float("nan")
            avdev = (values - mean).abs().mean() if not values.empty else float("nan")
            out_rows.append({
                "category": category,
                "metric": spec.key,
                "mean": mean,
                "avdev": avdev,
                "n": int(values.count()),
            })
    return pd.DataFrame(out_rows)
