"""Comparison curves: group averages sliced along one axis.

There are three axes a group is defined by -- **temperature**, **RH**, and
**category**. Every comparison fixes two of them and overlays one curve per
value of the third:

======================  ==========================================
vary                    reads as
======================  ==========================================
``"temp"``              fixed RH + category, one curve per 25/35/45/60 C
``"rh"``                fixed temperature + category, one curve per 10/30/50 %
``"category"``          fixed condition, one curve per Pd / Pd|ParC / control
======================  ==========================================

Each curve is the (condition x category) group average across timepoints --
mean and AVDEV as produced by :mod:`qfn_aging.grouping`, which already
respects the user's transistor selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .schedule import parse_timepoint, sort_timepoints

_CONDITION_RE = re.compile(r"^(\d+)C_(\d+)RH$")

#: Study day for each timepoint, used as the x axis.
TIMEPOINT_DAYS: dict[str, int] = {"baseline": 0, "3D": 3, "6D": 6, "7D": 7,
                                  "9D": 9, "13D": 13}

VARY_AXES = ("temp", "rh", "category")

#: Human labels for the axis being varied.
AXIS_LABELS = {"temp": "Temperature", "rh": "Relative humidity", "category": "Category"}


#: JG_n03 threshold-voltage drift, expressed vs the baseline day, in mV.
#: These derived metrics let the Comparisons tab reproduce sensorlab's
#: dedicated "ΔVth [mV]" degradation plot (per-device shift, sign = physical:
#: a Vth that becomes more negative drifts negative). JG_n03 only, by request.
DRIFT_SHEET = "JG_n03"
DRIFT_SOURCE_METRICS = ("Vth_bg0", "Vth_bg4")
DRIFT_BASELINE_TP = "baseline"
DRIFT_SCALE = 1000.0                       # V -> mV
DRIFT_YLABEL = "ΔVth [mV]"


def _drift_metric_name(metric: str) -> str:
    """``"Vth_bg0"`` -> ``"dVth_bg0_mV"``."""
    return "d" + metric + "_mV"


#: Metric ids produced by :func:`add_jg_vth_drift`.
DRIFT_METRICS = tuple(_drift_metric_name(m) for m in DRIFT_SOURCE_METRICS)


def add_jg_vth_drift(tidy: pd.DataFrame, *,
                     sheet: str = DRIFT_SHEET,
                     metrics: tuple[str, ...] = DRIFT_SOURCE_METRICS,
                     baseline_tp: str = DRIFT_BASELINE_TP,
                     scale: float = DRIFT_SCALE) -> pd.DataFrame:
    """Append per-device JG_n03 Vth *drift* rows (ΔVth vs baseline, in mV).

    For every ``(esn, tnumber)`` the drift at a timepoint is
    ``(Vth_day - Vth_baseline_day) * scale``: the shift relative to that
    device's OWN baseline-day measurement. Sign is physical -- a Vth that
    moves more negative gives a negative drift -- and the baseline day is 0
    for every device by construction. Devices with no baseline reading get
    NaN, so they simply drop out of the average.

    The new rows carry metric ids ``dVth_bg0_mV`` / ``dVth_bg4_mV`` and flow
    through aggregation, the selection filter and plotting exactly like any
    measured metric. This is additive -- the original ``Vth_bg*`` rows stay.
    """
    if tidy.empty:
        return tidy
    sub = tidy[(tidy["sheet"] == sheet) & (tidy["metric"].isin(metrics))].copy()
    if sub.empty:
        return tidy

    sub["tnumber"] = sub["tnumber"].astype(int)
    base = (sub[sub["timepoint"].astype(str) == baseline_tp]
            .drop_duplicates(["esn", "tnumber", "metric"])
            .set_index(["esn", "tnumber", "metric"])["value"])
    keys = list(zip(sub["esn"], sub["tnumber"], sub["metric"]))
    base_vals = base.reindex(keys).to_numpy()
    sub["value"] = (sub["value"].to_numpy() - base_vals) * scale
    sub["metric"] = sub["metric"].map(_drift_metric_name)
    return pd.concat([tidy, sub], ignore_index=True)


def ylabel_for_metric(metric: str) -> str | None:
    """Pretty y-axis label for derived metrics; ``None`` = use the metric id."""
    if metric in DRIFT_METRICS:
        return DRIFT_YLABEL
    return None


def parse_condition(condition: str) -> tuple[int | None, int | None]:
    """``"25C_10RH"`` -> ``(25, 10)``; anything unrecognised -> ``(None, None)``."""
    m = _CONDITION_RE.match(str(condition))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def add_condition_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Split ``condition`` into ``temp_c`` / ``rh_pct`` and add ``day``."""
    df = df.copy()
    parsed = df["condition"].map(parse_condition)
    df["temp_c"] = [p[0] for p in parsed]
    df["rh_pct"] = [p[1] for p in parsed]
    df["day"] = df["timepoint"].map(parse_timepoint)
    return df


@dataclass(frozen=True)
class ComparisonSpec:
    """What to draw. ``fixed`` holds the two axes being held constant."""

    vary: str                      # "temp" | "rh" | "category"
    metric: str
    sheet: str
    series: str = ""
    fixed: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.vary not in VARY_AXES:
            raise ValueError(f"vary must be one of {VARY_AXES}, got {self.vary!r}")

    @property
    def required_fixed(self) -> tuple[str, ...]:
        return tuple(a for a in ("temp", "rh", "category") if a != self.vary)

    def title(self) -> str:
        bits = []
        if "temp" in self.fixed:
            bits.append(f"{self.fixed['temp']}°C")
        if "rh" in self.fixed:
            bits.append(f"{self.fixed['rh']}% RH")
        if "category" in self.fixed:
            bits.append(str(self.fixed["category"]))
        where = ", ".join(bits)
        return f"{self.metric} — {AXIS_LABELS[self.vary]} comparison at {where}"


def _column_for(axis: str) -> str:
    return {"temp": "temp_c", "rh": "rh_pct", "category": "category"}[axis]


def build_curves(agg: pd.DataFrame, spec: ComparisonSpec) -> pd.DataFrame:
    """Slice aggregated group data down to the curves for one comparison.

    ``agg`` is the output of :func:`qfn_aging.grouping.aggregate_groups`.
    Returns one row per (curve value, timepoint) with mean/avdev/n, sorted so
    the curve values come out in a natural order.
    """
    if agg.empty:
        return pd.DataFrame(columns=["curve", "timepoint", "day", "mean", "avdev", "n"])

    df = add_condition_axes(agg)
    df = df[(df["metric"] == spec.metric) & (df["sheet"] == spec.sheet)]
    if "series" in df.columns:
        df = df[df["series"].fillna("") == (spec.series or "")]

    for axis in spec.required_fixed:
        if axis not in spec.fixed:
            raise ValueError(f"comparison varying {spec.vary!r} needs a fixed {axis!r}")
        df = df[df[_column_for(axis)] == spec.fixed[axis]]

    if df.empty:
        return pd.DataFrame(columns=["curve", "timepoint", "day", "mean", "avdev", "n"])

    out = df.rename(columns={_column_for(spec.vary): "curve"})
    out = out[["curve", "timepoint", "day", "mean", "avdev", "n"]].copy()
    out = out.dropna(subset=["curve"])

    # numeric axes sort numerically, category sorts alphabetically
    out = out.sort_values(["curve", "day"], kind="stable").reset_index(drop=True)
    return out


def available_options(agg: pd.DataFrame) -> dict[str, list]:
    """What can actually be plotted from this data -- drives the GUI pickers."""
    if agg.empty:
        return {"sheet": [], "metric": [], "series": [],
                "temp": [], "rh": [], "category": []}
    df = add_condition_axes(agg)
    return {
        "sheet": sorted(df["sheet"].dropna().unique()),
        "metric": sorted(df["metric"].dropna().unique()),
        "series": sorted(df.get("series", pd.Series(dtype=str)).fillna("").unique()),
        "temp": sorted(int(t) for t in df["temp_c"].dropna().unique()),
        "rh": sorted(int(r) for r in df["rh_pct"].dropna().unique()),
        "category": sorted(df["category"].dropna().unique()),
    }


def metrics_for_sheet(agg: pd.DataFrame, sheet: str) -> list[str]:
    if agg.empty:
        return []
    return sorted(agg.loc[agg["sheet"] == sheet, "metric"].dropna().unique())


def timepoints_present(agg: pd.DataFrame) -> list[str]:
    if agg.empty:
        return []
    return sort_timepoints(agg["timepoint"].dropna().unique())
