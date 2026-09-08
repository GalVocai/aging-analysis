"""Focused metric table and live category-summary calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from qfn_aging.device_metrics import (DEVICE_METRIC_KEYS, build_device_metric_table,
                                      category_summary)
from qfn_aging.selection import Selection


def _row(esn, tnumber, sheet, metric, value, series="", timepoint="3D"):
    return dict(esn=str(esn), tnumber=tnumber, timepoint=timepoint, run_key="r",
                mode="m", sheet=sheet, series=series, metric=metric, value=value)


def test_builds_requested_metrics_and_converts_jg_h2_to_mv():
    tidy = pd.DataFrame([
        _row("1236", 11, "JG_n03", "dVth_bg0_mV", -12.5),
        _row("1236", 11, "JG_n03", "dVth_bg4_mV", 25.0),
        _row("1236", 11, "JG_H2", "delta_Vth_H2", 0.125, "Vbg=0"),
        _row("1236", 11, "JG_H2", "delta_Vth_H2", 0.150, "Vbg=4"),
        _row("1236", 11, "JG_H2", "delta_Vth_recovery", 0.020, "Vbg=0"),
        _row("1236", 11, "JG_H2", "max_resp_H2_pct", 321.0, "Vbg=0"),
        _row("1236", 11, "JG_H2", "max_resp_recovery_pct", 999.0, "Vbg=0"),
        _row("1236", 11, "SE_n04", "response_delta", 4500.0),
        _row("1236", 11, "SE_n04", "t90_response_s", 120.0),
    ])

    wide = build_device_metric_table(tidy)
    row = wide.iloc[0]

    assert row["dvth03_bg0"] == pytest.approx(-12.5)
    assert row["dvth_h2_bg0"] == pytest.approx(125.0)
    assert row["dvth_h2_bg4"] == pytest.approx(150.0)
    assert row["dvth_rec_bg0"] == pytest.approx(20.0)
    assert row["max_resp_bg0"] == pytest.approx(321.0)
    assert row["delta_id"] == pytest.approx(4500.0)
    assert row["response_time"] == pytest.approx(120.0)
    assert "max_resp_recovery" not in DEVICE_METRIC_KEYS


def test_missing_metrics_remain_empty_columns():
    tidy = pd.DataFrame([_row("1236", 11, "SE_n04", "response_delta", 1.0)])
    wide = build_device_metric_table(tidy)

    assert list(wide.columns[3:]) == list(DEVICE_METRIC_KEYS)
    assert pd.isna(wide.iloc[0]["dvth03_bg0"])


def test_category_summary_respects_selection_and_uses_avdev():
    rows = pd.DataFrame([
        dict(timepoint="3D", esn="1236", tnumber=11, category="PdParC_H2",
             dvth03_bg0=10.0),
        dict(timepoint="3D", esn="1236", tnumber=12, category="PdParC_H2",
             dvth03_bg0=20.0),
        dict(timepoint="3D", esn="1250", tnumber=11, category="PdParC_H2",
             dvth03_bg0=90.0),
    ])
    selection = Selection()

    before = category_summary(rows, selection, "3D")
    metric = before[(before.category == "PdParC_H2")
                    & (before.metric == "dvth03_bg0")].iloc[0]
    assert metric["mean"] == pytest.approx(40.0)
    assert metric["avdev"] == pytest.approx(100 / 3)
    assert metric["n"] == 3

    selection.exclude("3D", "1250", 11)
    after = category_summary(rows, selection, "3D")
    metric = after[(after.category == "PdParC_H2")
                   & (after.metric == "dvth03_bg0")].iloc[0]
    assert metric["mean"] == pytest.approx(15.0)
    assert metric["avdev"] == pytest.approx(5.0)
    assert metric["n"] == 2
