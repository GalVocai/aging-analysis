"""Tests for reading sensorlab master workbooks into a tidy table."""

from __future__ import annotations

import pandas as pd
import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.grouping import add_group_keys, aggregate_groups
from qfn_aging.masters import load_masters, load_masters_cached


def _write_master(root, timepoint, run_key, condition, mode, sheets: dict):
    d = root / timepoint / run_key / condition / mode
    d.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(d / f"master_{mode}.xlsx") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)


@pytest.fixture
def analysis_root(tmp_path):
    # 1236 is PdParC_H2 @ 25C_10RH; two transistors, air + H2 sheets
    air = pd.DataFrame({"ESN": [1236, 1236], "TNumber": [11, 12],
                        "condition": ["25C_10RH"] * 2, "day": ["baseline"] * 2,
                        "Vth_bg4": [-1.0, -1.2], "flags_bg4": ["ok", "ok"]})
    h2 = pd.DataFrame({"ESN": [1236, 1236], "TNumber": [11, 12],
                       "condition": ["25C_10RH"] * 2, "day": ["baseline"] * 2,
                       "Vth_bg4": [-2.0, -2.2], "flags_bg4": ["ok", "ok"]})
    diag = pd.DataFrame({"ESN": [1236], "TNumber": [11], "day": ["baseline"],
                         "Vth_bg4": [999.0]})
    _write_master(tmp_path, "baseline", "25C_H2", "25C_10RH", "JG",
                  {"JG_n03": air, "JG_n05": h2, "JG_n03_groups": diag})
    return tmp_path


def test_sheets_are_kept_distinct(analysis_root):
    t = load_masters(analysis_root)
    assert set(t.sheet) == {"JG_n03", "JG_n05"}          # _groups excluded
    assert len(t) == 4                                    # 2 sheets x 2 transistors
    assert set(t["mode"]) == {"JG"}
    assert set(t.timepoint) == {"baseline"}


def test_flag_columns_are_not_treated_as_metrics(analysis_root):
    t = load_masters(analysis_root)
    assert set(t.metric) == {"Vth_bg4"}


def test_air_and_h2_never_average_together(analysis_root):
    t = load_masters(analysis_root)
    g = aggregate_groups(add_group_keys(t, DEFAULT))

    assert len(g) == 2
    by_sheet = {r.sheet: r for r in g.itertuples()}
    assert by_sheet["JG_n03"].mean == pytest.approx(-1.1)
    assert by_sheet["JG_n05"].mean == pytest.approx(-2.1)
    assert all(r.condition == "25C_10RH" for r in g.itertuples())
    assert all(r.category == "PdParC_H2" for r in g.itertuples())
    assert all(r.n == 2 for r in g.itertuples())          # counts transistors


def test_condition_comes_from_roster_not_the_file(tmp_path):
    # file claims the wrong condition; the roster must win
    wrong = pd.DataFrame({"ESN": [1236], "TNumber": [11],
                          "condition": ["99C_99RH"], "day": ["baseline"],
                          "Vth_bg4": [-1.0]})
    _write_master(tmp_path, "baseline", "25C_H2", "99C_99RH", "JG", {"JG_n03": wrong})

    t = load_masters(tmp_path)
    assert "condition" not in t.columns
    g = aggregate_groups(add_group_keys(t, DEFAULT))
    assert g.iloc[0]["condition"] == "25C_10RH"


def test_series_subkeys_are_preserved(tmp_path):
    zozo = pd.DataFrame({"ESN": [1236] * 4, "TNumber": [11] * 4,
                         "day": ["baseline"] * 4,
                         "segment": [1, 2, 1, 2], "state_Vj": [-0.4, -0.4, -0.2, -0.2],
                         "tau": [10.0, 20.0, 30.0, 40.0]})
    _write_master(tmp_path, "baseline", "25C_H2", "25C_10RH", "ZOZO", {"ZOZO_n01": zozo})

    t = load_masters(tmp_path)
    assert t.series.nunique() == 4          # each (segment, state) is its own curve
    g = aggregate_groups(add_group_keys(t, DEFAULT))
    assert len(g) == 4
    assert set(g.n) == {1}


def test_empty_root_returns_empty_frame_with_schema(tmp_path):
    t = load_masters(tmp_path)
    assert t.empty
    assert "metric" in t.columns and "sheet" in t.columns


def test_unreadable_workbook_is_reported_and_other_data_still_loads(analysis_root):
    bad_dir = analysis_root / "3D" / "control" / "25C_10RH" / "JG"
    bad_dir.mkdir(parents=True)
    bad = bad_dir / "master_JG.xlsx"
    bad.write_bytes(b"this is not an Excel workbook")
    lines: list[str] = []

    tidy = load_masters(analysis_root, log=lines.append)

    assert not tidy.empty
    assert any("could not read" in line and str(bad) in line for line in lines)


def test_validated_cache_is_reused_and_invalidated(tmp_path, monkeypatch):
    import qfn_aging.masters as masters

    workbook = tmp_path / "analysis" / "3D" / "run" / "25C_10RH" / "JG" \
        / "master_JG.xlsx"
    workbook.parent.mkdir(parents=True)
    workbook.write_bytes(b"version one")
    expected = pd.DataFrame({"esn": ["1236"], "tnumber": [11], "value": [1.0]})
    calls: list[int] = []

    def fake_load(*args, **kwargs):
        calls.append(1)
        return expected.copy()

    monkeypatch.setattr(masters, "load_masters", fake_load)
    cache = tmp_path / "cache"

    first = load_masters_cached(tmp_path / "analysis", cache_dir=cache)
    second = load_masters_cached(tmp_path / "analysis", cache_dir=cache)
    assert first.equals(expected) and second.equals(expected)
    assert len(calls) == 1

    workbook.write_bytes(b"version two is a different size")
    load_masters_cached(tmp_path / "analysis", cache_dir=cache)
    assert len(calls) == 2
