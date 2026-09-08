"""Tests for the Comparisons tab: option wiring and live re-aggregation."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qfn_aging.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    return MainWindow(selection_path=tmp_path / "selection.json",
                      settings_path=tmp_path / "settings.json")


def _tidy():
    """Two devices per condition, four transistors each, one metric."""
    rows = []
    # 1236/1250 are PdParC_H2 @ 25C_10RH; 1244/1245 are PdParC_H2 @ 35C_10RH
    for esn, value in (("1236", 1.0), ("1250", 3.0),
                       ("1244", 5.0), ("1245", 7.0)):
        for t in (11, 12, 31, 32):
            rows.append(dict(esn=esn, tnumber=t, timepoint="baseline",
                             run_key="r", mode="JG", sheet="JG_n03",
                             series="", metric="Vth_bg4", value=value))
    return pd.DataFrame(rows)


def _per(tidy):
    """The per-transistor summary populate() expects."""
    from qfn_aging.allocation import DEFAULT
    from qfn_aging.grouping import add_group_keys
    g = add_group_keys(tidy, DEFAULT)
    per = (g.groupby(["timepoint", "esn", "tnumber", "condition", "category"])
             .size().reset_index(name="metrics"))
    per["type"] = per["esn"].map(lambda e: DEFAULT.device(e).type)
    per["h2"] = per["esn"].map(lambda e: DEFAULT.device(e).h2)
    return per


def _load(win, tidy=None):
    win._tidy = _tidy() if tidy is None else tidy
    win._rebuild_aggregate()
    win._populate_comparison_options()


def test_tab_exists(win):
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Data", "Devices", "Comparisons"]


def test_options_are_populated_from_the_data(win):
    _load(win)
    sheets = [win.cmb_sheet.itemData(i) for i in range(win.cmb_sheet.count())]
    metrics = [win.cmb_metric.itemData(i) for i in range(win.cmb_metric.count())]
    assert sheets == ["JG_n03"]
    assert metrics == ["Vth_bg4"]


def test_the_varied_axis_is_hidden_from_the_fixed_pickers(win):
    _load(win)

    win.cmb_vary.setCurrentIndex(0)                    # temperature
    assert win.cmb_vary.currentData() == "temp"
    assert not win.cmb_fix_temp.isVisible() or not win.lbl_temp.isVisible()

    win.cmb_vary.setCurrentIndex(2)                    # category
    assert win.cmb_vary.currentData() == "category"
    assert not win.cmb_fix_cat.isVisible() or not win.lbl_cat.isVisible()


def test_spec_reflects_the_pickers(win):
    _load(win)
    win.cmb_vary.setCurrentIndex(0)                    # vary temperature
    spec = win.current_spec()

    assert spec.vary == "temp"
    assert spec.metric == "Vth_bg4"
    assert spec.sheet == "JG_n03"
    assert set(spec.fixed) == {"rh", "category"}       # the other two fixed


def test_curves_use_the_group_average(win):
    from qfn_aging.comparison import build_curves

    _load(win)
    win.cmb_vary.setCurrentIndex(0)
    curves = build_curves(win._agg, win.current_spec())

    by_temp = {int(r.curve): r.mean for r in curves.itertuples()}
    assert by_temp[25] == pytest.approx(2.0)           # (1.0 + 3.0) / 2
    assert by_temp[35] == pytest.approx(6.0)           # (5.0 + 7.0) / 2


def test_excluding_a_transistor_moves_the_curve(win):
    """The whole chain: untick in Devices -> average changes -> curve moves."""
    from qfn_aging.comparison import build_curves

    _load(win)
    win.cmb_vary.setCurrentIndex(0)
    before = build_curves(win._agg, win.current_spec())
    n_before = int(before.loc[before["curve"] == 25, "n"].iloc[0])
    mean_before = float(before.loc[before["curve"] == 25, "mean"].iloc[0])

    # drop every transistor of the device that sits at 1.0
    win.selection.exclude_device("baseline", "1236", [11, 12, 31, 32], reason="test")
    win._rebuild_aggregate()

    after = build_curves(win._agg, win.current_spec())
    n_after = int(after.loc[after["curve"] == 25, "n"].iloc[0])
    mean_after = float(after.loc[after["curve"] == 25, "mean"].iloc[0])

    assert n_before == 8 and n_after == 4
    assert mean_before == pytest.approx(2.0)
    assert mean_after == pytest.approx(3.0)           # only the 3.0 device left


def test_refresh_draws_without_error_on_empty_data(win):
    win._tidy = pd.DataFrame()
    win._rebuild_aggregate()
    win._refresh_comparison()                          # must not raise
    assert win._agg.empty


def test_selection_change_triggers_a_redraw(win, monkeypatch):
    tidy = _tidy()
    _load(win, tidy)
    win.populate(_per(tidy))
    calls: list[int] = []
    monkeypatch.setattr(win, "_refresh_comparison", lambda: calls.append(1))

    win.tree_for("baseline", "25C_10RH").topLevelItem(0).child(0) \
        .setCheckState(0, Qt.Unchecked)
    assert calls, "unticking must refresh the comparison plot"


# -- axis-limit preservation -----------------------------------------------

def _draw(win, tidy=None, vary=0):
    _load(win, tidy)
    win.cmb_vary.setCurrentIndex(vary)
    win._refresh_comparison()


def test_limits_survive_a_same_metric_redraw(win):
    """Zoom, then a selection-style redraw keeps the view (not autoscale)."""
    _draw(win)
    win._ax.set_ylim(10.0, 20.0)
    win._refresh_comparison()                       # same spec -> preserve
    assert win._ax.get_ylim() == pytest.approx((10.0, 20.0))


def test_reset_view_autoscales_back_to_the_data(win):
    _draw(win)
    win._ax.set_ylim(10.0, 20.0)
    win._reset_view()                               # force a one-shot fit
    assert win._ax.get_ylim() != pytest.approx((10.0, 20.0))


def test_lock_axes_keeps_limits_even_when_the_metric_changes(win):
    tidy = _tidy()
    other = tidy.copy()
    other["metric"] = "Vth_bg0"
    other["value"] = tidy["value"] * 100.0          # a very different y range
    both = pd.concat([tidy, other], ignore_index=True)

    _draw(win, both)
    win.cmb_metric.setCurrentIndex(win.cmb_metric.findData("Vth_bg4"))
    win._ax.set_ylim(10.0, 20.0)
    win.btn_lock_axes.setChecked(True)              # freeze

    win.cmb_metric.setCurrentIndex(win.cmb_metric.findData("Vth_bg0"))  # triggers redraw
    assert win._ax.get_ylim() == pytest.approx((10.0, 20.0))


def test_unlocked_metric_change_autoscales(win):
    tidy = _tidy()
    other = tidy.copy()
    other["metric"] = "Vth_bg0"
    other["value"] = tidy["value"] * 100.0
    both = pd.concat([tidy, other], ignore_index=True)

    _draw(win, both)
    win.cmb_metric.setCurrentIndex(win.cmb_metric.findData("Vth_bg4"))
    win._ax.set_ylim(10.0, 20.0)                    # a range that fits neither metric

    win.cmb_metric.setCurrentIndex(win.cmb_metric.findData("Vth_bg0"))  # key changes
    assert win._ax.get_ylim() != pytest.approx((10.0, 20.0))
