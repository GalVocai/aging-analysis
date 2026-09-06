"""Tests for the GUI's per-day, per-condition device trees.

Headless (offscreen Qt platform); no window is shown. These cover the
behaviour the user relies on -- ticking a device or a single channel out on
one specific day and having that reach the saved selection and the
aggregation without touching the other days.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qfn_aging.gui import ConditionTree, MainWindow  # noqa: E402
from qfn_aging.selection import Selection, load_selection  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated(qapp, tmp_path):
    """A MainWindow whose saved state lives under tmp_path.

    Essential: selections auto-save, so a test that ticks a checkbox on a
    default-constructed window would write into the user's real
    ``~/.qfn_aging/selection.json``.
    """
    def make():
        return MainWindow(selection_path=tmp_path / "selection.json",
                          settings_path=tmp_path / "settings.json")
    return make


def _rows(esn, condition, category, timepoint="baseline",
          tnums=(11, 12, 31, 32), type_="A", h2=True):
    return [{"esn": esn, "tnumber": t, "timepoint": timepoint,
             "condition": condition, "category": category,
             "type": type_, "h2": h2, "metrics": 50} for t in tnums]


@pytest.fixture
def frame():
    """Two days, so per-day behaviour is exercised."""
    rows = []
    for tp in ("baseline", "3D"):
        rows += _rows("1236", "25C_10RH", "PdParC_H2", tp)
        rows += _rows("1250", "25C_10RH", "PdParC_H2", tp)
        rows += _rows("1019", "60C_30RH", "ParC_only", tp, type_=None, h2=False)
    return pd.DataFrame(rows)


@pytest.fixture
def tree(qapp, frame):
    sub = frame[(frame["condition"] == "25C_10RH")
                & (frame["timepoint"] == "baseline")].copy()
    return ConditionTree(sub, Selection(), "baseline")


# -- tree structure --------------------------------------------------------

def test_tree_nests_transistors_under_devices(tree):
    assert tree.topLevelItemCount() == 2
    dev = tree.topLevelItem(0)
    assert dev.text(0) == "ESN 1236"
    assert dev.childCount() == 4
    assert [dev.child(i).text(0) for i in range(4)] == ["T11", "T12", "T31", "T32"]


def test_everything_starts_included(tree):
    dev = tree.topLevelItem(0)
    assert dev.checkState(0) == Qt.Checked
    assert tree.counts() == (8, 8)


def test_unticking_a_device_excludes_all_its_channels(tree):
    tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)
    assert tree._sel.excluded_for("baseline", "1236") == [11, 12, 31, 32]
    assert tree._sel.excluded_for("baseline", "1250") == []
    assert tree.counts() == (4, 8)


def test_unticking_one_channel_leaves_the_device_partial(tree):
    dev = tree.topLevelItem(0)
    dev.child(1).setCheckState(0, Qt.Unchecked)

    assert tree._sel.excluded_for("baseline", "1236") == [12]
    assert dev.checkState(0) == Qt.PartiallyChecked
    assert dev.text(4) == "3/4 in"
    assert tree.counts() == (7, 8)


def test_reticking_a_device_restores_every_channel(tree):
    dev = tree.topLevelItem(0)
    dev.setCheckState(0, Qt.Unchecked)
    dev.setCheckState(0, Qt.Checked)
    assert tree._sel.excluded_for("baseline", "1236") == []
    assert tree.counts() == (8, 8)


def test_the_tree_only_touches_its_own_day(tree):
    tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)
    assert tree._sel.excluded_for("3D", "1236") == []
    assert all(tp == "baseline" for tp, _, _ in tree._sel.excluded)


def test_excluded_rows_are_greyed(tree):
    dev = tree.topLevelItem(0)
    normal = dev.child(0).foreground(0).color().name()
    dev.child(0).setCheckState(0, Qt.Unchecked)
    assert dev.child(0).foreground(0).color().name() != normal


def test_set_all_and_counts(tree):
    tree.set_all(False)
    assert tree.counts() == (0, 8)
    tree.set_all(True)
    assert tree.counts() == (8, 8)


def test_refresh_reflects_an_externally_changed_selection(tree):
    tree._sel.exclude("baseline", "1250", 31, reason="external")
    tree.refresh()

    dev = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if tree.topLevelItem(i).text(0) == "ESN 1250")
    assert dev.checkState(0) == Qt.PartiallyChecked
    assert tree.counts() == (7, 8)


# -- day tabs --------------------------------------------------------------

def test_one_tab_per_day_each_holding_its_conditions(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)

    days = [win.day_tabs.tabText(i).split("  (")[0]
            for i in range(win.day_tabs.count())]
    assert days == ["baseline", "3D"]

    for day in days:
        conds = [win.cond_tab_widgets[day].tabText(i).split("  (")[0]
                 for i in range(win.cond_tab_widgets[day].count())]
        assert conds == ["25C_10RH", "60C_30RH"]

    assert set(win.trees) == {(d, c) for d in days for c in ("25C_10RH", "60C_30RH")}


def test_tab_titles_carry_counts(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)
    assert "(12/12)" in win.day_tabs.tabText(0)
    assert "(8/8)" in win.cond_tab_widgets["baseline"].tabText(0)


def test_excluding_on_one_day_only_changes_that_days_tabs(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)

    win.trees[("3D", "25C_10RH")].topLevelItem(0).setCheckState(0, Qt.Unchecked)

    assert win.trees[("baseline", "25C_10RH")].counts() == (8, 8)
    assert win.trees[("3D", "25C_10RH")].counts() == (4, 8)
    assert "(8/12)" in win.day_tabs.tabText(1)
    assert "(12/12)" in win.day_tabs.tabText(0)


def test_status_line_reports_the_current_day(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)
    win.trees[("3D", "25C_10RH")].topLevelItem(0).child(0).setCheckState(0, Qt.Unchecked)

    win.day_tabs.setCurrentIndex(1)
    msg = win.statusBar().currentMessage()
    assert msg.startswith("3D:")
    assert "1 excluded on this day" in msg


# -- persistence -----------------------------------------------------------

def test_ticking_auto_saves_without_pressing_save(qapp, frame, tmp_path):
    path = tmp_path / "selection.json"
    win = MainWindow(selection_path=path, settings_path=tmp_path / "settings.json")
    win.populate(frame)

    win.trees[("baseline", "25C_10RH")].topLevelItem(0).child(0) \
        .setCheckState(0, Qt.Unchecked)

    assert path.is_file(), "selection was not written on change"
    back = load_selection(path)
    assert not back.is_included("baseline", "1236", 11)
    assert back.is_included("3D", "1236", 11)


def test_a_fresh_window_reopens_with_previous_exclusions(qapp, frame, tmp_path):
    path = tmp_path / "selection.json"
    settings = tmp_path / "settings.json"

    first = MainWindow(selection_path=path, settings_path=settings)
    first.populate(frame)
    first.trees[("3D", "60C_30RH")].topLevelItem(0).setCheckState(0, Qt.Unchecked)

    second = MainWindow(selection_path=path, settings_path=settings)
    second.populate(frame)
    assert second.trees[("3D", "60C_30RH")].counts() == (0, 4)
    assert second.trees[("baseline", "60C_30RH")].counts() == (4, 4)


def test_data_root_is_remembered_between_sessions(qapp, tmp_path):
    settings = tmp_path / "settings.json"
    path = tmp_path / "selection.json"

    first = MainWindow(selection_path=path, settings_path=settings)
    first.root_edit.setText("C:/some/data/root")
    first._remember_settings()

    second = MainWindow(selection_path=path, settings_path=settings)
    assert second.root_edit.text() == "C:/some/data/root"


def test_close_persists_even_without_any_tick(qapp, frame, tmp_path):
    path = tmp_path / "selection.json"
    win = MainWindow(selection_path=path, settings_path=tmp_path / "settings.json")
    win.populate(frame)
    win.selection.exclude("baseline", "1250", 31, reason="set programmatically")
    win.close()

    assert not load_selection(path).is_included("baseline", "1250", 31)


def test_settings_roundtrip(tmp_path):
    from qfn_aging.selection import load_settings, save_settings

    p = tmp_path / "settings.json"
    assert load_settings(p) == {}
    save_settings({"data_root": "C:/data", "make_plots": False}, p)
    back = load_settings(p)
    assert back["data_root"] == "C:/data"
    assert back["make_plots"] is False


def test_corrupt_settings_file_is_ignored(tmp_path):
    from qfn_aging.selection import load_settings

    p = tmp_path / "settings.json"
    p.write_text("{ broken", encoding="utf-8")
    assert load_settings(p) == {}


# -- the whole chain -------------------------------------------------------

def test_gui_exclusion_changes_only_that_days_group_average(qapp, frame, isolated):
    from qfn_aging.allocation import DEFAULT
    from qfn_aging.grouping import add_group_keys, aggregate_groups
    from qfn_aging.selection import apply_selection

    win = isolated()
    win.populate(frame)

    tidy = pd.DataFrame([
        dict(esn="1236", tnumber=t, timepoint=tp, sheet="JG_n03", series="",
             metric="Vth_bg4", value=v)
        for tp in ("baseline", "3D")
        for t, v in zip((11, 12, 31, 32), (1.0, 2.0, 3.0, 10.0))
    ])

    def group_mean(day):
        g = aggregate_groups(add_group_keys(
            apply_selection(tidy, win.selection), DEFAULT))
        row = g[g["timepoint"] == day].iloc[0]
        return row["mean"], int(row["n"])

    assert group_mean("baseline") == (pytest.approx(4.0), 4)
    assert group_mean("3D") == (pytest.approx(4.0), 4)

    # drop the outlier on 3D only
    win.trees[("3D", "25C_10RH")].topLevelItem(0).child(3) \
        .setCheckState(0, Qt.Unchecked)

    assert group_mean("baseline") == (pytest.approx(4.0), 4)
    assert group_mean("3D") == (pytest.approx(2.0), 3)


# -- sync across days ------------------------------------------------------

def test_sync_off_by_default_keeps_exclusions_per_day(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)
    assert win.sync_days is False

    win.trees[("3D", "25C_10RH")].topLevelItem(0).child(0).setCheckState(0, Qt.Unchecked)

    assert win.selection.excluded_for("3D", "1236") == [11]
    assert win.selection.excluded_for("baseline", "1236") == []


def test_sync_on_applies_a_channel_tick_to_every_day(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)
    win.btn_sync.setChecked(True)                      # turn universal mode on
    assert win.sync_days is True

    win.trees[("3D", "25C_10RH")].topLevelItem(0).child(0).setCheckState(0, Qt.Unchecked)

    # excluded on BOTH days, and both day tabs reflect it
    assert win.selection.excluded_for("3D", "1236") == [11]
    assert win.selection.excluded_for("baseline", "1236") == [11]
    assert win.trees[("baseline", "25C_10RH")].counts() == (7, 8)
    assert win.trees[("3D", "25C_10RH")].counts() == (7, 8)


def test_sync_on_applies_a_whole_device_tick_to_every_day(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)
    win.btn_sync.setChecked(True)

    win.trees[("baseline", "25C_10RH")].topLevelItem(0).setCheckState(0, Qt.Unchecked)

    for day in ("baseline", "3D"):
        assert win.selection.excluded_for(day, "1236") == [11, 12, 31, 32]


def test_sync_is_non_destructive_when_toggled_off_again(qapp, frame, isolated):
    win = isolated()
    win.populate(frame)

    # a per-day exclusion made before sync exists
    win.trees[("baseline", "25C_10RH")].topLevelItem(0).child(1).setCheckState(0, Qt.Unchecked)
    # turning sync on and back off must not rewrite existing days
    win.btn_sync.setChecked(True)
    win.btn_sync.setChecked(False)

    assert win.selection.excluded_for("baseline", "1236") == [12]
    assert win.selection.excluded_for("3D", "1236") == []
