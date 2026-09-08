"""Comparisons-tab formatting: draw modes, palettes, curve filter, titles.

The plotting half checks what actually lands on the axes; the GUI half checks
that an edit survives a metric switch and a restart, which is the whole point
of storing it per metric.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from qfn_aging.comparison import ComparisonSpec, build_curves
from qfn_aging.comparison_plot import (DEFAULT_XLABEL, PlotOptions, axis_labels_for,
                                       plot_comparison, title_for)


def _agg(days=("baseline", "3D", "7D"), temps=(25, 35, 45, 60),
         metric="Vth_bg4", sheet="JG_n03"):
    day_of = {"baseline": 0, "3D": 3, "7D": 7}
    rows = []
    for temp in temps:
        for tp in days:
            rows.append(dict(timepoint=tp, condition=f"{temp}C_30RH",
                             category="PdParC_H2", sheet=sheet, series="",
                             metric=metric, mean=-1.0 - 0.001 * temp * day_of[tp],
                             avdev=0.05, n=8))
    return pd.DataFrame(rows)


def _spec(metric="Vth_bg4", sheet="JG_n03"):
    return ComparisonSpec(vary="temp", metric=metric, sheet=sheet,
                          fixed={"rh": 30, "category": "PdParC_H2"})


def _draw(options=None, agg=None, spec=None):
    spec = spec or _spec()
    fig = plot_comparison(build_curves(agg if agg is not None else _agg(), spec),
                          spec, options=options)
    return fig.axes[0]


def _data_lines(ax):
    """The curve line of each errorbar series, in draw order.

    ``ax.lines`` also holds the caps, and the label lives on the container --
    so go through the containers, which is what errorbar actually returns.
    """
    lines = [c.lines[0] for c in ax.containers]
    assert lines, "no curves were drawn"        # never let a test pass vacuously
    return lines


# -- draw modes ------------------------------------------------------------

@pytest.mark.parametrize("mode, linestyle, has_marker", [
    ("line+points", "-", True),
    ("line", "-", False),
    ("points", "None", True),
])
def test_draw_mode_sets_line_and_marker(mode, linestyle, has_marker):
    ax = _draw(PlotOptions(draw=mode))
    line = _data_lines(ax)[0]

    assert line.get_linestyle() == linestyle
    assert (line.get_marker() not in ("", "None")) is has_marker


def test_a_single_timepoint_still_draws_markers_in_line_mode():
    """A line through one point is invisible, so markers win over the mode."""
    ax = _draw(PlotOptions(draw="line"), agg=_agg(days=("baseline",)))
    line = _data_lines(ax)[0]

    assert line.get_marker() == "o"
    assert line.get_linestyle() == "None"


# -- palettes --------------------------------------------------------------

def test_paper_palette_is_sensorlabs():
    from sensorlab.plotting.style import PUB_PALETTE

    ax = _draw(PlotOptions(palette="paper"))
    colors = [ln.get_color() for ln in _data_lines(ax)]

    assert colors == list(PUB_PALETTE[:len(colors)])


def test_green_palette_is_still_available():
    ax = _draw(PlotOptions(palette="green"))
    colors = [ln.get_color() for ln in _data_lines(ax)]

    assert colors[0] == "#a8ddb5"          # light -> dark ramp, unchanged
    assert colors[-1] == "#0f5132"


# -- curve filter ----------------------------------------------------------

def test_hidden_curves_are_dropped_from_the_plot_and_legend():
    ax = _draw(PlotOptions(hidden=["60"]))
    labels = [t.get_text() for t in ax.get_legend().get_texts()]

    assert labels == ["25°C", "35°C", "45°C"]


def test_hiding_a_curve_does_not_recolour_the_others():
    """Colours are assigned over every value, so the survivors keep theirs."""
    full = {ln.get_label(): ln.get_color() for ln in _data_lines(_draw(PlotOptions()))}
    part = {ln.get_label(): ln.get_color()
            for ln in _data_lines(_draw(PlotOptions(hidden=["35"])))}

    assert part == {k: v for k, v in full.items() if k != "35°C"}


def test_hidden_matches_across_int_and_str():
    """Settings come back from JSON as strings; the data holds ints."""
    assert PlotOptions(hidden=["60"]).hides(60)
    assert PlotOptions(hidden=[60]).hides("60")


# -- title -----------------------------------------------------------------

def test_custom_title_replaces_the_generated_one():
    ax = _draw(PlotOptions(title="Threshold drift, 30% RH"))
    assert ax.get_title() == "Threshold drift, 30% RH"


def test_blank_title_falls_back_to_the_generated_one():
    spec = _spec()
    assert title_for(spec, PlotOptions(title="   ")) == spec.title()


@pytest.mark.parametrize("bold, italic, weight, style", [
    (True, False, "bold", "normal"),
    (False, True, "normal", "italic"),
    (False, False, "normal", "normal"),
    (True, True, "bold", "italic"),
])
def test_title_bold_and_italic(bold, italic, weight, style):
    ax = _draw(PlotOptions(title_bold=bold, title_italic=italic))

    assert ax.title.get_fontweight() == weight
    assert ax.title.get_fontstyle() == style


def test_the_empty_plot_keeps_the_custom_title():
    spec = _spec()
    empty = pd.DataFrame(columns=["curve", "timepoint", "day", "mean", "avdev", "n"])
    fig = plot_comparison(empty, spec, options=PlotOptions(title="Nothing here"))

    assert fig.axes[0].get_title() == "Nothing here"


# -- axis labels -----------------------------------------------------------

def test_axis_labels_default_to_the_automatic_ones():
    ax = _draw(PlotOptions())

    assert ax.get_xlabel() == DEFAULT_XLABEL
    assert ax.get_ylabel() == "Vth_bg4"          # falls back to the metric id


def test_axis_labels_can_be_overridden():
    ax = _draw(PlotOptions(xlabel="Ageing time [days]", ylabel="Threshold shift [V]"))

    assert ax.get_xlabel() == "Ageing time [days]"
    assert ax.get_ylabel() == "Threshold shift [V]"


def test_a_blank_axis_label_falls_back_rather_than_blanking_the_axis():
    """Whitespace is an empty override, not a request for an unlabelled axis."""
    x, y = axis_labels_for(_spec(), "\u0394Vth [mV]", PlotOptions(xlabel="  ", ylabel=""))

    assert x == DEFAULT_XLABEL
    assert y == "\u0394Vth [mV]"                     # the caller's pretty label wins


def test_an_overridden_y_label_beats_the_derived_metric_label():
    x, y = axis_labels_for(_spec(), "\u0394Vth [mV]", PlotOptions(ylabel="My own label"))

    assert (x, y) == (DEFAULT_XLABEL, "My own label")


def test_overridden_axis_labels_keep_the_paper_typography():
    ax = _draw(PlotOptions(xlabel="Ageing time [days]"))

    assert ax.xaxis.label.get_fontsize() == 16.0
    assert ax.xaxis.label.get_fontweight() == "bold"


# -- axes ------------------------------------------------------------------

def test_tick_steps_are_applied():
    ax = _draw(PlotOptions(x_step=2.0, y_step=0.05))

    xticks = list(ax.xaxis.get_majorticklocs())
    yticks = list(ax.yaxis.get_majorticklocs())
    assert xticks[1] - xticks[0] == pytest.approx(2.0)
    assert yticks[1] - yticks[0] == pytest.approx(0.05)
    assert 0.0 in xticks


def test_zero_tick_step_leaves_matplotlib_alone():
    auto = list(_draw(PlotOptions()).xaxis.get_majorticklocs())
    assert auto == list(_draw(PlotOptions(x_step=0.0)).xaxis.get_majorticklocs())


def test_minor_ticks_and_grid_can_be_turned_off():
    on = _draw(PlotOptions(minor_ticks=True, grid=True))
    off = _draw(PlotOptions(minor_ticks=False, grid=False))

    assert len(on.xaxis.get_minorticklocs()) > 0
    assert len(off.xaxis.get_minorticklocs()) == 0
    assert on.xaxis.get_gridlines()[0].get_visible()
    assert not off.xaxis.get_gridlines()[0].get_visible()


# -- GUI wiring ------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from qfn_aging.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _tidy_two_metrics():
    rows = []
    for esn, base in (("1236", 1.0), ("1250", 3.0), ("1244", 5.0), ("1245", 7.0)):
        for tnum in (11, 12, 31, 32):
            for metric, bump in (("Vth_bg0", 0.0), ("Vth_bg4", 0.5)):
                rows.append(dict(esn=esn, tnumber=tnum, timepoint="baseline",
                                 run_key="r", mode="JG", sheet="JG_n03",
                                 series="", metric=metric, value=base + bump))
    return pd.DataFrame(rows)


@pytest.fixture
def win(qapp, tmp_path):
    w = MainWindow(selection_path=tmp_path / "selection.json",
                   settings_path=tmp_path / "settings.json")
    w._tidy = _tidy_two_metrics()
    w._rebuild_aggregate()
    w._populate_comparison_options()
    return w


def _select_metric(win, metric):
    win.cmb_metric.setCurrentIndex(win.cmb_metric.findData(metric))


def test_a_title_is_remembered_per_metric(win):
    _select_metric(win, "Vth_bg0")
    win.ed_title.setText("Drift at zero back-gate")
    win._format_changed()

    _select_metric(win, "Vth_bg4")
    assert win.ed_title.text() == ""            # a different metric, its own title

    _select_metric(win, "Vth_bg0")
    assert win.ed_title.text() == "Drift at zero back-gate"


def test_title_and_style_survive_a_restart(qapp, tmp_path):
    settings = tmp_path / "settings.json"
    first = MainWindow(selection_path=tmp_path / "selection.json",
                       settings_path=settings)
    first._tidy = _tidy_two_metrics()
    first._rebuild_aggregate()
    first._populate_comparison_options()
    _select_metric(first, "Vth_bg4")
    first.ed_title.setText("Kept across restarts")
    first.btn_title_italic.setChecked(True)
    first.cmb_draw.setCurrentIndex(first.cmb_draw.findData("points"))
    first._format_changed()
    first.close()

    again = MainWindow(selection_path=tmp_path / "selection.json",
                       settings_path=settings)
    again._tidy = _tidy_two_metrics()
    again._rebuild_aggregate()
    again._populate_comparison_options()
    _select_metric(again, "Vth_bg4")

    assert again.ed_title.text() == "Kept across restarts"
    assert again.btn_title_italic.isChecked()
    assert again.cmb_draw.currentData() == "points"      # global, not per metric
    assert json.loads(settings.read_text())["comparison"]["draw"] == "points"


def test_a_default_only_metric_is_not_written_to_settings(win, tmp_path):
    _select_metric(win, "Vth_bg0")
    win.ed_title.setText("x")
    win._format_changed()
    assert win._comparison_prefs()["per_metric"]

    win._reset_title()                          # back to every default
    assert win._comparison_prefs()["per_metric"] == {}


def test_curve_filter_lists_the_values_in_the_data(win):
    win.cmb_vary.setCurrentIndex(win.cmb_vary.findData("temp"))
    assert sorted(win._curve_checks) == ["25", "35"]
    assert all(chk.isChecked() for chk in win._curve_checks.values())


def test_curve_filter_rebuilds_when_the_varied_axis_changes(win):
    win.cmb_vary.setCurrentIndex(win.cmb_vary.findData("temp"))
    assert sorted(win._curve_checks) == ["25", "35"]

    win.cmb_vary.setCurrentIndex(win.cmb_vary.findData("rh"))
    assert sorted(win._curve_checks) == ["10"]          # no stale 25 / 35


def test_unticking_a_curve_hides_it_and_is_remembered(win):
    win.cmb_vary.setCurrentIndex(win.cmb_vary.findData("temp"))
    win._curve_checks["35"].setChecked(False)

    assert win._comparison_prefs()["hidden"]["temp"] == ["35"]
    labels = [t.get_text() for t in win._ax.get_legend().get_texts()]
    assert labels == ["25°C"]

    win._curve_checks["35"].setChecked(True)
    assert win._comparison_prefs()["hidden"]["temp"] == []


def test_axis_labels_are_remembered_per_metric(win):
    _select_metric(win, "Vth_bg0")
    win.ed_xlabel.setText("Ageing time [days]")
    win.ed_ylabel.setText("Vth at Vbg=0 [V]")
    win._format_changed()

    _select_metric(win, "Vth_bg4")
    assert (win.ed_xlabel.text(), win.ed_ylabel.text()) == ("", "")

    _select_metric(win, "Vth_bg0")
    assert win.ed_xlabel.text() == "Ageing time [days]"
    assert win.ed_ylabel.text() == "Vth at Vbg=0 [V]"
    assert win._ax.get_ylabel() == "Vth at Vbg=0 [V]"


def test_axis_labels_survive_a_restart(qapp, tmp_path):
    settings = tmp_path / "settings.json"

    def build():
        w = MainWindow(selection_path=tmp_path / "selection.json",
                       settings_path=settings)
        w._tidy = _tidy_two_metrics()
        w._rebuild_aggregate()
        w._populate_comparison_options()
        return w

    first = build()
    _select_metric(first, "Vth_bg4")
    first.ed_ylabel.setText("Kept [V]")
    first._format_changed()
    first.close()

    again = build()
    _select_metric(again, "Vth_bg4")
    assert again.ed_ylabel.text() == "Kept [V]"
    assert json.loads(settings.read_text())["comparison"]["per_metric"][
        "JG_n03|Vth_bg4"]["ylabel"] == "Kept [V]"


def test_auto_clears_the_title_and_both_axis_labels(win):
    _select_metric(win, "Vth_bg0")
    win.ed_title.setText("t")
    win.ed_xlabel.setText("x")
    win.ed_ylabel.setText("y")
    win._format_changed()

    win._reset_title()

    assert (win.ed_title.text(), win.ed_xlabel.text(), win.ed_ylabel.text()) == ("", "", "")
    assert win._comparison_prefs()["per_metric"] == {}       # nothing left to store
    assert win._ax.get_xlabel() == "Study day"


def test_the_y_placeholder_shows_what_empty_would_give(win):
    _select_metric(win, "Vth_bg0")
    assert win.ed_ylabel.placeholderText() == "Vth_bg0"


# -- format follows you across Vbg series ----------------------------------

def _tidy_two_series():
    """One metric measured at two back-gates -- the Vbg=0 / Vbg=4 case."""
    rows = []
    for esn, base in (("1236", 1.0), ("1250", 3.0), ("1244", 5.0), ("1245", 7.0)):
        for tnum in (11, 12, 31, 32):
            for series, bump in (("Vbg=0", 0.0), ("Vbg=4", 0.5)):
                rows.append(dict(esn=esn, tnumber=tnum, timepoint="baseline",
                                 run_key="r", mode="JG_H2", sheet="JG_H2_n05",
                                 series=series, metric="delta_Vth_H2",
                                 value=base + bump))
    return pd.DataFrame(rows)


@pytest.fixture
def win_series(qapp, tmp_path):
    w = MainWindow(selection_path=tmp_path / "selection.json",
                   settings_path=tmp_path / "settings.json")
    w._tidy = _tidy_two_series()
    w._rebuild_aggregate()
    w._populate_comparison_options()
    return w


def _select_series(win, series):
    win.cmb_series.setCurrentIndex(win.cmb_series.findData(series))


def test_the_key_ignores_the_series(win_series):
    """Vbg=0 and Vbg=4 are the same quantity, so they share one entry."""
    _select_series(win_series, "Vbg=0")
    key_bg0 = win_series._opts_key
    _select_series(win_series, "Vbg=4")

    assert win_series._opts_key == key_bg0
    assert "Vbg" not in key_bg0


def test_title_and_labels_survive_a_vbg_switch(win_series):
    _select_series(win_series, "Vbg=0")
    win_series.ed_title.setText("H2 response fade")
    win_series.ed_ylabel.setText("dVth [V]")
    win_series._format_changed()

    _select_series(win_series, "Vbg=4")
    assert win_series.ed_title.text() == "H2 response fade"
    assert win_series.ed_ylabel.text() == "dVth [V]"
    assert win_series._ax.get_title() == "H2 response fade"

    _select_series(win_series, "Vbg=0")
    assert win_series.ed_title.text() == "H2 response fade"


def test_bold_after_a_vbg_switch_keeps_the_typed_text(win_series):
    """The reported symptom: Bold appeared to wipe the custom title."""
    _select_series(win_series, "Vbg=0")
    win_series.ed_title.setText("Typed by hand")
    win_series._format_changed()
    _select_series(win_series, "Vbg=4")

    win_series.btn_title_bold.setChecked(not win_series.btn_title_bold.isChecked())

    assert win_series.ed_title.text() == "Typed by hand"
    assert win_series._ax.get_title() == "Typed by hand"


def test_repopulating_keeps_the_chosen_series(win_series):
    """A reload used to drop you back to the first Vbg without saying so."""
    _select_series(win_series, "Vbg=4")
    win_series._populate_comparison_options()

    assert win_series.cmb_series.currentData() == "Vbg=4"


def test_an_entry_written_under_the_old_series_key_is_still_read(win_series):
    """Settings saved before the key dropped the series must keep working."""
    prefs = win_series._comparison_prefs()
    prefs["per_metric"]["JG_H2_n05|delta_Vth_H2|Vbg=4"] = {
        "title": "From the old format", "bold": True, "italic": False,
        "xlabel": "", "ylabel": "", "x_step": 0.0, "y_step": 0.0}
    win_series._opts_key = None                  # force a reload
    win_series._refresh_comparison()

    assert win_series.ed_title.text() == "From the old format"


def test_per_metric_widgets_are_disabled_without_a_metric(qapp, tmp_path):
    """No metric means nowhere to file an edit, so do not accept one."""
    w = MainWindow(selection_path=tmp_path / "selection.json",
                   settings_path=tmp_path / "settings.json")
    w._refresh_comparison()

    assert w.current_spec() is None
    assert not w.ed_title.isEnabled()
    assert not w.ed_ylabel.isEnabled()

    w._tidy = _tidy_two_series()
    w._rebuild_aggregate()
    w._populate_comparison_options()
    assert w.ed_title.isEnabled()
