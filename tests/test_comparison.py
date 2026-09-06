"""Tests for comparison-curve slicing and rendering."""

from __future__ import annotations

import pandas as pd
import pytest

from qfn_aging.comparison import (DRIFT_METRICS, ComparisonSpec, add_condition_axes,
                                  add_jg_vth_drift, available_options, build_curves,
                                  parse_condition, timepoints_present, ylabel_for_metric)


def _agg_row(timepoint, condition, category, metric="Vth_bg4", sheet="JG_n03",
             mean=-1.0, avdev=0.1, n=12, series=""):
    return dict(timepoint=timepoint, condition=condition, category=category,
                sheet=sheet, series=series, metric=metric, mean=mean,
                avdev=avdev, n=n)


@pytest.fixture
def agg():
    rows = []
    for temp in (25, 35, 45, 60):
        for rh in (10, 30, 50):
            for cat in ("PdParC_H2", "Pd_only"):
                for tp, day in (("baseline", 0), ("3D", 3)):
                    rows.append(_agg_row(tp, f"{temp}C_{rh}RH", cat,
                                         mean=-1.0 - 0.001 * temp * day))
    return pd.DataFrame(rows)


# -- condition parsing -----------------------------------------------------

def test_parse_condition():
    assert parse_condition("25C_10RH") == (25, 10)
    assert parse_condition("60C_50RH") == (60, 50)
    assert parse_condition("nonsense") == (None, None)


def test_add_condition_axes(agg):
    df = add_condition_axes(agg)
    assert set(df["temp_c"].unique()) == {25, 35, 45, 60}
    assert set(df["rh_pct"].unique()) == {10, 30, 50}
    assert set(df["day"].unique()) == {0, 3}


# -- the three comparison modes -------------------------------------------

def test_vary_temperature_at_fixed_rh_and_category(agg):
    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 30, "category": "PdParC_H2"})
    c = build_curves(agg, spec)

    assert sorted(c["curve"].unique()) == [25, 35, 45, 60]
    assert sorted(c["day"].unique()) == [0, 3]
    assert len(c) == 8                       # 4 temperatures x 2 days


def test_vary_rh_at_fixed_temperature_and_category(agg):
    spec = ComparisonSpec(vary="rh", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"temp": 45, "category": "PdParC_H2"})
    c = build_curves(agg, spec)

    assert sorted(c["curve"].unique()) == [10, 30, 50]
    assert len(c) == 6


def test_vary_category_at_a_fixed_condition(agg):
    spec = ComparisonSpec(vary="category", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"temp": 45, "rh": 30})
    c = build_curves(agg, spec)

    assert sorted(c["curve"].unique()) == ["PdParC_H2", "Pd_only"]
    assert len(c) == 4


def test_each_mode_fixes_exactly_the_other_two_axes():
    assert ComparisonSpec(vary="temp", metric="m", sheet="s").required_fixed == \
        ("rh", "category")
    assert ComparisonSpec(vary="rh", metric="m", sheet="s").required_fixed == \
        ("temp", "category")
    assert ComparisonSpec(vary="category", metric="m", sheet="s").required_fixed == \
        ("temp", "rh")


def test_missing_fixed_axis_is_rejected(agg):
    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 30})           # category missing
    with pytest.raises(ValueError, match="category"):
        build_curves(agg, spec)


def test_unknown_vary_axis_is_rejected():
    with pytest.raises(ValueError, match="vary must be one of"):
        ComparisonSpec(vary="pressure", metric="m", sheet="s")


# -- slicing correctness ---------------------------------------------------

def test_sheets_are_never_mixed(agg):
    """JG_n03 (air) and JG_n05 (H2) must not end up on one curve."""
    extra = agg.copy()
    extra["sheet"] = "JG_n05"
    extra["mean"] = -9.0
    both = pd.concat([agg, extra], ignore_index=True)

    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 30, "category": "PdParC_H2"})
    c = build_curves(both, spec)
    assert (c["mean"] > -5).all()
    assert len(c) == 8


def test_series_subkey_is_respected():
    rows = [_agg_row("baseline", "25C_30RH", "PdParC_H2", metric="tau",
                     sheet="ZOZO_n01", series="segment=1", mean=10.0),
            _agg_row("baseline", "25C_30RH", "PdParC_H2", metric="tau",
                     sheet="ZOZO_n01", series="segment=2", mean=20.0)]
    df = pd.DataFrame(rows)

    spec = ComparisonSpec(vary="category", metric="tau", sheet="ZOZO_n01",
                          series="segment=2", fixed={"temp": 25, "rh": 30})
    c = build_curves(df, spec)
    assert len(c) == 1
    assert c.iloc[0]["mean"] == 20.0


def test_empty_combination_returns_an_empty_frame_not_an_error(agg):
    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 99, "category": "PdParC_H2"})
    c = build_curves(agg, spec)
    assert c.empty
    assert list(c.columns) == ["curve", "timepoint", "day", "mean", "avdev", "n"]


def test_empty_input_is_handled(agg):
    spec = ComparisonSpec(vary="temp", metric="m", sheet="s",
                          fixed={"rh": 30, "category": "x"})
    assert build_curves(pd.DataFrame(), spec).empty


# -- option discovery ------------------------------------------------------

def test_available_options(agg):
    o = available_options(agg)
    assert o["temp"] == [25, 35, 45, 60]
    assert o["rh"] == [10, 30, 50]
    assert o["category"] == ["PdParC_H2", "Pd_only"]
    assert o["sheet"] == ["JG_n03"]


def test_timepoints_come_back_in_study_order(agg):
    assert timepoints_present(agg) == ["baseline", "3D"]


# -- rendering -------------------------------------------------------------

def test_plot_renders_one_line_per_curve(agg):
    from qfn_aging.comparison_plot import plot_comparison

    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 30, "category": "PdParC_H2"})
    fig = plot_comparison(build_curves(agg, spec), spec, ylabel="Vth [V]")
    ax = fig.axes[0]

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["25°C", "35°C", "45°C", "60°C"]
    assert "Temperature comparison" in ax.get_title()
    assert ax.get_xlabel() == "Study day"
    assert ax.get_ylabel() == "Vth [V]"


def test_single_timepoint_is_drawn_as_markers_with_a_note():
    from qfn_aging.comparison_plot import plot_comparison

    rows = [_agg_row("baseline", f"{t}C_30RH", "PdParC_H2") for t in (25, 60)]
    spec = ComparisonSpec(vary="temp", metric="Vth_bg4", sheet="JG_n03",
                          fixed={"rh": 30, "category": "PdParC_H2"})
    fig = plot_comparison(build_curves(pd.DataFrame(rows), spec), spec)
    ax = fig.axes[0]

    notes = [t.get_text() for t in ax.texts]
    assert any("no trend yet" in n for n in notes)


def test_empty_curves_render_a_message_not_a_crash():
    from qfn_aging.comparison_plot import plot_comparison

    spec = ComparisonSpec(vary="temp", metric="m", sheet="s",
                          fixed={"rh": 30, "category": "x"})
    fig = plot_comparison(pd.DataFrame(
        columns=["curve", "timepoint", "day", "mean", "avdev", "n"]), spec)
    assert any("No data" in t.get_text() for t in fig.axes[0].texts)


# -- JG_n03 ΔVth drift metric ----------------------------------------------

def _tidy_row(esn, tnum, tp, metric, value, sheet="JG_n03", series=""):
    return dict(esn=str(esn), tnumber=tnum, timepoint=tp, sheet=sheet,
                series=series, metric=metric, value=value)


def test_add_jg_vth_drift_is_baseline_relative_mv():
    tidy = pd.DataFrame([
        _tidy_row("A", 11, "baseline", "Vth_bg0", -0.50),
        _tidy_row("A", 11, "3D",       "Vth_bg0", -0.40),   # +0.10 V -> +100 mV
        _tidy_row("A", 11, "baseline", "Vth_bg4", -0.60),
        _tidy_row("A", 11, "3D",       "Vth_bg4", -0.70),   # -0.10 V -> -100 mV
        # a non-JG_n03 Vth-like metric must NOT get a drift twin
        _tidy_row("A", 11, "3D", "delta_Vth_H2", 0.2, sheet="JG_H2"),
    ])
    out = add_jg_vth_drift(tidy)

    bg0 = out[out["metric"] == "dVth_bg0_mV"].set_index("timepoint")["value"]
    assert bg0["baseline"] == 0.0                 # every device is 0 at baseline
    assert bg0["3D"] == pytest.approx(100.0)      # sign is physical (moved +)

    bg4 = out[out["metric"] == "dVth_bg4_mV"].set_index("timepoint")["value"]
    assert bg4["3D"] == pytest.approx(-100.0)     # moved more negative -> negative

    assert set(DRIFT_METRICS) <= set(out["metric"])
    assert "ddelta_Vth_H2_mV" not in set(out["metric"])
    # original rows are untouched (additive transform)
    assert (out["metric"] == "Vth_bg0").sum() == 2


def test_add_jg_vth_drift_missing_baseline_gives_nan():
    tidy = pd.DataFrame([_tidy_row("A", 11, "3D", "Vth_bg0", -0.4)])
    out = add_jg_vth_drift(tidy)
    assert out[out["metric"] == "dVth_bg0_mV"]["value"].isna().all()


def test_add_jg_vth_drift_noop_when_no_jg_n03():
    tidy = pd.DataFrame([_tidy_row("A", 11, "3D", "loop_area", 1.0, sheet="JGH_n02")])
    out = add_jg_vth_drift(tidy)
    assert list(out["metric"]) == ["loop_area"]


def test_ylabel_for_metric_maps_only_drift_metrics():
    assert ylabel_for_metric("dVth_bg0_mV") == "ΔVth [mV]"
    assert ylabel_for_metric("dVth_bg4_mV") == "ΔVth [mV]"
    assert ylabel_for_metric("Vth_bg0") is None
    assert ylabel_for_metric("delta_Vth_H2") is None
