"""Tests for analyzing only a new day, and for skipping already-done runs."""

from __future__ import annotations

import pandas as pd
import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.discovery import available_timepoints, discover
from qfn_aging.runner import analyze_all, has_output

H2 = [("n01", "SE_STATE"), ("n02", "JGH"), ("n03", "JG"), ("n04", "SE_STATE"),
      ("n05", "JG"), ("n06", "JG"), ("n07", "SE_STATE")]
CTL = [("n01", "SE_STATE"), ("n02", "JGH"), ("n03", "JG")]


def _make_run(root, timepoint, run_key, majors):
    inner = root / timepoint / run_key / f"E1000_2026_07_25_N01_{run_key}"
    inner.mkdir(parents=True)
    for major, mode in majors:
        (inner / f"E1000_2026_07_25_N01_{major}_{mode}_.xlsx").touch()
    (inner / "script.xlsx").touch()


@pytest.fixture
def tree(tmp_path):
    for tp in ("Baseline", "3D"):
        for key in ("25C_H2", "35C_H2", "45C_H2", "60C_H2"):
            _make_run(tmp_path, tp, key, H2)
        for key in ("control_set1", "control_set2"):
            _make_run(tmp_path, tp, key, CTL)
    return tmp_path


def _write_masters(out_root, timepoint, out_group, conditions):
    for c in conditions:
        d = out_root / timepoint / out_group / c / "JG"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ESN": [1], "TNumber": [11], "day": [timepoint], "Vth_bg4": [1.0]}) \
            .to_excel(d / "master_JG.xlsx", index=False)


def test_available_timepoints_in_study_order(tree):
    assert available_timepoints(tree) == ["baseline", "3D"]


def test_discover_can_target_one_timepoint(tree):
    all_runs, _ = discover(tree)
    assert len(all_runs) == 12                       # 6 runs x 2 days

    only_3d, _ = discover(tree, timepoints=["3D"])
    assert len(only_3d) == 6
    assert {r.timepoint for r in only_3d} == {"3D"}


def test_discover_unknown_timepoint_yields_nothing(tree):
    runs, _ = discover(tree, timepoints=["99D"])
    assert runs == []


def test_has_output_is_false_before_anything_is_written(tree, tmp_path):
    runs, _ = discover(tree, timepoints=["baseline"])
    out = tmp_path / "out"
    assert not any(has_output(rf, out, DEFAULT) for rf in runs)


def test_has_output_distinguishes_the_two_control_runs(tree, tmp_path):
    """Both write into 'control'; set 2 must not look done just because
    set 1 has been written."""
    out = tmp_path / "out"
    runs, _ = discover(tree, timepoints=["baseline"])
    set1 = next(r for r in runs if r.run_key == "control_set1")
    set2 = next(r for r in runs if r.run_key == "control_set2")
    assert set1.out_group == set2.out_group == "control"

    c1 = {DEFAULT.device(e).condition for e in DEFAULT.runs["control_run1"].cards.values()}
    _write_masters(out, "baseline", "control", c1)

    assert has_output(set1, out, DEFAULT)
    assert not has_output(set2, out, DEFAULT)        # the bug this guards


def test_skip_existing_leaves_finished_runs_alone(tree, tmp_path, monkeypatch):
    """analyze_all must not re-run what is already done."""
    out = tmp_path / "out"
    c1 = {DEFAULT.device(e).condition for e in DEFAULT.runs["control_run1"].cards.values()}
    _write_masters(out, "baseline", "control", c1)

    called: list[str] = []

    def fake_analyze_run(rf, out_root, alloc=DEFAULT, **kw):
        called.append(f"{rf.timepoint}/{rf.run_key}")
        from qfn_aging.runner import RunResult
        return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {})

    monkeypatch.setattr("qfn_aging.runner.analyze_run", fake_analyze_run)
    analyze_all(tree, out, timepoints=["baseline"], skip_existing=True)

    assert "baseline/control_set1" not in called     # already had output
    assert "baseline/control_set2" in called         # its own conditions missing
    assert "baseline/25C_H2" in called


def test_analyzing_only_a_new_day(tree, tmp_path, monkeypatch):
    out = tmp_path / "out"
    called: list[str] = []

    def fake_analyze_run(rf, out_root, alloc=DEFAULT, **kw):
        called.append(f"{rf.timepoint}/{rf.run_key}")
        from qfn_aging.runner import RunResult
        return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {})

    monkeypatch.setattr("qfn_aging.runner.analyze_run", fake_analyze_run)
    analyze_all(tree, out, timepoints=["3D"])

    assert len(called) == 6
    assert all(c.startswith("3D/") for c in called)


def test_progress_callback_reports_each_run(tree, tmp_path, monkeypatch):
    out = tmp_path / "out"
    seen: list[tuple[int, int, str]] = []

    def fake_analyze_run(rf, out_root, alloc=DEFAULT, **kw):
        from qfn_aging.runner import RunResult
        return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {})

    monkeypatch.setattr("qfn_aging.runner.analyze_run", fake_analyze_run)
    analyze_all(tree, out, timepoints=["3D"], progress=lambda *a: seen.append(a))

    assert seen[0][:2] == (0, 6)
    assert seen[-1] == (6, 6, "done")


def test_nothing_to_do_is_reported_not_crashed(tree, tmp_path):
    out = tmp_path / "out"
    for group, rid in (("25C_H2", "H2_run1_25C"), ("35C_H2", "H2_run2_35C"),
                       ("45C_H2", "H2_run3_45C"), ("60C_H2", "H2_run4_60C")):
        conds = {DEFAULT.device(e).condition for e in DEFAULT.runs[rid].cards.values()}
        _write_masters(out, "3D", group, conds)
    for rid in ("control_run1", "control_run2"):
        conds = {DEFAULT.device(e).condition for e in DEFAULT.runs[rid].cards.values()}
        _write_masters(out, "3D", "control", conds)

    lines: list[str] = []
    res = analyze_all(tree, out, timepoints=["3D"], skip_existing=True, log=lines.append)
    assert res == []
    assert any("already has output" in ln for ln in lines)


# -- timepoints are derived, not hardcoded ---------------------------------

def test_any_day_folder_is_recognised(tmp_path):
    """The measured schedule drifted from the plan -- 6D/9D never ran, while
    7D and 22D did. A hardcoded list silently ignored those folders."""
    from qfn_aging.schedule import normalize_timepoint, parse_timepoint

    assert parse_timepoint("Baseline") == 0
    assert parse_timepoint("22D") == 22
    assert parse_timepoint("100D") == 100
    assert normalize_timepoint("22D") == "22D"
    assert normalize_timepoint("Baseline") == "baseline"


def test_non_timepoint_folders_are_still_ignored():
    from qfn_aging.schedule import parse_timepoint

    for name in ("analysis", "Haborer", "screening", "", "D", "abc"):
        assert parse_timepoint(name) is None


def test_timepoints_sort_by_day_not_alphabetically():
    from qfn_aging.schedule import sort_timepoints

    assert sort_timepoints(["13D", "3D", "baseline", "22D", "7D"]) == \
        ["baseline", "3D", "7D", "13D", "22D"]


def test_discovery_picks_up_an_unplanned_day(tmp_path):
    for key in ("25C_H2", "35C_H2", "45C_H2", "60C_H2"):
        _make_run(tmp_path, "22D", key, H2)
    for key in ("control_set1", "control_set2"):
        _make_run(tmp_path, "22D", key, CTL)

    assert available_timepoints(tmp_path) == ["22D"]
    runs, warnings = discover(tmp_path)
    assert len(runs) == 6
    assert {r.timepoint for r in runs} == {"22D"}
    assert warnings == []


def test_majors_accept_an_unplanned_day():
    from qfn_aging.schedule import majors_for

    h2 = next(d for d in DEFAULT.devices.values() if d.h2)
    ctl = next(d for d in DEFAULT.devices.values() if not d.h2)

    assert majors_for(h2, "22D") == ("n01", "n02", "n03", "n04", "n05", "n06")
    assert majors_for(ctl, "22D") == ("n01", "n02", "n03")
    # STEPS stays a baseline-only characterisation
    assert "n07" in majors_for(h2, "baseline")


def test_unparseable_timepoint_still_raises():
    from qfn_aging.schedule import majors_for

    h2 = next(d for d in DEFAULT.devices.values() if d.h2)
    with pytest.raises(ValueError, match="expected 'baseline' or"):
        majors_for(h2, "someday")
