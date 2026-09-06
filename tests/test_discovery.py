"""Unit tests for run-folder discovery.

Built on a synthetic tree so they run without the OneDrive data present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qfn_aging.discovery import discover, verify

H2_MAJORS = [("n01", "SE_STATE"), ("n02", "JGH"), ("n03", "JG"), ("n04", "SE_STATE"),
             ("n05", "JG"), ("n06", "JG"), ("n07", "SE_STATE")]
CONTROL_MAJORS = [("n01", "SE_STATE"), ("n02", "JGH"), ("n03", "JG")]


def _make_run(root: Path, timepoint: str, run_key: str, majors, script="script.xlsx"):
    inner = root / timepoint / run_key / f"E1000_2026_07_25_N01_{run_key}"
    inner.mkdir(parents=True)
    for major, mode in majors:
        (inner / f"E1000_2026_07_25_N01_{major}_{mode}_.xlsx").touch()
    if script:
        (inner / script).touch()
    return inner


@pytest.fixture
def tree(tmp_path):
    for key in ("25C_H2", "35C_H2", "45C_H2", "60C_H2"):
        _make_run(tmp_path, "Baseline", key, H2_MAJORS)
    for key in ("control_set1", "control_set2"):
        _make_run(tmp_path, "Baseline", key, CONTROL_MAJORS)
    return tmp_path


def test_discovers_all_runs_and_maps_run_ids(tree):
    runs, warnings = discover(tree)
    assert warnings == []
    assert len(runs) == 6

    by_key = {r.run_key: r for r in runs}
    assert by_key["25C_H2"].run_id == "H2_run1_25C"
    assert by_key["60C_H2"].run_id == "H2_run4_60C"
    assert by_key["control_set1"].run_id == "control_run1"
    assert by_key["control_set2"].run_id == "control_run2"
    assert all(r.timepoint == "baseline" for r in runs)


def test_majors_and_script_are_separated(tree):
    runs, _ = discover(tree)
    by_key = {r.run_key: r for r in runs}

    assert by_key["25C_H2"].majors == ["n01", "n02", "n03", "n04", "n05", "n06", "n07"]
    assert by_key["control_set1"].majors == ["n01", "n02", "n03"]
    assert by_key["25C_H2"].script.name == "script.xlsx"

    n03 = by_key["25C_H2"].result("n03")
    assert n03.mode_token == "JG"
    assert n03.mode == "JG_air"          # canonical name from schedule.MAJOR_NAMES
    assert by_key["25C_H2"].result("n07").mode == "STEPS"


def test_control_runs_share_one_output_folder(tree):
    runs, _ = discover(tree)
    by_key = {r.run_key: r for r in runs}

    assert by_key["control_set1"].out_group == "control"
    assert by_key["control_set2"].out_group == "control"
    # hydrogen runs stay separate, one per temperature
    assert by_key["25C_H2"].out_group == "25C_H2"
    assert by_key["60C_H2"].out_group == "60C_H2"


def test_verify_passes_on_a_complete_tree(tree):
    runs, _ = discover(tree)
    assert verify(runs) == []


def test_verify_reports_a_missing_major(tree):
    runs, _ = discover(tree)
    target = next(r for r in runs if r.run_key == "35C_H2")
    target.result("n05").path.unlink()

    runs2, _ = discover(tree)
    problems = verify(runs2)
    assert any("35C_H2" in p and "missing n05" in p for p in problems)


def test_control_run_with_h2_majors_is_flagged_as_unexpected(tree):
    runs, _ = discover(tree)
    folder = next(r for r in runs if r.run_key == "control_set1").path
    (folder / "E1000_2026_07_25_N01_n05_JG_.xlsx").touch()

    runs2, _ = discover(tree)
    assert any("control_set1" in p and "unexpected n05" in p for p in verify(runs2))


def test_lock_files_and_unknown_folders_are_ignored(tree):
    folder = next(r for r in discover(tree)[0] if r.run_key == "25C_H2").path
    (folder / "~$E1000_2026_07_25_N01_n03_JG_.xlsx").touch()
    (tree / "Haborer").mkdir()
    (tree / "Haborer" / "screening").mkdir()

    runs, warnings = discover(tree)
    assert warnings == []
    assert len(runs) == 6
    assert next(r for r in runs if r.run_key == "25C_H2").majors.count("n03") == 1


def test_missing_root_reports_warning(tmp_path):
    runs, warnings = discover(tmp_path / "nope")
    assert runs == []
    assert any("does not exist" in w for w in warnings)


def test_unmatched_run_folder_warns_instead_of_raising(tree):
    _make_run(tree, "Baseline", "99C_H2", H2_MAJORS)
    runs, warnings = discover(tree)
    assert len(runs) == 6
    assert any("99C_H2" in w for w in warnings)
