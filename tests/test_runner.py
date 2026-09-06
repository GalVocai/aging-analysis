"""Tests for the sensorlab runner -- focused on the project-separation
guarantees, which are the part that must not silently regress.

These do not invoke sensorlab's analysis itself (that needs real instrument
files); they verify the wiring that keeps the two projects apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.runner import _check_outside_sensorlab_config, build_condition_cache

pytest.importorskip("sensorlab", reason="sensorlab must be installed to build a ConditionCache")


def test_condition_cache_carries_the_full_roster(tmp_path):
    cache = build_condition_cache(tmp_path, DEFAULT)
    mapping = cache.as_dict()

    assert len(mapping) == len(DEFAULT.devices) == 75
    assert mapping["1236"] == "25C_10RH"
    assert set(mapping.values()) == set(DEFAULT.conditions())
    assert cache.missing([d.esn for d in DEFAULT.devices.values()]) == []


def test_condition_cache_path_is_inside_our_output_not_sensorlab(tmp_path):
    cache = build_condition_cache(tmp_path, DEFAULT)
    sensorlab_cfg = Path.home() / ".sensorlab"

    assert Path(cache.path).parent == tmp_path
    assert sensorlab_cfg not in Path(cache.path).resolve().parents


def test_building_the_cache_writes_nothing(tmp_path):
    build_condition_cache(tmp_path, DEFAULT)
    assert list(tmp_path.iterdir()) == []


def test_guard_rejects_sensorlab_config_dir():
    cfg = Path.home() / ".sensorlab"
    with pytest.raises(ValueError, match="entirely separate"):
        _check_outside_sensorlab_config(cfg)
    with pytest.raises(ValueError, match="entirely separate"):
        _check_outside_sensorlab_config(cfg / "output" / "deep")


def test_guard_allows_ordinary_paths(tmp_path):
    _check_outside_sensorlab_config(tmp_path)
    _check_outside_sensorlab_config(tmp_path / "analysis" / "baseline")


def test_build_condition_cache_refuses_sensorlab_config_root():
    with pytest.raises(ValueError):
        build_condition_cache(Path.home() / ".sensorlab", DEFAULT)


def test_plot_style_is_ours_not_sensorlabs(tmp_path):
    """Figures must not depend on ~/.sensorlab/plot_style.json."""
    import json
    import sensorlab.pipeline as slp
    from qfn_aging.runner import _own_plot_style

    ours = tmp_path / "plot_style.json"
    ours.write_text(json.dumps({"linewidth": 7.25, "se_color": "#123456"}))

    before = slp.load_style()
    with _own_plot_style(ours):
        inside = slp.load_style()
        assert inside.linewidth == 7.25
        assert inside.se_color == "#123456"

    after = slp.load_style()
    assert after.linewidth == before.linewidth      # original restored
    assert slp.load_style is not None


def test_plot_style_falls_back_to_defaults_when_our_file_is_absent(tmp_path):
    import sensorlab.pipeline as slp
    from sensorlab.plotting.style import StyleConfig
    from qfn_aging.runner import _own_plot_style

    with _own_plot_style(tmp_path / "does_not_exist.json"):
        assert slp.load_style().linewidth == StyleConfig().linewidth


def test_plot_style_is_restored_even_on_error(tmp_path):
    import sensorlab.pipeline as slp
    from qfn_aging.runner import _own_plot_style

    original = slp.load_style
    with pytest.raises(RuntimeError):
        with _own_plot_style(tmp_path / "x.json"):
            raise RuntimeError("boom")
    assert slp.load_style is original


def test_shipped_style_has_no_other_studys_devices():
    """The style file was seeded from sensorlab's; its per-device entries
    referenced that study's ESNs and must not leak into this project."""
    import json
    from qfn_aging.runner import STYLE_PATH

    if not STYLE_PATH.is_file():
        pytest.skip("no style file shipped")
    data = json.loads(STYLE_PATH.read_text(encoding="utf-8"))
    roster = set(DEFAULT.devices)
    for key in ("stack_hidden",):
        for esn in (data.get(key) or {}):
            assert str(esn) in roster, f"{key} references non-roster ESN {esn}"


def test_merged_control_runs_cover_disjoint_conditions():
    """control_set1 and control_set2 share one output folder, so their
    conditions must not overlap or masters would overwrite each other."""
    from qfn_aging.runner import _condition_clashes
    from qfn_aging.discovery import RunFolder

    runs = [
        RunFolder(path=Path("a"), timepoint="baseline", run_key="control_set1",
                  run_id="control_run1", results=[]),
        RunFolder(path=Path("b"), timepoint="baseline", run_key="control_set2",
                  run_id="control_run2", results=[]),
    ]
    assert _condition_clashes(runs, DEFAULT) == []

    c1 = {DEFAULT.device(e).condition for e in DEFAULT.runs["control_run1"].cards.values()}
    c2 = {DEFAULT.device(e).condition for e in DEFAULT.runs["control_run2"].cards.values()}
    assert c1 & c2 == set()
    assert c1 | c2 == set(DEFAULT.conditions())      # together they cover all 12


def test_clash_is_reported_when_two_runs_share_a_condition():
    from qfn_aging.runner import _condition_clashes
    from qfn_aging.discovery import RunFolder

    # same run id twice under different keys -> guaranteed overlap
    runs = [
        RunFolder(path=Path("a"), timepoint="baseline", run_key="control_set1",
                  run_id="control_run1", results=[]),
        RunFolder(path=Path("b"), timepoint="baseline", run_key="control_setX",
                  run_id="control_run1", results=[]),
    ]
    problems = _condition_clashes(runs, DEFAULT)
    assert problems and "would overwrite" in problems[0]
