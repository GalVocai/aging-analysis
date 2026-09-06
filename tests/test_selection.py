"""Tests for per-day transistor selection and its effect on averaging."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.grouping import add_group_keys, aggregate_groups
from qfn_aging.selection import (ALL_DAYS, Selection, apply_selection, load_selection)


def _tidy(timepoints=("baseline",)):
    """1236 and 1250 are both PdParC_H2 @ 25C_10RH -> one group."""
    rows = []
    for tp in timepoints:
        for esn, base in (("1236", 1.0), ("1250", 3.0)):
            for i, t in enumerate((11, 12)):
                rows.append(dict(esn=esn, tnumber=t, timepoint=tp, sheet="JG_n03",
                                 series="", metric="Vth_bg4", value=base + i))
    return pd.DataFrame(rows)


def _mean(df, timepoint):
    g = aggregate_groups(add_group_keys(df, DEFAULT))
    row = g[g["timepoint"] == timepoint].iloc[0]
    return row["mean"], int(row["n"])


# -- basics ----------------------------------------------------------------

def test_nothing_excluded_by_default():
    sel = Selection()
    assert len(sel) == 0
    assert sel.is_included("baseline", "1236", 11)
    assert apply_selection(_tidy(), sel)["included"].all()


def test_excluded_rows_are_kept_but_not_averaged():
    sel = Selection()
    sel.exclude("baseline", "1250", 12, reason="user call")

    df = apply_selection(_tidy(), sel)
    assert len(df) == 4                        # nothing dropped
    assert df["included"].sum() == 3
    assert df.loc[~df["included"], "exclude_reason"].iloc[0] == "user call"

    mean, n = _mean(df, "baseline")
    assert n == 3
    assert mean == pytest.approx((1.0 + 2.0 + 3.0) / 3)


# -- the point of this model: per-day independence -------------------------

def test_excluding_on_one_day_leaves_the_other_untouched():
    sel = Selection()
    sel.exclude("3D", "1236", 11, reason="bad at 3D only")

    df = apply_selection(_tidy(("baseline", "3D")), sel)

    assert _mean(df, "baseline") == (pytest.approx(2.5), 4)
    mean_3d, n_3d = _mean(df, "3D")
    assert n_3d == 3
    assert mean_3d == pytest.approx((2.0 + 3.0 + 4.0) / 3)


def test_the_same_channel_can_differ_between_days():
    sel = Selection()
    sel.exclude("3D", "1236", 11)

    assert sel.is_included("baseline", "1236", 11)
    assert not sel.is_included("3D", "1236", 11)


def test_all_days_exclusion_covers_every_timepoint():
    sel = Selection()
    sel.exclude(ALL_DAYS, "1236", 11, reason="never worked")

    assert not sel.is_included("baseline", "1236", 11)
    assert not sel.is_included("3D", "1236", 11)
    assert not sel.is_included("13D", "1236", 11)
    assert sel.reason("6D", "1236", 11) == "never worked"


def test_including_clears_an_all_days_exclusion():
    """Otherwise re-ticking a globally excluded row would appear to do nothing."""
    sel = Selection()
    sel.exclude(ALL_DAYS, "1236", 11)
    sel.include("baseline", "1236", 11)
    assert sel.is_included("baseline", "1236", 11)
    assert sel.is_included("3D", "1236", 11)


def test_count_and_clear_are_per_day():
    sel = Selection()
    sel.exclude("baseline", "1236", 11)
    sel.exclude("3D", "1236", 12)
    sel.exclude(ALL_DAYS, "1250", 11)

    assert sel.count_for("baseline") == 2          # own + all-days
    assert sel.count_for("3D") == 2
    assert len(sel) == 3

    sel.clear("3D")
    assert sel.count_for("3D") == 1                # the all-days one remains
    assert sel.count_for("baseline") == 2


def test_excluded_for_lists_a_devices_channels_on_a_day():
    sel = Selection()
    sel.exclude("baseline", "1250", 11)
    sel.exclude(ALL_DAYS, "1250", 12)
    sel.exclude("3D", "1250", 31)

    assert sel.excluded_for("baseline", "1250") == [11, 12]
    assert sel.excluded_for("3D", "1250") == [12, 31]


def test_device_level_helpers():
    sel = Selection()
    sel.exclude_device("3D", "1250", [11, 12], reason="all channels")
    assert sel.excluded_for("3D", "1250") == [11, 12]
    assert sel.excluded_for("baseline", "1250") == []

    sel.include_device("3D", "1250", [11, 12])
    assert len(sel) == 0


def test_group_with_everything_excluded_disappears():
    sel = Selection()
    sel.exclude_device("baseline", "1236", [11, 12])
    sel.exclude_device("baseline", "1250", [11, 12])

    g = aggregate_groups(add_group_keys(apply_selection(_tidy(), sel), DEFAULT))
    assert g.empty


# -- persistence -----------------------------------------------------------

def test_roundtrip_keeps_the_day(tmp_path):
    sel = Selection()
    sel.exclude("3D", "1236", 11, reason="noisy")
    sel.exclude(ALL_DAYS, "1019", 31)
    p = tmp_path / "selection.json"
    sel.save(p)

    back = load_selection(p)
    assert len(back) == 2
    assert not back.is_included("3D", "1236", 11)
    assert back.is_included("baseline", "1236", 11)
    assert not back.is_included("baseline", "1019", 31)
    assert back.reason("3D", "1236", 11) == "noisy"


def test_legacy_file_without_timepoint_loads_as_all_days(tmp_path):
    """Selections saved before this was per-day must keep working."""
    p = tmp_path / "selection.json"
    p.write_text(json.dumps({"excluded": [
        {"esn": "1236", "tnumber": 11, "reason": "excluded in GUI"},
    ]}), encoding="utf-8")

    sel = load_selection(p)
    assert len(sel) == 1
    assert not sel.is_included("baseline", "1236", 11)
    assert not sel.is_included("3D", "1236", 11)
    assert sel.timepoints() == [ALL_DAYS]


def test_saved_file_records_the_day_and_a_version(tmp_path):
    sel = Selection()
    sel.exclude("6D", "1236", 12)
    p = tmp_path / "s.json"
    sel.save(p)

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["excluded"][0]["timepoint"] == "6D"


def test_missing_or_corrupt_file_means_nothing_excluded(tmp_path):
    assert len(load_selection(tmp_path / "absent.json")) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert len(load_selection(bad)) == 0


def test_selection_path_is_ours_not_sensorlabs():
    from pathlib import Path

    from qfn_aging.selection import DEFAULT_SELECTION_PATH

    assert DEFAULT_SELECTION_PATH.parent.name == ".qfn_aging"
    assert (Path.home() / ".sensorlab") not in DEFAULT_SELECTION_PATH.parents


def test_key_types_are_normalised(tmp_path):
    """The GUI may hand back ints or strings; both must match."""
    sel = Selection()
    sel.exclude("3D", 1236, "11")            # type: ignore[arg-type]
    assert not sel.is_included("3D", "1236", 11)

    sel.save(tmp_path / "s.json")
    rec = json.loads((tmp_path / "s.json").read_text())["excluded"][0]
    assert rec["esn"] == "1236" and rec["tnumber"] == 11
