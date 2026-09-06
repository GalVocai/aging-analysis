"""Unit tests for measurement-mode scheduling."""

from __future__ import annotations

from qfn_aging.allocation import DEFAULT, PDPARC_H2, PARC_ONLY
from qfn_aging.schedule import TIMEPOINTS, build_plan, majors_for


def test_control_device_gets_n01_n03_at_every_timepoint():
    control = DEFAULT.devices_by_category(PARC_ONLY)[0]
    assert control.h2 is False
    for tp in TIMEPOINTS:
        assert majors_for(control, tp) == ("n01", "n02", "n03")


def test_h2_device_gets_steps_only_at_baseline():
    h2_device = DEFAULT.devices_by_category(PDPARC_H2)[0]
    assert h2_device.h2 is True

    assert majors_for(h2_device, "baseline") == ("n01", "n02", "n03", "n04", "n05", "n06", "n07")
    for tp in ("3D", "6D", "9D", "13D"):
        assert majors_for(h2_device, tp) == ("n01", "n02", "n03", "n04", "n05", "n06")


def test_any_day_number_is_accepted():
    """Days are parsed from the label, so the study can add days the
    original plan never listed (7D and 22D were measured; 6D and 9D were not)."""
    device = DEFAULT.devices_by_category(PARC_ONLY)[0]
    for tp in ("baseline", "3D", "7D", "20D", "22D", "100D"):
        assert majors_for(device, tp) == ("n01", "n02", "n03")


def test_a_non_day_label_still_raises():
    device = DEFAULT.devices_by_category(PARC_ONLY)[0]
    for bad in ("someday", "analysis", "", "D"):
        try:
            majors_for(device, bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_full_plan_row_count():
    t = len(TIMEPOINTS)
    # control: 25 devices (12 ParC-only + 13 Pd|ParC-noH2 physical) x 3 majors x t timepoints
    # h2: 50 devices (12 Pd-only + 38 Pd|ParC-H2) x (6 majors x t timepoints + 1 extra STEPS at baseline)
    plan = build_plan(DEFAULT)
    assert len(plan) == 25 * 3 * t + 50 * (6 * t + 1)

    steps_rows = [r for r in plan if r.major == "n07"]
    assert {r.timepoint for r in steps_rows} == {"baseline"}
    assert len(steps_rows) == 50  # every H2 device, baseline only
