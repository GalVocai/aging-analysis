"""Measurement-mode scheduling: which majors run on which device, when.

**Timepoints are derived from the folder name, not a fixed list.** Any
folder called ``baseline`` or ``<n>D`` is a timepoint, ordered by ``n``
(baseline = day 0). The measured schedule already departed from the plan --
6D and 9D were never run, while 7D and 22D were -- so hardcoding the days
just meant new folders were silently ignored. ``PLANNED_TIMEPOINTS`` is kept
for reference only; nothing validates against it.

Major numbering mirrors the existing sensorlab convention (see that
project's protocol.py / SESSION_HANDOFF.md):
    n01 ZOZO, n02 JGH, n03 JG (air), n04 SE, n05 JG (H2 exposure),
    n06 JG (recovery), n07 STEPS

Rule (confirmed 2026-07-26, amended 2026-07-26 -- no STEPS at 13D):
  * Control devices (``device.h2 is False`` -- ParC-only and Pd|ParC
    without H2) run n01-n03 at EVERY timepoint.
  * Hydrogen devices (``device.h2 is True`` -- Pd-only and Pd|ParC with H2)
    run n01-n06 at every timepoint, PLUS n07 (STEPS) only at the baseline
    timepoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from .allocation import Allocation, DeviceRecord

import re

#: The originally planned schedule. Reference only -- discovery accepts any
#: ``baseline`` / ``<n>D`` folder, so the study can extend without a code change.
PLANNED_TIMEPOINTS: tuple[str, ...] = ("baseline", "3D", "6D", "9D", "13D")

#: Backwards-compatible alias; prefer the parsing helpers below.
TIMEPOINTS: tuple[str, ...] = PLANNED_TIMEPOINTS

BASELINE = "baseline"
_DAY_RE = re.compile(r"^(\d+)\s*D$", re.IGNORECASE)


def parse_timepoint(name: str) -> int | None:
    """Study day for a folder name, or ``None`` if it is not a timepoint.

    ``"Baseline"`` -> 0, ``"3D"`` -> 3, ``"22D"`` -> 22, ``"analysis"`` -> None.
    """
    text = str(name).strip()
    if text.lower() == BASELINE:
        return 0
    m = _DAY_RE.match(text)
    return int(m.group(1)) if m else None


def normalize_timepoint(name: str) -> str | None:
    """Canonical label for a timepoint folder, or ``None``."""
    day = parse_timepoint(name)
    if day is None:
        return None
    return BASELINE if day == 0 else f"{day}D"


def sort_timepoints(timepoints) -> list[str]:
    """Timepoints in study order (by day), unknown ones last by name."""
    def key(tp):
        day = parse_timepoint(tp)
        return (0, day, "") if day is not None else (1, 0, str(tp))
    return sorted(dict.fromkeys(timepoints), key=key)

MAJOR_NAMES: dict[str, str] = {
    "n01": "ZOZO",
    "n02": "JGH",
    "n03": "JG_air",
    "n04": "SE",
    "n05": "JG_H2",
    "n06": "JG_recovery",
    "n07": "STEPS",
}

CONTROL_MAJORS: tuple[str, ...] = ("n01", "n02", "n03")
H2_CORE_MAJORS: tuple[str, ...] = ("n01", "n02", "n03", "n04", "n05", "n06")
H2_EXTRA_MAJOR: str = "n07"


def majors_for(device: DeviceRecord, timepoint: str) -> tuple[str, ...]:
    """Which majors run on this device at this timepoint.

    Any ``baseline`` / ``<n>D`` label is accepted -- the study is allowed to
    add days the original plan did not list.
    """
    day = parse_timepoint(timepoint)
    if day is None:
        raise ValueError(
            f"unknown timepoint {timepoint!r}; expected 'baseline' or '<n>D'")

    if not device.h2:
        return CONTROL_MAJORS

    majors = list(H2_CORE_MAJORS)
    if day == 0:                       # STEPS is a baseline-only characterisation
        majors.append(H2_EXTRA_MAJOR)
    return tuple(majors)


@dataclass(frozen=True)
class PlanRow:
    esn: str
    condition: str
    category: str
    h2: bool
    timepoint: str
    major: str
    mode: str


def build_plan(alloc: Allocation, timepoints: tuple[str, ...] = TIMEPOINTS) -> list[PlanRow]:
    """Every (device, timepoint, major) measurement the study needs -- the
    execution checklist. One row per instrument file expected."""
    rows: list[PlanRow] = []
    for device in alloc.devices.values():
        for tp in timepoints:
            for major in majors_for(device, tp):
                rows.append(PlanRow(
                    esn=device.esn, condition=device.condition, category=device.category,
                    h2=device.h2, timepoint=tp, major=major, mode=MAJOR_NAMES[major],
                ))
    return rows
