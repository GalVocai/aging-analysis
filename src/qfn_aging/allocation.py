"""Single source of truth for the QFN reliability-study device roster.

Transcribed from ``docs/QFN_Reliability_Allocation_Plan_EN.docx`` (Master
Allocation Table, hydrogen-run card assignments, control-run card
assignments, and daily schedule) into
``data/qfn_reliability_allocation.json``, then re-derived at load time and
cross-checked against every total the document itself states (75 physical
packages, 12 ParC-only, 12 Pd-only, 38 Pd|ParC with H2 split 26 Type A / 12
Type B, 12 Pd|ParC without H2 = 13 physical packages because 1234+1252 are
one 2T unit, 18 Type B total). A mismatch raises at import time -- a typo in
the roster fails loud, not as a silently wrong master workbook.

This module only answers "what is this device / what runs is it in" --
it does NOT decide which measurement modes apply to which category. That
scheduling logic (control devices -> n01-n03 only; H2 devices -> n01-n06,
plus n07/STEPS on the first and last day of exposure only) is still open.

Condition labels here (``"25C_10RH"`` etc.) are plain strings, one per
temp/RH cell (12 total) -- deliberately shaped like a flat ESN->label
mapping so they can be plugged into whatever per-device condition scheme
this project ends up using, via :meth:`Allocation.condition_map`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "qfn_reliability_allocation.json"

# Device categories (the ``category`` field on every DeviceRecord).
PARC_ONLY = "ParC_only"
PD_ONLY = "Pd_only"
PDPARC_NO_H2 = "PdParC_noH2"
PDPARC_H2 = "PdParC_H2"


@dataclass(frozen=True)
class DeviceRecord:
    esn: str
    temp_c: int
    rh_pct: int
    category: str                  # PARC_ONLY | PD_ONLY | PDPARC_NO_H2 | PDPARC_H2
    type: str | None               # "A" | "B" | None (ParC-only / Pd-only have no type)
    h2: bool                       # exposed to hydrogen in this study
    pair_group: str | None         # e.g. "2T_1234_1252" for devices sharing one physical 2T package
    condition: str                 # "25C_10RH" etc.


@dataclass(frozen=True)
class Run:
    run_id: str
    kind: str                      # "hydrogen" | "control"
    day: int
    order: int                     # run order within the day
    temp_c: int | None             # None for control runs (span multiple temps)
    capacity_packages: int
    cards: dict[str, str]          # card number (str) -> esn


class Allocation:
    """Loaded roster: devices, runs, and the daily schedule."""

    def __init__(self, meta: dict, devices: dict[str, DeviceRecord], runs: dict[str, Run],
                 schedule: list[dict]):
        self.meta = meta
        self.devices = devices
        self.runs = runs
        self.schedule = schedule

    # -- lookups ---------------------------------------------------------

    def device(self, esn: str) -> DeviceRecord:
        return self.devices[str(esn)]

    def devices_by_category(self, category: str) -> list[DeviceRecord]:
        return [d for d in self.devices.values() if d.category == category]

    def devices_in_condition(self, condition: str) -> list[DeviceRecord]:
        return [d for d in self.devices.values() if d.condition == condition]

    def runs_for_device(self, esn: str) -> list[tuple[Run, str]]:
        """(run, card) pairs a device is measured on -- usually one, two for
        the card-117 reuse case (25C H2 run -> Control Run 2)."""
        esn = str(esn)
        out = []
        for run in self.runs.values():
            for card, dev_esn in run.cards.items():
                if dev_esn == esn:
                    out.append((run, card))
        return out

    def pair_members(self, pair_group: str) -> list[DeviceRecord]:
        return [d for d in self.devices.values() if d.pair_group == pair_group]

    def condition_map(self) -> dict[str, str]:
        """ESN -> condition label."""
        return {esn: d.condition for esn, d in self.devices.items()}

    def conditions(self) -> list[str]:
        """All distinct condition labels, sorted (temp ascending, then RH)."""
        temps = sorted(self.meta["temps_c"])
        rhs = sorted(self.meta["rh_pct"])
        return [f"{t}C_{r}RH" for t in temps for r in rhs]

    def group_devices(self) -> dict[tuple[str, str], list[DeviceRecord]]:
        """(condition, category) -> devices in that group. Not every
        combination is populated -- only groups with at least one device
        are keys of the returned dict."""
        groups: dict[tuple[str, str], list[DeviceRecord]] = {}
        for d in self.devices.values():
            groups.setdefault((d.condition, d.category), []).append(d)
        return groups


def load_allocation(path: Path = DATA_PATH) -> Allocation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    devices = {
        esn: DeviceRecord(
            esn=d["esn"], temp_c=d["temp_c"], rh_pct=d["rh_pct"],
            category=d["category"], type=d["type"], h2=d["h2"],
            pair_group=d["pair_group"], condition=d["condition"],
        )
        for esn, d in data["devices"].items()
    }
    runs = {
        run_id: Run(
            run_id=run_id, kind=r["kind"], day=r["day"], order=r["order"],
            temp_c=r["temp_c"], capacity_packages=r["capacity_packages"],
            cards=dict(r["cards"]),
        )
        for run_id, r in data["runs"].items()
    }

    alloc = Allocation(meta=data["meta"], devices=devices, runs=runs, schedule=data["schedule"])
    _verify(alloc)
    return alloc


def _verify(alloc: Allocation) -> None:
    """Re-derive every total the source document states. Fails loud on a
    transcription error instead of silently shipping a wrong roster."""

    def count(pred):
        return sum(1 for d in alloc.devices.values() if pred(d))

    checks = {
        "physical packages": (count(lambda d: True), 75),
        "ParC only": (count(lambda d: d.category == PARC_ONLY), 12),
        "Pd only": (count(lambda d: d.category == PD_ONLY), 12),
        "Type A with H2": (count(lambda d: d.category == PDPARC_H2 and d.type == "A"), 26),
        "Type B with H2": (count(lambda d: d.category == PDPARC_H2 and d.type == "B"), 12),
        "Type B without H2": (count(lambda d: d.category == PDPARC_NO_H2 and d.type == "B"), 6),
        "Pd|ParC with H2": (count(lambda d: d.category == PDPARC_H2), 38),
        "Pd|ParC without H2 (physical)": (count(lambda d: d.category == PDPARC_NO_H2), 13),
        "Type B total": (count(lambda d: d.type == "B"), 18),
    }
    bad = [f"{name}: got={got} want={want}" for name, (got, want) in checks.items() if got != want]
    assert not bad, "QFN allocation roster failed verification:\n" + "\n".join(bad)

    pair_ids = {d.pair_group for d in alloc.devices.values() if d.pair_group}
    for run in alloc.runs.values():
        assert len(run.cards) == run.capacity_packages, (
            f"{run.run_id}: {len(run.cards)} card slots != capacity_packages={run.capacity_packages}"
        )
        for card, esn in run.cards.items():
            assert esn in alloc.devices, f"{run.run_id} card {card} -> unknown ESN {esn}"

    equiv_a_units = count(lambda d: d.type == "A") - sum(
        len(alloc.pair_members(g)) - 1 for g in pair_ids
    )
    assert equiv_a_units == 32, f"equivalent Type A units: got={equiv_a_units} want=32"


DEFAULT = load_allocation()
