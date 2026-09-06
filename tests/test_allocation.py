"""Unit tests for the QFN reliability-study device roster."""

from __future__ import annotations

import json

import pytest

from qfn_aging.allocation import (
    DATA_PATH,
    PARC_ONLY,
    PD_ONLY,
    PDPARC_H2,
    PDPARC_NO_H2,
    DEFAULT,
    load_allocation,
)


def test_default_roster_matches_document_totals():
    assert len(DEFAULT.devices) == 75
    assert len(DEFAULT.devices_by_category(PARC_ONLY)) == 12
    assert len(DEFAULT.devices_by_category(PD_ONLY)) == 12
    assert len(DEFAULT.devices_by_category(PDPARC_H2)) == 38
    assert len(DEFAULT.devices_by_category(PDPARC_NO_H2)) == 13  # physical packages (2T pair)
    assert sum(1 for d in DEFAULT.devices.values() if d.type == "B") == 18


def test_twelve_conditions_temp_x_rh():
    conds = DEFAULT.conditions()
    assert len(conds) == 12
    assert conds[0] == "25C_10RH"
    assert conds[-1] == "60C_50RH"
    assert {d.condition for d in DEFAULT.devices.values()} == set(conds)


def test_2t_pair_shares_condition_and_pair_group():
    a, b = DEFAULT.pair_members("2T_1234_1252")
    assert {a.esn, b.esn} == {"1234", "1252"}
    assert a.condition == b.condition == "60C_30RH"
    assert a.category == b.category == PDPARC_NO_H2
    assert a.h2 is False and b.h2 is False


def test_card_117_reused_between_h2_run_and_control_run2():
    device_runs = {run.run_id: card for run, card in DEFAULT.runs_for_device("1397")}
    assert device_runs == {"control_run2": "117"}
    device_runs_1257 = {run.run_id: card for run, card in DEFAULT.runs_for_device("1257")}
    assert device_runs_1257 == {"H2_run1_25C": "117"}


def test_condition_map_shape():
    cmap = DEFAULT.condition_map()
    assert cmap["1236"] == "25C_10RH"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in cmap.items())


def test_hydrogen_run_capacities():
    caps = {r.run_id: r.capacity_packages for r in DEFAULT.runs.values() if r.kind == "hydrogen"}
    assert caps == {
        "H2_run1_25C": 14,
        "H2_run2_35C": 12,
        "H2_run3_45C": 12,
        "H2_run4_60C": 12,
    }


def test_verification_catches_corrupted_roster(tmp_path):
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for esn, d in list(data["devices"].items()):
        if d["category"] == PDPARC_H2 and d["type"] == "A":
            del data["devices"][esn]
            break
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AssertionError, match="Type A with H2"):
        load_allocation(p)
