"""Unit tests for the (condition x category) group-average layer."""

from __future__ import annotations

import pandas as pd
import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.grouping import add_group_keys, aggregate_groups


def test_group_devices_covers_every_device_exactly_once():
    groups = DEFAULT.group_devices()
    total = sum(len(v) for v in groups.values())
    assert total == len(DEFAULT.devices)
    for (condition, category), devices in groups.items():
        for d in devices:
            assert d.condition == condition
            assert d.category == category


def test_aggregate_groups_mean_and_avdev():
    # real group: ('25C_30RH', 'PdParC_H2') = esns 1239, 1241, 1257, 1258
    df = pd.DataFrame({
        "esn": ["1239", "1241", "1257", "1258"],
        "timepoint": ["baseline"] * 4,
        "metric": ["Vth"] * 4,
        "value": [1.0, 2.0, 3.0, 6.0],
    })
    df = add_group_keys(df, DEFAULT)
    assert set(df["condition"]) == {"25C_30RH"}
    assert set(df["category"]) == {"PdParC_H2"}

    out = aggregate_groups(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["timepoint"] == "baseline"
    assert row["condition"] == "25C_30RH"
    assert row["category"] == "PdParC_H2"
    assert row["metric"] == "Vth"
    assert row["n"] == 4
    assert row["mean"] == pytest.approx(3.0)
    # mean=3.0; abs devs = [2,1,0,3] -> avdev = 6/4 = 1.5
    assert row["avdev"] == pytest.approx(1.5)


def test_aggregate_groups_keeps_groups_separate():
    # two different real groups, one row each, same metric/timepoint
    df = pd.DataFrame({
        "esn": ["1236", "1239"],  # 25C_10RH/PdParC_H2, 25C_30RH/PdParC_H2
        "timepoint": ["baseline", "baseline"],
        "metric": ["Vth", "Vth"],
        "value": [1.0, 5.0],
    })
    df = add_group_keys(df, DEFAULT)
    out = aggregate_groups(df)
    assert len(out) == 2
    assert set(out["condition"]) == {"25C_10RH", "25C_30RH"}
    for _, row in out.iterrows():
        assert row["n"] == 1
        assert row["avdev"] == 0.0
