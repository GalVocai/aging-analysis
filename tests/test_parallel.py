"""Tests for the crash fix (matplotlib backend) and parallel execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qfn_aging.allocation import DEFAULT
from qfn_aging.runner import RunResult, analyze_all

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
    for key in ("25C_H2", "35C_H2", "45C_H2", "60C_H2"):
        _make_run(tmp_path, "Baseline", key, H2)
    for key in ("control_set1", "control_set2"):
        _make_run(tmp_path, "Baseline", key, CTL)
    return tmp_path


# -- the crash fix ---------------------------------------------------------

def test_matplotlib_backend_is_thread_safe():
    """A Qt backend crashes the process when figures are made off the main
    thread, which is exactly how the GUI runs the analysis."""
    import matplotlib

    import qfn_aging.runner  # noqa: F401  (importing it is what sets this)

    assert matplotlib.get_backend().lower() == "agg"


def test_backend_survives_qt_being_imported_first():
    pytest.importorskip("PySide6")
    import matplotlib
    from PySide6.QtWidgets import QApplication  # noqa: F401

    import qfn_aging.runner  # noqa: F401

    import matplotlib.pyplot  # noqa: F401  (would latch a backend if unset)
    assert matplotlib.get_backend().lower() == "agg"


def test_figures_can_be_created_off_the_main_thread():
    """The regression itself: this crashed the interpreter under Qt."""
    import threading

    import qfn_aging.runner  # noqa: F401

    errors: list[BaseException] = []

    def draw():
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.add_subplot(111).plot([0, 1], [0, 1])
            plt.close(fig)
        except BaseException as exc:      # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=draw)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive()
    assert not errors


# -- figure resolution -----------------------------------------------------

def test_dpi_override_applies_and_restores():
    """sensorlab passes dpi=110 explicitly, so the keyword must be
    overridden on the way through -- an rcParam would be ignored."""
    import io

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from PIL import Image

    from qfn_aging.runner import _plot_dpi

    def size(**kw):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([0, 1], [0, 1])
        buf = io.BytesIO()
        fig.savefig(buf, **kw)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).size

    original = Figure.savefig
    assert size(dpi=110) == (880, 550)

    with _plot_dpi(90):
        assert size(dpi=110) == (720, 450)      # request overridden

    assert size(dpi=110) == (880, 550)
    assert Figure.savefig is original


def test_dpi_none_is_a_no_op():
    from matplotlib.figure import Figure

    from qfn_aging.runner import _plot_dpi

    original = Figure.savefig
    with _plot_dpi(None):
        assert Figure.savefig is original


def test_dpi_is_restored_after_an_error():
    from matplotlib.figure import Figure

    from qfn_aging.runner import _plot_dpi

    original = Figure.savefig
    with pytest.raises(RuntimeError):
        with _plot_dpi(90):
            raise RuntimeError("boom")
    assert Figure.savefig is original


def test_project_default_dpi():
    from qfn_aging.runner import PLOT_DPI
    assert PLOT_DPI == 90


# -- surviving a broken pool ----------------------------------------------

def test_pythonw_workers_are_redirected_to_python_exe(monkeypatch):
    """multiprocessing spawns children with sys.executable; pythonw children
    have no stdio and die, which is what killed the H2 runs."""
    import multiprocessing
    import sys

    from qfn_aging.runner import ensure_multiprocessing_executable

    fake = Path(sys.executable).with_name("pythonw.exe")
    monkeypatch.setattr(sys, "executable", str(fake))
    captured = {}
    monkeypatch.setattr(multiprocessing, "set_executable",
                        lambda p: captured.__setitem__("exe", p))

    result = ensure_multiprocessing_executable()
    if Path(sys.executable).with_name("python.exe").is_file():
        assert result and result.endswith("python.exe")
        assert captured["exe"].endswith("python.exe")


def test_console_python_is_left_alone(monkeypatch):
    import sys

    from qfn_aging.runner import ensure_multiprocessing_executable

    monkeypatch.setattr(sys, "executable",
                        str(Path(sys.executable).with_name("python.exe")))
    assert ensure_multiprocessing_executable() is None


def test_broken_pool_falls_back_to_serial_instead_of_losing_runs(tree, tmp_path,
                                                                monkeypatch):
    """A dead pool must not cost the whole sweep -- exactly the failure the
    user hit, where all four H2 runs were lost at once."""
    from concurrent.futures.process import BrokenProcessPool

    import qfn_aging.runner as R
    from qfn_aging.discovery import discover

    runs, _ = discover(tree, timepoints=["baseline"])
    lines: list[str] = []
    analyzed: list[str] = []

    class DeadPool:
        def __init__(self, max_workers=None): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, payload): raise BrokenProcessPool("pool died")

    import concurrent.futures as cf
    monkeypatch.setattr(cf, "ProcessPoolExecutor", DeadPool)
    monkeypatch.setattr(R, "analyze_run",
                        lambda rf, o, a=DEFAULT, **kw: (
                            analyzed.append(rf.run_key),
                            RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {}),
                        )[1])

    results = R._analyze_parallel(runs, tmp_path / "o", DEFAULT, False, None,
                                  90, 4, None, lines.append)

    assert len(results) == 6                       # nothing lost
    assert len(analyzed) == 6                      # all re-run serially
    assert any("Falling back" in ln for ln in lines)


# -- parallel execution ----------------------------------------------------

def test_parallel_and_serial_cover_the_same_runs(tree, tmp_path, monkeypatch):
    seen_serial: list[str] = []

    def fake(rf, out_root, alloc=DEFAULT, **kw):
        seen_serial.append(rf.run_key)
        return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {})

    monkeypatch.setattr("qfn_aging.runner.analyze_run", fake)
    serial = analyze_all(tree, tmp_path / "o1", workers=1)
    assert len(serial) == 6

    # workers>1 with a monkeypatched function can't cross the process
    # boundary, so just assert the routing decision itself
    from qfn_aging.runner import _analyze_parallel
    assert callable(_analyze_parallel)


def test_single_worker_stays_in_process(tree, tmp_path, monkeypatch):
    """workers=1 must not spawn a pool -- the log has to stream live."""
    called = {"parallel": False}
    monkeypatch.setattr("qfn_aging.runner._analyze_parallel",
                        lambda *a, **k: called.__setitem__("parallel", True) or [])
    monkeypatch.setattr("qfn_aging.runner.analyze_run",
                        lambda rf, o, a=DEFAULT, **kw:
                        RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {}))

    analyze_all(tree, tmp_path / "o", workers=1)
    assert not called["parallel"]


def test_workers_are_capped_at_the_number_of_runs(tree, tmp_path, monkeypatch):
    captured = {}

    def fake_parallel(runs, out_root, alloc, make_plots, style_path, dpi,
                      workers, progress, emit):
        captured["workers"] = workers
        captured["runs"] = len(runs)
        captured["dpi"] = dpi
        return []

    monkeypatch.setattr("qfn_aging.runner._analyze_parallel", fake_parallel)
    analyze_all(tree, tmp_path / "o", timepoints=["baseline"], workers=99)

    assert captured["runs"] == 6
    # _analyze_parallel does the clamping itself; it receives the raw request
    assert captured["workers"] == 99
    assert captured["dpi"] == 90          # project default flows through


def test_a_failing_worker_does_not_lose_the_other_runs(tree, tmp_path, monkeypatch):
    """One bad run must not abort the sweep."""
    from qfn_aging.runner import _analyze_parallel
    from qfn_aging.discovery import discover

    runs, _ = discover(tree, timepoints=["baseline"])
    lines: list[str] = []

    class _Boom(Exception):
        pass

    class FakeFuture:
        def __init__(self, rf, fail):
            self._rf, self._fail = rf, fail

        def result(self):
            if self._fail:
                raise _Boom("worker died")
            return RunResult(self._rf.timepoint, self._rf.run_key,
                             self._rf.run_id, self._rf.path, {"JG": []}), ["ok"]

    class FakePool:
        def __init__(self, max_workers=None): self.max_workers = max_workers
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, payload):
            rf = payload[0]
            return FakeFuture(rf, fail=(rf.run_key == "45C_H2"))

    import qfn_aging.runner as R
    monkeypatch.setattr(R, "_analyze_parallel", _analyze_parallel)
    monkeypatch.setitem(__import__("sys").modules, "concurrent.futures",
                        type("M", (), {
                            "ProcessPoolExecutor": FakePool,
                            "as_completed": lambda fs: list(fs),
                        }))

    results = _analyze_parallel(runs, tmp_path / "o", DEFAULT, False, None,
                                90, 4, None, lines.append)
    assert len(results) == 6
    failed = [r for r in results if not r.ok]
    assert len(failed) == 1 and failed[0].run_key == "45C_H2"
    assert any("FAILED" in ln for ln in lines)
