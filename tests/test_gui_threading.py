"""Regression tests for the background-work threading in the GUI.

The window froze after a *successful* analysis because the completion
handler was a local closure: Qt has no receiver QObject for one, so it runs
the handler directly on the emitting (worker) thread, where the handler's
``QThread.wait()`` waited on its own thread forever.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qfn_aging.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    return MainWindow(selection_path=tmp_path / "selection.json",
                      settings_path=tmp_path / "settings.json")


def _pump(app, predicate, timeout_ms=15000):
    """Spin the event loop until predicate() or timeout. Returns success."""
    from PySide6.QtCore import QElapsedTimer
    t = QElapsedTimer()
    t.start()
    while not predicate():
        app.processEvents()
        if t.elapsed() > timeout_ms:
            return False
        QThread.msleep(10)
    return True


def test_completion_handler_is_a_bound_method_not_a_closure(win):
    """A closure would be invoked on the worker thread -- the deadlock."""
    assert hasattr(win, "_on_worker_done")
    assert callable(win._on_worker_done)
    # bound to the window, which lives on the UI thread
    assert win._on_worker_done.__self__ is win


def test_background_job_completes_and_unfreezes_the_ui(win, qapp):
    """The regression: after the job finished the window never came back."""
    marks: list[str] = []
    win._load_results = lambda: marks.append("loaded")     # skip disk work

    win._run_bg(lambda emit: emit("working"), "job finished")

    assert _pump(qapp, lambda: win._thread is None), \
        "worker thread was never cleaned up -- UI is deadlocked"

    assert "job finished" in win.log.toPlainText()
    assert marks == ["loaded"]
    assert win.btn_analyze.isEnabled()          # controls re-enabled


def test_failure_is_reported_and_ui_recovers(win, qapp):
    called: list[str] = []
    win._load_results = lambda: called.append("loaded")

    def boom(emit):
        raise RuntimeError("kaboom")

    win._run_bg(boom, "should not appear")

    assert _pump(qapp, lambda: win._thread is None)
    text = win.log.toPlainText()
    assert "FAILED" in text and "kaboom" in text
    assert "should not appear" not in text
    assert called == []                          # no reload after a failure
    assert win.btn_analyze.isEnabled()


def test_log_lines_arrive_in_order_before_the_done_message(win, qapp):
    win._load_results = lambda: None

    def job(emit):
        for i in range(5):
            emit(f"line {i}")

    win._run_bg(job, "ALL DONE")
    assert _pump(qapp, lambda: win._thread is None)

    lines = [ln for ln in win.log.toPlainText().splitlines() if ln.strip()]
    assert lines[-1] == "ALL DONE", (
        "the done message must come last; if it appears earlier the handler "
        "is running on the wrong thread"
    )
    assert lines[:5] == [f"line {i}" for i in range(5)]


def test_thread_is_released_so_a_second_run_can_start(win, qapp):
    win._load_results = lambda: None

    win._run_bg(lambda emit: emit("first"), "first done")
    assert _pump(qapp, lambda: win._thread is None)

    win._run_bg(lambda emit: emit("second"), "second done")
    assert _pump(qapp, lambda: win._thread is None)

    assert "second done" in win.log.toPlainText()
