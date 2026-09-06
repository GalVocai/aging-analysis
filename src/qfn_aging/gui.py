"""QFN-AGING desktop GUI (PySide6).

Thin frontend over the headless core -- it contains no analysis logic. Its
job is to let the user (a) point at the data, (b) run the analysis, and
(c) decide which transistors take part in the group averages.

Run with::

    python -m qfn_aging.gui

Two tabs:

* **Data** -- pick the data root, discover runs, check them against the
  schedule, and run sensorlab's analysis (on a worker thread so the window
  stays responsive).
* **Devices** -- a tab per measurement day, and within each day a sub-tab
  per condition holding that condition's devices with their transistors
  nested underneath. **Exclusions are per day**: the same channel is ticked
  independently at baseline and at 3D, because a transistor can read fine
  early and misbehave later. Ticking a device toggles all four of its
  channels; a device shows a partially-checked box when only some are in.
  Saved to ``~/.qfn_aging/selection.json`` and applied to every subsequent
  aggregation.
* **Comparisons** -- fix two of (temperature, RH, category) and overlay the
  third, as group averages across days.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressDialog, QPushButton, QSpinBox, QStatusBar,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .allocation import DEFAULT, Allocation
from .comparison import (ComparisonSpec, add_jg_vth_drift, available_options,
                         build_curves, metrics_for_sheet, ylabel_for_metric)
from .comparison_plot import plot_comparison
from .discovery import available_timepoints, discover, verify
from .grouping import add_group_keys, aggregate_groups
from .masters import load_masters
from .runner import analyze_all, has_output
from .schedule import sort_timepoints
from .selection import (DEFAULT_SELECTION_PATH, Selection, apply_selection,
                        load_selection, load_settings, save_settings)
from .theme import MUTED, STYLESHEET

#: (esn, tnumber) for a row; devices carry tnumber=None.
_KEY = Qt.UserRole + 1

TREE_COLUMNS = ["Device / Transistor", "Category", "Type", "H2", "Metrics"]
_GREY = QBrush(QColor(MUTED))
_NORMAL = QBrush(QColor("#1f2933"))


class _Cancelled(Exception):
    """Raised out of the progress callback when the user hits Cancel."""


# --------------------------------------------------------------------------
# background work
# --------------------------------------------------------------------------

class _Worker(QObject):
    """Runs a callable off the UI thread, streaming log lines back."""

    line = Signal(str)
    done = Signal(bool, str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn(self.line.emit)
        except Exception:
            self.done.emit(False, traceback.format_exc())
        else:
            self.done.emit(True, "")


# --------------------------------------------------------------------------
# one condition's device tree
# --------------------------------------------------------------------------

class ConditionTree(QTreeWidget):
    """Devices of a single condition, transistors nested underneath.

    Check state is a view of :class:`~qfn_aging.selection.Selection`; the
    tree never holds its own copy, so a selection reloaded from disk shows
    up simply by calling :meth:`refresh`.
    """

    changed = Signal()

    def __init__(self, df: pd.DataFrame, selection: Selection, timepoint: str,
                 targets=None):
        super().__init__()
        self._sel = selection
        self._df = df
        self._tp = timepoint          # exclusions apply to THIS day only...
        # ...unless a targets hook says otherwise (the "sync across days"
        # button makes a tick apply to every measured day). Given this tree's
        # day it returns the list of days the change should be written to.
        self._targets = targets or (lambda tp: [tp])
        self._syncing = False

        self.setColumnCount(len(TREE_COLUMNS))
        self.setHeaderLabels(TREE_COLUMNS)
        self.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setIndentation(18)
        self.setRootIsDecorated(True)
        self._build()
        self.itemChanged.connect(self._on_item_changed)

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        self._syncing = True
        self.clear()
        for esn, rows in self._df.groupby("esn", sort=True):
            first = rows.iloc[0]
            dev = QTreeWidgetItem(self, [
                f"ESN {esn}", str(first["category"]), str(first["type"] or ""),
                "yes" if first["h2"] else "no", "",
            ])
            dev.setData(0, _KEY, (str(esn), None))
            dev.setFlags(dev.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            bold = QFont(); bold.setBold(True)
            dev.setFont(0, bold)

            for _, r in rows.sort_values("tnumber").iterrows():
                ch = QTreeWidgetItem(dev, [
                    f"T{int(r['tnumber'])}", "", "", "", str(int(r["metrics"])),
                ])
                ch.setData(0, _KEY, (str(esn), int(r["tnumber"])))
                ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
            dev.setExpanded(False)
        self._syncing = False
        self.refresh()
        for i in range(len(TREE_COLUMNS)):
            self.resizeColumnToContents(i)

    # -- selection <-> view ---------------------------------------------

    def refresh(self) -> None:
        """Re-read the Selection and restyle every row."""
        self._syncing = True
        for i in range(self.topLevelItemCount()):
            dev = self.topLevelItem(i)
            n_in = 0
            for j in range(dev.childCount()):
                ch = dev.child(j)
                esn, tnum = ch.data(0, _KEY)
                included = self._sel.is_included(self._tp, esn, tnum)
                ch.setCheckState(0, Qt.Checked if included else Qt.Unchecked)
                self._style(ch, included)
                n_in += included

            total = dev.childCount()
            state = (Qt.Checked if n_in == total else
                     Qt.Unchecked if n_in == 0 else Qt.PartiallyChecked)
            dev.setCheckState(0, state)
            self._style(dev, n_in > 0)
            dev.setText(4, f"{n_in}/{total} in")
        self._syncing = False

    @staticmethod
    def _style(item: QTreeWidgetItem, included: bool) -> None:
        brush = _NORMAL if included else _GREY
        for c in range(item.columnCount()):
            item.setForeground(c, brush)

    # -- write helpers: honour the "sync across days" target list ---------

    def _set_channel(self, esn: str, tnum: int, included: bool) -> None:
        for tp in self._targets(self._tp):
            if included:
                self._sel.include(tp, esn, tnum)
            else:
                self._sel.exclude(tp, esn, tnum, reason=f"excluded in GUI ({tp})")

    def _set_device(self, esn: str, tnums: list[int], included: bool) -> None:
        for tp in self._targets(self._tp):
            if included:
                self._sel.include_device(tp, esn, tnums)
            else:
                self._sel.exclude_device(tp, esn, tnums,
                                         reason=f"excluded in GUI ({tp})")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        esn, tnum = item.data(0, _KEY)
        checked = item.checkState(0) == Qt.Checked

        if tnum is None:                       # a device row: apply to all its channels
            if item.checkState(0) == Qt.PartiallyChecked:
                return
            tnums = [item.child(j).data(0, _KEY)[1] for j in range(item.childCount())]
            self._set_device(esn, tnums, checked)
        else:
            self._set_channel(esn, tnum, checked)

        self.refresh()
        self.changed.emit()

    # -- bulk ------------------------------------------------------------

    def set_all(self, included: bool) -> None:
        for _, r in self._df.iterrows():
            self._set_channel(r["esn"], int(r["tnumber"]), included)
        self.refresh()
        self.changed.emit()

    def set_selected(self, included: bool) -> None:
        """Apply to whatever rows are highlighted (devices expand to channels)."""
        keys: list[tuple[str, int]] = []
        for item in self.selectedItems():
            esn, tnum = item.data(0, _KEY)
            if tnum is None:
                keys += [(esn, item.child(j).data(0, _KEY)[1])
                         for j in range(item.childCount())]
            else:
                keys.append((esn, tnum))
        if not keys:
            return
        for esn, tnum in keys:
            self._set_channel(esn, tnum, included)
        self.refresh()
        self.changed.emit()

    def counts(self) -> tuple[int, int]:
        """(included, total) transistors in this condition."""
        total = len(self._df)
        inc = sum(self._sel.is_included(self._tp, r["esn"], int(r["tnumber"]))
                  for _, r in self._df.iterrows())
        return inc, total


# --------------------------------------------------------------------------
# main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    #: emitted from the worker thread; queued delivery marshals it to the UI
    _analysis_progress = Signal(int, int, str)

    def __init__(self, alloc: Allocation = DEFAULT,
                 selection_path: Path = DEFAULT_SELECTION_PATH,
                 settings_path: Path | None = None):
        super().__init__()
        self.alloc = alloc
        #: where user state lives; injectable so tests never touch the real file
        self.selection_path = Path(selection_path)
        self.settings_path = settings_path
        self.selection = load_selection(self.selection_path)
        self.trees: dict[tuple[str, str], ConditionTree] = {}
        self.cond_tab_widgets: dict[str, QTabWidget] = {}
        self._thread: QThread | None = None
        self.progress: QProgressDialog | None = None
        self._cancel_requested = False
        self._worker: _Worker | None = None
        self._done_msg = ""
        self._tidy: pd.DataFrame | None = None
        self._agg: pd.DataFrame = pd.DataFrame()
        self._filling = False
        #: when True, ticking a channel applies to every measured day, not
        #: just the active one (the "Sync across days" button)
        self.sync_days = False
        #: comparison-tab axis state. Redraws reuse one axes and keep the
        #: user's zoom/limits instead of snapping back to autoscale on every
        #: selection or picker change. `_plot_key` tracks the plotted metric
        #: so the view is only kept while the axes still mean the same thing.
        self._plot_key = None
        self._lock_axes = False           # hard-freeze limits across everything
        self._autoscale_once = False      # one-shot: "Reset view" forces a fit
        self._analysis_progress.connect(self._on_analysis_progress)

        self.setWindowTitle("QFN-AGING")
        self.resize(1180, 760)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "data" / "qfn.ico")))
        self.setStatusBar(QStatusBar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_devices_tab(), "Devices")
        self.tabs.addTab(self._build_compare_tab(), "Comparisons")
        self.setCentralWidget(self.tabs)

        # reopen where the user left off
        self.settings = (load_settings(self.settings_path) if self.settings_path
                         else load_settings())
        last_root = self.settings.get("data_root", "")
        if last_root:
            self.root_edit.setText(last_root)
            self.chk_plots.setChecked(self.settings.get("make_plots", True))
            self.chk_skip.setChecked(self.settings.get("skip_existing", True))
            self.spn_workers.setValue(self.settings.get("workers", self.spn_workers.value()))
        self.root_edit.editingFinished.connect(self._remember_settings)
        self._refresh_status()

    def _remember_settings(self) -> None:
        self.settings.update({
            "data_root": self.root_edit.text().strip(),
            "make_plots": self.chk_plots.isChecked(),
            "skip_existing": self.chk_skip.isChecked(),
            "workers": self.spn_workers.value(),
        })
        if self.settings_path:
            save_settings(self.settings, self.settings_path)
        else:
            save_settings(self.settings)

    def closeEvent(self, event):          # noqa: N802  (Qt naming)
        """Persist selection and preferences on the way out, always."""
        self.selection.save(self.selection_path)
        self._remember_settings()
        super().closeEvent(event)

    # -- tab 1: data -----------------------------------------------------

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        head = QLabel("Data source")
        head.setProperty("heading", True)
        v.addWidget(head)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Data root (the folder containing Baseline/, 3D/, ...)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(QLabel("Data root:"))
        row.addWidget(self.root_edit, 1)
        row.addWidget(browse)
        v.addLayout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_discover = QPushButton("Discover runs")
        self.btn_discover.clicked.connect(self._discover)
        self.btn_analyze = QPushButton("Run analysis")
        self.btn_analyze.setProperty("primary", True)
        self.btn_analyze.clicked.connect(self._analyze)
        self.chk_plots = QCheckBox("Also generate plots")
        self.chk_plots.setChecked(True)
        self.chk_skip = QCheckBox("Skip already-analyzed runs")
        self.chk_skip.setChecked(True)
        self.chk_skip.setToolTip(
            "Only analyze runs that have no master workbooks yet — so a new "
            "day can be processed without redoing the earlier ones."
        )
        self.btn_load = QPushButton("Load results")
        self.btn_load.clicked.connect(self._load_results)

        self.cmb_timepoint = QComboBox()
        self.cmb_timepoint.setToolTip("Which day(s) to analyze")
        self.cmb_timepoint.addItem("All timepoints", None)

        self.spn_workers = QSpinBox()
        self.spn_workers.setRange(1, max(1, (os.cpu_count() or 4)))
        self.spn_workers.setValue(min(6, max(1, (os.cpu_count() or 4))))
        self.spn_workers.setToolTip(
            "Run this many runs at once. Plotting is CPU-bound and the runs "
            "are independent, so more processes means less waiting.\n"
            "With more than 1, the log arrives per run instead of live."
        )

        row2.addWidget(self.btn_discover)
        row2.addWidget(QLabel("Timepoint:"))
        row2.addWidget(self.cmb_timepoint)
        row2.addWidget(self.btn_analyze)
        row2.addWidget(self.chk_plots)
        row2.addWidget(self.chk_skip)
        row2.addWidget(QLabel("Parallel:"))
        row2.addWidget(self.spn_workers)
        row2.addWidget(self.btn_load)
        row2.addStretch(1)
        v.addLayout(row2)

        log_head = QLabel("Activity")
        log_head.setProperty("heading", True)
        v.addWidget(log_head)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Discover runs to see what is on disk…")
        v.addWidget(self.log, 1)
        return w

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select the aging data root",
                                             self.root_edit.text().strip())
        if d:
            self.root_edit.setText(d)
            self._remember_settings()

    def _root(self) -> Path | None:
        t = self.root_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "No data root", "Pick the data root folder first.")
            return None
        return Path(t)

    def _emit(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _fill_timepoints(self, root: Path) -> None:
        """Offer only the days that actually exist on disk."""
        current = self.cmb_timepoint.currentData()
        self.cmb_timepoint.blockSignals(True)
        self.cmb_timepoint.clear()
        self.cmb_timepoint.addItem("All timepoints", None)
        for tp in available_timepoints(root):
            self.cmb_timepoint.addItem(tp, tp)
        if current:
            i = self.cmb_timepoint.findData(current)
            if i >= 0:
                self.cmb_timepoint.setCurrentIndex(i)
        self.cmb_timepoint.blockSignals(False)

    def _discover(self) -> None:
        root = self._root()
        if root is None:
            return
        self.log.clear()
        self._fill_timepoints(root)
        runs, warnings = discover(root, self.alloc)
        for w in warnings:
            self._emit(f"WARNING: {w}")
        if not runs:
            self._emit("No runs found.")
            return
        out_root = root / "analysis"
        for rf in runs:
            state = "analyzed" if has_output(rf, out_root, self.alloc) else "NOT analyzed"
            self._emit(f"{rf.timepoint:9} {rf.run_key:14} -> {rf.out_group:9} "
                       f"majors={','.join(rf.majors):31} [{state}]")
        problems = verify(runs, self.alloc)
        self._emit("")
        self._emit("Schedule check: " + ("OK" if not problems else "\n  ".join(problems)))

        if (root / "analysis").is_dir():
            self._emit("")
            self._emit("Existing analysis found — loading the device list…")
            self._load_results()
        else:
            self._emit("")
            self._emit("No analysis yet — click 'Run analysis' to generate it.")

    def _analyze(self) -> None:
        root = self._root()
        if root is None or self._busy():
            return
        out = root / "analysis"
        make_plots = self.chk_plots.isChecked()
        skip_existing = self.chk_skip.isChecked()
        workers = self.spn_workers.value()
        tp = self.cmb_timepoint.currentData()
        timepoints = [tp] if tp else None

        self.log.clear()
        self._emit(f"Analyzing {tp or 'all timepoints'} into {out} …")
        if workers > 1:
            self._emit(f"Using {workers} parallel processes "
                       f"(log arrives per run, not live).")
        if skip_existing:
            self._emit("Runs that already have masters will be skipped.")

        self.progress = QProgressDialog("Starting…", "Cancel", 0, 1, self)
        self.progress.setWindowTitle("Running analysis")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setValue(0)

        def job(emit):
            def on_progress(done, total, label):
                # marshalled onto the UI thread by the queued signal below
                self._analysis_progress.emit(done, total, label)
                if self._cancel_requested:
                    raise _Cancelled()

            analyze_all(root, out, self.alloc, make_plots=make_plots,
                        timepoints=timepoints, skip_existing=skip_existing,
                        workers=workers, progress=on_progress, log=emit)

        self._cancel_requested = False
        self.progress.canceled.connect(self._request_cancel)
        self._run_bg(job, "Analysis finished.")

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self._emit("Cancel requested — stopping after the current run…")

    def _on_analysis_progress(self, done: int, total: int, label: str) -> None:
        if self.progress is None:
            return
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        if label == "done":
            self.progress.setLabelText("Finishing…")
        else:
            self.progress.setLabelText(f"Run {done + 1} of {total}\n{label}")

    def _busy(self) -> bool:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return True
        return False

    def _run_bg(self, fn, done_msg: str) -> None:
        self._set_enabled(False)
        self._done_msg = done_msg
        self._thread = QThread()
        worker = _Worker(fn)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.line.connect(self._emit)

        # Connect to a BOUND METHOD of this window, never a local closure.
        #
        # Qt picks the connection type from the receiver's thread affinity.
        # A plain function has no receiver QObject, so Qt runs it directly on
        # the *emitting* thread -- i.e. the worker. The handler then called
        # QThread.wait() on its own thread, which waits forever: the window
        # froze after the analysis had already finished successfully.
        # MainWindow lives on the UI thread, so this resolves to a queued
        # connection and the handler runs where it belongs.
        worker.done.connect(self._on_worker_done, Qt.QueuedConnection)
        self._worker = worker            # keep a reference alive
        self._thread.start()

    def _on_worker_done(self, ok: bool, err: str) -> None:
        """Runs on the UI thread (queued connection -- see :meth:`_run_bg`)."""
        if self.progress is not None:
            self.progress.close()
            self.progress = None

        if ok:
            self._emit(self._done_msg)
        elif "_Cancelled" in err:
            self._emit("Analysis cancelled. Output written so far is kept.")
        else:
            self._emit(f"FAILED:\n{err}")

        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(5000):
                self._emit("WARNING: worker thread did not stop cleanly.")
            self._thread = None
        self._worker = None
        self._set_enabled(True)

        if ok:
            self._load_results()

    def _set_enabled(self, on: bool) -> None:
        for b in (self.btn_discover, self.btn_analyze, self.btn_load):
            b.setEnabled(on)

    # -- tab 2: devices, one sub-tab per condition -----------------------

    def _build_devices_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        self.dev_hint = QLabel(
            "Exclusions apply to the selected day only — a channel can "
            "be in at baseline and out at 3D.")
        self.dev_hint.setProperty("hint", True)
        v.addWidget(self.dev_hint)

        # day tabs on the outside, condition tabs within each day
        self.day_tabs = QTabWidget()
        self.day_tabs.setTabPosition(QTabWidget.North)
        self.day_tabs.currentChanged.connect(self._refresh_status)
        v.addWidget(self.day_tabs, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        for text, slot in (("Include selected", lambda: self._bulk_sel(True)),
                           ("Exclude selected", lambda: self._bulk_sel(False)),
                           ("Include all in tab", lambda: self._bulk_tab(True)),
                           ("Exclude all in tab", lambda: self._bulk_tab(False)),
                           ("Expand all", lambda: self._expand(True)),
                           ("Collapse all", lambda: self._expand(False)),
                           ("Save selection", self._save_selection),
                           ("Reload selection", self._reload_selection)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)

        # universal-exclusion toggle: when on, a tick applies to every day
        self.btn_sync = QPushButton("Sync across days")
        self.btn_sync.setCheckable(True)
        self.btn_sync.setToolTip(
            "When on, checking or unchecking a channel applies to EVERY "
            "measured day at once, so a bad transistor is dropped everywhere.")
        self.btn_sync.toggled.connect(self._toggle_sync)
        row.addWidget(self.btn_sync)

        row.addStretch(1)
        v.addLayout(row)

        self.placeholder = QLabel("Load results on the Data tab to list devices.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        v.addWidget(self.placeholder)
        return w

    # -- tab 3: comparisons ----------------------------------------------

    def _build_compare_tab(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        head = QLabel("Group comparison")
        head.setProperty("heading", True)
        v.addWidget(head)

        hint = QLabel("Fix two axes, overlay the third. Curves are group "
                      "averages (mean ± AVDEV) and honour your Devices selection.")
        hint.setProperty("hint", True)
        v.addWidget(hint)

        pick = QHBoxLayout()
        pick.setSpacing(8)
        self.cmb_vary = QComboBox()
        for label, key in (("Temperature", "temp"), ("Humidity", "rh"),
                           ("Category", "category")):
            self.cmb_vary.addItem(f"Compare {label}", key)
        self.cmb_sheet = QComboBox()
        self.cmb_metric = QComboBox()
        self.cmb_series = QComboBox()
        for lbl, wid in (("Vary:", self.cmb_vary), ("Sheet:", self.cmb_sheet),
                         ("Metric:", self.cmb_metric), ("Series:", self.cmb_series)):
            pick.addWidget(QLabel(lbl))
            pick.addWidget(wid)
        pick.addStretch(1)
        v.addLayout(pick)

        fixed = QHBoxLayout()
        fixed.setSpacing(8)
        self.lbl_temp = QLabel("Temperature:")
        self.cmb_fix_temp = QComboBox()
        self.lbl_rh = QLabel("Humidity:")
        self.cmb_fix_rh = QComboBox()
        self.lbl_cat = QLabel("Category:")
        self.cmb_fix_cat = QComboBox()
        for lbl, wid in ((self.lbl_temp, self.cmb_fix_temp), (self.lbl_rh, self.cmb_fix_rh),
                         (self.lbl_cat, self.cmb_fix_cat)):
            fixed.addWidget(lbl)
            fixed.addWidget(wid)
        # keep the user's zoom/limits across redraws
        self.btn_lock_axes = QPushButton("Lock axes")
        self.btn_lock_axes.setCheckable(True)
        self.btn_lock_axes.setToolTip(
            "Freeze the current x/y limits so redraws (a selection change or a "
            "picker change) keep your zoom instead of autoscaling.")
        self.btn_lock_axes.toggled.connect(self._toggle_lock_axes)
        self.btn_reset_view = QPushButton("Reset view")
        self.btn_reset_view.setToolTip("Autoscale the axes to fit the current data.")
        self.btn_reset_view.clicked.connect(self._reset_view)

        self.btn_save_plot = QPushButton("Save figure…")
        self.btn_save_plot.clicked.connect(self._save_comparison)
        fixed.addStretch(1)
        fixed.addWidget(self.btn_lock_axes)
        fixed.addWidget(self.btn_reset_view)
        fixed.addWidget(self.btn_save_plot)
        v.addLayout(fixed)

        self.figure = plot_comparison(
            pd.DataFrame(columns=["curve", "timepoint", "day", "mean", "avdev", "n"]),
            ComparisonSpec(vary="temp", metric="—", sheet="—",
                           fixed={"rh": "—", "category": "—"}))
        # one persistent axes: redraws clear and reuse it (not the whole
        # figure), which keeps the nav-toolbar and any restored limits valid
        self._ax = self.figure.axes[0]
        self.canvas = FigureCanvasQTAgg(self.figure)
        v.addWidget(NavigationToolbar2QT(self.canvas, w))
        v.addWidget(self.canvas, 1)

        for combo in (self.cmb_vary, self.cmb_sheet, self.cmb_metric, self.cmb_series,
                      self.cmb_fix_temp, self.cmb_fix_rh, self.cmb_fix_cat):
            combo.currentIndexChanged.connect(self._refresh_comparison)
        self.cmb_sheet.currentIndexChanged.connect(self._sheet_changed)
        self.cmb_vary.currentIndexChanged.connect(self._vary_changed)
        return w

    def _rebuild_aggregate(self) -> None:
        """Recompute group averages from the loaded tidy table + selection."""
        if getattr(self, "_tidy", None) is None or self._tidy.empty:
            self._agg = pd.DataFrame()
            return
        self._agg = aggregate_groups(
            add_group_keys(apply_selection(self._tidy, self.selection), self.alloc))

    def _populate_comparison_options(self) -> None:
        opts = available_options(getattr(self, "_agg", pd.DataFrame()))
        self._filling = True
        for combo, values in ((self.cmb_sheet, opts["sheet"]),
                              (self.cmb_fix_temp, opts["temp"]),
                              (self.cmb_fix_rh, opts["rh"]),
                              (self.cmb_fix_cat, opts["category"])):
            keep = combo.currentData()
            combo.clear()
            for v in values:
                combo.addItem(str(v), v)
            i = combo.findData(keep)
            combo.setCurrentIndex(max(i, 0))
        self._filling = False
        self._sheet_changed()
        self._vary_changed()

    def _sheet_changed(self) -> None:
        if getattr(self, "_filling", False):
            return
        sheet = self.cmb_sheet.currentData()
        metrics = metrics_for_sheet(getattr(self, "_agg", pd.DataFrame()), sheet or "")
        keep = self.cmb_metric.currentData()
        self._filling = True
        self.cmb_metric.clear()
        for m in metrics:
            self.cmb_metric.addItem(m, m)
        i = self.cmb_metric.findData(keep)
        self.cmb_metric.setCurrentIndex(max(i, 0))

        agg = getattr(self, "_agg", pd.DataFrame())
        series = []
        if not agg.empty and "series" in agg.columns:
            series = sorted(agg.loc[agg["sheet"] == sheet, "series"].fillna("").unique())
        self.cmb_series.clear()
        for s in series:
            self.cmb_series.addItem(s or "(none)", s)
        self._filling = False
        self._refresh_comparison()

    def _vary_changed(self) -> None:
        """Grey out the axis being varied -- it cannot also be fixed."""
        vary = self.cmb_vary.currentData()
        for axis, lbl, combo in (("temp", self.lbl_temp, self.cmb_fix_temp),
                                 ("rh", self.lbl_rh, self.cmb_fix_rh),
                                 ("category", self.lbl_cat, self.cmb_fix_cat)):
            on = axis != vary
            lbl.setVisible(on)
            combo.setVisible(on)
        self._refresh_comparison()

    def current_spec(self) -> ComparisonSpec | None:
        vary = self.cmb_vary.currentData()
        metric = self.cmb_metric.currentData()
        sheet = self.cmb_sheet.currentData()
        if not (vary and metric and sheet):
            return None
        fixed: dict = {}
        if vary != "temp" and self.cmb_fix_temp.currentData() is not None:
            fixed["temp"] = self.cmb_fix_temp.currentData()
        if vary != "rh" and self.cmb_fix_rh.currentData() is not None:
            fixed["rh"] = self.cmb_fix_rh.currentData()
        if vary != "category" and self.cmb_fix_cat.currentData() is not None:
            fixed["category"] = self.cmb_fix_cat.currentData()
        return ComparisonSpec(vary=vary, metric=metric, sheet=sheet,
                              series=self.cmb_series.currentData() or "", fixed=fixed)

    def _refresh_comparison(self) -> None:
        if getattr(self, "_filling", False) or not hasattr(self, "canvas"):
            return
        spec = self.current_spec()
        agg = getattr(self, "_agg", pd.DataFrame())
        empty = pd.DataFrame(columns=["curve", "timepoint", "day", "mean", "avdev", "n"])
        curves = empty if (spec is None or agg.empty) else build_curves(agg, spec)

        # Keep the view the user set. The axes are reused (cleared, not
        # recreated) so a redraw doesn't snap back to autoscale under them.
        # Limits are restored while the plotted metric is unchanged (so a
        # selection tweak keeps the zoom), or always while "Lock axes" is on.
        # "Reset view" forces a single autoscale via `_autoscale_once`.
        key = (spec.sheet, spec.metric, spec.series) if spec else None
        ax = self._ax
        prev = (ax.get_xlim(), ax.get_ylim())
        restore = (not self._autoscale_once
                   and (self._lock_axes or key == self._plot_key))

        ax.clear()
        if spec is not None:
            plot_comparison(curves, spec, ylabel=ylabel_for_metric(spec.metric), ax=ax)
        if restore:
            ax.set_xlim(prev[0])
            ax.set_ylim(prev[1])

        self._plot_key = key
        self._autoscale_once = False
        self.canvas.draw_idle()

    def _toggle_lock_axes(self, on: bool) -> None:
        """Freeze/unfreeze the comparison axis limits across redraws."""
        self._lock_axes = on
        self.statusBar().showMessage(
            "Axis limits locked — redraws keep your zoom" if on
            else "Axis limits unlocked — redraws autoscale to the data", 4000)

    def _reset_view(self) -> None:
        """Autoscale the comparison axes to the current data, once."""
        self._autoscale_once = True
        self._refresh_comparison()

    def _save_comparison(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        default = f"{spec.sheet}_{spec.metric}_by_{spec.vary}.png".replace("/", "-")
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", default,
                                              "PNG image (*.png)")
        if path:
            self.figure.savefig(path, dpi=150)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def _load_results(self) -> None:
        root = self._root()
        if root is None:
            return
        analysis = root / "analysis"
        if not analysis.is_dir():
            QMessageBox.warning(self, "No analysis", f"{analysis} does not exist yet.\n"
                                                     "Run the analysis first.")
            return

        # Reading ~130 workbooks takes several seconds. Without visible
        # progress the window just freezes and looks like the click did
        # nothing, so show a real bar with a per-file label.
        dlg = QProgressDialog("Loading master workbooks…", "Cancel", 0, 100, self)
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setValue(0)
        QApplication.processEvents()

        cancelled = False

        def on_progress(done: int, total: int, label: str) -> None:
            nonlocal cancelled
            if dlg.wasCanceled():
                cancelled = True
                raise _Cancelled()
            dlg.setMaximum(max(total, 1))
            dlg.setValue(done)
            dlg.setLabelText(f"Reading master {done + 1} of {total}\n{label}")
            QApplication.processEvents()

        try:
            tidy = load_masters(analysis, progress=on_progress)
        except _Cancelled:
            dlg.close()
            self._emit("Loading cancelled.")
            self.statusBar().showMessage("Loading cancelled", 4000)
            return
        finally:
            if not cancelled:
                dlg.close()

        if tidy.empty:
            QMessageBox.warning(self, "Empty", "No master workbooks found.")
            return

        self.statusBar().showMessage("Building the device list…")
        QApplication.processEvents()
        # keep the raw tidy table (+ derived JG_n03 ΔVth-in-mV metrics): the
        # comparison tab re-aggregates from it every time the selection
        # changes, without re-reading any workbook
        self._tidy = add_jg_vth_drift(tidy)
        # device list is built from the measured metrics only, so the
        # per-transistor metric count isn't inflated by the derived drift rows
        keyed = add_group_keys(tidy, self.alloc)
        per = (keyed.groupby(["timepoint", "esn", "tnumber", "condition", "category"])
                    .size().reset_index(name="metrics"))
        per["type"] = per["esn"].map(lambda e: self.alloc.device(e).type)
        per["h2"] = per["esn"].map(lambda e: self.alloc.device(e).h2)
        self.populate(per)

        self._rebuild_aggregate()
        self._populate_comparison_options()
        self._emit(f"Loaded {len(per)} transistors from {analysis}")
        self.tabs.setCurrentIndex(1)          # jump straight to the device list

    def populate(self, per: pd.DataFrame) -> None:
        """Build day tabs, each holding that day's condition sub-tabs.

        ``per`` must carry a ``timepoint`` column; exclusions are per day, so
        the same transistor appears once per measured day and is ticked
        independently in each.
        """
        self.day_tabs.clear()
        self.trees.clear()
        self.cond_tab_widgets.clear()
        self.placeholder.setVisible(False)

        days = sort_timepoints(per["timepoint"].unique())
        for day in days:
            day_rows = per[per["timepoint"] == day]
            cond_tabs = QTabWidget()
            cond_tabs.setTabPosition(QTabWidget.North)
            cond_tabs.currentChanged.connect(self._refresh_status)

            for condition in self.alloc.conditions():
                sub = day_rows[day_rows["condition"] == condition]
                if sub.empty:
                    continue
                tree = ConditionTree(sub.copy(), self.selection, day,
                                     targets=self._sync_targets)
                tree.changed.connect(self._on_selection_changed)
                self.trees[(day, condition)] = tree
                cond_tabs.addTab(tree, condition)

            self.cond_tab_widgets[day] = cond_tabs
            self.day_tabs.addTab(cond_tabs, day)

        self._refresh_tab_titles()
        self._refresh_status()

    def _measured_days(self) -> list[str]:
        """Days that actually have trees, in study order."""
        return sort_timepoints({d for d, _ in self.trees})

    def _sync_targets(self, day: str) -> list[str]:
        """Which days a tick on ``day`` writes to: just it, or all of them."""
        return self._measured_days() if self.sync_days else [day]

    def current_day(self) -> str | None:
        i = self.day_tabs.currentIndex()
        return self.day_tabs.tabText(i).split("  (")[0] if i >= 0 else None

    def _current_tree(self) -> ConditionTree | None:
        holder = self.day_tabs.currentWidget()
        if not isinstance(holder, QTabWidget):
            return None
        w = holder.currentWidget()
        return w if isinstance(w, ConditionTree) else None

    def _bulk_sel(self, included: bool) -> None:
        t = self._current_tree()
        if t is not None:
            t.set_selected(included)

    def _bulk_tab(self, included: bool) -> None:
        t = self._current_tree()
        if t is not None:
            t.set_all(included)

    def _expand(self, on: bool) -> None:
        t = self._current_tree()
        if t is None:
            return
        t.expandAll() if on else t.collapseAll()

    def _toggle_sync(self, on: bool) -> None:
        """Turn universal (all-days) exclusion on/off. Non-destructive: it
        only changes where FUTURE ticks are written — existing per-day
        selections are left exactly as they are until you touch them."""
        self.sync_days = on
        if on:
            self.dev_hint.setText(
                "SYNC ON — checking/unchecking a channel now applies to ALL "
                f"days ({', '.join(self._measured_days())}) at once.")
            self.statusBar().showMessage(
                "Sync across days is ON — ticks now apply to every day", 5000)
        else:
            self.dev_hint.setText(
                "Exclusions apply to the selected day only — a channel can "
                "be in at baseline and out at 3D.")
            self.statusBar().showMessage(
                "Sync across days is OFF — exclusions are per day again", 4000)

    def _on_selection_changed(self) -> None:
        """Every tick is persisted immediately.

        Requiring an explicit Save is how exclusions get lost: you untick a
        transistor, close the window, and the decision is gone. Auto-saving
        means the app always reopens with the choices you last made.
        """
        self.selection.save(self.selection_path)
        # a synced tick changes other days too; restyle every tree so the
        # boxes on the inactive day tabs match what was just written
        if self.sync_days:
            for tree in self.trees.values():
                tree.refresh()
        self._rebuild_aggregate()
        self._refresh_comparison()
        self._refresh_status()
        self.statusBar().showMessage(
            self.statusBar().currentMessage() + "  ·  saved", 3000)

    def _save_selection(self) -> None:
        self.selection.save(self.selection_path)
        self.statusBar().showMessage(f"Saved to {self.selection_path}", 5000)

    def _reload_selection(self) -> None:
        self.selection = load_selection(self.selection_path)
        for tree in self.trees.values():
            tree._sel = self.selection
            tree.refresh()
        self._rebuild_aggregate()
        self._refresh_comparison()
        self._refresh_status()

    def _refresh_tab_titles(self) -> None:
        for day, cond_tabs in self.cond_tab_widgets.items():
            for i in range(cond_tabs.count()):
                tree = cond_tabs.widget(i)
                if isinstance(tree, ConditionTree):
                    inc, total = tree.counts()
                    base = cond_tabs.tabText(i).split("  (")[0]
                    cond_tabs.setTabText(i, f"{base}  ({inc}/{total})")

        for i in range(self.day_tabs.count()):
            day = self.day_tabs.tabText(i).split("  (")[0]
            trees = [t for (d, _), t in self.trees.items() if d == day]
            inc = sum(t.counts()[0] for t in trees)
            total = sum(t.counts()[1] for t in trees)
            self.day_tabs.setTabText(i, f"{day}  ({inc}/{total})")

    def _refresh_status(self) -> None:
        self._refresh_tab_titles()
        day = self.current_day()
        days = sorted({d for d, _ in self.trees})
        if day:
            trees = [t for (d, _), t in self.trees.items() if d == day]
            inc = sum(t.counts()[0] for t in trees)
            total = sum(t.counts()[1] for t in trees)
            self.statusBar().showMessage(
                f"{day}: {inc}/{total} transistors in · "
                f"{self.selection.count_for(day)} excluded on this day · "
                f"{len(self.selection)} total across {len(days)} day(s) · auto-saved"
            )
        else:
            self.statusBar().showMessage("No results loaded")


def _hide_console() -> None:
    """Hide the console window this process owns, if any.

    The app is launched with ``python.exe`` rather than ``pythonw.exe`` on
    purpose: multiprocessing spawns its workers with ``sys.executable``, and
    pythonw children have no usable stdio and die on bootstrap, taking the
    whole pool down. Keeping a real console and hiding it gives working
    workers with no black window on screen.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)      # SW_HIDE
    except Exception:
        pass          # cosmetic only -- never block startup over it


def main() -> int:
    _hide_console()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
