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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QFileDialog, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressDialog,
                               QDoubleSpinBox, QPushButton, QSpinBox, QStatusBar, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .allocation import DEFAULT, Allocation
from .comparison import (AXIS_LABELS, ComparisonSpec, add_jg_vth_drift,
                         available_options, build_curves, metrics_for_sheet,
                         ylabel_for_metric)
from .comparison_plot import (DEFAULT_XLABEL, DRAW_MODES, PALETTES, PlotOptions,
                              auto_ylabel, figure_dpi, plot_comparison)
from .device_metrics import (CATEGORY_ORDER, DEVICE_METRICS, build_device_metric_table,
                             category_summary, format_metric)
from .discovery import available_timepoints, discover, verify
from .grouping import add_group_keys, aggregate_groups
from .masters import load_masters_cached
from .runner import analyze_all, has_output
from .schedule import sort_timepoints
from .selection import (ALL_DAYS, DEFAULT_SELECTION_PATH, Selection, apply_selection,
                         load_selection, load_settings, save_settings)
from .theme import MUTED, STYLESHEET

#: (esn, tnumber) for a row; devices carry tnumber=None.
_KEY = Qt.UserRole + 1
_VALUE = Qt.UserRole + 2

_BASE_TREE_COLUMNS = ["Device / Transistor", "Category", "Type", "H2", "Metrics"]
TREE_COLUMNS = _BASE_TREE_COLUMNS + [spec.label for spec in DEVICE_METRICS]
_METRIC_START = len(_BASE_TREE_COLUMNS)
_GREY = QBrush(QColor(MUTED))
_NORMAL = QBrush(QColor("#1f2933"))

_CATEGORY_LABELS = {
    "ParC_only": "ParC only",
    "Pd_only": "Pd only",
    "PdParC_H2": "PdParC H2",
    "PdParC_noH2": "PdParC no H2",
}


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

class _MetricTreeItem(QTreeWidgetItem):
    """Sort metric columns numerically while keeping formatted display text."""

    def __lt__(self, other):
        column = self.treeWidget().sortColumn() if self.treeWidget() else 0
        left = self.data(column, _VALUE)
        right = other.data(column, _VALUE)
        if left is not None or right is not None:
            if left is None:
                return False
            if right is None:
                return True
            return float(left) < float(right)
        return self.text(column).casefold() < other.text(column).casefold()


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
        self._change_affects_all_days = False
        self._category_filter: str | None = None

        self.setColumnCount(len(TREE_COLUMNS))
        self.setHeaderLabels(TREE_COLUMNS)
        self.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setIndentation(18)
        self.setRootIsDecorated(True)
        self._build()
        self.itemChanged.connect(self._on_item_changed)
        self.setSortingEnabled(True)
        self.sortItems(0, Qt.AscendingOrder)

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        self._syncing = True
        self.clear()
        for esn, rows in self._df.groupby("esn", sort=True):
            first = rows.iloc[0]
            dev = _MetricTreeItem(self, [
                f"ESN {esn}", str(first["category"]), str(first["type"] or ""),
                "yes" if first["h2"] else "no", "", *(["—"] * len(DEVICE_METRICS)),
            ])
            dev.setData(0, _KEY, (str(esn), None))
            dev.setFlags(dev.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            bold = QFont(); bold.setBold(True)
            for column in range(len(TREE_COLUMNS)):
                dev.setFont(column, bold)

            for _, r in rows.sort_values("tnumber").iterrows():
                values = [r.get(spec.key, float("nan")) for spec in DEVICE_METRICS]
                ch = _MetricTreeItem(dev, [
                    f"T{int(r['tnumber'])}", "", "", "", str(int(r["metrics"])),
                    *[format_metric(value, spec)
                      for value, spec in zip(values, DEVICE_METRICS)],
                ])
                ch.setData(0, _KEY, (str(esn), int(r["tnumber"])))
                ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
                for offset, value in enumerate(values):
                    if not pd.isna(value):
                        ch.setData(_METRIC_START + offset, _VALUE, float(value))
            dev.setExpanded(False)
        self._syncing = False
        self.refresh()
        for i in range(len(_BASE_TREE_COLUMNS)):
            self.resizeColumnToContents(i)
        for i in range(_METRIC_START, len(TREE_COLUMNS)):
            self.setColumnWidth(i, 132)

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
            for offset, spec in enumerate(DEVICE_METRICS):
                column = _METRIC_START + offset
                values = [dev.child(j).data(column, _VALUE)
                          for j in range(dev.childCount())
                          if dev.child(j).checkState(0) == Qt.Checked
                          and dev.child(j).data(column, _VALUE) is not None]
                mean = sum(values) / len(values) if values else float("nan")
                dev.setText(column, format_metric(mean, spec))
                dev.setData(column, _VALUE, None if pd.isna(mean) else float(mean))
        self._syncing = False

    @staticmethod
    def _style(item: QTreeWidgetItem, included: bool) -> None:
        brush = _NORMAL if included else _GREY
        for c in range(item.columnCount()):
            item.setForeground(c, brush)

    # -- write helpers: honour the "sync across days" target list ---------

    def _set_channel(self, esn: str, tnum: int, included: bool) -> None:
        if included and (ALL_DAYS, str(esn), int(tnum)) in self._sel.excluded:
            self._change_affects_all_days = True
        for tp in self._targets(self._tp):
            if included:
                self._sel.include(tp, esn, tnum)
            else:
                self._sel.exclude(tp, esn, tnum, reason=f"excluded in GUI ({tp})")

    def _set_device(self, esn: str, tnums: list[int], included: bool) -> None:
        if included and any((ALL_DAYS, str(esn), int(t)) in self._sel.excluded
                            for t in tnums):
            self._change_affects_all_days = True
        for tp in self._targets(self._tp):
            if included:
                self._sel.include_device(tp, esn, tnums)
            else:
                self._sel.exclude_device(tp, esn, tnums,
                                         reason=f"excluded in GUI ({tp})")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        self._change_affects_all_days = False
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
        self._change_affects_all_days = False
        rows = self._df
        if self._category_filter is not None:
            rows = rows[rows["category"] == self._category_filter]
        for _, r in rows.iterrows():
            self._set_channel(r["esn"], int(r["tnumber"]), included)
        self.refresh()
        self.changed.emit()

    def set_selected(self, included: bool) -> None:
        """Apply to whatever rows are highlighted (devices expand to channels)."""
        self._change_affects_all_days = False
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

    def set_category_filter(self, category: str | None) -> None:
        """Show only one device category; selection/statistics stay unchanged."""
        self._category_filter = category
        for i in range(self.topLevelItemCount()):
            device = self.topLevelItem(i)
            device.setHidden(category is not None and device.text(1) != category)


class CategorySummaryTable(QTableWidget):
    """Live mean ± AVDEV table for the four categories in one condition."""

    def __init__(self, df: pd.DataFrame, selection: Selection, timepoint: str):
        super().__init__(len(CATEGORY_ORDER), 1 + len(DEVICE_METRICS))
        self._df = df
        self._sel = selection
        self._tp = timepoint
        self.setHorizontalHeaderLabels(
            ["Category"] + [spec.label for spec in DEVICE_METRICS])
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setMaximumHeight(176)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for column in range(1, self.columnCount()):
            self.horizontalHeader().setSectionResizeMode(column, QHeaderView.Interactive)
            self.setColumnWidth(column, 165)
        for row, category in enumerate(CATEGORY_ORDER):
            item = QTableWidgetItem(_CATEGORY_LABELS.get(category, category))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.setItem(row, 0, item)
        self.refresh()

    def set_selection(self, selection: Selection) -> None:
        self._sel = selection
        self.refresh()

    def set_category_filter(self, category: str | None) -> None:
        for row, row_category in enumerate(CATEGORY_ORDER):
            self.setRowHidden(row, category is not None and row_category != category)

    def refresh(self) -> None:
        summary = category_summary(self._df, self._sel, self._tp)
        by_key = {(r.category, r.metric): r for r in summary.itertuples()}
        for row, category in enumerate(CATEGORY_ORDER):
            for offset, spec in enumerate(DEVICE_METRICS, start=1):
                result = by_key.get((category, spec.key))
                if result is None or result.n == 0:
                    text = "—"
                    tooltip = "No included measurements"
                else:
                    mean = format_metric(result.mean, spec)
                    avdev = format_metric(result.avdev, spec)
                    text = f"{mean} ± {avdev}  (n={result.n})"
                    tooltip = f"Mean {mean}; AVDEV {avdev}; included n={result.n}"
                item = self.item(row, offset) or QTableWidgetItem()
                item.setText(text)
                item.setToolTip(tooltip)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(row, offset, item)


class ConditionPanel(QWidget):
    """One condition's transistor tree with its category summary below."""

    def __init__(self, df: pd.DataFrame, selection: Selection, timepoint: str,
                 targets=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.tree = ConditionTree(df, selection, timepoint, targets=targets)
        self.summary = CategorySummaryTable(df, selection, timepoint)
        label = QLabel("Category summary — mean ± AVDEV over included transistors")
        label.setProperty("hint", True)
        layout.addWidget(self.tree, 1)
        layout.addWidget(label)
        layout.addWidget(self.summary)

    def refresh(self) -> None:
        self.tree.refresh()
        self.summary.refresh()

    def set_selection(self, selection: Selection) -> None:
        self.tree._sel = selection
        self.summary.set_selection(selection)

    def set_category_filter(self, category: str | None) -> None:
        self.tree.set_category_filter(category)
        self.summary.set_category_filter(category)


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
        self.panels: dict[tuple[str, str], ConditionPanel] = {}
        self._condition_frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.cond_tab_widgets: dict[str, QTabWidget] = {}
        self._thread: QThread | None = None
        self.progress: QProgressDialog | None = None
        self._cancel_requested = False
        self._worker: _Worker | None = None
        self._done_msg = ""
        self._tidy: pd.DataFrame | None = None
        self._keyed_tidy: pd.DataFrame | None = None
        self._keyed_source_id: int | None = None
        self._agg: pd.DataFrame = pd.DataFrame()
        self._agg_ready = False
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
        #: which (sheet, metric, series) the format widgets currently describe,
        #: so switching metric stores the old one before loading the new
        self._opts_key: str | None = None
        #: curve-visibility checkboxes, rebuilt whenever the varied axis changes
        self._curve_checks: dict[str, QCheckBox] = {}
        self._analysis_progress.connect(self._on_analysis_progress)

        self.setWindowTitle("QFN-AGING")
        self.resize(1180, 760)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "data" / "qfn.ico")))
        self.setStatusBar(QStatusBar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_devices_tab(), "Devices")
        self.tabs.addTab(self._build_compare_tab(), "Comparisons")
        self.tabs.currentChanged.connect(self._main_tab_changed)
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
        category = self.settings.get("device_category")
        category_index = self.cmb_device_category.findData(category)
        self.cmb_device_category.setCurrentIndex(max(category_index, 0))
        self._apply_comparison_settings()
        self.root_edit.editingFinished.connect(self._remember_settings)
        self._refresh_status()

    def _remember_settings(self) -> None:
        self.settings.update({
            "data_root": self.root_edit.text().strip(),
            "make_plots": self.chk_plots.isChecked(),
            "skip_existing": self.chk_skip.isChecked(),
            "workers": self.spn_workers.value(),
            "device_category": self.cmb_device_category.currentData(),
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

            analyze_all(root, out, self.alloc, make_plots=make_plots,
                        timepoints=timepoints, skip_existing=skip_existing,
                        workers=workers, progress=on_progress, log=emit,
                        cancelled=lambda: self._cancel_requested)

        self._cancel_requested = False
        self.progress.canceled.connect(self._request_cancel)
        self._run_bg(job, "Analysis finished.")

    def _request_cancel(self) -> None:
        self._cancel_requested = True
        self._emit("Cancel requested — stopping before more runs are started…")

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
        elif "_Cancelled" in err or "AnalysisCancelled" in err:
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

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show category:"))
        self.cmb_device_category = QComboBox()
        self.cmb_device_category.addItem("All categories", None)
        for category in CATEGORY_ORDER:
            self.cmb_device_category.addItem(_CATEGORY_LABELS.get(category, category), category)
        self.cmb_device_category.setToolTip(
            "Visual filter only. It does not include or exclude transistors from statistics.")
        self.cmb_device_category.currentIndexChanged.connect(
            self._device_category_changed)
        filter_row.addWidget(self.cmb_device_category)
        filter_row.addStretch(1)
        v.addLayout(filter_row)

        # day tabs on the outside, condition tabs within each day
        self.day_tabs = QTabWidget()
        self.day_tabs.setTabPosition(QTabWidget.North)
        self.day_tabs.currentChanged.connect(self._day_changed)
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

    def _device_category_changed(self) -> None:
        category = self.cmb_device_category.currentData()
        for panel in self.panels.values():
            panel.set_category_filter(category)
        if hasattr(self, "settings"):
            self._remember_settings()
        self._refresh_status()

    # -- tab 3: comparisons ----------------------------------------------

    #: Format settings that are a matter of taste and apply to every metric.
    _GLOBAL_FORMAT_DEFAULTS = {"draw": "line+points", "palette": "paper",
                               "minor_ticks": True, "grid": True}
    #: Format settings that only make sense per metric (a Y tick step for
    #: volts is meaningless for a percentage), stored under its key.
    _METRIC_FORMAT_DEFAULTS = {"title": "", "bold": True, "italic": False,
                               "xlabel": "", "ylabel": "",
                               "x_step": 0.0, "y_step": 0.0}

    def _comparison_prefs(self) -> dict:
        """The persisted Comparisons-tab block, created on first use."""
        if not hasattr(self, "settings"):
            self.settings = {}
        prefs = self.settings.setdefault("comparison", {})
        prefs.setdefault("per_metric", {})
        prefs.setdefault("hidden", {})
        for key, default in self._GLOBAL_FORMAT_DEFAULTS.items():
            prefs.setdefault(key, default)
        return prefs

    @staticmethod
    def _comparison_key(spec: ComparisonSpec) -> str:
        """Identity of 'the same metric' -- what a saved title is filed under.

        Sheet is part of it because one metric name can appear on two sheets.
        The series (Vbg) deliberately is NOT: flipping between Vbg=0 and
        Vbg=4 is looking at the same quantity, so the title and axis labels
        follow you across instead of blanking out.
        """
        return f"{spec.sheet}|{spec.metric}"

    def _build_title_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Title:"))
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText(
            "(automatic \u2014 type to override, saved per metric)")
        self.ed_title.setClearButtonEnabled(True)
        self.ed_title.textEdited.connect(self._format_changed)
        row.addWidget(self.ed_title, 1)

        self.btn_title_bold = QPushButton("B")
        self.btn_title_bold.setToolTip("Bold title")
        font = self.btn_title_bold.font()
        font.setBold(True)
        self.btn_title_bold.setFont(font)
        self.btn_title_italic = QPushButton("I")
        self.btn_title_italic.setToolTip("Italic title")
        font = self.btn_title_italic.font()
        font.setItalic(True)
        self.btn_title_italic.setFont(font)
        for btn in (self.btn_title_bold, self.btn_title_italic):
            btn.setCheckable(True)
            btn.setFixedWidth(34)
            btn.toggled.connect(self._format_changed)
            row.addWidget(btn)

        self.btn_title_reset = QPushButton("Auto")
        self.btn_title_reset.setToolTip(
            "Drop the custom title and axis labels, back to the generated ones.")
        self.btn_title_reset.clicked.connect(self._reset_title)
        row.addWidget(self.btn_title_reset)

        row.addWidget(QLabel("X axis:"))
        self.ed_xlabel = QLineEdit()
        self.ed_xlabel.setPlaceholderText(DEFAULT_XLABEL)
        row.addWidget(self.ed_xlabel, 1)
        row.addWidget(QLabel("Y axis:"))
        self.ed_ylabel = QLineEdit()
        # placeholder is refilled per metric in _refresh_comparison
        self.ed_ylabel.setPlaceholderText("(metric)")
        row.addWidget(self.ed_ylabel, 1)
        for edit in (self.ed_xlabel, self.ed_ylabel):
            edit.setClearButtonEnabled(True)
            edit.textEdited.connect(self._format_changed)
        return row

    def _build_format_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Draw:"))
        self.cmb_draw = QComboBox()
        for label, key in (("Line + points", "line+points"), ("Line only", "line"),
                           ("Points only", "points")):
            self.cmb_draw.addItem(label, key)
        row.addWidget(self.cmb_draw)

        row.addWidget(QLabel("Colours:"))
        self.cmb_palette = QComboBox()
        self.cmb_palette.addItem("Paper (sensorlab)", "paper")
        self.cmb_palette.addItem("Project green", "green")
        row.addWidget(self.cmb_palette)

        row.addWidget(QLabel("X tick step:"))
        self.spn_xstep = QDoubleSpinBox()
        row.addWidget(self.spn_xstep)
        row.addWidget(QLabel("Y tick step:"))
        self.spn_ystep = QDoubleSpinBox()
        row.addWidget(self.spn_ystep)
        for spn in (self.spn_xstep, self.spn_ystep):
            spn.setRange(0.0, 1e6)
            spn.setDecimals(4)
            spn.setSingleStep(0.5)
            spn.setSpecialValueText("auto")     # 0 reads as "auto", not "0"
            spn.setFixedWidth(90)
            spn.valueChanged.connect(self._format_changed)

        self.chk_minor_ticks = QCheckBox("Minor ticks")
        self.chk_grid = QCheckBox("Grid")
        for chk in (self.chk_minor_ticks, self.chk_grid):
            chk.setChecked(True)
            chk.toggled.connect(self._format_changed)
            row.addWidget(chk)

        for cmb in (self.cmb_draw, self.cmb_palette):
            cmb.currentIndexChanged.connect(self._format_changed)
        row.addStretch(1)
        #: everything that is filed per metric -- greyed out until there is
        #: one, so a title typed too early cannot be silently discarded
        self._metric_format_widgets = (self.ed_title, self.btn_title_bold,
                                       self.btn_title_italic, self.btn_title_reset,
                                       self.ed_xlabel, self.ed_ylabel,
                                       self.spn_xstep, self.spn_ystep)
        return row

    def _build_curve_filter_row(self) -> QWidget:
        """Host for the per-value "show this curve" boxes.

        The boxes depend on which axis is varied and on what the data holds,
        so they are rebuilt on each redraw rather than created once here.
        """
        self.curve_filter = QWidget()
        lay = QHBoxLayout(self.curve_filter)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.lbl_curve_filter = QLabel("Show:")
        lay.addWidget(self.lbl_curve_filter)
        lay.addStretch(1)
        return self.curve_filter

    def _rebuild_curve_filter(self, vary: str, values: list) -> None:
        """One checkbox per curve value present in the data.

        Rebuilt in place: the previous boxes are dropped, so switching the
        varied axis cannot leave stale values (25/35/45 C) on screen next to
        the new ones (10/30/50 %RH).
        """
        lay = self.curve_filter.layout()
        for chk in self._curve_checks.values():
            lay.removeWidget(chk)
            chk.deleteLater()
        self._curve_checks = {}

        hidden = {str(h) for h in self._comparison_prefs()["hidden"].get(vary, [])}
        self.lbl_curve_filter.setText(
            f"Show {AXIS_LABELS[vary].lower()}:" if values else "Show:")
        was_filling = self._filling
        self._filling = True                 # setChecked must not trigger a redraw
        for i, value in enumerate(values):
            chk = QCheckBox(str(value))
            chk.setChecked(str(value) not in hidden)
            chk.toggled.connect(self._curve_visibility_changed)
            lay.insertWidget(i + 1, chk)     # before the trailing stretch
            self._curve_checks[str(value)] = chk
        self._filling = was_filling

    def _curve_visibility_changed(self) -> None:
        if getattr(self, "_filling", False):
            return
        spec = self.current_spec()
        if spec is None:
            return
        hidden = sorted(k for k, chk in self._curve_checks.items() if not chk.isChecked())
        self._comparison_prefs()["hidden"][spec.vary] = hidden
        self._remember_settings()
        self._refresh_comparison()

    def _current_options(self, spec: ComparisonSpec | None) -> PlotOptions:
        """Read the format widgets (plus stored curve visibility) into one object."""
        prefs = self._comparison_prefs()
        hidden = prefs["hidden"].get(spec.vary, []) if spec is not None else []
        return PlotOptions(
            title=self.ed_title.text(),
            title_bold=self.btn_title_bold.isChecked(),
            title_italic=self.btn_title_italic.isChecked(),
            xlabel=self.ed_xlabel.text(),
            ylabel=self.ed_ylabel.text(),
            draw=self.cmb_draw.currentData() or "line+points",
            palette=self.cmb_palette.currentData() or "paper",
            x_step=float(self.spn_xstep.value()),
            y_step=float(self.spn_ystep.value()),
            minor_ticks=self.chk_minor_ticks.isChecked(),
            grid=self.chk_grid.isChecked(),
            hidden=list(hidden),
        )

    def _apply_comparison_settings(self) -> None:
        """Push the persisted global format prefs into the widgets.

        Called once after settings are loaded -- the compare tab is built
        before that, so it starts on defaults.
        """
        prefs = self._comparison_prefs()
        self._filling = True
        for combo, key in ((self.cmb_draw, "draw"), (self.cmb_palette, "palette")):
            i = combo.findData(prefs.get(key))
            combo.setCurrentIndex(max(i, 0))
        self.chk_minor_ticks.setChecked(bool(prefs.get("minor_ticks", True)))
        self.chk_grid.setChecked(bool(prefs.get("grid", True)))
        self._filling = False

    def _load_metric_format(self, key: str) -> None:
        """Fill the per-metric widgets from what was saved for *key*."""
        per_metric = self._comparison_prefs()["per_metric"]
        saved = per_metric.get(key)
        if saved is None:
            # entries written while the key still carried the series; the
            # first one wins and is re-filed under the new key on next save
            legacy = [v for k, v in sorted(per_metric.items())
                      if k.startswith(f"{key}|")]
            saved = legacy[0] if legacy else {}
        self._filling = True
        self.ed_title.setText(str(saved.get("title", "")))
        self.btn_title_bold.setChecked(bool(saved.get("bold", True)))
        self.btn_title_italic.setChecked(bool(saved.get("italic", False)))
        self.ed_xlabel.setText(str(saved.get("xlabel", "")))
        self.ed_ylabel.setText(str(saved.get("ylabel", "")))
        self.spn_xstep.setValue(float(saved.get("x_step", 0.0) or 0.0))
        self.spn_ystep.setValue(float(saved.get("y_step", 0.0) or 0.0))
        self._filling = False

    def _store_metric_format(self, key: str) -> None:
        """Persist the per-metric widgets under *key*, and the global ones.

        An entry holding nothing but defaults is deleted rather than written,
        so the settings file does not fill with empty records for every metric
        the user merely clicked through.
        """
        prefs = self._comparison_prefs()
        entry = {"title": self.ed_title.text(),
                 "bold": self.btn_title_bold.isChecked(),
                 "italic": self.btn_title_italic.isChecked(),
                 "xlabel": self.ed_xlabel.text(),
                 "ylabel": self.ed_ylabel.text(),
                 "x_step": float(self.spn_xstep.value()),
                 "y_step": float(self.spn_ystep.value())}
        if entry == self._METRIC_FORMAT_DEFAULTS:
            prefs["per_metric"].pop(key, None)
        else:
            prefs["per_metric"][key] = entry
        prefs.update(draw=self.cmb_draw.currentData() or "line+points",
                     palette=self.cmb_palette.currentData() or "paper",
                     minor_ticks=self.chk_minor_ticks.isChecked(),
                     grid=self.chk_grid.isChecked())
        self._remember_settings()

    def _format_changed(self) -> None:
        """Any format widget moved: save it against the current metric, redraw."""
        if getattr(self, "_filling", False):
            return
        if self._opts_key:
            self._store_metric_format(self._opts_key)
        self._refresh_comparison()

    def _reset_title(self) -> None:
        """Back to the generated title and axis labels for this metric."""
        self._filling = True
        for edit in (self.ed_title, self.ed_xlabel, self.ed_ylabel):
            edit.clear()
        self._filling = False
        self._format_changed()

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

        v.addLayout(self._build_title_row())
        v.addLayout(self._build_format_row())
        v.addWidget(self._build_curve_filter_row())

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

    def _rebuild_aggregate(self,
                           scopes: set[tuple[str, str]] | None = None) -> None:
        """Recompute all averages, or only affected (day, condition) scopes."""
        if getattr(self, "_tidy", None) is None or self._tidy.empty:
            self._agg = pd.DataFrame()
            self._keyed_tidy = None
            self._keyed_source_id = None
            self._agg_ready = True
            return
        if self._keyed_tidy is None or self._keyed_source_id != id(self._tidy):
            self._keyed_tidy = add_group_keys(self._tidy, self.alloc)
            self._keyed_source_id = id(self._tidy)
            scopes = None

        if not scopes or self._agg.empty:
            self._agg = aggregate_groups(apply_selection(self._keyed_tidy, self.selection))
            self._agg_ready = True
            return

        source_mask = pd.Series(False, index=self._keyed_tidy.index)
        old_mask = pd.Series(False, index=self._agg.index)
        for timepoint, condition in scopes:
            source_mask |= ((self._keyed_tidy["timepoint"] == timepoint)
                            & (self._keyed_tidy["condition"] == condition))
            old_mask |= ((self._agg["timepoint"] == timepoint)
                         & (self._agg["condition"] == condition))
        refreshed = aggregate_groups(
            apply_selection(self._keyed_tidy.loc[source_mask], self.selection))
        self._agg = pd.concat([self._agg.loc[~old_mask], refreshed], ignore_index=True)
        self._agg_ready = True

    def _main_tab_changed(self, index: int) -> None:
        """Build comparison aggregates lazily, only when that tab is opened."""
        if index != 2 or self._agg_ready or self._tidy is None:
            return
        self.statusBar().showMessage("Preparing comparison averages…")
        QApplication.processEvents()
        self._rebuild_aggregate()
        self._populate_comparison_options()
        self.statusBar().showMessage("Comparison averages ready", 3000)

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
        keep_series = self.cmb_series.currentData()
        self.cmb_series.clear()
        for s in series:
            self.cmb_series.addItem(s or "(none)", s)
        # without this a repopulate silently drops you back to the first Vbg,
        # which looks exactly like the plot losing your settings
        i = self.cmb_series.findData(keep_series)
        self.cmb_series.setCurrentIndex(max(i, 0))
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

        # A different metric means a different saved title / tick steps, so
        # swap the format widgets over before they are read back below.
        opts_key = self._comparison_key(spec) if spec is not None else None
        for widget in getattr(self, "_metric_format_widgets", ()):
            widget.setEnabled(opts_key is not None)
        if opts_key != self._opts_key:
            self._load_metric_format(opts_key or "")
            self._opts_key = opts_key
        options = self._current_options(spec)

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
            auto_y = ylabel_for_metric(spec.metric)
            # show what leaving the box empty would give, so the override is
            # an obvious edit rather than a guess
            self.ed_ylabel.setPlaceholderText(auto_ylabel(spec, auto_y))
            plot_comparison(curves, spec, ylabel=auto_y, ax=ax, options=options)
            # the boxes list every value in the data, ticked or not, so a
            # hidden curve can always be brought back
            self._rebuild_curve_filter(
                spec.vary, list(dict.fromkeys(curves["curve"].tolist())))
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
            # paper-style resolution, same as the pipeline's figures
            self.figure.savefig(path, dpi=figure_dpi())
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
            tidy = load_masters_cached(analysis, progress=on_progress, log=self._emit)
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
        self._keyed_tidy = None
        self._keyed_source_id = None
        self._agg = pd.DataFrame()
        self._agg_ready = False
        # device list is built from the measured metrics only, so the
        # per-transistor metric count isn't inflated by the derived drift rows
        keyed = add_group_keys(tidy, self.alloc)
        per = (keyed.groupby(["timepoint", "esn", "tnumber", "condition", "category"])
                    .size().reset_index(name="metrics"))
        per["type"] = per["esn"].map(lambda e: self.alloc.device(e).type)
        per["h2"] = per["esn"].map(lambda e: self.alloc.device(e).h2)
        metric_table = build_device_metric_table(self._tidy)
        per = per.merge(metric_table, on=["timepoint", "esn", "tnumber"], how="left")
        self.populate(per)

        # Comparisons are built lazily when that tab is opened.  The Devices
        # view should not pay the full-study aggregation cost up front.
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
        self.panels.clear()
        self._condition_frames.clear()
        self.cond_tab_widgets.clear()
        self.placeholder.setVisible(False)

        days = sort_timepoints(per["timepoint"].unique())
        for day in days:
            day_rows = per[per["timepoint"] == day]
            cond_tabs = QTabWidget()
            cond_tabs.setTabPosition(QTabWidget.North)

            for condition in self.alloc.conditions():
                sub = day_rows[day_rows["condition"] == condition]
                if sub.empty:
                    continue
                self._condition_frames[(day, condition)] = sub.copy()
                # Creating all 72 rich tables up front costs tens of seconds.
                # A light placeholder is replaced only when its tab is opened.
                cond_tabs.addTab(QWidget(), condition)

            self.cond_tab_widgets[day] = cond_tabs
            cond_tabs.currentChanged.connect(
                lambda index, current_day=day:
                self._condition_tab_changed(current_day, index))
            self.day_tabs.addTab(cond_tabs, day)

        self._ensure_current_panel()
        self._refresh_tab_titles()
        self._refresh_status()

    def _condition_tab_changed(self, day: str, index: int) -> None:
        self._ensure_condition_panel(day, index)
        self._refresh_status()

    def _day_changed(self, _index: int) -> None:
        self._ensure_current_panel()
        self._refresh_status()

    def _ensure_current_panel(self) -> ConditionPanel | None:
        day = self.current_day()
        if day is None:
            return None
        tabs = self.cond_tab_widgets.get(day)
        if tabs is None:
            return None
        return self._ensure_condition_panel(day, tabs.currentIndex())

    def _ensure_condition_panel(self, day: str, index: int) -> ConditionPanel | None:
        tabs = self.cond_tab_widgets.get(day)
        if tabs is None or index < 0 or index >= tabs.count():
            return None
        condition = tabs.tabText(index).split("  (")[0]
        key = (day, condition)
        if key in self.panels:
            return self.panels[key]
        rows = self._condition_frames.get(key)
        if rows is None:
            return None

        panel = ConditionPanel(rows, self.selection, day, targets=self._sync_targets)
        panel.tree.changed.connect(self._on_selection_changed)
        title = tabs.tabText(index)
        old = tabs.widget(index)
        tabs.blockSignals(True)
        tabs.removeTab(index)
        tabs.insertTab(index, panel, title)
        tabs.setCurrentIndex(index)
        tabs.blockSignals(False)
        old.deleteLater()
        self.panels[key] = panel
        self.trees[key] = panel.tree
        panel.set_category_filter(self.cmb_device_category.currentData())
        return panel

    def tree_for(self, day: str, condition: str) -> ConditionTree:
        """Return a condition tree, materialising its lazy panel if needed."""
        tabs = self.cond_tab_widgets[day]
        index = next(i for i in range(tabs.count())
                     if tabs.tabText(i).split("  (")[0] == condition)
        panel = self._ensure_condition_panel(day, index)
        if panel is None:
            raise KeyError((day, condition))
        return panel.tree

    def _measured_days(self) -> list[str]:
        """Days present in the loaded per-transistor data, in study order."""
        return sort_timepoints({d for d, _ in self._condition_frames})

    def _sync_targets(self, day: str) -> list[str]:
        """Which days a tick on ``day`` writes to: just it, or all of them."""
        return self._measured_days() if self.sync_days else [day]

    def current_day(self) -> str | None:
        i = self.day_tabs.currentIndex()
        return self.day_tabs.tabText(i).split("  (")[0] if i >= 0 else None

    def _current_tree(self) -> ConditionTree | None:
        panel = self._ensure_current_panel()
        return panel.tree if panel is not None else None

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
            for panel in self.panels.values():
                panel.refresh()
        else:
            for panel in self.panels.values():
                panel.summary.refresh()

        sender = self.sender()
        if isinstance(sender, ConditionTree) and not sender._df.empty:
            condition = str(sender._df.iloc[0]["condition"])
            days = (self._measured_days()
                    if self.sync_days or sender._change_affects_all_days
                    else [sender._tp])
            scopes = {(day, condition) for day in days}
        else:
            scopes = None
        if self._agg_ready:
            self._rebuild_aggregate(scopes)
            self._refresh_comparison()
        self._refresh_status()
        self.statusBar().showMessage(
            self.statusBar().currentMessage() + "  ·  saved", 3000)

    def _save_selection(self) -> None:
        self.selection.save(self.selection_path)
        self.statusBar().showMessage(f"Saved to {self.selection_path}", 5000)

    def _reload_selection(self) -> None:
        self.selection = load_selection(self.selection_path)
        for panel in self.panels.values():
            panel.set_selection(self.selection)
        if self._agg_ready:
            self._rebuild_aggregate()
            self._refresh_comparison()
        self._refresh_status()

    def _refresh_tab_titles(self) -> None:
        for day, cond_tabs in self.cond_tab_widgets.items():
            for i in range(cond_tabs.count()):
                condition = cond_tabs.tabText(i).split("  (")[0]
                frame = self._condition_frames.get((day, condition))
                if frame is not None:
                    inc, total = self._frame_counts(frame, day)
                    cond_tabs.setTabText(i, f"{condition}  ({inc}/{total})")

        for i in range(self.day_tabs.count()):
            day = self.day_tabs.tabText(i).split("  (")[0]
            counts = [self._frame_counts(frame, day)
                      for (d, _), frame in self._condition_frames.items() if d == day]
            inc = sum(c[0] for c in counts)
            total = sum(c[1] for c in counts)
            self.day_tabs.setTabText(i, f"{day}  ({inc}/{total})")

    def _frame_counts(self, frame: pd.DataFrame, day: str) -> tuple[int, int]:
        total = len(frame)
        included = sum(self.selection.is_included(day, str(r.esn), int(r.tnumber))
                       for r in frame.itertuples())
        return included, total

    def _refresh_status(self) -> None:
        self._refresh_tab_titles()
        day = self.current_day()
        days = self._measured_days()
        if day:
            counts = [self._frame_counts(frame, day)
                      for (d, _), frame in self._condition_frames.items() if d == day]
            inc = sum(c[0] for c in counts)
            total = sum(c[1] for c in counts)
            category = self.cmb_device_category.currentData()
            filter_text = (_CATEGORY_LABELS.get(category, category)
                           if category else "all categories")
            self.statusBar().showMessage(
                f"{day}: {inc}/{total} transistors in · "
                f"{self.selection.count_for(day)} excluded on this day · "
                f"filter: {filter_text} · "
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
