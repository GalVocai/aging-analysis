"""Run sensorlab's Stage-1 analysis over the discovered QFN-AGING runs.

sensorlab is used as an external, read-only library: this project imports
it to compute metrics, and never writes into its source tree, its output
folders, or its user config. See [[feedback-dont-mix-projects]] -- the two
projects stay entirely separate.

Concretely, the separation is enforced by:

* **Passing our own** :class:`ConditionCache`, built in memory from this
  project's roster. ``analyze_folder`` only falls back to sensorlab's
  ``~/.sensorlab/conditions.json`` when ``conditions`` is ``None``, so
  passing one means that file is never read and never written. Its ``path``
  is pointed inside our own output root as a second line of defence.
* **``make_plots=False``**. Plot rendering is the only part of Stage 1 that
  touches ``~/.sensorlab/plot_style.json``; masters-only skips it entirely.
  This project draws its own comparison plots anyway.
* **Only calling Stage 1.** sensorlab's Stage 2 (``run_degradation``)
  writes ``~/.sensorlab/degradation_labels.json``; we never call it -- the
  (condition x category) aggregation lives in this project's
  :mod:`qfn_aging.grouping`.
* **An explicit ``out_dir``** under this project's output root, so nothing
  is written next to the raw OneDrive data either.

Output layout::

    <out_root>/<timepoint>/<run_key>/<condition>/<MODE>/master_<MODE>.xlsx
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Force a non-interactive matplotlib backend BEFORE pyplot is ever imported.
#
# With PySide6 installed, matplotlib auto-selects a Qt backend. sensorlab's
# analysis creates figures, and this project runs that analysis on a worker
# thread to keep the window responsive -- but Qt figure creation off the
# main thread crashes the whole process, taking the GUI down mid-run. Agg
# is thread-safe and renders to file, which is all we need here.
#
# This must stay at module import: gui.py imports this module before any
# analysis starts, and sensorlab is imported lazily inside the functions
# below, so by the time its plotting code loads pyplot the backend is set.
import matplotlib

matplotlib.use("Agg", force=True)

from .allocation import DEFAULT, Allocation  # noqa: E402
from .discovery import RunFolder, discover  # noqa: E402

# Guard: nothing this module produces may live under sensorlab's user config.
_FORBIDDEN_ROOT = Path.home() / ".sensorlab"

#: This project's own plot style. sensorlab's ``analyze_folder`` calls
#: ``load_style()`` with no arguments, which would read
#: ``~/.sensorlab/plot_style.json`` -- read-only and harmless, but it would
#: make our figures change whenever the sensorlab GUI's style is edited.
#: :func:`_own_plot_style` redirects that lookup here instead. If the file
#: does not exist, sensorlab's built-in ``StyleConfig`` defaults are used,
#: so QFN-AGING figures never depend on the other project's settings.
STYLE_PATH = Path(__file__).parent / "data" / "plot_style.json"

# sensorlab writes one master folder per analysis mode. Several JG majors
# intentionally share one workbook, while JG_H2 is derived when both the air
# and hydrogen sweeps are present.
_MASTER_MODE_BY_MAJOR = {
    "n01": "ZOZO",
    "n02": "JGH",
    "n03": "JG",
    "n04": "SE",
    "n05": "JG",
    "n06": "JG",
    "n07": "STEPS",
}

#: Output resolution for the per-device figures.
#:
#: sensorlab hardcodes ``dpi=110`` at each ``savefig`` call, which dominates
#: rasterisation cost -- rendering time scales with pixel count, and plotting
#: is ~83% of a run. Dropping to 90 roughly halves it (measured 132ms -> 67ms
#: per figure) at a small cost in sharpness, which is a good trade for
#: reading degradation trends. :func:`_plot_dpi` applies it without touching
#: sensorlab; set to ``None`` to keep whatever sensorlab asks for.
PLOT_DPI: int | None = 90


class AnalysisCancelled(RuntimeError):
    """The user cancelled a sweep between runs."""


def _run_identity(rf: RunFolder) -> tuple[str, str, str]:
    """Stable identity for a discovered run, including its measurement day."""
    return rf.timepoint, rf.run_key, str(rf.path)


def ensure_multiprocessing_executable() -> str | None:
    """Make sure worker processes are spawned with a usable interpreter.

    On Windows, ``multiprocessing`` starts children with ``sys.executable``.
    Under ``pythonw.exe`` there is no console, so the child's stdio handles
    are invalid and it dies during bootstrap -- the pool then reports
    ``BrokenProcessPool`` and every run in flight is lost. Pointing at the
    console ``python.exe`` next to it fixes the children without changing
    how the GUI itself was launched.

    Returns the executable now in use, or ``None`` if nothing was changed.
    """
    import multiprocessing
    import sys

    exe = Path(sys.executable)
    if exe.name.lower() != "pythonw.exe":
        return None
    console = exe.with_name("python.exe")
    if not console.is_file():
        return None
    multiprocessing.set_executable(str(console))
    return str(console)


def _check_outside_sensorlab_config(path: Path) -> None:
    p = Path(path).resolve()
    if p == _FORBIDDEN_ROOT or _FORBIDDEN_ROOT in p.parents:
        raise ValueError(
            f"refusing to write inside sensorlab's config directory: {p}. "
            "The two projects are kept entirely separate."
        )


def build_condition_cache(out_root: Path, alloc: Allocation = DEFAULT):
    """A ConditionCache holding this study's 12 conditions, in memory.

    Its ``path`` is inside our own output root, so it can never clobber
    sensorlab's own cache even if something called ``.save()`` on it.
    """
    from sensorlab.io import ConditionCache  # imported lazily: read-only use

    path = Path(out_root) / "_conditions.json"
    _check_outside_sensorlab_config(path)
    return ConditionCache(path=path, mapping=alloc.condition_map())


@contextlib.contextmanager
def _plot_dpi(dpi: int | None):
    """Force every figure saved inside this block to a given resolution.

    sensorlab passes ``dpi=110`` explicitly at each call site, so an rcParam
    would be ignored -- the keyword has to be overridden on the way through.
    Patching ``Figure.savefig`` is contained (restored on exit, including on
    error) and leaves sensorlab's source untouched.
    """
    if not dpi:
        yield
        return

    from matplotlib.figure import Figure

    original = Figure.savefig

    def patched(self, *args, **kwargs):
        kwargs["dpi"] = dpi
        return original(self, *args, **kwargs)

    Figure.savefig = patched
    try:
        yield
    finally:
        Figure.savefig = original


@contextlib.contextmanager
def _own_plot_style(style_path: Path | None = None):
    """Point sensorlab's style lookup at this project's style file.

    ``analyze_folder`` resolves ``load_style()`` through its own module
    namespace, so swapping that name for the duration of the call keeps our
    figures independent of ``~/.sensorlab/plot_style.json``. Nothing is
    written to either location, and the original is always restored.

    If sensorlab's internals change and the name is gone, this becomes a
    no-op -- figures would fall back to sensorlab's style rather than
    breaking the run.
    """
    import sensorlab.pipeline as sl_pipeline
    from sensorlab.plotting.style import load_style as real_load_style

    path = Path(style_path or STYLE_PATH)
    if not hasattr(sl_pipeline, "load_style"):
        yield
        return

    original = sl_pipeline.load_style
    sl_pipeline.load_style = lambda *_a, **_kw: real_load_style(path)
    try:
        yield
    finally:
        sl_pipeline.load_style = original


def _run_one(args: tuple) -> tuple["RunResult", list[str]]:
    """Analyze one run, collecting its log instead of streaming it.

    Module-level and self-contained so it can be sent to a worker process:
    a live log callback is not picklable, so the lines are gathered here and
    handed back with the result.
    """
    rf, out_root, alloc, make_plots, style_path, dpi = args
    lines: list[str] = []
    result = analyze_run(rf, out_root, alloc, make_plots=make_plots,
                         style_path=style_path, dpi=dpi, log=lines.append)
    return result, lines


@dataclass(frozen=True)
class RunResult:
    timepoint: str
    run_key: str
    run_id: str | None
    folder: Path
    masters: dict[str, list[str]]     # mode -> master paths written
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def analyze_run(rf: RunFolder, out_root: str | Path, alloc: Allocation = DEFAULT,
                make_plots: bool = True, style_path: Path | None = None,
                dpi: int | None = PLOT_DPI,
                log: Callable[[str], None] | None = None) -> RunResult:
    """Run sensorlab Stage 1 on one discovered run folder.

    With ``make_plots=True`` sensorlab also writes its per-device analysis
    figures next to each master, under ``<condition>/<MODE>/by_device/``.
    """
    from sensorlab.pipeline import analyze_folder

    emit = log or (lambda _m: None)
    out_root = Path(out_root)
    out_dir = out_root / rf.timepoint / rf.out_group
    _check_outside_sensorlab_config(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_condition_cache(out_root, alloc)

    emit(f"[{rf.timepoint}/{rf.run_key}] analyzing {len(rf.results)} majors "
         f"-> {rf.timepoint}/{rf.out_group}")
    try:
        with _own_plot_style(style_path), _plot_dpi(dpi if make_plots else None):
            masters = analyze_folder(
                str(rf.path),
                out_dir=str(out_dir),
                conditions=conditions,
                make_plots=make_plots,
                day=rf.timepoint,
                log=emit,
            )
    except Exception as exc:  # one bad run must not abort the whole sweep
        emit(f"[{rf.timepoint}/{rf.run_key}] FAILED: {type(exc).__name__}: {exc}")
        return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, {}, error=str(exc))

    return RunResult(rf.timepoint, rf.run_key, rf.run_id, rf.path, masters)


def has_output(rf: RunFolder, out_root: str | Path,
               alloc: Allocation = DEFAULT) -> bool:
    """True if **this run's own** conditions already have master workbooks.

    Checking the output folder alone would be wrong: the two control runs
    share one ``control`` folder, so as soon as set 1 has written, set 2
    would look done and be skipped even though its devices were never
    analyzed. Their conditions are disjoint, so ask per condition instead.
    """
    base = Path(out_root) / rf.timepoint / rf.out_group
    if not base.is_dir():
        return False
    if rf.run_id is None:
        return any(base.glob("*/*/master_*.xlsx"))

    conditions = {alloc.device(e).condition for e in alloc.runs[rf.run_id].cards.values()}
    modes = {_MASTER_MODE_BY_MAJOR[m] for m in rf.majors if m in _MASTER_MODE_BY_MAJOR}
    if {"n03", "n05"}.issubset(rf.majors):
        modes.add("JG_H2")
    if not modes:
        return False

    # A single leftover workbook is not a completed run. Require the exact
    # master for every mode that the discovered input files can produce, in
    # every condition assigned to this run. ``WP`` remains optional because
    # a script workbook may legitimately contain no States sheet.
    return all(
        (base / condition / mode / f"master_{mode}.xlsx").is_file()
        for condition in conditions
        for mode in modes
    )


def analyze_all(root: str | Path, out_root: str | Path, alloc: Allocation = DEFAULT,
                make_plots: bool = True, style_path: Path | None = None,
                timepoints: list[str] | None = None, skip_existing: bool = False,
                workers: int = 1, dpi: int | None = PLOT_DPI,
                progress: Callable[[int, int, str], None] | None = None,
                log: Callable[[str], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> list[RunResult]:
    """Discover runs under ``root`` and analyze each one.

    ``timepoints`` restricts the work to those days -- pass ``["3D"]`` when
    a new day has been measured and the earlier days' output should be left
    alone. ``skip_existing`` additionally skips any run that already has
    masters written, so a re-run only fills in what is missing.

    ``workers`` > 1 fans the runs out across processes. Plotting is ~83% of
    the cost and is CPU-bound, so this is where the wall-clock time goes;
    the runs are independent (own input folder, own output subtree) so there
    is nothing to coordinate. With ``workers=1`` everything stays in-process
    and the log streams live.

    ``progress`` is called as ``progress(done, total, label)``, so a GUI can
    show a bar over what is otherwise a multi-minute job.

    ``cancelled`` is polled between serial runs and between completed
    parallel jobs. Pending parallel futures are cancelled; workers already
    executing a run are allowed to finish without corrupting their output.
    """
    emit = log or (lambda _m: None)
    runs, warnings = discover(root, alloc, timepoints=timepoints)
    for w in warnings:
        emit(f"WARNING: {w}")
    if not runs:
        emit("no runs discovered"
             + (f" for timepoint(s) {', '.join(timepoints)}" if timepoints else ""))
        return []

    if skip_existing:
        keep = [rf for rf in runs if not has_output(rf, out_root, alloc)]
        for rf in runs:
            if rf not in keep:
                emit(f"skipping {rf.timepoint}/{rf.run_key} — already analyzed")
        runs = keep
        if not runs:
            emit("nothing to do: every discovered run already has output")
            return []

    emit(f"analyzing {len(runs)} runs, {sum(len(r.results) for r in runs)} result files")
    for clash in _condition_clashes(runs, alloc):
        emit(f"WARNING: {clash}")

    if workers and workers > 1 and len(runs) > 1:
        return _analyze_parallel(runs, out_root, alloc, make_plots, style_path,
                                 dpi, workers, progress, emit, cancelled)

    results: list[RunResult] = []
    for i, rf in enumerate(runs):
        if cancelled is not None and cancelled():
            emit("analysis cancelled; output written so far is kept")
            raise AnalysisCancelled("analysis cancelled")
        if progress is not None:
            progress(i, len(runs), f"{rf.timepoint} / {rf.run_key}")
        results.append(analyze_run(rf, out_root, alloc, make_plots=make_plots,
                                   style_path=style_path, dpi=dpi, log=log))
    if progress is not None:
        progress(len(runs), len(runs), "done")
    return results


def _analyze_parallel(runs, out_root, alloc, make_plots, style_path, dpi,
                      workers: int, progress, emit, cancelled=None) -> list[RunResult]:
    """Run the folders across processes.

    The runs are independent -- each reads its own folder and writes its own
    output subtree -- so this is a straight fan-out. Plotting dominates the
    cost and is CPU-bound, which is exactly what extra processes help with
    (threads would not, because of the GIL).

    Progress is reported as runs *complete*, not as they start, since with a
    pool there is no meaningful "current" run.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    workers = max(1, min(workers, len(runs)))
    swapped = ensure_multiprocessing_executable()
    if swapped:
        emit(f"spawning workers with {swapped} (pythonw cannot host them)")
    emit(f"running {len(runs)} runs across {workers} processes")

    payload = [(rf, str(out_root), alloc, make_plots, style_path, dpi) for rf in runs]
    results: list[RunResult] = []
    # A run key repeats at every timepoint (for example ``25C_H2``). Tracking
    # only the key made a broken-pool fallback incorrectly treat the same run
    # on every other day as already complete.
    completed: set[tuple[str, str, str]] = set()
    done = 0
    if progress is not None:
        progress(0, len(runs), f"starting {workers} workers")

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, p): p[0] for p in payload}
            for fut in as_completed(futures):
                if cancelled is not None and cancelled():
                    for pending in futures:
                        pending.cancel()
                    emit("analysis cancelled; waiting for active workers to finish")
                    raise AnalysisCancelled("analysis cancelled")
                rf = futures[fut]
                try:
                    result, lines = fut.result()
                except BrokenProcessPool:
                    raise                       # pool-level: fall back below
                except Exception as exc:        # one bad run; keep the others
                    emit(f"[{rf.timepoint}/{rf.run_key}] FAILED: "
                         f"{type(exc).__name__}: {exc}")
                    result = RunResult(rf.timepoint, rf.run_key, rf.run_id,
                                       rf.path, {}, error=str(exc))
                else:
                    for ln in lines:
                        emit(ln)
                results.append(result)
                completed.add(_run_identity(rf))
                done += 1
                if progress is not None:
                    progress(done, len(runs), f"{rf.timepoint} / {rf.run_key} finished")
    except BrokenProcessPool as exc:
        # The pool itself died -- typically because the interpreter hosting
        # it cannot spawn children. Falling back keeps the sweep alive
        # instead of losing every run that was in flight.
        emit(f"WARNING: parallel execution failed ({exc}). "
             f"Falling back to one process at a time — this is slower but reliable.")
        remaining = [rf for rf in runs if _run_identity(rf) not in completed]
        for i, rf in enumerate(remaining):
            if cancelled is not None and cancelled():
                emit("analysis cancelled; output written so far is kept")
                raise AnalysisCancelled("analysis cancelled")
            if progress is not None:
                progress(len(completed) + i, len(runs),
                         f"{rf.timepoint} / {rf.run_key} (serial fallback)")
            results.append(analyze_run(rf, out_root, alloc, make_plots=make_plots,
                                       style_path=style_path, dpi=dpi, log=emit))
        if progress is not None:
            progress(len(runs), len(runs), "done")

    return results


def _condition_clashes(runs: list[RunFolder], alloc: Allocation) -> list[str]:
    """Runs sharing an output folder must cover disjoint conditions.

    Masters are written per condition, so two runs landing in the same
    ``<timepoint>/<out_group>`` with a condition in common would have the
    second silently overwrite the first. That is safe today (the control
    sets split 25/35 C vs 45/60 C) but must not regress unnoticed.
    """
    seen: dict[tuple[str, str, str], str] = {}
    problems: list[str] = []
    for rf in runs:
        if rf.run_id is None:
            continue
        for esn in alloc.runs[rf.run_id].cards.values():
            key = (rf.timepoint, rf.out_group, alloc.device(esn).condition)
            prior = seen.setdefault(key, rf.run_key)
            if prior != rf.run_key:
                msg = (f"{rf.timepoint}/{rf.out_group}: '{prior}' and '{rf.run_key}' "
                       f"both write condition {key[2]} -- masters would overwrite")
                if msg not in problems:
                    problems.append(msg)
    return problems
