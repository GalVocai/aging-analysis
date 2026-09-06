"""Find the measurement run folders and result files in the aging data tree.

Layout this expects (verified against the real tree 2026-07-27)::

    <root>/
        Baseline/                       <- timepoint folder
            25C_H2/<run folder>/        <- run folder holds the result files
                E<id>_<date>_N0<n>_n0<major>_<MODE>_.xlsx
                ...
                <script>.xlsx           <- no _n0X_ token
            control_set1/<run folder>/
        3D/ 7D/                         <- later timepoints, same layout

Discovery is deliberately tolerant: unknown timepoint or run folders are
reported rather than raising, so a new timepoint appearing on disk with a
slightly different name shows up as a warning instead of vanishing.

This module only *locates and labels* files. It does not read measurement
data and it makes no judgement about data quality -- metrics come from
sensorlab's analysis, and which transistors count is decided downstream by
the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .allocation import DEFAULT, Allocation
from .schedule import MAJOR_NAMES, normalize_timepoint, sort_timepoints

# `E1195_2026_07_25_N02_n03_JG_.xlsx` -> major 3, mode token "JG"
_RESULT_RE = re.compile(r"_n0*(\d+)_([A-Za-z_]+?)_?\.xlsx$", re.IGNORECASE)
_TEMP_RUN_RE = re.compile(r"^(\d+)C[_-]?H2$", re.IGNORECASE)
_CONTROL_RUN_RE = re.compile(r"^control[_-]?set[_-]?(\d+)$", re.IGNORECASE)


def _is_junk(name: str) -> bool:
    """Excel lock files and hidden temporaries."""
    return name.startswith("~$") or name.startswith(".")


@dataclass(frozen=True)
class ResultFile:
    path: Path
    major: str          # "n01".."n07"
    mode_token: str     # raw token from the filename, e.g. "JG", "SE_STATE"

    @property
    def mode(self) -> str:
        """Canonical mode name for this major (see schedule.MAJOR_NAMES)."""
        return MAJOR_NAMES.get(self.major, self.mode_token)


@dataclass(frozen=True)
class RunFolder:
    path: Path
    timepoint: str            # "baseline", "3D", ...
    run_key: str              # folder name, e.g. "25C_H2", "control_set1"
    run_id: str | None        # allocation run id, e.g. "H2_run1_25C"; None if unmatched
    results: list[ResultFile] = field(default_factory=list)
    script: Path | None = None

    @property
    def out_group(self) -> str:
        """Output folder name for this run's analysis.

        The control sets are split across two instrument runs only because
        of card capacity -- analytically they are one population, so they
        share a single ``control`` output folder. Their conditions are
        disjoint (set 1 = 25/35 C, set 2 = 45/60 C), so nothing collides.
        Hydrogen runs keep their own folder, one per temperature.
        """
        if self.run_id and self.run_id.startswith("control"):
            return "control"
        return self.run_key

    @property
    def majors(self) -> list[str]:
        return sorted({r.major for r in self.results})

    def result(self, major: str) -> ResultFile | None:
        for r in self.results:
            if r.major == major:
                return r
        return None


def _match_run_id(run_key: str, alloc: Allocation) -> str | None:
    """Map a run folder name to an allocation run id, derived from the
    allocation itself rather than a hardcoded table."""
    m = _TEMP_RUN_RE.match(run_key)
    if m:
        temp = int(m.group(1))
        for run in alloc.runs.values():
            if run.kind == "hydrogen" and run.temp_c == temp:
                return run.run_id
        return None

    m = _CONTROL_RUN_RE.match(run_key)
    if m:
        n = int(m.group(1))
        controls = sorted(
            (r for r in alloc.runs.values() if r.kind == "control"),
            key=lambda r: (r.day, r.order),
        )
        if 1 <= n <= len(controls):
            return controls[n - 1].run_id
    return None


def _normalize_timepoint(name: str) -> str | None:
    """Any ``baseline`` / ``<n>D`` folder is a timepoint (see schedule.py)."""
    return normalize_timepoint(name)


def _scan_run_folder(folder: Path, timepoint: str, run_key: str,
                     run_id: str | None) -> RunFolder:
    results: list[ResultFile] = []
    script: Path | None = None

    for f in sorted(folder.glob("*.xlsx")):
        if _is_junk(f.name):
            continue
        m = _RESULT_RE.search(f.name)
        if m:
            results.append(ResultFile(
                path=f, major=f"n{int(m.group(1)):02d}", mode_token=m.group(2).strip("_"),
            ))
        else:
            script = f  # the one .xlsx with no _n0X_ token

    return RunFolder(path=folder, timepoint=timepoint, run_key=run_key,
                     run_id=run_id, results=results, script=script)


def discover(root: str | Path, alloc: Allocation = DEFAULT,
             timepoints: list[str] | None = None,
             ) -> tuple[list[RunFolder], list[str]]:
    """Walk the aging tree. Returns (run folders, warnings).

    ``timepoints`` limits the walk to those timepoints (e.g. ``["3D"]`` to
    process only a newly measured day), leaving existing output for the
    others untouched. ``None`` means every timepoint present on disk.
    """
    root = Path(root)
    runs: list[RunFolder] = []
    warnings: list[str] = []
    wanted = set(timepoints) if timepoints else None

    if not root.is_dir():
        return runs, [f"data root does not exist: {root}"]

    for tp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        timepoint = _normalize_timepoint(tp_dir.name)
        if timepoint is None:
            continue  # Haborer/, screening/, anything not a timepoint
        if wanted is not None and timepoint not in wanted:
            continue

        for run_dir in sorted(p for p in tp_dir.iterdir() if p.is_dir()):
            run_id = _match_run_id(run_dir.name, alloc)
            if run_id is None:
                warnings.append(
                    f"{timepoint}/{run_dir.name}: folder name does not match any "
                    f"allocation run -- skipped"
                )
                continue

            # the result files live one level down, in the instrument's own folder
            inner = [p for p in run_dir.iterdir() if p.is_dir()]
            targets = inner or [run_dir]
            found_any = False
            for t in targets:
                rf = _scan_run_folder(t, timepoint, run_dir.name, run_id)
                if rf.results:
                    runs.append(rf)
                    found_any = True
            if not found_any:
                warnings.append(f"{timepoint}/{run_dir.name}: no result files found")

    return runs, warnings


def available_timepoints(root: str | Path) -> list[str]:
    """Timepoint folders present on disk, in study order."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = {_normalize_timepoint(p.name) for p in root.iterdir() if p.is_dir()}
    return sort_timepoints(tp for tp in found if tp)


def verify(runs: list[RunFolder], alloc: Allocation = DEFAULT) -> list[str]:
    """Check discovered runs against what the schedule says should exist.

    Reports missing/unexpected majors per run. Returns a list of problem
    strings -- empty means everything lines up.
    """
    from .schedule import majors_for

    problems: list[str] = []
    for rf in runs:
        if rf.run_id is None:
            continue
        run = alloc.runs[rf.run_id]
        devices = [alloc.device(e) for e in run.cards.values()]

        # a run is homogeneous in h2 by construction; the schedule for the
        # whole run is only well-defined if that holds
        if len({d.h2 for d in devices}) > 1:
            problems.append(f"{rf.timepoint}/{rf.run_key}: mixes H2 and control devices")

        expected = set(majors_for(devices[0], rf.timepoint)) if devices else set()
        found = set(rf.majors)
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing:
            problems.append(f"{rf.timepoint}/{rf.run_key}: missing {', '.join(missing)}")
        if extra:
            problems.append(f"{rf.timepoint}/{rf.run_key}: unexpected {', '.join(extra)}")
        if rf.script is None:
            problems.append(f"{rf.timepoint}/{rf.run_key}: no script workbook found")
    return problems
