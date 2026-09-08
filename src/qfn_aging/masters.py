"""Read sensorlab's master workbooks back into one tidy table.

Layout produced by :mod:`qfn_aging.runner`::

    <analysis_root>/<timepoint>/<run_key>/<condition>/<MODE>/master_<MODE>.xlsx

A master workbook can hold **several sheets**, one per major of the same
kind -- e.g. the JG master of a hydrogen run contains ``JG_n03`` (air),
``JG_n05`` (under H2) and ``JG_n06`` (recovery). Those are different
physical measurements and must never be averaged together, so the sheet
name is carried through and becomes part of the grouping key.

Some masters also key rows by more than (ESN, TNumber): ZOZO emits one row
per settling segment and working-point state. Those sub-keys are folded
into a ``series`` string so that each (metric, series) pair is a distinct
curve rather than being silently averaged across segments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

# Columns that identify a row rather than measure something.
_ID_COLS = {"ESN", "TNumber", "condition", "day"}
# Extra row keys that subdivide a metric into separate curves.
_SERIES_COLS = ("state_Vj", "segment", "Vbg", "part")
CACHE_VERSION = 1
CACHE_DIR = Path.home() / ".qfn_aging" / "cache"


def _series_label(row: pd.Series, cols: list[str]) -> str:
    if not cols:
        return ""
    return "|".join(f"{c}={row[c]}" for c in cols)


def _tidy_sheet(df: pd.DataFrame, timepoint: str, run_key: str, mode: str,
                sheet: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    series_cols = [c for c in _SERIES_COLS if c in df.columns]
    id_cols = set(_ID_COLS) | set(series_cols)
    # flags/notes are diagnostics, not metrics; keep only numeric measurements
    metric_cols = [
        c for c in df.columns
        if c not in id_cols
        and not str(c).startswith("flags")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        return pd.DataFrame()

    out = df.melt(
        id_vars=[c for c in df.columns if c in id_cols],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    out["series"] = (
        out.apply(lambda r: _series_label(r, series_cols), axis=1) if series_cols else ""
    )
    out = out.rename(columns={"ESN": "esn", "TNumber": "tnumber"})
    out["esn"] = out["esn"].astype(str)
    out["timepoint"] = timepoint
    out["run_key"] = run_key
    out["mode"] = mode
    out["sheet"] = sheet

    keep = ["esn", "tnumber", "timepoint", "run_key", "mode", "sheet",
            "series", "metric", "value"]
    return out[[c for c in keep if c in out.columns]].dropna(subset=["value"])


def _master_paths(analysis_root: Path, modes: list[str] | None = None) -> list[Path]:
    return [p for p in sorted(analysis_root.glob("*/*/*/*/master_*.xlsx"))
            if not p.name.startswith("~$")
            and not (modes and p.parent.name not in modes)]


def _fingerprint(analysis_root: Path, paths: list[Path]) -> str:
    """Cheap invalidation key from every master's path, size and mtime."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        record = f"{path.relative_to(analysis_root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def load_masters(analysis_root: str | Path,
                 modes: list[str] | None = None,
                 progress: Callable[[int, int, str], None] | None = None,
                 log: Callable[[str], None] | None = None,
                 ) -> pd.DataFrame:
    """Load every master under ``analysis_root`` into one tidy frame.

    Returns columns: esn, tnumber, timepoint, run_key, mode, sheet, series,
    metric, value. ``condition`` is deliberately NOT taken from the files --
    it comes from this project's roster via
    :func:`qfn_aging.grouping.add_group_keys`, so the roster stays the single
    source of truth.

    ``progress`` is called as ``progress(done, total, label)`` before each
    workbook is read, so a GUI can show real progress rather than freezing.
    Reading ~130 workbooks takes several seconds.

    Unreadable workbooks are skipped so one damaged file does not hide all
    usable results, but every skip is reported through ``log`` when supplied.
    """
    analysis_root = Path(analysis_root)
    frames: list[pd.DataFrame] = []
    emit = log or (lambda _message: None)

    paths = _master_paths(analysis_root, modes)
    total = len(paths)

    for done, path in enumerate(paths):
        mode = path.parent.name
        if progress is not None:
            progress(done, total, f"{path.parents[2].name} / {path.parents[1].name} / {mode}")
        # <root>/<timepoint>/<run_key>/<condition>/<MODE>/master_<MODE>.xlsx
        timepoint = path.parents[3].name
        run_key = path.parents[2].name
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            emit(f"WARNING: could not read {path}: {type(exc).__name__}: {exc}")
            continue
        for sheet, df in sheets.items():
            # sensorlab's population diagnostics, not measurements
            if sheet.endswith("_groups"):
                continue
            tidy = _tidy_sheet(df, timepoint, run_key, mode, sheet)
            if not tidy.empty:
                frames.append(tidy)

    if progress is not None:
        progress(total, total, "done")

    if not frames:
        return pd.DataFrame(columns=["esn", "tnumber", "timepoint", "run_key",
                                     "mode", "sheet", "series", "metric", "value"])
    return pd.concat(frames, ignore_index=True)


def load_masters_cached(analysis_root: str | Path,
                        cache_dir: str | Path = CACHE_DIR,
                        progress: Callable[[int, int, str], None] | None = None,
                        log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Load all masters, reusing a validated local pickle on later launches.

    The cache lives outside OneDrive by default.  It is accepted only when
    every master path, size and modification timestamp still matches, and is
    rebuilt automatically after analysis output changes.
    """
    analysis_root = Path(analysis_root).resolve()
    cache_dir = Path(cache_dir)
    emit = log or (lambda _message: None)
    paths = _master_paths(analysis_root)
    fingerprint = _fingerprint(analysis_root, paths)
    root_id = hashlib.sha256(str(analysis_root).casefold().encode("utf-8")).hexdigest()[:16]
    pickle_path = cache_dir / f"masters_{root_id}.pkl"
    manifest_path = cache_dir / f"masters_{root_id}.json"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (manifest.get("version") == CACHE_VERSION
                 and manifest.get("root") == str(analysis_root)
                 and manifest.get("fingerprint") == fingerprint
                 and pickle_path.is_file())
    except (OSError, ValueError, TypeError):
        valid = False

    if valid:
        try:
            if progress is not None:
                progress(0, 1, "Loading validated local cache")
            tidy = pd.read_pickle(pickle_path)
            if progress is not None:
                progress(1, 1, "done")
            emit(f"Loaded cached master data ({len(tidy):,} rows)")
            return tidy
        except Exception as exc:
            emit(f"WARNING: could not read master cache; rebuilding: "
                 f"{type(exc).__name__}: {exc}")

    tidy = load_masters(analysis_root, progress=progress, log=log)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_pickle = pickle_path.with_suffix(".pkl.tmp")
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tidy.to_pickle(tmp_pickle)
        tmp_manifest.write_text(json.dumps({
            "version": CACHE_VERSION,
            "root": str(analysis_root),
            "fingerprint": fingerprint,
            "workbooks": len(paths),
            "rows": len(tidy),
        }, indent=2), encoding="utf-8")
        tmp_pickle.replace(pickle_path)
        tmp_manifest.replace(manifest_path)
        emit(f"Updated local master cache ({len(tidy):,} rows)")
    except Exception as exc:
        emit(f"WARNING: could not update master cache: {type(exc).__name__}: {exc}")
    return tidy
