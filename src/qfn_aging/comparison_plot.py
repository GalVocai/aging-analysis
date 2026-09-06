"""Render comparison curves.

One line per value of the axis being varied, mean +/- AVDEV, against study
day. Colours follow the project's green identity: the ordered axes
(temperature, RH) use a light-to-dark green ramp so the reading order is
obvious, while categories -- which have no natural order -- get distinct
hues.

A single-timepoint study (baseline only) draws markers rather than lines,
so the plot is honest about there being no trend yet instead of showing a
flat line that looks like "no degradation".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .comparison import AXIS_LABELS, ComparisonSpec  # noqa: E402

#: Ordered axes read light -> dark; unordered categories get their own hues.
_GREEN_RAMP = ("#a8ddb5", "#5fbb7a", "#2b9348", "#0f5132")
_CATEGORY_COLORS = {
    "ParC_only": "#7f8c8d",
    "Pd_only": "#2b9348",
    "PdParC_noH2": "#4c8faf",
    "PdParC_H2": "#0f5132",
}

_CURVE_UNITS = {"temp": "°C", "rh": "%RH"}


def _colors(spec: ComparisonSpec, values: list) -> list[str]:
    if spec.vary == "category":
        return [_CATEGORY_COLORS.get(str(v), "#0f5132") for v in values]
    n = max(len(values), 1)
    if n <= len(_GREEN_RAMP):
        step = max(1, len(_GREEN_RAMP) // n)
        return [_GREEN_RAMP[min(i * step, len(_GREEN_RAMP) - 1)] for i in range(n)]
    cmap = plt.get_cmap("Greens")
    return [cmap(0.30 + 0.65 * i / (n - 1)) for i in range(n)]


def _curve_label(spec: ComparisonSpec, value) -> str:
    unit = _CURVE_UNITS.get(spec.vary, "")
    return f"{value}{unit}" if unit else str(value)


def plot_comparison(curves, spec: ComparisonSpec, ylabel: str | None = None,
                    ax=None) -> Figure:
    """Draw one comparison. ``curves`` is the output of ``build_curves``."""
    if ax is None:
        # Figure() rather than plt.subplots(): pyplot keeps every figure it
        # creates in a global registry until explicitly closed, which leaks
        # when the GUI redraws on each selection change.
        fig = Figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if curves.empty:
        ax.text(0.5, 0.5, "No data for this combination",
                ha="center", va="center", transform=ax.transAxes,
                color="#9aa5b1", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(spec.title())
        return fig

    values = list(dict.fromkeys(curves["curve"].tolist()))
    single_point = curves["day"].nunique() <= 1

    # With one timepoint every curve sits on the same x and the markers and
    # error bars pile up illegibly. Spread them out so each group can still
    # be read -- the offsets vanish as soon as there is a second day.
    n = max(len(values), 1)
    offsets = ([(i - (n - 1) / 2) * (0.6 / max(n - 1, 1)) for i in range(n)]
               if single_point and n > 1 else [0.0] * n)

    for color, value, dx in zip(_colors(spec, values), values, offsets):
        part = curves[curves["curve"] == value].sort_values("day")
        ax.errorbar(
            part["day"] + dx, part["mean"], yerr=part["avdev"].abs(),
            label=_curve_label(spec, value), color=color,
            marker="o", markersize=7 if single_point else 5,
            linestyle="none" if single_point else "-",
            linewidth=2.0, capsize=3, elinewidth=1.2,
        )

    ax.set_xlabel("Study day")
    ax.set_ylabel(ylabel or spec.metric)
    ax.set_title(spec.title())
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(title=AXIS_LABELS[spec.vary], frameon=False, fontsize=9)

    if single_point:
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.text(0.99, 0.02, "baseline only — no trend yet",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#9aa5b1", style="italic")

    fig.tight_layout()
    return fig


def save_comparison(curves, spec: ComparisonSpec, out_path: str,
                    ylabel: str | None = None, dpi: int = 110) -> str:
    """Render one comparison straight to a file (no pyplot state involved)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plot_comparison(curves, spec, ylabel=ylabel)
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=dpi)
    return out_path
