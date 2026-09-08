"""Render comparison curves.

One line per value of the axis being varied, mean +/- AVDEV, against study
day. Two palettes are offered: sensorlab's academic-paper palette (the
default, so a figure from this tab sits next to a pipeline figure without
looking foreign) and the project's own green identity, where the ordered
axes read light-to-dark and unordered categories get distinct hues.

A single-timepoint study (baseline only) draws markers rather than lines,
so the plot is honest about there being no trend yet instead of showing a
flat line that looks like "no degradation". That override wins over the
requested draw mode -- a line through one point is invisible.

Typography and frame come from sensorlab's publication style, so a figure
saved here matches the ones the pipeline produces: boxed axes, inward major
and minor ticks, and bold axis titles at paper sizes. Everything the user
can steer from the GUI travels in one :class:`PlotOptions`, which is also
what gets persisted -- the plotting code never reads settings itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from .comparison import AXIS_LABELS, ComparisonSpec  # noqa: E402

#: Ordered axes read light -> dark; unordered categories get their own hues.
_GREEN_RAMP = ("#a8ddb5", "#5fbb7a", "#2b9348", "#0f5132")
_CATEGORY_COLORS = {
    "ParC_only": "#7f8c8d",
    "Pd_only": "#2b9348",
    "PdParC_noH2": "#4c8faf",
    "PdParC_H2": "#0f5132",
}

#: Used only if sensorlab is unreachable, so the paper palette still works.
_FALLBACK_PUB_PALETTE = ("#000000", "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
                         "#ff7f0e", "#e377c2", "#17becf", "#8c564b", "#bcbd22")

_CURVE_UNITS = {"temp": "°C", "rh": "%RH"}

#: Fallback legend size if sensorlab is unavailable -- the old look, not a crash.
_PLAIN_LEGEND_SIZE = 9

#: The x axis is always the study timeline, whatever the metric.
DEFAULT_XLABEL = "Study day"

PALETTES = ("paper", "green")
DRAW_MODES = ("line+points", "line", "points")

#: How each draw mode maps to matplotlib. ``marker`` is forced on anyway when
#: the study has a single timepoint.
_DRAW = {
    "line+points": {"linestyle": "-", "marker": "o"},
    "line": {"linestyle": "-", "marker": ""},
    "points": {"linestyle": "none", "marker": "o"},
}


@dataclass
class PlotOptions:
    """Everything the Comparisons tab lets the user steer, in one place.

    Defaults reproduce the tab's behaviour before any of it was editable, so
    ``PlotOptions()`` is always a safe argument.
    """

    #: "" = use the spec's generated title.
    title: str = ""
    title_bold: bool = True
    title_italic: bool = False
    #: "" = the axis keeps its automatic label.
    xlabel: str = ""
    ylabel: str = ""
    draw: str = "line+points"
    palette: str = "paper"
    #: 0 = let matplotlib choose the tick interval.
    x_step: float = 0.0
    y_step: float = 0.0
    minor_ticks: bool = True
    grid: bool = True
    #: Curve values to leave out. Compared stringified, so a saved "60" still
    #: matches a numeric 60 read back from the data later.
    hidden: list = field(default_factory=list)

    def hides(self, value) -> bool:
        return str(value) in {str(h) for h in self.hidden}


def _paper_style():
    """sensorlab's academic-paper style, read from *this project's* style file.

    Deliberately ``runner.STYLE_PATH`` and not ``~/.sensorlab/plot_style.json``,
    for the same reason :func:`qfn_aging.runner._own_plot_style` exists: the
    two apps keep independent styles. The bundled file carries no ``pub_*``
    keys, so those fall back to sensorlab's paper defaults.

    Returns ``None`` if sensorlab cannot be imported, leaving the plot
    unstyled rather than breaking the tab.
    """
    try:
        from sensorlab.plotting.style import load_style
    except Exception:
        return None
    from .runner import STYLE_PATH

    style = load_style(STYLE_PATH)
    # sensorlab drops in-figure titles because papers use captions; here the
    # title is how you tell which comparison you are looking at, so keep it
    # and let the paper sizing apply to it.
    style.pub_show_title = True
    return style


def _paper_colors(style, n: int) -> list:
    """sensorlab's categorical paper palette, n colours long."""
    if style is not None:
        return [style.pub_color(i) for i in range(n)]
    pal = _FALLBACK_PUB_PALETTE
    return [pal[i % len(pal)] for i in range(n)]


def _green_colors(spec: ComparisonSpec, values: list) -> list:
    if spec.vary == "category":
        return [_CATEGORY_COLORS.get(str(v), "#0f5132") for v in values]
    n = max(len(values), 1)
    if n <= len(_GREEN_RAMP):
        step = max(1, len(_GREEN_RAMP) // n)
        return [_GREEN_RAMP[min(i * step, len(_GREEN_RAMP) - 1)] for i in range(n)]
    cmap = plt.get_cmap("Greens")
    return [cmap(0.30 + 0.65 * i / (n - 1)) for i in range(n)]


def _colors(spec: ComparisonSpec, values: list, options: PlotOptions, style) -> list:
    """Colour per curve value, assigned over *all* values.

    Hiding a curve must not recolour the ones still on screen, so the caller
    passes the full value list and drops the hidden ones afterwards.
    """
    if options.palette == "green":
        return _green_colors(spec, values)
    return _paper_colors(style, len(values))


def _curve_label(spec: ComparisonSpec, value) -> str:
    unit = _CURVE_UNITS.get(spec.vary, "")
    return f"{value}{unit}" if unit else str(value)


def _style_title(ax, options: PlotOptions, style) -> None:
    """Size the title from the paper style, then apply the user's B / I."""
    if style is not None:
        ax.title.set_fontsize(style.pub_title_size)
    ax.title.set_fontweight("bold" if options.title_bold else "normal")
    ax.title.set_fontstyle("italic" if options.title_italic else "normal")


def title_for(spec: ComparisonSpec, options: PlotOptions | None = None) -> str:
    """The title actually drawn: the user's override, else the generated one."""
    override = (options.title if options else "").strip()
    return override or spec.title()


def auto_ylabel(spec: ComparisonSpec, ylabel: str | None = None) -> str:
    """The y label used when the user has not overridden it."""
    return ylabel or spec.metric


def axis_labels_for(spec: ComparisonSpec, ylabel: str | None = None,
                    options: PlotOptions | None = None) -> tuple[str, str]:
    """(x, y) axis labels actually drawn -- overrides win, blanks fall back."""
    x_override = (options.xlabel if options else "").strip()
    y_override = (options.ylabel if options else "").strip()
    return (x_override or DEFAULT_XLABEL,
            y_override or auto_ylabel(spec, ylabel))


def plot_comparison(curves, spec: ComparisonSpec, ylabel: str | None = None,
                    ax=None, options: PlotOptions | None = None) -> Figure:
    """Draw one comparison. ``curves`` is the output of ``build_curves``."""
    options = options or PlotOptions()
    if ax is None:
        # Figure() rather than plt.subplots(): pyplot keeps every figure it
        # creates in a global registry until explicitly closed, which leaks
        # when the GUI redraws on each selection change.
        fig = Figure(figsize=(9, 5.5))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    style = _paper_style()

    if curves.empty:
        ax.text(0.5, 0.5, "No data for this combination",
                ha="center", va="center", transform=ax.transAxes,
                color="#9aa5b1", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_for(spec, options))
        _style_title(ax, options, style)
        return fig

    all_values = list(dict.fromkeys(curves["curve"].tolist()))
    colors = _colors(spec, all_values, options, style)
    shown = [(c, v) for c, v in zip(colors, all_values) if not options.hides(v)]
    single_point = curves["day"].nunique() <= 1

    # With one timepoint every curve sits on the same x and the markers and
    # error bars pile up illegibly. Spread them out so each group can still
    # be read -- the offsets vanish as soon as there is a second day.
    n = max(len(shown), 1)
    offsets = ([(i - (n - 1) / 2) * (0.6 / max(n - 1, 1)) for i in range(n)]
               if single_point and n > 1 else [0.0] * n)

    draw = _DRAW.get(options.draw, _DRAW["line+points"])
    # a line through a single point draws nothing -- markers regardless
    marker = "o" if single_point else draw["marker"]
    linestyle = "none" if single_point else draw["linestyle"]

    for (color, value), dx in zip(shown, offsets):
        part = curves[curves["curve"] == value].sort_values("day")
        ax.errorbar(
            part["day"] + dx, part["mean"], yerr=part["avdev"].abs(),
            label=_curve_label(spec, value), color=color,
            marker=marker, markersize=7 if single_point else 5,
            linestyle=linestyle,
            linewidth=2.0, capsize=3, elinewidth=1.2,
        )

    x_label, y_label = axis_labels_for(spec, ylabel, options)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_for(spec, options))

    if options.x_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(options.x_step))
    if options.y_step > 0 and ax.get_yscale() != "log":
        ax.yaxis.set_major_locator(MultipleLocator(options.y_step))

    if style is not None:
        # after the labels: apply_publication sizes them and builds the frame.
        # Minor ticks are the user's call here, not the style file's.
        style.pub_minor_ticks = options.minor_ticks
        style.apply_publication(ax)
        style.apply_legend(ax, fontsize=_PLAIN_LEGEND_SIZE,
                           title=AXIS_LABELS[spec.vary])
    else:
        ax.legend(title=AXIS_LABELS[spec.vary], frameon=False,
                  fontsize=_PLAIN_LEGEND_SIZE)
    _style_title(ax, options, style)

    if options.grid:
        ax.grid(True, which="major", ls=":", alpha=0.35, color="#888888")

    if single_point:
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.text(0.99, 0.02, "baseline only — no trend yet",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#9aa5b1", style="italic")

    fig.tight_layout()
    return fig


def figure_dpi(fallback: float = 110) -> float:
    """Saved-figure resolution from the paper style (300), else *fallback*."""
    style = _paper_style()
    return float(style.dpi) if style is not None else float(fallback)


def save_comparison(curves, spec: ComparisonSpec, out_path: str,
                    ylabel: str | None = None, dpi: float | None = None,
                    options: PlotOptions | None = None) -> str:
    """Render one comparison straight to a file (no pyplot state involved)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plot_comparison(curves, spec, ylabel=ylabel, options=options)
    FigureCanvasAgg(fig)
    fig.savefig(out_path, dpi=figure_dpi() if dpi is None else dpi)
    return out_path
