"""Per-method comparison plots: trade-off scatter and summary bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from .style import INK, METHOD_COLORS, MUTED


def tradeoff(
    ax: Axes,
    summary: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: dict[str, str] = METHOD_COLORS,
    size: float = 28,
    highlight: str = "Segger",
) -> None:
    """One point per method with direct labels, e.g. PMR against specificity.

    The dotted diagonal of the axes marks the direction in which both readouts improve.

    Parameters
    ----------
    summary
        One row per method (index), with columns ``x`` and ``y``; rows with NaN are skipped.
    highlight
        Method drawn larger, with its label in its colour.
    """
    summary = summary.dropna(subset=[x, y])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls=(0, (1, 2)), color=MUTED, lw=0.7, zorder=0)
    texts = []
    for method, row in summary.iterrows():
        big = method == highlight
        color = colors.get(method, MUTED)
        ax.scatter(row[x], row[y], s=size * (1.6 if big else 1), color=color, lw=0, zorder=3)
        texts.append(ax.text(row[x], row[y], method, color=color if big else INK, zorder=4))
    pad_x = 0.12 * np.ptp(summary[x]) or 1.0
    pad_y = 0.12 * np.ptp(summary[y]) or 0.1
    ax.set_xlim(summary[x].min() - pad_x, summary[x].max() + pad_x)
    ax.set_ylim(summary[y].min() - pad_y, summary[y].max() + pad_y)
    adjust_text(
        texts, x=summary[x].to_numpy(), y=summary[y].to_numpy(), ax=ax,
        expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5),
    )
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)


def bars(
    ax: Axes,
    values: pd.Series,
    low: pd.Series | None = None,
    high: pd.Series | None = None,
    title: str | None = None,
    log: bool = False,
    reference: str | None = "Segger",
    colors: dict[str, str] = METHOD_COLORS,
    labels: dict[str, str] | None = None,
    rotation: float = 90,
) -> None:
    """One bar per method with error bars and the reference method as a dashed line.

    Parameters
    ----------
    values
        Value per method, in plotting order. Methods with NaN keep their slot.
    low, high
        Lower and upper end of the error bar per method (e.g. quartiles).
    log
        Log-scaled axis; the axis starts and ends on a decade.
    reference
        Method whose value is marked across the panel, None for no line.
    labels
        Short tick labels, e.g. ``{"10x Nucleus": "10x Nuc."}``.
    rotation
        Angle of the tick labels.
    """
    x = np.arange(len(values))
    finite = values.notna().to_numpy()
    base = 0.0
    if log:
        span = pd.concat([values, low if low is not None else values, high if high is not None else values])
        span = span[span > 0]
        lo_dec, hi_dec = np.floor(np.log10(span.min())), np.ceil(np.log10(span.max()))
        base = 10.0**lo_dec
        ax.set_yscale("log")
        ax.set_ylim(base, 10.0**hi_dec)
        ax.yaxis.set_major_locator(LogLocator(numticks=int(hi_dec - lo_dec) + 1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    heights = values.to_numpy(dtype=float) - base
    ax.bar(
        x[finite], heights[finite], bottom=base, width=0.72,
        color=[colors.get(m, MUTED) for m in values.index[finite]], lw=0,
    )
    if low is not None and high is not None:
        low, high = low.reindex(values.index), high.reindex(values.index)
        err = np.clip(np.vstack([values - low, high - values]), 0, None)[:, finite]
        ax.errorbar(x[finite], values[finite], yerr=err, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6)
    if reference in values.index and np.isfinite(values[reference]):
        ax.axhline(values[reference], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    if not log:
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xticks(x)
    ax.set_xticklabels([(labels or {}).get(m, m) for m in values.index], rotation=rotation)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, len(values) - 0.4)
    if title:
        ax.set_title(title, pad=4)
