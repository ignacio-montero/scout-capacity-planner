"""Number and date formatters: one function per quantity (DESIGN_SYSTEM.md section 3.3).

Pure functions with no Streamlit import, shared by the app (labels, tables,
KPI cards), the chart titles in :mod:`scout_planner.charts` and the README
script. Having exactly one formatter per quantity is what keeps "94.3%" from
showing up as "0.943" on one page and "94%" on another.

Signs: every signed value uses the ASCII hyphen ``-``. ``st.metric`` decides
the direction of its delta arrow from a leading ``-``; a Unicode minus would
show a drop as a green up-arrow.
"""

from __future__ import annotations

import datetime as dt
import math

__all__ = [
    "fmt_cost_k",
    "fmt_count",
    "fmt_days",
    "fmt_days_delta",
    "fmt_duration",
    "fmt_growth",
    "fmt_money",
    "fmt_month",
    "fmt_pct",
    "fmt_pts",
    "fmt_when",
]

DASH = "—"  # shown for a value that doesn't exist (not started, no results)


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_pct(fraction: float | None, decimals: int = 1) -> str:
    """A rate stored as a fraction: ``0.943 -> "94.3%"``."""
    if _missing(fraction):
        return DASH
    return f"{fraction * 100:.{decimals}f}%"


def fmt_pts(delta: float | None) -> str:
    """Difference between two rates in percentage points: ``-0.016 -> "-1.6 pts"``.

    Always signed (``"+0.4 pts"``) except a difference that rounds to zero,
    which reads ``"0.0 pts"`` so it carries neither an up nor a down arrow.
    """
    if _missing(delta):
        return DASH
    points = round(delta * 100, 1)
    if points == 0:
        return "0.0 pts"
    return f"{points:+.1f} pts"


def fmt_cost_k(value: float | None, *, signed: bool = False) -> str:
    """Money in thousands: ``1_240_000 -> "1,240k"``, ``4_500 -> "4.5k"``.

    Under 10k one decimal is kept, so small amounts don't all read "0k".
    ``signed=True`` adds ``+`` to positive values (for differences).
    """
    if _missing(value):
        return DASH
    k = value / 1000
    decimals = 0 if abs(k) >= 10 else 1
    sign = "+" if signed else ""
    text = f"{k:{sign},.{decimals}f}k"
    return "0k" if text in ("+0.0k", "-0.0k", "0.0k") else text


def fmt_money(value: float | None) -> str:
    """Money in a form or table cell: plain integer with separator, ``4500 -> "4,500"``."""
    if _missing(value):
        return DASH
    return f"{value:,.0f}"


def fmt_days(value: float | None, *, short: bool = False) -> str:
    """Turnaround: ``17.0 -> "17.0 days"`` (KPI) or ``"17.0 d"`` (tables, axes)."""
    if _missing(value):
        return DASH
    return f"{value:.1f} d" if short else f"{value:.1f} days"


def fmt_days_delta(value: float | None, *, short: bool = False) -> str:
    """A signed difference in days: ``3.0 -> "+3.0 days"``."""
    if _missing(value):
        return DASH
    unit = "d" if short else "days"
    if round(value, 1) == 0:
        return f"0.0 {unit}"
    return f"{value:+.1f} {unit}"


def fmt_growth(value: float | None) -> str:
    """Growth multiplier, up to one decimal, trailing zero dropped: ``4.0 -> "4x"``."""
    if _missing(value):
        return DASH
    text = f"{round(value, 1):.1f}".rstrip("0").rstrip(".")
    return f"{text}x"


def fmt_count(value: float | None, *, signed: bool = False) -> str:
    """A count with thousands separator: ``3120 -> "3,120"``."""
    if _missing(value):
        return DASH
    rounded = round(value)
    sign = "+" if signed and rounded != 0 else ""
    return f"{rounded:{sign},d}"


def fmt_month(day: dt.date | dt.datetime | None, *, with_year: bool = True) -> str:
    """``2027-03-01 -> "Mar 2027"`` (or ``"Mar"``)."""
    if day is None:
        return DASH
    return f"{day:%b %Y}" if with_year else f"{day:%b}"


def fmt_when(moment: dt.datetime | None) -> str:
    """A timestamp: ``"03 Oct 10:15"``."""
    if moment is None:
        return DASH
    return f"{moment:%d %b %H:%M}"


def fmt_duration(seconds: float | None) -> str:
    """Elapsed time: ``134 -> "2m 14s"``, ``18 -> "18s"``, ``3720 -> "1h 2m"``."""
    if _missing(seconds):
        return DASH
    total = max(0, round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
