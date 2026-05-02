from __future__ import annotations
from typing import Any


# ----------------------------------------------------------------------------
# SAFE NUMBER FORMATTERS
# ----------------------------------------------------------------------------

def format_number(value: Any, decimals: int = 1) -> str:
    try:
        if value is None:
            return "—"
        num = float(value)
        return f"{num:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_int(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "—"


def format_percent(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{int(round(float(value)))}%"
    except (TypeError, ValueError):
        return "—"


# ----------------------------------------------------------------------------
# STATION NAME FORMATTER
# ----------------------------------------------------------------------------

def format_station_name(raw_name: str) -> str:
    if not raw_name:
        return "Unknown"

    cleaned = str(raw_name).replace("_", " ").strip()

    # remove unwanted suffixes
    for suffix in [" Delhi DPCC", " Delhi CPCB", " Delhi IMD", " Delhi IITM"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]

    cleaned = cleaned.replace("(T3)", "T3")

    return " ".join(cleaned.split())


# ----------------------------------------------------------------------------
# DATE FORMATTER
# ----------------------------------------------------------------------------

def format_date(date_obj) -> str:
    try:
        return date_obj.strftime("%d %b %Y")
    except Exception:
        return "—"