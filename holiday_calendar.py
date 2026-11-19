"""Official rest days from the annual production calendars, five-day week."""

import numpy as np
import pandas as pd

TYPES = {
    "new_year": ["01-01", "01-02"],
    "mourning": ["01-20"],
    "womens": ["03-08"],
    "novruz": ["03-20", "03-21", "03-22", "03-23", "03-24"],
    "victory": ["05-09"],
    "independence": ["05-28"],
    "salvation": ["06-15"],
    "army": ["06-26"],
    "victory_nov": ["11-08"],
    "flag": ["11-09"],
    "solidarity": ["12-31"],
}

NAMED = {
    "ramadan": ["2024-04-10", "2024-04-11", "2025-03-30", "2025-03-31", "2026-03-20", "2026-03-21"],
    "qurban": ["2024-06-16", "2024-06-17", "2025-06-06", "2025-06-07", "2026-05-27", "2026-05-28"],
}

TRANSFERRED = {
    "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-03-25", "2024-03-26",
    "2024-06-18", "2024-06-19",
    "2024-11-11",
    "2025-01-03",
    "2025-03-10", "2025-03-25", "2025-03-26",
    "2025-04-01",
    "2025-06-09", "2025-06-16",
    "2025-11-10", "2025-11-11",
    "2026-03-09", "2026-03-25", "2026-03-26", "2026-03-27", "2026-03-30",
    "2026-05-11", "2026-05-29",
    "2026-11-10",
}


def _typed_days():
    out = {}
    for year in (2024, 2025, 2026):
        for name, month_days in TYPES.items():
            for month_day in month_days:
                out[pd.Timestamp(f"{year}-{month_day}")] = name
    for name, days in NAMED.items():
        for day in days:
            out[pd.Timestamp(day)] = name
    for day in TRANSFERRED:
        stamp = pd.Timestamp(day)
        out.setdefault(stamp, "transferred")
    return out


DAY_TYPE = _typed_days()
REST_DAYS = np.array(sorted(DAY_TYPE), dtype="datetime64[ns]")


def type_of(dates):
    stamps = pd.to_datetime(dates)
    return stamps.map(DAY_TYPE).fillna("").to_numpy()


def days_to_rest(dates):
    values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    if len(REST_DAYS) == 0:
        return np.zeros(len(values))
    idx = np.searchsorted(REST_DAYS, values)
    prev_i = np.clip(idx - 1, 0, len(REST_DAYS) - 1)
    next_i = np.clip(idx, 0, len(REST_DAYS) - 1)
    prev_gap = (values - REST_DAYS[prev_i]) / np.timedelta64(1, "D")
    next_gap = (REST_DAYS[next_i] - values) / np.timedelta64(1, "D")
    use_prev = prev_gap <= next_gap
    signed = np.where(use_prev, -prev_gap, next_gap).astype(float)
    on_rest = np.isin(values, REST_DAYS)
    signed = np.where(on_rest, 0.0, signed)
    return np.clip(signed, -14, 14)
