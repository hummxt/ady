"""Locked backtest. Folds, metric, and ship rule stay fixed."""

from pathlib import Path

import numpy as np
import pandas as pd

import build_submission as template

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"

FOLDS = (
    ("mimic", "2025-03-01", "2025-07-01", 0, None),
    ("summer", "2025-07-01", "2025-11-01", 0, None),
    ("winter", "2025-11-01", "2026-03-01", 20, None),
)
ROUTE_FOLD = ("route", "2026-02-22", "2026-03-01", 0, (27, 28))
SHIP_MEAN_GAIN = 0.002
SHIP_FOLD_REGRESSION = 0.001
SHIP_ROUTE_REGRESSION = 0.02


def load_train():
    train, _ = template.load_frames()
    return train


def slice_fold(train, fold):
    name, origin, end, min_days, trains = fold
    origin = pd.Timestamp(origin)
    end = pd.Timestamp(end)
    history = train[train["trip_date"] < origin].copy()
    future = train[(train["trip_date"] >= origin) & (train["trip_date"] < end)].copy()
    if trains is not None:
        keep = set(trains)
        history = history[history["train_n"].isin(keep)].copy()
        future = future[future["train_n"].isin(keep)].copy()
    if min_days:
        counts = history.groupby("train_n")["trip_date"].nunique()
        allowed = set(counts[counts >= min_days].index)
        future = future[future["train_n"].isin(allowed)].copy()
    return name, history, future


def template_predict(history, target):
    base = template.predict_from_history(history, target, 4, template.MAIN_TRAINS)
    pred = template.same_month_level(history, target, base, 2)
    pred = template.resort_level(history, target, pred)
    pred = template.gabala_march_single(history, target, pred)
    pred = template.gabala_novruz(history, target, pred)
    pred = template.gabala_tail(history, target, pred)
    pred = template.morning_break(history, target, pred)
    return np.asarray(pred, dtype=float)


def base_predict(history, target):
    base = template.predict_from_history(history, target, 4, template.MAIN_TRAINS)
    return np.asarray(template.same_month_level(history, target, base, 2), dtype=float)


def score_fold(future, pred):
    pred = np.asarray(pred, dtype=float)
    out = {
        "rmse": template.weighted_rmse(future["occupancy"], pred, future["n_segments"]),
        "n": int(future.groupby(["trip_date", "train_n"]).ngroups),
    }
    groups = {
        "main": future["train_n"].isin(template.MAIN_TRAINS),
        "gabala": future["train_n"].isin([709, 710]),
        "new": future["train_n"].isin([27, 28]),
    }
    for key, mask in groups.items():
        if int(mask.sum()) < 20:
            out[key] = np.nan
            continue
        out[key] = template.weighted_rmse(
            future.loc[mask, "occupancy"], pred[mask.to_numpy()], future.loc[mask, "n_segments"]
        )
    return out


def log_result(experiment, fold, scores, notes=""):
    row = {
        "experiment": experiment,
        "fold": fold,
        "rmse": round(float(scores["rmse"]), 6),
        "main": scores.get("main", np.nan),
        "gabala": scores.get("gabala", np.nan),
        "new": scores.get("new", np.nan),
        "n": scores.get("n", 0),
        "notes": notes,
    }
    frame = pd.DataFrame([row])
    frame.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def trip_errors(future, pred):
    frame = future.copy()
    frame["pred"] = np.asarray(pred, dtype=float)
    rows = []
    for (date, train_n), g in frame.groupby(["trip_date", "train_n"]):
        w = 1.0 / g["n_segments"].to_numpy(dtype=float)
        err = g["occupancy"].to_numpy(dtype=float) - g["pred"].to_numpy(dtype=float)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "train_n": int(train_n),
                "dow": int(g["dow"].iloc[0]),
                "sse": float(np.sum(w * err ** 2)),
                "gap": float(np.sum(w * err) / np.sum(w)),
            }
        )
    return pd.DataFrame(rows)


def top_error_sources(future, pred, limit=3):
    trips = trip_errors(future, pred)
    trips["month"] = trips["date"].dt.to_period("M").astype(str)
    total = float(trips["sse"].sum())
    grouped = (
        trips.groupby(["train_n", "month"], as_index=False)["sse"]
        .sum()
        .sort_values("sse", ascending=False)
    )
    lines = []
    for _, row in grouped.head(limit).iterrows():
        share = 100.0 * row["sse"] / total if total else 0.0
        lines.append(f"train {int(row.train_n)} {row.month} sse {row.sse:.2f} ({share:.1f}%)")
    return lines
