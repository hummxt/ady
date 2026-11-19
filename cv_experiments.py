import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import cv_harness as h
import holiday_calendar as cal

train = h.load_train()
FOLDS = list(h.FOLDS) + [h.ROUTE_FOLD]


def shrink(value, count, prior, strength):
    return (count * value + strength * prior) / (count + strength)


def prepare_fold(fold):
    name, history, future = h.slice_fold(train, fold)
    full = h.template_predict(history, future)
    base = h.base_predict(history, future)
    return name, history, future, full, base


def log(experiment, name, future, pred, notes):
    scores = h.score_fold(future, pred)
    h.log_result(experiment, name, scores, notes)
    print(
        f"{experiment:22} {name:7} {scores['rmse']:.4f} "
        f"main {scores['main'] if scores['main']==scores['main'] else float('nan'):.4f} "
        f"gabala {scores['gabala'] if scores['gabala']==scores['gabala'] else float('nan'):.4f}"
    )
    return scores


cached = [prepare_fold(fold) for fold in FOLDS]


def holiday_adjust(history, future, pred, strength):
    hist = history.copy()
    hist_pred = h.base_predict(history, hist)
    hist["pred"] = hist_pred
    hist["gap"] = hist["occupancy"] - hist["pred"]
    hist["ht"] = cal.type_of(hist["trip_date"])
    hist = hist[hist["ht"] != ""]
    out = np.asarray(pred, dtype=float).copy()
    if hist.empty:
        return out
    hist["weekend"] = hist["dow"].isin([5, 6]).astype(int)
    grouped = hist.groupby(["train_n", "ht", "weekend"])["gap"].agg(["mean", "count"])
    future_ht = cal.type_of(future["trip_date"])
    future_we = future["dow"].isin([5, 6]).astype(int).to_numpy()
    trains = future["train_n"].to_numpy()
    for i in range(len(future)):
        key = (trains[i], future_ht[i], int(future_we[i]))
        if future_ht[i] == "" or key not in grouped.index:
            continue
        mean, count = grouped.loc[key]
        out[i] = out[i] + shrink(float(mean), float(count), 0.0, strength)
    return np.clip(out, 0, 2.5)


print("holiday additive on template")
for strength in (4, 8, 16):
    for name, history, future, full, base in cached:
        pred = holiday_adjust(history, future, full, strength)
        log(f"holiday_s{strength}", name, future, pred, "additive type effect")


def recent_scale(history, future, pred, trains, lookback, shrink_strength):
    origin = future["trip_date"].min()
    window = history[history["trip_date"] >= origin - pd.Timedelta(days=lookback)]
    window = window[window["train_n"].isin(trains)]
    out = np.asarray(pred, dtype=float).copy()
    if len(window) < 30:
        return out
    past_pred = h.template_predict(history[history["trip_date"] < window["trip_date"].min()], window)
    ratio_num = float(window["occupancy"].mean())
    ratio_den = float(np.mean(past_pred))
    if ratio_den <= 0.05:
        return out
    raw = ratio_num / ratio_den
    ratio = shrink(raw, float(len(window) / 8), 1.0, shrink_strength)
    mask = future["train_n"].isin(trains).to_numpy()
    out[mask] = out[mask] * ratio
    return np.clip(out, 0, 2.5)


print("gabala recent scale")
for name, history, future, full, base in cached:
    pred = recent_scale(history, future, full, {709, 710}, 45, 8)
    log("gabala_scale_45", name, future, pred, "recent/template ratio")


def train_bias(history, future, pred, lookback, strength, trains):
    origin = future["trip_date"].min()
    window = history[history["trip_date"] >= origin - pd.Timedelta(days=lookback)].copy()
    if trains is not None:
        window = window[window["train_n"].isin(trains)]
    out = np.asarray(pred, dtype=float).copy()
    if len(window) < 40:
        return out
    past_hist = history[history["trip_date"] < window["trip_date"].min()]
    window_pred = h.template_predict(past_hist, window)
    window = window.copy()
    window["gap"] = window["occupancy"].to_numpy() - window_pred
    bias = window.groupby("train_n")["gap"].agg(["mean", "count"])
    trains_arr = future["train_n"].to_numpy()
    for i, train_n in enumerate(trains_arr):
        if train_n not in bias.index:
            continue
        if trains is not None and train_n not in trains:
            continue
        mean, count = bias.loc[train_n]
        out[i] = out[i] + shrink(float(mean), float(count) / 8.0, 0.0, strength)
    return np.clip(out, 0, 2.5)


def asymmetric_gabala(history, future, pred, lookback, horizon, strength):
    origin = future["trip_date"].min()
    window = history[
        (history["trip_date"] >= origin - pd.Timedelta(days=lookback))
        & history["train_n"].isin([709, 710])
    ]
    out = np.asarray(pred, dtype=float).copy()
    if len(window) < 20:
        return out
    past_hist = history[history["trip_date"] < window["trip_date"].min()]
    past_pred = h.template_predict(past_hist, window)
    den = float(np.mean(past_pred))
    if den <= 0.05:
        return out
    raw = float(window["occupancy"].mean()) / den
    if raw >= 1:
        return out
    ratio = shrink(raw, float(len(window) / 8.0), 1.0, strength)
    near = future["trip_date"] <= origin + pd.Timedelta(days=horizon)
    mask = future["train_n"].isin([709, 710]).to_numpy() & near.to_numpy()
    out[mask] = out[mask] * ratio
    return np.clip(out, 0, 2.5)


print("asymmetric gabala scale")
for horizon in (30, 60, 120):
    for name, history, future, full, base in cached:
        pred = asymmetric_gabala(history, future, full, 45, horizon, 6)
        log(f"gabala_down_{horizon}", name, future, pred, "scale down only, short horizon")


print("train bias")
for name, history, future, full, base in cached:
    pred = train_bias(history, future, full, 60, 12, None)
    log("train_bias_60", name, future, pred, "shrunk train bias")
    pred = train_bias(history, future, full, 60, 12, {709, 710})
    log("gabala_bias_60", name, future, pred, "shrunk gabala bias")
