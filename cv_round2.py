import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import cv_harness as h
import holiday_calendar as cal

train = h.load_train()
FOLDS = list(h.FOLDS)


def shrink(value, count, prior, strength):
    return (count * value + strength * prior) / (count + strength)


def family_of(train_n):
    if train_n in h.template.MAIN_TRAINS:
        return "main"
    if train_n in (709, 710):
        return "gabala"
    return "new"


def trip_frame(frame, pred):
    out = frame.copy()
    out["pred"] = np.asarray(pred, dtype=float)
    rows = []
    for (date, train_n), g in out.groupby(["trip_date", "train_n"]):
        w = 1.0 / g["n_segments"].to_numpy(dtype=float)
        y = float(np.sum(w * g["occupancy"]) / np.sum(w))
        p = float(np.sum(w * g["pred"]) / np.sum(w))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "train_n": int(train_n),
                "dow": int(g["dow"].iloc[0]),
                "gap": y - p,
                "family": family_of(int(train_n)),
                "ht": cal.type_of(pd.Series([date]))[0],
                "weekend": int(g["dow"].iloc[0] in (5, 6)),
            }
        )
    return pd.DataFrame(rows)


def holiday_v2(history, future, pred, strength):
    base = h.base_predict(history, history)
    trips = trip_frame(history, base)
    trips = trips[trips["ht"] != ""]
    out = np.asarray(pred, dtype=float).copy()
    if trips.empty:
        return out
    fam = trips.groupby(["family", "ht", "weekend"])["gap"].agg(["mean", "count"])
    own = trips.groupby(["train_n", "ht", "weekend"])["gap"].agg(["mean", "count"])
    ht = cal.type_of(future["trip_date"])
    we = future["dow"].isin([5, 6]).astype(int).to_numpy()
    trains = future["train_n"].to_numpy()
    for i in range(len(future)):
        if ht[i] == "":
            continue
        fam_key = (family_of(int(trains[i])), ht[i], int(we[i]))
        prior = 0.0
        if fam_key in fam.index:
            mean, count = fam.loc[fam_key]
            prior = shrink(float(mean), float(count), 0.0, strength)
        key = (int(trains[i]), ht[i], int(we[i]))
        delta = prior
        if key in own.index:
            mean, count = own.loc[key]
            delta = shrink(float(mean), float(count), prior, strength)
        out[i] = out[i] + delta
    return np.clip(out, 0, 2.5)


cached = []
for fold in FOLDS:
    name, history, future = h.slice_fold(train, fold)
    full = h.template_predict(history, future)
    cached.append((name, history, future, full))


def report(experiment, pred_fn, notes):
    scores = []
    for name, history, future, full in cached:
        pred = pred_fn(history, future, full)
        sc = h.score_fold(future, pred)
        h.log_result(experiment, name, sc, notes)
        scores.append(sc["rmse"])
        print(f"{experiment:20} {name:7} {sc['rmse']:.4f} main {sc['main']:.4f} gabala {sc['gabala']:.4f}")
    print("  mean", round(float(np.mean(scores)), 4))


print("lgbm")
import lightgbm as lgb


def features(frame, history):
    seg = (
        history.groupby(["seg", "dow"])["occupancy"].mean()
        if len(history)
        else pd.Series(dtype=float)
    )
    table = frame[["train_n", "segment_order", "dow", "month", "from_station_id", "to_station_id", "seg"]].copy()
    codes = {name: i + 1 for i, name in enumerate(sorted(cal.TYPES) + ["ramadan", "qurban", "transferred"])}
    table["ht"] = pd.Series(cal.type_of(frame["trip_date"])).map(codes).fillna(0).astype(int).to_numpy()
    table["days"] = cal.days_to_rest(frame["trip_date"])
    table["trend"] = (frame["trip_date"] - pd.Timestamp("2024-06-14")).dt.days.to_numpy()
    if len(seg):
        mapped = frame.set_index(["seg", "dow"]).index.map(seg)
        table["seg_dow"] = np.asarray(mapped, dtype=float)
    else:
        table["seg_dow"] = np.nan
    table["seg_dow"] = table["seg_dow"].fillna(table["seg_dow"].mean() if table["seg_dow"].notna().any() else 0.5)
    return table.drop(columns=["seg"])


def lgbm_blend(history, future, full, weight):
    origin = future["trip_date"].min()
    cal_start = origin - pd.Timedelta(days=120)
    window = history[history["trip_date"] >= cal_start].copy()
    past = history[history["trip_date"] < cal_start]
    if len(window) < 400 or len(past) < 400:
        return full
    window_pred = h.template_predict(past, window)
    x_train = features(window, past)
    x_train["template"] = window_pred
    y = window["occupancy"].to_numpy(dtype=float) - window_pred
    x_test = features(future, history)
    x_test["template"] = full
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=0,
        verbosity=-1,
    )
    model.fit(x_train, y)
    residual = model.predict(x_test)
    return np.clip(full + weight * residual, 0, 2.5)


for weight in (0.25, 0.5, 1.0):
    report(f"lgbm_w{weight}", lambda hist, fut, full, w=weight: lgbm_blend(hist, fut, full, w), "residual lightgbm")
