import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import build_submission as template
import cv_harness as h
import holiday_calendar as cal

train = h.load_train()
FOLDS = list(h.FOLDS) + [h.ROUTE_FOLD]


def shrink(value, count, prior, strength):
    return (count * value + strength * prior) / (count + strength)


def resort(history, target, pred, gabala_weight):
    out = template.resort_level(history, target, pred)
    if gabala_weight == 0.85:
        return out
    rebuilt = template.resort_level(history, target, pred)
    no_year = rebuilt.copy()
    gabala = history[(history["train_n"].isin([709, 710])) & (history["trip_date"].dt.year == 2025)]
    if len(gabala) == 0 or gabala_weight == 0.85:
        return out
    cell = gabala.groupby(["seg", "month", "dow"])["occupancy"].mean()
    table = target[["seg", "month", "dow"]].reset_index(drop=True)
    table = table.merge(cell.rename("season").reset_index(), on=["seg", "month", "dow"], how="left")
    season = table["season"].to_numpy(dtype=float)
    is_gabala = target["train_n"].isin([709, 710]).to_numpy()
    use = is_gabala & ~np.isnan(season)
    base = np.asarray(pred, dtype=float)
    mixed = np.asarray(rebuilt, dtype=float).copy()
    mixed[use] = (1.0 - gabala_weight) * base[use] + gabala_weight * season[use]
    return np.clip(mixed, 0, 2.5)


def predict(history, target, gabala_weight):
    base = template.predict_from_history(history, target, 4, template.MAIN_TRAINS)
    pred = template.same_month_level(history, target, base, 2)
    pred = resort(history, target, pred, gabala_weight)
    pred = template.gabala_march_single(history, target, pred)
    pred = template.gabala_novruz(history, target, pred)
    pred = template.gabala_tail(history, target, pred)
    pred = template.morning_break(history, target, pred)
    return np.asarray(pred, dtype=float)


def holiday_layer(history, future, pred, strength):
    hist_pred = h.base_predict(history, history)
    gap = history["occupancy"].to_numpy(dtype=float) - hist_pred
    ht = cal.type_of(history["trip_date"])
    keep = ht != ""
    frame = history.loc[keep, ["train_n", "dow"]].copy()
    frame["gap"] = gap[keep]
    frame["ht"] = ht[keep]
    frame["weekend"] = frame["dow"].isin([5, 6]).astype(int)
    grouped = frame.groupby(["train_n", "ht", "weekend"])["gap"].agg(["mean", "count"])
    out = np.asarray(pred, dtype=float).copy()
    future_ht = cal.type_of(future["trip_date"])
    future_we = future["dow"].isin([5, 6]).astype(int).to_numpy()
    trains = future["train_n"].to_numpy()
    index = grouped.index
    means = grouped["mean"].to_numpy()
    counts = grouped["count"].to_numpy()
    lookup = {key: (means[i], counts[i]) for i, key in enumerate(index)}
    for i in range(len(future)):
        key = (int(trains[i]), future_ht[i], int(future_we[i]))
        if future_ht[i] == "" or key not in lookup:
            continue
        mean, count = lookup[key]
        out[i] = out[i] + shrink(float(mean), float(count), 0.0, strength)
    return np.clip(out, 0, 2.5)


def oos_holiday(history, future, pred, strength):
    months = sorted(history["trip_date"].dt.to_period("M").unique())
    oos = np.full(len(history), np.nan)
    history = history.copy()
    history["_row"] = np.arange(len(history))
    for period in months:
        start = period.to_timestamp()
        block = history[history["trip_date"].dt.to_period("M") == period]
        past = history[history["trip_date"] < start]
        if len(past) < 800 or len(block) == 0:
            continue
        oos[block["_row"].to_numpy()] = h.template_predict(
            past.drop(columns="_row"), block.drop(columns="_row")
        )
    gap = history["occupancy"].to_numpy(dtype=float) - oos
    ht = cal.type_of(history["trip_date"])
    keep = (ht != "") & ~np.isnan(gap)
    frame = history.loc[keep, ["train_n", "dow"]].copy()
    frame["gap"] = gap[keep]
    frame["ht"] = ht[keep]
    frame["weekend"] = frame["dow"].isin([5, 6]).astype(int)
    if frame.empty:
        return np.asarray(pred, dtype=float)
    grouped = frame.groupby(["train_n", "ht", "weekend"])["gap"].agg(["mean", "count"])
    out = np.asarray(pred, dtype=float).copy()
    future_ht = cal.type_of(future["trip_date"])
    future_we = future["dow"].isin([5, 6]).astype(int).to_numpy()
    trains = future["train_n"].to_numpy()
    lookup = {
        key: (float(grouped.loc[key, "mean"]), float(grouped.loc[key, "count"]))
        for key in grouped.index
    }
    for i in range(len(future)):
        key = (int(trains[i]), future_ht[i], int(future_we[i]))
        if future_ht[i] == "" or key not in lookup:
            continue
        mean, count = lookup[key]
        out[i] = out[i] + shrink(mean, count, 0.0, strength)
    return np.clip(out, 0, 2.5)


cached = []
for fold in FOLDS:
    name, history, future = h.slice_fold(train, fold)
    cached.append((name, history, future))


def run(experiment, fn, notes):
    scores = {}
    for name, history, future in cached:
        pred = fn(history, future)
        sc = h.score_fold(future, pred)
        h.log_result(experiment, name, sc, notes)
        scores[name] = sc["rmse"]
        print(
            f"{experiment:22} {name:7} {sc['rmse']:.4f} "
            f"main {sc['main'] if sc['main']==sc['main'] else float('nan'):.4f} "
            f"gabala {sc['gabala'] if sc['gabala']==sc['gabala'] else float('nan'):.4f}"
        )
    core = [scores[name] for name in ("mimic", "summer", "winter")]
    print("  mean", round(float(np.mean(core)), 4))
    return scores


def candidate(history, future):
    return holiday_layer(history, future, predict(history, future, 0.0), 16)


if __name__ == "__main__":
    run("template_locked", lambda hist, fut: h.template_predict(hist, fut), "reference")
