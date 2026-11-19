import numpy as np
import pandas as pd

import build_submission as b
import cv_harness as h

train = h.load_train()
FOLDS = list(h.FOLDS) + [h.ROUTE_FOLD]


def cached_folds():
    rows = []
    for fold in FOLDS:
        name, history, future = h.slice_fold(train, fold)
        pred = b.forecast(history, future)
        rows.append((name, history, future, pred))
    return rows


CACHE = cached_folds()


def report(name, pred_fn):
    scores = {}
    for fold_name, history, future, base in CACHE:
        pred = pred_fn(history, future, base)
        scores[fold_name] = h.score_fold(future, pred)["rmse"]
        if fold_name == "mimic":
            early = (future["trip_date"] < "2025-04-16").to_numpy()
            scores["early"] = h.score_fold(future.loc[early], pred[early])["rmse"]
            scores["late"] = h.score_fold(future.loc[~early], pred[~early])["rmse"]
    core = np.mean([scores["mimic"], scores["summer"], scores["winter"]])
    print(
        f"{name:22} late {scores['late']:.4f} early {scores['early']:.4f} "
        f"mimic {scores['mimic']:.4f} summer {scores['summer']:.4f} "
        f"winter {scores['winter']:.4f} route {scores['route']:.4f} mean {core:.4f}"
    )
    return scores


def last_year_table(history):
    hist = history.copy()
    hist["year"] = hist["trip_date"].dt.year
    return hist.groupby(["train_n", "seg", "month", "dow", "year"])["occupancy"].mean()


def blend_last_year(history, future, base, weight, trains, lower_only, months):
    table = last_year_table(history)
    out = np.asarray(base, dtype=float).copy()
    years = future["trip_date"].dt.year.to_numpy()
    trains_arr = future["train_n"].to_numpy()
    segs = future["seg"].to_numpy()
    months_arr = future["month"].to_numpy()
    dows = future["dow"].to_numpy()
    allowed = set(trains) if trains is not None else None
    for i in range(len(future)):
        if allowed is not None and int(trains_arr[i]) not in allowed:
            continue
        if months is not None and int(months_arr[i]) not in months:
            continue
        key = (int(trains_arr[i]), segs[i], int(months_arr[i]), int(dows[i]), int(years[i]) - 1)
        if key not in table.index:
            continue
        ly = float(table.loc[key])
        if lower_only and ly >= out[i]:
            continue
        out[i] = (1.0 - weight) * out[i] + weight * ly
    return np.clip(out, 0, 2.5)


print("baseline")
report("forecast", lambda hst, fut, base: base)

for weight in (0.25, 0.5, 1.0):
    report(
        f"gab_w{weight}",
        lambda hst, fut, base, w=weight: blend_last_year(hst, fut, base, w, {709, 710}, False, None),
    )
    report(
        f"gab_low_w{weight}",
        lambda hst, fut, base, w=weight: blend_last_year(hst, fut, base, w, {709, 710}, True, None),
    )
    report(
        f"gab_spr_w{weight}",
        lambda hst, fut, base, w=weight: blend_last_year(hst, fut, base, w, {709, 710}, True, {4, 5, 6}),
    )

report(
    "main_low_w0.5",
    lambda hst, fut, base: blend_last_year(hst, fut, base, 0.5, set(b.MAIN_TRAINS), True, None),
)
