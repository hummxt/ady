import numpy as np
import pandas as pd
import lightgbm as lgb

import build_submission as b
import cv_harness as h

train = h.load_train()


def oos_template(history):
    pred = np.full(len(history), np.nan)
    frame = history.copy()
    frame["_row"] = np.arange(len(frame))
    for period in sorted(frame["trip_date"].dt.to_period("M").unique()):
        start = period.to_timestamp()
        block = frame[frame["trip_date"].dt.to_period("M") == period]
        past = frame[frame["trip_date"] < start]
        if len(past) < 800 or len(block) == 0:
            continue
        pred[block["_row"].to_numpy()] = b.forecast(past.drop(columns="_row"), block.drop(columns="_row"))
    return pred


def design(frame, template_value):
    return pd.DataFrame(
        {
            "train_n": frame["train_n"].to_numpy(),
            "segment_order": frame["segment_order"].to_numpy(),
            "dow": frame["dow"].to_numpy(),
            "month": frame["month"].to_numpy(),
            "from_station_id": frame["from_station_id"].to_numpy(),
            "to_station_id": frame["to_station_id"].to_numpy(),
            "template": np.asarray(template_value, dtype=float),
            "trend": (frame["trip_date"] - pd.Timestamp("2024-06-14")).dt.days.to_numpy(),
        }
    )


def fit_predict(history, future, base, leaves, child):
    oos = oos_template(history)
    ok = ~np.isnan(oos)
    if int(ok.sum()) < 500:
        return np.zeros(len(future))
    y = history.loc[ok, "occupancy"].to_numpy(dtype=float) - oos[ok]
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=leaves,
        min_child_samples=child,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=0,
        verbosity=-1,
    )
    model.fit(design(history.loc[ok], oos[ok]), y)
    return model.predict(design(future, base))


def evaluate(residual_fn):
    stored = []
    for fold in list(h.FOLDS) + [h.ROUTE_FOLD]:
        name, history, future = h.slice_fold(train, fold)
        base = b.forecast(history, future)
        residual = residual_fn(history, future, base)
        stored.append((name, future, base, residual))
        print("fit", name, flush=True)
    return stored


def show(tag, stored, weight):
    scores = {}
    for name, future, base, residual in stored:
        pred = np.clip(base + weight * residual, 0, 2.5)
        scores[name] = h.score_fold(future, pred)["rmse"]
        if name == "mimic":
            early = (future["trip_date"] < "2025-04-16").to_numpy()
            scores["early"] = h.score_fold(future.loc[early], pred[early])["rmse"]
            scores["late"] = h.score_fold(future.loc[~early], pred[~early])["rmse"]
    print(
        f"{tag:16} w {weight:.2f} late {scores['late']:.4f} early {scores['early']:.4f} "
        f"mimic {scores['mimic']:.4f} summer {scores['summer']:.4f} "
        f"winter {scores['winter']:.4f} route {scores['route']:.4f}",
        flush=True,
    )
    return scores


print("fitting", flush=True)
fitted = evaluate(lambda hist, fut, base: fit_predict(hist, fut, base, 15, 80))
for weight in (0.15, 0.25, 0.4, 0.6, 1.0):
    show("lgb", fitted, weight)
