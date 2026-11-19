import numpy as np

import build_submission as b
import cv_harness as h
import cv_search as s

train, test = h.load_train(), None
train = h.load_train()
_, test = b.load_frames()


def blend_skip(history, future, base):
    pred = s.blend_last_year(history, future, base, 0.5, {709, 710}, True, None)
    md = future["trip_date"].dt.strftime("%m-%d")
    skip = md.between("03-20", "03-30").to_numpy()
    out = np.asarray(pred, dtype=float).copy()
    base = np.asarray(base, dtype=float)
    out[skip] = base[skip]
    return out


for fold in list(h.FOLDS) + [h.ROUTE_FOLD]:
    name, history, future = h.slice_fold(train, fold)
    base = b.forecast(history, future)
    pred = blend_skip(history, future, base)
    score = h.score_fold(future, pred)["rmse"]
    line = f"{name} {score:.4f}"
    if name == "mimic":
        early = (future["trip_date"] < "2025-04-16").to_numpy()
        line += (
            f" early {h.score_fold(future.loc[early], pred[early])['rmse']:.4f}"
            f" late {h.score_fold(future.loc[~early], pred[~early])['rmse']:.4f}"
        )
    print(line)

full = b.forecast(train, test)
new = blend_skip(train, test, full)
pub = (test["trip_date"] < "2026-04-16").to_numpy()
gab = test["train_n"].isin([709, 710]).to_numpy()
print("public gabala max", round(float(np.max(np.abs(new - full)[pub & gab])), 4))
print("public changed", int(np.sum(np.abs(new - full)[pub] > 1e-6)))
test = test.copy()
test["d"] = new - full
test["old"] = full
rows = []
import pandas as pd

part = test.loc[pub & gab]
for (dt, tn), g in part.groupby(["trip_date", "train_n"]):
    delta = float(g["d"].mean())
    if abs(delta) > 0.005:
        rows.append((str(pd.Timestamp(dt).date()), int(tn), round(float(g["old"].mean()), 3), round(delta, 3)))
print("moved", rows)
