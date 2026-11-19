import numpy as np
import pandas as pd

import build_submission as b
import cv_harness as h

train = h.load_train()


def score_slice(future, pred, start, end):
    mask = (future["trip_date"] >= start) & (future["trip_date"] < end)
    part = future.loc[mask]
    if len(part) == 0:
        return None
    values = np.asarray(pred, dtype=float)[mask.to_numpy()]
    return h.score_fold(part, values)


origin = pd.Timestamp("2025-03-01")
history = train[train["trip_date"] < origin]
future = train[(train["trip_date"] >= origin) & (train["trip_date"] < "2025-07-01")].copy()
base = b.forecast(history, future)
template = h.template_predict(history, future)
for label, pred in [("forecast", base), ("template", template)]:
    full = h.score_fold(future, pred)
    early = score_slice(future, pred, "2025-03-01", "2025-04-16")
    late = score_slice(future, pred, "2025-04-16", "2025-07-01")
    print(label, "full", round(full["rmse"], 4), "early", round(early["rmse"], 4), "late", round(late["rmse"], 4))
    print("  early", {k: round(early[k], 4) if early[k] == early[k] else None for k in ("main", "gabala", "new")})
    print("  late", {k: round(late[k], 4) if late[k] == late[k] else None for k in ("main", "gabala", "new")})

trips = h.trip_errors(future, base)
trips["month"] = trips["date"].dt.month
late = trips[trips["date"] >= "2025-04-16"]
early = trips[trips["date"] < "2025-04-16"]
for name, part in [("early", early), ("late", late)]:
    print("==", name)
    print(part.groupby("train_n").agg(n=("gap", "size"), gap=("gap", "mean"), sse=("sse", "sum")).round(3))
