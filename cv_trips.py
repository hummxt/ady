import pandas as pd

import build_submission as b
import cv_harness as h

train = h.load_train()
for fold in h.FOLDS:
    name, history, future = h.slice_fold(train, fold)
    pred = b.forecast(history, future)
    trips = h.trip_errors(future, pred).sort_values("sse", ascending=False).head(3)
    print(name)
    print(trips.assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))
