import numpy as np
import pandas as pd

import build_submission as b
import cv_harness as h
import holiday_calendar as cal

train = h.load_train()
expected = {"mimic": 0.1293, "summer": 0.0973, "winter": 0.0968, "route": 0.0594}
for fold in list(h.FOLDS) + [h.ROUTE_FOLD]:
    name, history, future = h.slice_fold(train, fold)
    pred = b.forecast(history, future)
    sc = h.score_fold(future, pred)
    h.log_result("ship_forecast", name, sc, "production forecast")
    print(name, round(sc["rmse"], 4), "expected", expected[name])
    print(" ", h.top_error_sources(future, pred))

_, test = b.load_frames()
old = h.template_predict(train, test)
new = b.forecast(train, test)
test = test.copy()
test["old"] = old
test["new"] = new
test["ht"] = cal.type_of(test["trip_date"])
settled = test["train_n"].isin([709, 710]) & (
    test["trip_date"].between("2026-03-01", "2026-04-15")
)
gap = np.abs(test.loc[settled, "new"] - test.loc[settled, "old"])
print("public gabala rows", int(settled.sum()), "max abs change", round(float(gap.max()), 6))
public = test["trip_date"] < "2026-04-16"
for label, mask in [
    ("public main", public & test["train_n"].isin(b.MAIN_TRAINS)),
    ("public gabala", public & test["train_n"].isin([709, 710])),
    ("public new", public & test["train_n"].isin([27, 28])),
    ("private gabala", ~public & test["train_n"].isin([709, 710])),
    ("private main", ~public & test["train_n"].isin(b.MAIN_TRAINS)),
]:
    print(label, "old", round(float(test.loc[mask, "old"].mean()), 3), "new", round(float(test.loc[mask, "new"].mean()), 3))
