import numpy as np
import pandas as pd

import build_submission as b
import cv_harness as h
import cv_protect
import holiday_calendar as cal

train, test = b.load_frames()
old = h.template_predict(train, test)
new = cv_protect.protected(train, test)
test = test.copy()
test["old"] = old
test["new"] = new
test["ht"] = cal.type_of(test["trip_date"])
public = test["trip_date"] < "2026-04-16"

rows = []
for (date, train_n), g in test.groupby(["trip_date", "train_n"]):
    rows.append(
        {
            "date": pd.Timestamp(date),
            "train_n": int(train_n),
            "dow": int(g["dow"].iloc[0]),
            "ht": g["ht"].iloc[0],
            "public": bool(g["trip_date"].iloc[0] < pd.Timestamp("2026-04-16")),
            "old": float(g["old"].mean()),
            "new": float(g["new"].mean()),
        }
    )
trips = pd.DataFrame(rows)
trips["delta"] = trips["new"] - trips["old"]
moved = trips[trips["delta"].abs() > 0.005].sort_values("delta")
print("trips moved", len(moved), "of", len(trips), "max", round(float(trips["delta"].abs().max()), 3))
print(moved.assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).round(3).to_string(index=False))
for label, mask in [
    ("public main", public & test["train_n"].isin(b.MAIN_TRAINS)),
    ("public gabala", public & test["train_n"].isin([709, 710])),
    ("public new", public & test["train_n"].isin([27, 28])),
    ("private main", ~public & test["train_n"].isin(b.MAIN_TRAINS)),
    ("private gabala", ~public & test["train_n"].isin([709, 710])),
    ("private new", ~public & test["train_n"].isin([27, 28])),
]:
    print(
        label,
        "old",
        round(float(test.loc[mask, "old"].mean()), 3),
        "new",
        round(float(test.loc[mask, "new"].mean()), 3),
    )
