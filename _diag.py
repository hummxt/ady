"""Yeni marşrutun gunluk seviyyesi ve 0.85 qaydasinin payi."""

import numpy as np
import pandas as pd
from build_submission import load_frames, HOLIDAYS, weighted_rmse

train, test = load_frames()
new = train[train["train_n"].isin([27, 28])].copy()
trip = (
    new.groupby(["trip_date", "train_n", "dow"], as_index=False)["occupancy"]
    .mean()
)
print("NEW ROUTE TRIPS")
for _, row in trip.sort_values(["trip_date", "train_n"]).iterrows():
    mark = "H" if row.trip_date in set(HOLIDAYS) else " "
    print(f"{mark} {row.trip_date.date()} t{int(row.train_n)} dow{int(row.dow)} {row.occupancy:.3f}")

gab = train[train["train_n"].isin([709, 710])]
for year in (2024, 2025, 2026):
    part = gab[gab["trip_date"].dt.year == year]
    if year < 2026:
        part = part[part["trip_date"].dt.month.isin([2, 3, 4])]
    daily = part.groupby("trip_date")["occupancy"].mean()
    print(f"\nGABALA {year} daily n={len(daily)} mean={daily.mean():.3f}")
    if year == 2025:
        march = daily[daily.index.month == 3]
        for day, value in march.items():
            print(f"  {day.date()} {value:.3f}")

sub = pd.read_csv("submission.csv")
merged = test.merge(sub, on="id")
public = merged[merged["trip_date"] < "2026-04-16"]
new_pub = public[public["train_n"].isin([27, 28])]
print("\nPUBLIC NEW pred describe")
print(new_pub["occupancy"].describe().round(3).to_string())
high = new_pub[new_pub["occupancy"] >= 0.5]
print("rows >=0.5", len(high), "unique dates", sorted(high["trip_date"].dt.date.unique().astype(str)))
print("mean on those", round(high["occupancy"].mean(), 3), "share rows", round(len(high) / len(new_pub), 3))

# trip weights
def trip_mean(frame):
    g = frame.groupby(["train_n", "trip_date"], as_index=False)["occupancy"].mean()
    return g["occupancy"].mean()

print("public new trip-mean of preds", round(trip_mean(new_pub), 3))
calm_like = new_pub[new_pub["occupancy"] < 0.5]
print("public new below 0.5 trip-mean", round(trip_mean(calm_like), 3), "trips", calm_like.groupby(["train_n", "trip_date"]).ngroups)
