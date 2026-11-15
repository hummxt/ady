"""Tree modelini kecmis pencerelerde indiki proqnozla muqayise edir."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from build_submission import (
    MAIN_TRAINS,
    load_frames,
    predict_from_history,
    same_month_level,
    resort_level,
    weighted_rmse,
)

CATS = ["train_n", "from_station_id", "to_station_id", "dow", "month"]


def current_prediction(history, future):
    base = predict_from_history(history, future, 4, MAIN_TRAINS)
    mixed = pd.Series(same_month_level(history, future, base, 2), index=future.index)
    return pd.Series(resort_level(history, future, mixed), index=future.index)


def design(history, frame):
    table = frame[
        ["seg", "train_n", "from_station_id", "to_station_id", "dow", "month", "segment_order", "n_segments", "capacity", "trip_date", "is_holiday", "is_novruz"]
    ].copy()
    table["seg_frac"] = table["segment_order"] / table["n_segments"]
    table["is_weekend"] = table["dow"].isin([5, 6]).astype(int)
    grouped = {
        "seg_dow": history.groupby(["seg", "dow"])["occupancy"].agg(seg_dow_mean="mean", seg_dow_n="count"),
        "train_dow": history.groupby(["train_n", "dow"])["occupancy"].mean(),
        "seg_month": history.groupby(["seg", "month"])["occupancy"].mean(),
    }
    table = table.merge(grouped["seg_dow"].reset_index(), on=["seg", "dow"], how="left")
    table = table.merge(grouped["train_dow"].rename("train_dow_mean").reset_index(), on=["train_n", "dow"], how="left")
    table = table.merge(grouped["seg_month"].rename("seg_month_mean").reset_index(), on=["seg", "month"], how="left")
    recent_cut = history["trip_date"].max() - pd.Timedelta(days=21)
    recent = history[history["trip_date"] >= recent_cut]
    recent_dow = recent.groupby(["seg", "dow"])["occupancy"].mean()
    table = table.merge(recent_dow.rename("recent_dow").reset_index(), on=["seg", "dow"], how="left")
    first_day = history.groupby("train_n")["trip_date"].min()
    table["days_running"] = (table["trip_date"] - table["train_n"].map(first_day)).dt.days
    columns = [
        "train_n", "from_station_id", "to_station_id", "dow", "month",
        "segment_order", "n_segments", "seg_frac", "capacity",
        "is_holiday", "is_novruz", "is_weekend",
        "seg_dow_mean", "seg_dow_n", "train_dow_mean", "seg_month_mean", "recent_dow", "days_running",
    ]
    return table[columns]


def fit_model(history):
    cutoff = history["trip_date"].max() - pd.Timedelta(days=21)
    core = history[history["trip_date"] < cutoff]
    tail = history[history["trip_date"] >= cutoff]
    x_core, y_core = design(core, core), core["occupancy"].to_numpy()
    x_tail, y_tail = design(history, tail), tail["occupancy"].to_numpy()
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_core, y_core,
        sample_weight=1 / core["n_segments"],
        eval_set=[(x_tail, y_tail)],
        eval_sample_weight=[1 / tail["n_segments"]],
        categorical_feature=[c for c in CATS if c in x_core.columns],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def score_window(train, origin, end):
    history = train[train["trip_date"] < origin]
    future = train[(train["trip_date"] >= origin) & (train["trip_date"] < end)].copy()
    stat = current_prediction(history, future).to_numpy()
    model = fit_model(history)
    tree = np.clip(model.predict(design(history, future)), 0, 2.5)
    blend = 0.5 * stat + 0.5 * tree
    print(f"\n{origin} -> {end}  rows {len(future)}")
    for name, pred in [("stat", stat), ("tree", tree), ("blend", blend)]:
        print(
            f"  {name:5} {weighted_rmse(future.occupancy, pred, future.n_segments):.4f}"
        )


def main():
    train, _ = load_frames()
    score_window(train, "2026-02-01", "2026-03-01")
    score_window(train, "2025-03-01", "2025-04-16")
    score_window(train, "2025-07-01", "2025-11-01")


if __name__ == "__main__":
    main()
