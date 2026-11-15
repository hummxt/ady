import numpy as np
import pandas as pd
from build_submission import MAIN_TRAINS, load_frames, predict_from_history, weighted_rmse

train, _ = load_frames()


def rmse(frame, pred):
    return weighted_rmse(frame.occupancy, pred, frame.n_segments)


def group_rmse(frame, pred):
    pred = np.asarray(pred, float)
    out = {"all": rmse(frame, pred)}
    for name, trains in [("main", MAIN_TRAINS), ("gabala", {709, 710}), ("new", {27, 28})]:
        mask = frame.train_n.isin(trains).to_numpy()
        if mask.sum() >= 15:
            out[name] = rmse(frame.loc[mask], pred[mask])
    return out


def forecast(history, future, cell_strength, scale_new):
    base = predict_from_history(history, future, 4, MAIN_TRAINS).to_numpy()
    cell = history.groupby(["seg", "month", "dow"]).occupancy.agg(["mean", "count"])
    keys = list(zip(future.seg, future.month, future.dow))
    means = np.array([cell.loc[k, "mean"] if k in cell.index else np.nan for k in keys])
    counts = np.array([cell.loc[k, "count"] if k in cell.index else 0.0 for k in keys])
    pred = (counts * np.where(np.isnan(means), base, means) + cell_strength * base) / (counts + cell_strength)

    if scale_new:
        gabala = history[history.train_n.isin([709, 710])]
        recent_cut = history.trip_date.max() - pd.Timedelta(days=40)
        recent = history[history.trip_date >= recent_cut]
        recent_dow = recent.groupby(["seg", "dow"]).occupancy.mean()
        recent_level = np.array([recent_dow.get(k, np.nan) for k in zip(future.seg, future.dow)])
        by_month = gabala.groupby("month").occupancy.mean()
        recent_months = recent[recent.train_n.isin([709, 710])].month
        if len(recent_months):
            anchor = float(gabala[gabala.month.isin(set(recent_months))].groupby("month").occupancy.mean().mean())
        else:
            anchor = np.nan
        target_level = future.month.map(by_month).to_numpy(dtype=float)
        factor = target_level / anchor if anchor and anchor > 0 else np.full(len(future), np.nan)
        is_new = future.train_n.isin([27, 28]).to_numpy()
        use = is_new & ~np.isnan(recent_level) & ~np.isnan(factor)
        # Weekdays stay near the observed level. Weekends follow the Gabala season.
        weekend = future.dow.isin([5, 6]).to_numpy()
        pred[use & weekend] = np.clip(recent_level[use & weekend] * factor[use & weekend], 0, 2.5)
        pred[use & ~weekend] = np.where(np.isnan(recent_level[use & ~weekend]), pred[use & ~weekend], recent_level[use & ~weekend])
    return np.clip(pred, 0, 2.5)


periods = list(pd.period_range("2025-03", "2026-02", freq="M"))
configs = [(4, False), (8, False), (2, False), (4, True), (8, True)]
bucket = {cfg: [] for cfg in configs}
spring = {cfg: [] for cfg in configs}

for period in periods:
    origin = period.to_timestamp()
    end = (period + 1).to_timestamp()
    history = train[train.trip_date < origin]
    future = train[(train.trip_date >= origin) & (train.trip_date < end)].copy()
    print(f"\n{period}")
    for cfg in configs:
        pred = forecast(history, future, cfg[0], cfg[1])
        scores = group_rmse(future, pred)
        bucket[cfg].append(scores["all"])
        if period.month in (3, 4, 5, 6):
            spring[cfg].append(scores["all"])
        text = " ".join(f"{k} {v:.3f}" for k, v in scores.items())
        print(f"  strength {cfg[0]} scale {cfg[1]}: {text}")

print("\nMEAN ALL MONTHS / SPRING")
for cfg in configs:
    print(cfg, round(float(np.mean(bucket[cfg])), 4), "spring", round(float(np.mean(spring[cfg])), 4))
