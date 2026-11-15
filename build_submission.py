"""ADY doluluq proqnozu. Ishletmek: python build_submission.py"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "files"
OUT = ROOT / "submission.csv"

DAILY_START = pd.Timestamp("2024-06-14")
MAIN_TRAINS = {731, 732, 733, 734}
STABLE_TRAINS = {709, 710, 731, 732, 733, 734}

FIXED_HOLIDAYS = [(1, 1), (1, 2), (3, 8), (5, 9), (5, 28), (6, 15), (6, 26), (11, 8), (11, 9), (12, 31)]
NAMED_HOLIDAYS = {
    "novruz": [
        "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24",
        "2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-24",
        "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23", "2026-03-24",
    ],
    "ramadan": ["2024-04-10", "2024-04-11", "2025-03-30", "2025-03-31", "2026-03-20", "2026-03-21"],
    "qurban": ["2024-06-16", "2024-06-17", "2025-06-06", "2025-06-07", "2026-05-27", "2026-05-28"],
}


def weighted_rmse(y, pred, n_segments):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    weight = 1.0 / np.asarray(n_segments, dtype=float)
    return float(np.sqrt(np.sum(weight * (y - pred) ** 2) / np.sum(weight)))


def holiday_dates():
    days = [f"{year}-{month:02d}-{day:02d}" for year in (2024, 2025, 2026) for month, day in FIXED_HOLIDAYS]
    named = {name: pd.to_datetime(values) for name, values in NAMED_HOLIDAYS.items()}
    for values in NAMED_HOLIDAYS.values():
        days.extend(values)
    return pd.to_datetime(sorted(set(days))), named


HOLIDAYS, NAMED = holiday_dates()


def prepare(frame):
    frame = frame.copy()
    frame["trip_date"] = pd.to_datetime(frame["trip_date"])
    frame["dow"] = frame["trip_date"].dt.day_of_week.astype(int)
    frame["month"] = frame["trip_date"].dt.month.astype(int)
    frame["seg"] = (
        frame["train_n"].astype(str)
        + "-"
        + frame["from_station_id"].astype(str)
        + "-"
        + frame["to_station_id"].astype(str)
    )
    frame["is_holiday"] = frame["trip_date"].isin(HOLIDAYS).astype(int)
    frame["is_novruz"] = frame["trip_date"].isin(NAMED["novruz"]).astype(int)
    return frame


def shrink(mean, count, prior, strength):
    mean = np.asarray(mean, dtype=float)
    count = np.asarray(count, dtype=float)
    prior = np.asarray(prior, dtype=float)
    mean = np.where(np.isnan(mean), prior, mean)
    count = np.where(np.isnan(count), 0.0, count)
    return (count * mean + strength * prior) / (count + strength)


def predict_from_history(history, target, month_strength, adjust_trains):
    """Hefte gunu ortasi, esas qatarlarda eyni ayin kecmis seviyyesi ile."""
    src = history[history["trip_date"] >= DAILY_START]
    if len(src) < 500:
        src = history
    stable = src[src["train_n"].isin(STABLE_TRAINS)]
    if len(stable) < 500:
        stable = src

    by_seg_dow = src.groupby(["seg", "dow"])["occupancy"].mean().rename("dow_mean")
    by_train_dow = src.groupby(["train_n", "dow"])["occupancy"].mean().rename("train_dow")
    by_dow = stable.groupby("dow")["occupancy"].mean()
    by_seg = src.groupby("seg")["occupancy"].mean().rename("seg_mean")
    by_seg_month = (
        src.groupby(["seg", "month"])["occupancy"].agg(month_mean="mean", month_n="count").reset_index()
    )
    month_ratio = stable.groupby("month")["occupancy"].mean() / float(stable["occupancy"].mean())

    table = target[["seg", "train_n", "dow", "month"]].reset_index(drop=True)
    table = table.merge(by_seg_dow.reset_index(), on=["seg", "dow"], how="left")
    table = table.merge(by_train_dow.reset_index(), on=["train_n", "dow"], how="left")
    table = table.merge(by_seg.reset_index(), on="seg", how="left")
    table = table.merge(by_seg_month, on=["seg", "month"], how="left")

    level = table["dow_mean"].to_numpy(dtype=float)
    level = np.where(np.isnan(level), table["train_dow"].to_numpy(dtype=float), level)
    level = np.where(np.isnan(level), table["dow"].map(by_dow).to_numpy(dtype=float), level)
    level = np.where(np.isnan(level), float(stable["occupancy"].mean()), level)

    own_ratio = table["month_mean"].to_numpy(dtype=float) / table["seg_mean"].replace(0, np.nan).to_numpy(dtype=float)
    global_ratio = table["month"].map(month_ratio).fillna(1.0).to_numpy(dtype=float)
    ratio = shrink(own_ratio, table["month_n"].to_numpy(dtype=float), global_ratio, month_strength)

    pred = level.copy()
    adjust = table["train_n"].isin(adjust_trains).to_numpy()
    pred[adjust] = level[adjust] * ratio[adjust]
    return pd.Series(np.clip(pred, 0, 2.5), index=target.index)


def group_scores(future, pred):
    groups = {
        "main": future["train_n"].isin(MAIN_TRAINS),
        "gabala": future["train_n"].isin([709, 710]),
        "new": future["train_n"].isin([27, 28]),
    }
    parts = []
    for name, mask in groups.items():
        if int(mask.sum()) < 20:
            continue
        value = weighted_rmse(future.loc[mask, "occupancy"], pred.loc[mask], future.loc[mask, "n_segments"])
        parts.append(f"{name} {value:.4f}")
    return " | ".join(parts)


def load_frames():
    train = prepare(pd.read_csv(DATA / "train.csv"))
    test = prepare(pd.read_csv(DATA / "test.csv"))
    keep = set(test["train_n"].unique())
    train = train[train["train_n"].isin(keep)].copy()
    return train, test


def evaluate(train, month_strength, adjust_trains):
    windows = [
        ("jul-oct 2025", "2025-07-01", "2025-11-01"),
        ("nov-feb 2026", "2025-11-01", "2026-03-01"),
        ("feb 2026", "2026-02-01", "2026-03-01"),
    ]
    scores = {}
    for label, origin, end in windows:
        history = train[train["trip_date"] < origin]
        future = train[(train["trip_date"] >= origin) & (train["trip_date"] < end)]
        pred = predict_from_history(history, future, month_strength, adjust_trains)
        scores[label] = weighted_rmse(future["occupancy"], pred, future["n_segments"])
        print(f"  {label}: {scores[label]:.4f}  {group_scores(future, pred)}")
    # Yeni marşrut noyabrda hele yoxdur, ona gore secimi yay penceresi ve fevral ile edirik.
    choice = 0.5 * scores["jul-oct 2025"] + 0.5 * scores["feb 2026"]
    return choice


def main():
    check = weighted_rmse([1, 1, 1], [0, 0, 1], [2, 2, 1])
    if abs(check - np.sqrt(0.5)) > 1e-6:
        raise SystemExit("metric check failed")

    train, test = load_frames()
    print(f"train {len(train)} test {len(test)}")

    options = [
        (4, MAIN_TRAINS),
        (8, MAIN_TRAINS),
        (16, MAIN_TRAINS),
        (8, STABLE_TRAINS),
        (16, STABLE_TRAINS),
        (30, set()),
    ]
    best = None
    for strength, trains in options:
        name = "none" if not trains else ("main" if trains == MAIN_TRAINS else "stable")
        print(f"\nmonth strength {strength}, adjust {name}")
        choice = evaluate(train, strength, trains)
        print(f"  choice score {choice:.4f}")
        if best is None or choice < best[0]:
            best = (choice, strength, trains, name)

    strength, trains, name = best[1], best[2], best[3]
    print(f"\nusing month strength {strength}, adjust {name}")
    pred = predict_from_history(train, test, strength, trains)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    submission = sample[["id"]].copy()
    submission["occupancy"] = submission["id"].map(dict(zip(test["id"], pred.to_numpy())))
    if submission["occupancy"].isna().any() or len(submission) != len(test):
        raise SystemExit("submission ids do not match test")
    submission["occupancy"] = submission["occupancy"].clip(lower=0).round(6)
    submission.to_csv(OUT, index=False)
    print(f"saved {OUT}")
    print(submission["occupancy"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
