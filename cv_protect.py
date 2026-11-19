import numpy as np

import build_submission as b
import cv_harness as h
import cv_round3 as c

train = h.load_train()
BASE = {"mimic": 0.1303, "summer": 0.0973, "winter": 0.1036, "route": 0.0594}


def protected(history, future):
    raw = c.predict(history, future, 0.0)
    base = b.predict_from_history(history, future, 4, b.MAIN_TRAINS)
    level = b.same_month_level(history, future, base, 2)
    pre = c.resort(history, future, level, 0.0)
    pinned = np.abs(raw - np.asarray(pre, dtype=float)) > 1e-8
    adjusted = c.holiday_layer(history, future, raw, 16)
    out = adjusted.copy()
    out[pinned] = raw[pinned]
    return out

