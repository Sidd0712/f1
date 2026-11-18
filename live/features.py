"""In-race podium/winner features and scoring, shared by phase_3_live.ipynb and the laptop live mode."""

import numpy as np
import pandas as pd

EARLY_CUT = 0.5
EARLY_TRAIN_MAX = 0.7
K = ["Year", "Race", "LapNumber"]
D = ["Year", "Race", "Driver"]
EARLY = ["PriorLogit", "Rank", "GridDelta", "LapFrac", "GapToP3", "GapToP4", "GapBehind", "PitDebt", "Stops", "TyreLife", "SC", "RelPace"]


def build_features(laps, prior, race_laps=None):
    # laps: master-table rows; prior: Year, Race, Driver, Prior, GridPosition (+ FinalPosition when training)
    d = laps.dropna(subset=["Position", "Time"]).drop_duplicates(D + ["LapNumber"])
    keep = [c for c in ("Year", "Race", "Driver", "Prior", "GridPosition", "FinalPosition") if c in prior]
    d = d.merge(prior[keep], on=D, how="inner")
    if "FinalPosition" in d:
        d["Podium"] = (d["FinalPosition"] <= 3).astype(int)
        d["Win"] = (d["FinalPosition"] == 1).astype(int)

    d = d.sort_values(K + ["Position"]).reset_index(drop=True)
    d["Rank"] = d.groupby(K)["Position"].rank(method="first")
    d["DeltaToLeader"] = d["Time"] - d.groupby(K)["Time"].transform("min")
    d = d.sort_values(K + ["Rank"]).reset_index(drop=True)
    d["GapToAhead"] = d.groupby(K)["Time"].diff()
    d["GapBehind"] = d.groupby(K)["GapToAhead"].shift(-1)
    if race_laps is None:
        d["RaceLaps"] = d.groupby(["Year", "Race"])["LapNumber"].transform("max")
    else:
        d["RaceLaps"] = [race_laps[(y, r)] for y, r in zip(d["Year"], d["Race"])]
    d["LapFrac"], d["LapsLeft"] = d["LapNumber"] / d["RaceLaps"], d["RaceLaps"] - d["LapNumber"]
    d["Stops"] = d["Stint"].fillna(1) - 1
    d["SC"] = d["TrackStatus"].astype(str).str.contains("[45]").astype(int)
    d["TyreLife"] = d["TyreLife"].fillna(0)
    d["PriorLogit"] = np.log(d["Prior"].clip(1e-3, 1 - 1e-3) / (1 - d["Prior"].clip(1e-3, 1 - 1e-3)))
    d["GridDelta"] = (d["Rank"] - d["GridPosition"].where(d["GridPosition"] > 0, 20).fillna(20)).clip(-19, 19)

    # recent pace: median of the last 5 clean laps (no pit in/out, no SC), relative to the leader
    d = d.sort_values(D + ["LapNumber"]).reset_index(drop=True)
    pit = d["PitInTime"].notna() | d["PitOutTime"].notna()
    clean = d["LapTime"].where(~pit & (d["SC"] == 0) & (d["LapNumber"] > 1))
    d["Pace"] = clean.groupby([d["Year"], d["Race"], d["Driver"]]).transform(lambda s: s.rolling(5, min_periods=2).median())
    d["Pace"] = d.groupby(D)["Pace"].ffill()
    d["Closing"] = (d.groupby(D)["GapToAhead"].shift(3) - d["GapToAhead"]) / 3
    d = d.join(d[d["Rank"] == 1].set_index(K)["Pace"].rename("LeadPace"), on=K)
    d["RelPace"] = (d["Pace"] - d["LeadPace"]).clip(-2, 2).fillna(0)
    d = d.join(d[d["Rank"] <= 6].groupby(K)["Stops"].median().rename("TopStops"), on=K)
    d["PitDebt"] = (d["TopStops"] - d["Stops"]).clip(-1, 2)
    for r in (3, 4):
        d = d.join(d[d["Rank"] == r].set_index(K)["DeltaToLeader"].rename(f"D{r}"), on=K)
    d["GapToP3"], d["GapToP4"] = d["DeltaToLeader"] - d["D3"], d["D4"] - d["DeltaToLeader"]
    d = d.sort_values(K + ["Rank"]).reset_index(drop=True)
    d["TyreDeltaAhead"] = d["TyreLife"] - d.groupby(K)["TyreLife"].shift(1)
    d["PaceDeltaAhead"] = d["RelPace"] - d.groupby(K)["RelPace"].shift(1)
    return d


def fit_podium(tr):
    from xgboost import XGBClassifier

    tr = tr[tr["LapFrac"] <= EARLY_TRAIN_MAX]
    return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=50, eval_metric="logloss", random_state=42).fit(tr[EARLY], tr["Podium"])


def win_x(x):
    return np.nan_to_num(np.c_[x["PriorLogit"], np.log(x["Rank"]), np.log(x["Rank"]) * x["LapFrac"], x["GapToAhead"].clip(0, 30),
                               x["GapToAhead"].clip(0, 30) / (x["LapsLeft"] + 1), x["Closing"].clip(-3, 3) * x["LapsLeft"] / 10,
                               x["PaceDeltaAhead"] * x["LapsLeft"] / 10, x["TyreDeltaAhead"] / 10, x["PitDebt"], x["SC"]])


def fit_win(tr):
    from sklearn.linear_model import LogisticRegression

    tr = tr[tr["Rank"] <= 4]
    return LogisticRegression(C=1.0, max_iter=5000).fit(win_x(tr), tr["Win"])


def bucket(x):
    return np.minimum((x["LapFrac"] * 10).astype(int), 9)


def running_lookup(tr):
    # historical podium rate by running position and race-distance decile: the probability behind "trust the running order"
    return tr.groupby([tr["Rank"].clip(upper=20), bucket(tr)])["Podium"].mean(), bucket


def norm_to(p, total):
    # shift logits by one constant so a lap's probabilities sum to `total` (3 for podium, 1 for winner)
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lo, hi = -15.0, 15.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if (1 / (1 + np.exp(-(z + mid)))).sum() > total else (mid, hi)
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


def score_lap(g, podium_proba, win_proba, tab):
    # one lap's field: podium chance (model early, running order late) summing to 3, win chance over the top 4 summing to 1
    early = g["LapFrac"].iloc[0] <= EARLY_CUT
    if early:
        raw = podium_proba(g[EARLY])
    else:
        raw = np.array([tab.get((min(int(r), 20), int(b)), 0.0) for r, b in zip(g["Rank"], bucket(g))])
    podium = norm_to(np.asarray(raw, float), min(3, len(g)))
    win = np.zeros(len(g))
    top = (g["Rank"] <= 4).to_numpy()
    if top.any():
        win[top] = norm_to(np.asarray(win_proba(win_x(g[top])), float), 1)
    return podium, win
