"""Live race state: laps so far -> export-shaped lap states with podium/win chances."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from live import features as lf
from site_export.export import _clean


class ReplaySource:
    # a finished race fed back lap by lap, to test live mode between race weekends
    def __init__(self, laps):
        self.laps = laps

    def frames(self):
        for n in sorted(self.laps["LapNumber"].dropna().unique()):
            yield self.laps[self.laps["LapNumber"] <= n]


class LiveRace:
    def __init__(self, year, race, race_laps, prior, models_dir):
        from xgboost import XGBClassifier

        m = Path(models_dir)
        self.pod = XGBClassifier()
        self.pod.load_model(str(m / "podium_xgb.json"))
        w = json.loads((m / "win_logit.json").read_text())
        self.win_coef, self.win_b = np.array(w["coef"]), w["intercept"]
        self.tab = {tuple(int(x) for x in k.split(",")): v for k, v in json.loads((m / "running.json").read_text()).items()}
        self.year, self.race, self.race_laps = year, race, race_laps
        self.prior = prior[(prior["Year"] == year) & (prior["Race"] == race)]
        self.frame = None
        self.states = {}
        self.updated = None

    def _win(self, x):
        return 1 / (1 + np.exp(-(x @ self.win_coef + self.win_b)))

    def _emit(self, upto):
        d = lf.build_features(self.frame, self.prior, race_laps={(self.year, self.race): self.race_laps})
        raw = self.frame.drop_duplicates(["Driver", "LapNumber"]).set_index(["Driver", "LapNumber"])
        seen = set()
        out = []
        for n in sorted(d["LapNumber"].unique()):
            g = d[d["LapNumber"] == n].sort_values("Rank")
            here = set(g["Driver"])
            if n > upto or n in self.states:
                seen |= here
                continue
            podium, win = lf.score_lap(g, lambda x: self.pod.predict_proba(x)[:, 1], self._win, self.tab)
            cars = []
            for r in g.itertuples():
                row = raw.loc[(r.Driver, n)]
                cmp = row.get("Compound")
                cars.append(dict(d=r.Driver, pos=int(r.Rank), gap=None if r.Rank == 1 else r.GapToAhead, lead=r.DeltaToLeader,
                                 cmp=cmp[0] if isinstance(cmp, str) and cmp else None, age=int(r.TyreLife),
                                 pit=bool(row.get("HasPit", 0)) or pd.notna(row.get("PitOutTime")), out=False))
            prev = self.states.get(n - 1)
            prev_missing = {c["d"] for c in prev["cars"] if c["out"] or c.get("missing")} if prev else set()
            for drv in sorted(seen - here):
                cars.append(dict(d=drv, pos=None, gap=None, lead=None, cmp=None, age=None, pit=False, out=drv in prev_missing, missing=True))
            seen |= here
            st = _clean(dict(lap=int(n), sc=bool(g["SC"].max()), cars=cars, battles=[],
                             live=[dict(d=x, podium=p, win=w) for x, p, w in zip(g["Driver"], podium, win)]))
            self.states[n] = st
            out.append(st)
        self.updated = time.time()
        return out

    def update(self, frame):
        # a lap is final once every car that ran the lap before has completed it, or the leader is two laps further on
        self.frame = frame
        top = frame["LapNumber"].max()
        if top >= self.race_laps:
            return self._emit(top)
        rows = frame.dropna(subset=["Position"]).groupby("LapNumber")["Driver"].agg(set)
        upto = max(self.states) if self.states else 0
        for n in range(int(upto) + 1, int(top)):
            need = rows.get(n - 1, set()) if n > 1 else set(self.prior["Driver"])
            if not (need <= rows.get(n, set()) or top >= n + 2):
                break
            upto = n
        return self._emit(upto)

    def finish(self):
        return self._emit(self.frame["LapNumber"].max()) if self.frame is not None else []

    def snapshot(self):
        return _clean(dict(year=self.year, race=self.race, raceLaps=self.race_laps, updatedAt=self.updated,
                           laps={str(int(n)): s for n, s in sorted(self.states.items())}))
