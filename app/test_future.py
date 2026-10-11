import numpy as np
import pandas as pd

import future as fu


def make_model():
    feats = ["LogGrid", "SlotLogit", "QualiGapPct", "TeamPace", "TeamQuali", "TeamDnf", "DriverEdge", "CircuitMobility"]
    m = dict(features=feats, mu={f: 0.0 for f in feats}, sd={f: 1.0 for f in feats}, coef=[-0.3, 1.1, 0.0, 0.8, -0.5, 0.2, 0.1, 0.0], intercept=-1.0,
             ensemble=[dict(coef=[-0.3, 1.1, 0.0, 0.8, -0.5, 0.2, 0.1, 0.0], intercept=-1.0)] * 3,
             fills=dict(gap_global=1.0, TeamPace=2.0, TeamQuali=4.0, TeamDnf=0.3, DriverEdge=0.0, CircuitMobility=3.0),
             gap_by_slot={str(i): i * 0.2 for i in range(1, 7)}, slot_table={str(i): max(0.65 - 0.12 * i, 0.02) for i in range(1, 7)})
    drivers = pd.DataFrame({"Driver": list("ABCDEF"), "FullName": list("abcdef"), "TeamKey": ["T1", "T1", "T2", "T2", "T3", "T3"], "DriverEdge": 0.0,
                            "DriverRaces": 20, "DriverNew": [False] * 5 + [True], "Current": True})
    teams = pd.DataFrame({"TeamKey": ["T1", "T2", "T3"], "TeamPace": [15.0, 8.0, np.nan], "TeamQuali": [0.5, 2.0, np.nan], "TeamDnf": [0.1, 0.2, np.nan], "TeamNew": [False, False, True]})
    return fu.FutureModel(m, drivers, teams, pd.DataFrame({"Location": ["x"], "CircuitName": ["X"]}), pd.DataFrame())


def test_future():
    fm = make_model()
    entries = [dict(driver=d, team=t, grid=i + 1) for i, (d, t) in enumerate(zip("ABCDEF", ["T1", "T1", "T2", "T2", "T3", "T3"]))]
    out = fm.predict(entries)
    assert abs(sum(o["p"] for o in out) - 3.0) < 1e-6
    assert out[0]["p"] > out[3]["p"] > 0
    assert out[4]["new_team"] and out[5]["new_driver"]
    assert all(o["lo"] <= o["p"] <= o["hi"] for o in out)
    assert "<details" in fu.render_results(out, fu.TEAM_COLORS, {})


if __name__ == "__main__":
    test_future()
    print("ok")
