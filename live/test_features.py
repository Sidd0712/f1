import numpy as np
import pandas as pd

from live import features as lf


def toy(n_laps=6):
    rows = []
    for lap in range(1, n_laps + 1):
        for i, d in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            gap = [0.0, 1.0 + 0.3 * (n_laps - lap), 5.0, 9.0][i]
            rows.append(dict(Year=2025, Race="Toy GP", Driver=d, LapNumber=float(lap), Position=float(i + 1), LapTime=90.0 + i * 0.1,
                             Time=lap * 90.0 + sum([0.0, 1.0 + 0.3 * (n_laps - lap), 5.0, 9.0][: i + 1]), Stint=1.0 if lap < 4 or d != "AAA" else 2.0,
                             TyreLife=float(lap), TrackStatus="1", PitInTime=None, PitOutTime=None))
    return pd.DataFrame(rows)


def prior():
    return pd.DataFrame([dict(Year=2025, Race="Toy GP", Driver=d, Prior=p, GridPosition=float(g)) for d, p, g in
                         (("AAA", 0.7, 1), ("BBB", 0.6, 2), ("CCC", 0.4, 3), ("DDD", 0.3, 4))])


def test_gap_and_rank():
    d = lf.build_features(toy(), prior(), race_laps={(2025, "Toy GP"): 50})
    last = d[d.LapNumber == 6].set_index("Driver")
    assert last.loc["AAA", "Rank"] == 1 and np.isnan(last.loc["AAA", "GapToAhead"])
    assert abs(last.loc["CCC", "GapToAhead"] - 5.0) < 1e-9
    assert last.loc["AAA", "LapsLeft"] == 44


def test_no_look_ahead():
    full = lf.build_features(toy(6), prior(), race_laps={(2025, "Toy GP"): 50})
    part = lf.build_features(toy(6)[lambda x: x.LapNumber <= 4], prior(), race_laps={(2025, "Toy GP"): 50})
    cols = lf.EARLY + ["Closing", "PaceDeltaAhead", "TyreDeltaAhead"]
    a = full[full.LapNumber == 4].set_index("Driver")[cols].sort_index()
    b = part[part.LapNumber == 4].set_index("Driver")[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_norm_to_sums():
    p = lf.norm_to(np.array([0.9, 0.5, 0.4, 0.1]), 3)
    assert abs(p.sum() - 3) < 1e-3
