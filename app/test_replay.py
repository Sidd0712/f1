import pandas as pd

import replay as rp


def make_state():
    rows = []
    # B passes A on lap 4 (clean), C retires after lap 3, D pits on lap 5
    pos = {"A": [1, 1, 1, 2, 2, 2], "B": [2, 2, 2, 1, 1, 1], "C": [3, 3, 3, None, None, None], "D": [4, 4, 4, 3, 3, 3]}
    gap = {"A": [None, None, None, 0.6, 0.5, 0.5], "B": [None, 0.9, 0.7, None, None, None], "C": [None, 5, 5, None, None, None], "D": [None, 9, 9, 9, 9, 9]}
    for d, ps in pos.items():
        for lap, p in enumerate(ps):
            if p is None:
                continue
            rows.append(dict(Year=2023, Race="Test GP", Circuit="T", Driver=d, DriverNumber="1", BroadcastName=f"{d} DRIVER", TeamName="Team",
                             TeamColor="ff0000", LapNumber=lap, Position=p, GapToAhead=gap[d][lap], DeltaToLeader=None, Compound="SOFT",
                             TyreLife=lap + 1, HasPit=int(d == "D" and lap == 5), SC=False))
    return pd.DataFrame(rows)


def test_replay():
    v = rp.RaceView(make_state(), None, None)
    cards = v.cards(4)
    assert [c["code"] for c in cards if not c["out"]] == ["B", "A", "D"]
    assert [c["code"] for c in cards if c["out"]] == ["C"]
    assert [c["pos"] for c in v.cards(0)] == [1, 2, 3, 4]
    assert any("passes" in e["text"] for e in v.moments_upto(4))
    assert not any(e["lap"] > 2 for e in v.moments_upto(2))
    assert rp.team_color("zz") == rp.FALLBACK_COLOR
    assert "<details" in rp.render_grid(v.cards(4), 4)
    html_live = rp.render_grid(v.cards(4), 4, live={"B": (0.82, 0.41), "A": (0.7, 0.004), "C": (0.5, 0.2)})
    assert "Podium 82%" in html_live and "Win 41%" in html_live and "Win 0%" not in html_live
    assert "Podium 50%" not in html_live  # retired car gets no estimate
    bt = pd.DataFrame([dict(Target="Podium", Window="0-50%", Model=.75, RunningOrder=.725, Gain=.025, CI_Lo=.008, CI_Hi=.043, Ship=True),
                       dict(Target="Winner", Window="70-100%", Model=.915, RunningOrder=.887, Gain=.028, CI_Lo=-.005, CI_Hi=.068, Ship=False)])
    note = rp.live_note(bt)
    assert "+2.5 pts" in note and "experimental" in note


if __name__ == "__main__":
    test_replay()
    print("ok")
