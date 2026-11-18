import json
import shutil
from pathlib import Path

import numpy as np

from live.engine import LiveRace, ReplaySource
from live.test_features import prior, toy

MODELS = Path(__file__).resolve().parent.parent / "site" / "public" / "data" / "livemodels"


def race():
    laps = toy(8)
    laps["Compound"], laps["HasPit"] = "MEDIUM", 0
    return LiveRace(2025, "Toy GP", 50, prior(), MODELS), laps


def test_one_state_per_lap_and_sums():
    lr, laps = race()
    states = []
    for frame in ReplaySource(laps).frames():
        states += lr.update(frame)
    states += lr.finish()
    assert [s["lap"] for s in states] == list(range(1, 9))
    for s in states:
        pod = sum(x["podium"] for x in s["live"])
        win = sum(x["win"] for x in s["live"])
        assert abs(pod - 3) < 1e-3 and abs(win - 1) < 1e-3
        assert s["cars"][0]["pos"] == 1 and s["cars"][0]["gap"] is None


def test_json_has_no_nan():
    lr, laps = race()
    for frame in ReplaySource(laps).frames():
        lr.update(frame)
    out = lr.snapshot()
    assert "NaN" not in json.dumps(out) and out["laps"]


def test_driver_missing_twice_is_out():
    lr, laps = race()
    laps = laps[~((laps.Driver == "DDD") & (laps.LapNumber >= 5))]
    for frame in ReplaySource(laps).frames():
        lr.update(frame)
    last = lr.snapshot()["laps"][str(7)]
    ddd = [c for c in last["cars"] if c["d"] == "DDD"]
    assert ddd and ddd[0]["out"] is True


def staggered_frames(laps, behind="DDD", lag=2):
    # leader at lap n while one car is `lag` laps behind on track, as in a real session
    top = int(laps.LapNumber.max())
    for n in range(1, top + lag + 1):
        yield laps[((laps.Driver != behind) & (laps.LapNumber <= n)) | ((laps.Driver == behind) & (laps.LapNumber <= n - lag))]


def test_lapped_car_is_not_marked_out():
    lr, laps = race()
    for frame in staggered_frames(laps):
        lr.update(frame)
    snap = lr.snapshot()["laps"]
    for n in range(1, 8):
        ddd = [c for c in snap[str(n)]["cars"] if c["d"] == "DDD"]
        assert ddd and not ddd[0]["out"] and ddd[0]["pos"] is not None, n


def test_final_lap_emitted_when_leader_reaches_race_distance():
    lr, laps = race()
    lr.race_laps = 8
    for frame in ReplaySource(laps).frames():
        lr.update(frame)
    assert "8" in lr.snapshot()["laps"]
