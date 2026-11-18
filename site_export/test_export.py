import json
import math

import pandas as pd
import pytest

from site_export import export as ex


def state_df():
    rows = []
    for race, circ, year in (("British Grand Prix", "Silverstone Circuit", 2020), ("70th Anniversary Grand Prix", "Silverstone Circuit", 2020)):
        for lap in range(0, 3):
            for i, (d, team, col) in enumerate((("HAM", "Mercedes", "00d2be"), ("VER", "Red Bull Racing", "1e41ff"), ("BOT", "Mercedes", "00d2be"))):
                pos = float(i + 1)
                if d == "BOT" and lap > 0:
                    pos = float("nan")
                rows.append(dict(Year=year, Race=race, Circuit=circ, Driver=d, DriverNumber=str(10 + i), BroadcastName=f"X {d}", TeamName=team, TeamColor=col,
                                 LapNumber=float(lap), Position=pos, GapToAhead=None if pos != pos or pos == 1 else 1.5, DeltaToLeader=0.0 if pos == 1 else 1.5,
                                 Compound="SOFT", TyreLife=float(lap + 1), HasPit=0, SC=False))
    return pd.DataFrame(rows)


def results_df():
    rows = []
    for rnd, race, circ in ((4, "British Grand Prix", "Silverstone Circuit"), (5, "70th Anniversary Grand Prix", "Silverstone Circuit"), (6, "Spanish Grand Prix", "Circuit de Barcelona-Catalunya")):
        for i, d in enumerate(("VER", "HAM", "BOT")):
            rows.append(dict(Year=2020, Round=rnd, Race=race, Location="x", CircuitName=circ, EventDate=f"2020-08-{rnd:02d}", Driver=d, DriverNumber="1", FullName=d,
                             TeamName="T", TeamKey="T", GridPosition=float(i + 1), FinalPosition=i + 1, Status="Finished", Points=25.0 - i, QualiPosition=float(i + 1), QualiGapPct=0.0))
    return pd.DataFrame(rows)


def live_df():
    return pd.DataFrame([dict(Year=2020, Race="British Grand Prix", LapNumber=1.0, Driver="HAM", Rank=1.0, PodiumP=0.9, WinP=0.6, PodiumSource="model")])


def pairs_df():
    return pd.DataFrame([dict(Year=2020, Race="British Grand Prix", Circuit="Silverstone Circuit", LapNumber=2, Trailing="VER", Leading="HAM", Gap=1.5, Closing=0.2,
                              TyreDelta=3.0, Clean=True, Y=0, P=0.12, Band="Medium")])


def prior_df():
    return pd.DataFrame([dict(Year=2020, Race="British Grand Prix", Round=4.0, Driver=d, Prior=p, GridPosition=float(g), FinalPosition=float(f))
                         for d, p, g, f in (("HAM", 0.8, 1, 2), ("VER", 0.7, 2, 1), ("BOT", 0.5, 3, 3))])


def test_slug():
    assert ex.slug(2026, 7) == "2026-07"


def test_team_table_colors():
    t = ex.team_table(state_df())["2020"]
    assert t["Mercedes"]["color"] == "#00D2BE"
    assert t["Mercedes"]["drivers"] == ["BOT", "HAM"]
    assert int(t["Mercedes"]["accent"][1:3], 16) < int(t["Mercedes"]["color"][1:3], 16) or t["Mercedes"]["color"][1:3] == "00"


def test_index_has_no_results():
    idx = ex.race_index(results_df(), state_df(), live_df(), prior_df())
    blob = json.dumps(idx)
    for k in ("winner", "FinalPosition", "finish", "podium", "Points"):
        assert k not in blob


def test_same_circuit_twice():
    idx = ex.race_index(results_df(), state_df(), live_df(), prior_df())
    slugs = [r["slug"] for r in idx if r["circuit"] == "Silverstone Circuit"]
    assert slugs == ["2020-04", "2020-05"]


def test_results_only_race():
    idx = {r["race"]: r for r in ex.race_index(results_df(), state_df(), live_df(), prior_df())}
    assert idx["Spanish Grand Prix"]["hasLaps"] is False
    assert idx["British Grand Prix"]["hasLaps"] is True
    assert idx["British Grand Prix"]["hasLive"] is True and idx["70th Anniversary Grand Prix"]["hasLive"] is False


def test_meta_has_no_final_positions():
    meta = ex.race_meta(2020, "British Grand Prix", state_df(), results_df(), prior_df())
    assert "FinalPosition" not in json.dumps(meta) and "finish" not in meta
    assert meta["grid"] == ["HAM", "VER", "BOT"]
    assert meta["slug"] == "2020-04" and meta["laps"] == 2
    assert {p["code"] for p in meta["prior"]} == {"HAM", "VER", "BOT"}


def test_meta_without_lap0():
    s = state_df()
    s = s[s["LapNumber"] > 0]
    meta = ex.race_meta(2020, "British Grand Prix", s, results_df(), prior_df())
    assert meta["grid"][:2] == ["HAM", "VER"]


def test_lap1_retiree_kept():
    meta = ex.race_meta(2020, "British Grand Prix", state_df(), results_df(), prior_df())
    assert "BOT" in [d["code"] for d in meta["drivers"]]
    chunks = ex.lap_chunks(state_df()[lambda d: d.Race == "British Grand Prix"], pairs_df(), live_df())
    lap1 = {c["d"]: c for c in chunks[0]["laps"]["1"]["cars"]}
    assert lap1["BOT"]["out"] is True


def test_chunks_cover_all_laps():
    s = state_df()[lambda d: d.Race == "British Grand Prix"]
    chunks = ex.lap_chunks(s, pairs_df(), live_df(), size=2)
    keys = sorted(int(k) for c in chunks for k in c["laps"])
    assert keys == [0, 1, 2]
    assert "0" in chunks[0]["laps"]
    lap2 = chunks[1]["laps"]["2"]
    assert lap2["battles"][0]["t"] == "VER" and lap2["battles"][0]["p"] == 0.12
    assert chunks[0]["laps"]["1"]["live"][0] == {"d": "HAM", "podium": 0.9, "win": 0.6}


def test_result_separate():
    r = ex.race_result(2020, "British Grand Prix", results_df())
    assert [f["code"] for f in r["finish"]] == ["VER", "HAM", "BOT"]


def test_no_nan_in_json():
    out = ex.dump({"a": float("nan"), "b": [math.inf, 1.234567]})
    assert json.loads(out) == {"a": None, "b": [None, 1.2346]}


def jolpica_payload():
    races = [
        {"season": "2026", "round": "22", "raceName": "Las Vegas Grand Prix", "date": "2026-11-21", "time": "06:00:00Z",
         "Circuit": {"circuitId": "vegas", "circuitName": "Las Vegas Strip Street Circuit", "Location": {"locality": "Las Vegas"}},
         "Qualifying": {"date": "2026-11-20", "time": "06:00:00Z"}},
        {"season": "2026", "round": "23", "raceName": "Qatar Grand Prix", "date": "2026-11-29", "time": "16:00:00Z",
         "Circuit": {"circuitName": "Losail International Circuit", "Location": {"locality": "Al Daayen"}},
         "Qualifying": {"date": "2026-11-28", "time": "18:00:00Z"}, "Sprint": {"date": "2026-11-28", "time": "14:00:00Z"}},
    ]
    return {"MRData": {"RaceTable": {"Races": races}}}


def test_calendar_parse():
    cal = ex.calendar_from_jolpica(jolpica_payload())
    assert cal[0]["race"] == "Las Vegas Grand Prix" and cal[0]["round"] == 22
    assert cal[0]["sessions"] == {"quali": "2026-11-20T06:00:00Z", "sprint": None, "race": "2026-11-21T06:00:00Z"}
    assert cal[1]["sessions"]["sprint"] == "2026-11-28T14:00:00Z"
    assert cal[0]["circuitId"] == "vegas" and cal[1]["circuitId"] is None


def podium_frames():
    model = json.dumps({"features": ["LogGrid"], "mu": {"LogGrid": 1.0}, "sd": {"LogGrid": 1.0}, "coef": [-1.0], "intercept": 0.0, "ensemble": [],
                        "fills": {}, "gap_by_slot": {"1": 0.0}, "slot_table": {"1": 0.6}})
    drivers = pd.DataFrame([dict(Driver="HAM", FullName="Lewis Hamilton", TeamName="Ferrari", TeamKey="Ferrari", DriverNumber="44", DriverEdge=0.1, DriverRaces=20, DriverNew=False, Current=True)])
    teams = pd.DataFrame([dict(TeamKey="Ferrari", TeamPace=0.5, TeamQuali=0.4, TeamDnf=0.1, TeamSupport=10.0, TeamNew=False, SeasonRaces=20)])
    circuits = pd.DataFrame([dict(Location="monza", CircuitName="Autodromo Nazionale di Monza", CircuitMobility=0.3)])
    grids = pd.DataFrame([dict(Year=2026, Race="X", Round=1.0, Driver="HAM", TeamKey="Ferrari", GridPosition=3.0)])
    return model, drivers, teams, circuits, grids


def test_bundle_roundtrip():
    b = ex.podium_bundle(*podium_frames())
    assert set(b) == {"model", "drivers", "teams", "circuits", "recentGrids"}
    assert b["model"]["coef"] == [-1.0] and b["drivers"][0]["Driver"] == "HAM"


def tables():
    model, drivers, teams, circuits, grids = podium_frames()
    return {
        "f1_replay_lap_state": state_df(), "f1_race_results": results_df(), "f1_live_predictions": live_df(), "f1_battle_pairs": pairs_df(),
        "f1_podium_prior_oof": prior_df(), "f1_podium_model": pd.DataFrame([{"Json": model}]), "f1_podium_driver_form": drivers,
        "f1_podium_team_form": teams, "f1_podium_circuits": circuits, "f1_podium_recent_grids": grids,
        "f1_live_backtest": pd.DataFrame([dict(Target="podium", Window="0-50", Model=0.756, RunningOrder=0.731, Gain=0.025, CI_Lo=0.012, CI_Hi=0.039, Races=112, Ship=True)]),
        "f1_podium_backtest": pd.DataFrame([dict(Year=2025, Model="model", Races=24, auc=0.9, brier=0.08, logloss=0.3)]),
        "f1_battle_bands": pd.DataFrame([dict(Band="Low", P_Lo=0.0, P_Hi=0.05, ObservedRate=0.027, Pairs=100)]),
    }


def test_write_layout(tmp_path):
    summary = ex.write_site_data(tables(), ex.calendar_from_jolpica(jolpica_payload()), str(tmp_path), "2026-11-16")
    idx = json.loads((tmp_path / "index.json").read_text(encoding="utf8"))
    assert idx["asOf"] == "2026-11-16" and len(idx["calendar"]) == 2 and "2020" in idx["teams"]
    for r in idx["races"]:
        d = tmp_path / "races" / r["slug"]
        assert d.exists() == r["hasLaps"]
        if r["hasLaps"]:
            assert (d / "meta.json").exists() and (d / "result.json").exists() and list(d.glob("laps-*.json"))
    for f in tmp_path.rglob("*.json"):
        json.loads(f.read_text(encoding="utf8"))
    assert "Finished" not in (tmp_path / "index.json").read_text(encoding="utf8")
    assert json.loads((tmp_path / "podium.json").read_text(encoding="utf8"))["model"]["coef"] == [-1.0]
    assert json.loads((tmp_path / "howweknow.json").read_text(encoding="utf8"))["live"][0]["Gain"] == 0.025
    assert summary["races"] == 2


def test_running_order_table():
    t = ex.running_order_table(state_df(), results_df())
    assert set(t) == {"podium", "win"}
    assert t["win"]["9"]["2"] == 1.0
    assert t["win"]["9"]["1"] == 0.0
    assert t["podium"]["0"]["1"] == 1.0
    assert "NaN" not in ex.dump(t)


def test_weekend_grid():
    cal = ex.calendar_from_jolpica(jolpica_payload())
    res = results_df()
    assert ex.weekend_grid(res, cal, "2026-11-17") is None
    extra = res[res["Round"] == 4].assign(Year=2026, Round=22, Race="Las Vegas Grand Prix", FinalPosition=None, QualiPosition=[3.0, 1.0, 2.0])
    wg = ex.weekend_grid(pd.concat([res, extra]), cal, "2026-11-17")
    assert wg["round"] == 22 and [g["driver"] for g in wg["grid"]] == ["HAM", "BOT", "VER"]
    assert [g["grid"] for g in wg["grid"]] == [1, 2, 3]


def test_final_lap_in_its_own_chunk():
    s = state_df()[lambda d: d.Race == "British Grand Prix"]
    chunks = ex.lap_chunks(s, pairs_df(), live_df(), size=10)
    assert chunks[-1]["chunk"] == "final" and list(chunks[-1]["laps"]) == ["2"]
    assert all("2" not in c["laps"] for c in chunks[:-1])


def strat_state():
    rows = []
    for race, year in (("A GP", 2024), ("B GP", 2025)):
        for d, pit_lap, c2 in (("AAA", 5, "HARD"), ("BBB", 6, "HARD"), ("CCC", None, None)):
            for lap in range(1, 11):
                cmp = "MEDIUM" if pit_lap is None or lap < pit_lap else c2
                rows.append(dict(Year=year, Race=race, Circuit="Monza", Driver=d, LapNumber=float(lap), Position=float(["AAA", "BBB", "CCC"].index(d) + 1),
                                 Compound=cmp, TyreLife=1.0, HasPit=int(pit_lap == lap), SC=False))
    return pd.DataFrame(rows)


def test_strategy_profiles():
    p = ex.strategy_profiles(strat_state())["Monza"]
    assert p["races"] == 2 and p["years"] == [2024, 2025] and p["laps"] == 10
    assert p["stops"] == {"0": 0.3333, "1": 0.6667}
    top = p["plans"][0]
    assert top["seq"] == "M-H" and top["stops"] == 1 and abs(top["share"] - 0.6667) < 1e-3
    assert top["stints"] == [4.5, 5.5]


def test_strategy_skips_unknown_compounds():
    s = strat_state()
    s.loc[(s.Driver == "AAA") & (s.LapNumber >= 5), "Compound"] = None
    p = ex.strategy_profiles(s)["Monza"]
    assert all("n" not in pl["seq"] for pl in p["plans"])


def test_compound_letters():
    assert [ex.compound_letter(c) for c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "nan", "UNKNOWN", None, "TEST_UNKNOWN")] == \
        ["S", "M", "H", "I", "W", "S", "S", "S", None, None, None, None]
