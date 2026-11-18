"""Turn the pipeline's tables into the static site's JSON files (pure pandas)."""

import json
import math

import numpy as np
import pandas as pd

KEY = ["Year", "Race"]


def slug(year, rnd):
    return f"{int(year)}-{int(rnd):02d}"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NaT:
        return None
    return o


def dump(obj):
    return json.dumps(_clean(obj), separators=(",", ":"), ensure_ascii=False)


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else v


def _hex(c):
    c = str(c or "").strip().lstrip("#")
    return "#" + c.upper() if len(c) == 6 else "#888888"


def _darken(h, f=0.5):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(int(r * f), int(g * f), int(b * f))


TYRES = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W", "HYPERSOFT": "S", "ULTRASOFT": "S", "SUPERSOFT": "S", "SUPERHARD": "H"}


def compound_letter(c):
    # 2018's hyper/ultra/super-soft count as Soft; unknown or missing -> None
    return TYRES.get(str(c).upper()) if c is not None else None


def team_table(state):
    out = {}
    for (year, team), g in state.groupby(["Year", "TeamName"]):
        col = _hex(g["TeamColor"].dropna().mode().iloc[0] if g["TeamColor"].notna().any() else None)
        out.setdefault(str(int(year)), {})[team] = {"color": col, "accent": _darken(col), "drivers": sorted(g["Driver"].dropna().unique().tolist())}
    return out


def _round_of(results, year, race):
    r = results[(results["Year"] == year) & (results["Race"] == race)]
    return int(r["Round"].iloc[0]) if len(r) else 0


def race_index(results, state, live, prior):
    laps = state.groupby(KEY)["LapNumber"].max()
    circ = state.groupby(KEY)["Circuit"].first()
    live_keys = set(map(tuple, live[KEY].drop_duplicates().values.tolist()))
    prior_keys = set(map(tuple, prior[KEY].drop_duplicates().values.tolist()))
    rows = []
    for (year, rnd, race), g in results.groupby(["Year", "Round", "Race"]):
        k = (year, race)
        has = k in laps.index
        rows.append(dict(slug=slug(year, rnd), year=int(year), round=int(rnd), race=race, circuit=circ.get(k, g["CircuitName"].iloc[0]),
                         location=g["Location"].iloc[0], date=str(g["EventDate"].iloc[0]), laps=int(laps[k]) if has else None,
                         hasLaps=bool(has), hasLive=k in live_keys, hasPrior=k in prior_keys))
    return sorted(rows, key=lambda r: (r["year"], r["round"]))


def _slot_rates(prior, year):
    old = prior[prior["Year"] < year].dropna(subset=["GridPosition", "FinalPosition"])
    if old.empty:
        return {}
    return (old["FinalPosition"] <= 3).groupby(old["GridPosition"].astype(int)).mean().to_dict()


def race_meta(year, race, state, results, prior):
    s = state[(state["Year"] == year) & (state["Race"] == race)]
    res = results[(results["Year"] == year) & (results["Race"] == race)]
    names = dict(zip(res["Driver"], res["FullName"]))
    first = s.sort_values("LapNumber").drop_duplicates("Driver")
    drivers = [dict(code=r.Driver, number=r.DriverNumber, name=names.get(r.Driver, r.BroadcastName), team=r.TeamName, color=_hex(r.TeamColor)) for r in first.itertuples()]
    start = s[s["LapNumber"] == s["LapNumber"].min()].sort_values("Position", na_position="last")
    grid = start["Driver"].tolist()
    grid += [d["code"] for d in drivers if d["code"] not in grid]
    p = prior[(prior["Year"] == year) & (prior["Race"] == race)]
    pri = None
    if len(p):
        rates = _slot_rates(prior, year)
        pri = [dict(code=r.Driver, p=r.Prior, baseline=rates.get(int(r.GridPosition)) if pd.notna(r.GridPosition) else None) for r in p.itertuples()]
    return dict(slug=slug(year, _round_of(results, year, race)), year=int(year), round=_round_of(results, year, race), race=race,
                circuit=s["Circuit"].iloc[0] if len(s) else None, laps=int(s["LapNumber"].max()) if len(s) else 0, drivers=drivers, grid=grid, prior=pri)


def lap_chunks(state, pairs, live, size=10):
    if state.empty:
        return []
    year, race = state["Year"].iloc[0], state["Race"].iloc[0]
    pr = pairs[(pairs["Year"] == year) & (pairs["Race"] == race)]
    lv = live[(live["Year"] == year) & (live["Race"] == race)]
    laps = {}
    for lap, g in state.groupby("LapNumber"):
        g = g.sort_values("Position", na_position="last")
        cars = [dict(d=r.Driver, pos=int(r.Position) if pd.notna(r.Position) else None, gap=_num(r.GapToAhead), lead=_num(r.DeltaToLeader),
                     cmp=compound_letter(r.Compound), age=int(r.TyreLife) if pd.notna(r.TyreLife) else None,
                     pit=bool(r.HasPit), out=bool(pd.isna(r.Position))) for r in g.itertuples()]
        b = pr[pr["LapNumber"] == lap]
        battles = [dict(t=r.Trailing, l=r.Leading, gap=r.Gap, closing=r.Closing, tyre=r.TyreDelta, p=r.P, band=r.Band) for r in b.itertuples()]
        l = lv[lv["LapNumber"] == lap]
        lives = [dict(d=r.Driver, podium=r.PodiumP, win=r.WinP) for r in l.itertuples()]
        laps[int(lap)] = dict(sc=bool(g["SC"].fillna(False).any()), cars=cars, battles=battles, live=lives)
    last = max(laps)
    out = []
    for k in range(0, last // size + 1):
        lo, hi = k * size, k * size + size - 1
        sel = {str(n): v for n, v in laps.items() if lo <= n <= hi and n != last}
        if sel:
            out.append(_clean(dict(chunk=k, **{"from": lo, "to": hi}, laps=sel)))
    # the final lap is its own file so the finishing order only loads at the flag
    out.append(_clean(dict(chunk="final", **{"from": last, "to": last}, laps={str(last): laps[last]})))
    return out


def race_result(year, race, results):
    r = results[(results["Year"] == year) & (results["Race"] == race)].sort_values("FinalPosition")
    return dict(slug=slug(year, _round_of(results, year, race)),
                finish=[dict(code=x.Driver, pos=_num(x.FinalPosition), status=x.Status, points=x.Points) for x in r.itertuples()])


def running_order_table(state, results):
    s = state.dropna(subset=["Position"]).merge(results[["Year", "Race", "Driver", "FinalPosition"]], on=["Year", "Race", "Driver"], how="inner")
    s = s.dropna(subset=["FinalPosition"])
    laps = s.groupby(KEY)["LapNumber"].transform("max").clip(lower=1)
    s = s.assign(Dec=(10 * s["LapNumber"] / laps).astype(int).clip(0, 9), Rank=s["Position"].astype(int))
    out = {}
    for name, hit in (("podium", s["FinalPosition"] <= 3), ("win", s["FinalPosition"] == 1)):
        rates = hit.groupby([s["Dec"], s["Rank"]]).mean()
        tab = {}
        for (d, r), v in rates.items():
            tab.setdefault(str(int(d)), {})[str(int(r))] = float(v)
        out[name] = tab
    return out


def weekend_grid(results, calendar, as_of):
    up = [c for c in calendar if c["sessions"]["race"] and c["sessions"]["race"][:10] >= as_of]
    if not up:
        return None
    nxt = min(up, key=lambda c: (c["year"], c["round"]))
    r = results[(results["Year"] == nxt["year"]) & (results["Round"] == nxt["round"])].dropna(subset=["QualiPosition"]).sort_values("QualiPosition")
    if r.empty:
        return None
    return dict(year=nxt["year"], round=nxt["round"], grid=[dict(driver=x.Driver, team=x.TeamKey, grid=i + 1) for i, x in enumerate(r.itertuples())])


def strategy_profiles(state):
    # what top-10 finishers actually did at each circuit: stop counts, compound order, stint lengths
    rows = []
    for (year, race, circ), g in state.groupby(["Year", "Race", "Circuit"]):
        last = g["LapNumber"].max()
        for drv, x in g.sort_values("LapNumber").groupby("Driver"):
            end = x[x["LapNumber"] == x["LapNumber"].max()]
            if x["LapNumber"].max() < last or end["Position"].isna().all() or end["Position"].iloc[0] > 10:
                continue
            first = x[x["LapNumber"] >= 1]
            if first.empty:
                continue
            pits = first[first["HasPit"] == 1]["LapNumber"].tolist()
            seq = [compound_letter(first["Compound"].iloc[0])] + [compound_letter(c) for c in first[first["HasPit"] == 1]["Compound"]]
            if None in seq:
                continue
            bounds = [1.0] + pits + [last + 1]
            rows.append(dict(Circuit=circ, Year=int(year), Race=race, Laps=float(last), Seq="-".join(seq), Stops=len(pits),
                             Stints=[bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]))
    df = pd.DataFrame(rows)
    out = {}
    for circ, g in df.groupby("Circuit"):
        plans = []
        for seq, h in g.groupby("Seq"):
            n = len(h["Stints"].iloc[0])
            st = [float(np.median([s[i] for s in h["Stints"] if len(s) == n])) for i in range(n)]
            plans.append(dict(seq=seq, stops=int(h["Stops"].iloc[0]), share=round(len(h) / len(g), 4), stints=st))
        plans.sort(key=lambda p: -p["share"])
        stops = (g["Stops"].value_counts(normalize=True).sort_index().round(4))
        out[circ] = dict(races=int(g[["Year", "Race"]].drop_duplicates().shape[0]), years=[int(g["Year"].min()), int(g["Year"].max())],
                         laps=int(g["Laps"].median()), stops={str(int(k)): float(v) for k, v in stops.items()}, plans=plans[:3])
    return out


def _records(df):
    return _clean(df.astype(object).where(df.notna(), None).to_dict("records"))


def podium_bundle(model_json, drivers, teams, circuits, recent_grids):
    return dict(model=json.loads(model_json), drivers=_records(drivers), teams=_records(teams), circuits=_records(circuits), recentGrids=_records(recent_grids))


def how_we_know(live_bt, podium_bt, bands):
    return dict(live=_records(live_bt), podium=_records(podium_bt), bands=_records(bands))


def _iso(s):
    return f"{s['date']}T{s.get('time', '00:00:00Z')}" if s else None


def calendar_from_jolpica(payload):
    out = []
    for r in payload["MRData"]["RaceTable"]["Races"]:
        out.append(dict(year=int(r["season"]), round=int(r["round"]), race=r["raceName"], location=r["Circuit"]["Location"]["locality"], circuitId=r["Circuit"].get("circuitId"),
                        circuit=r["Circuit"]["circuitName"], sessions=dict(quali=_iso(r.get("Qualifying")), sprint=_iso(r.get("Sprint")), race=_iso(r))))
    return out


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(obj), encoding="utf8")


def write_site_data(tables, calendar, out_dir, as_of):
    from pathlib import Path

    out = Path(out_dir)
    t = tables
    state, results, live, pairs, prior = t["f1_replay_lap_state"], t["f1_race_results"], t["f1_live_predictions"], t["f1_battle_pairs"], t["f1_podium_prior_oof"]
    idx = race_index(results, state, live, prior)
    _write(out / "index.json", dict(asOf=as_of, races=idx, calendar=calendar, teams=team_table(state), runningOrder=running_order_table(state, results), weekendGrid=weekend_grid(pd.concat([results, t.get("f1_quali_results", pd.DataFrame())]), calendar, as_of)))
    _write(out / "podium.json", podium_bundle(t["f1_podium_model"]["Json"].iloc[0], t["f1_podium_driver_form"], t["f1_podium_team_form"], t["f1_podium_circuits"], t["f1_podium_recent_grids"]))
    _write(out / "howweknow.json", how_we_know(t["f1_live_backtest"], t["f1_podium_backtest"], t["f1_battle_bands"]))
    _write(out / "strategy.json", strategy_profiles(state))
    files, n = 3, 0
    by_race = dict(tuple(state.groupby(KEY)))
    for r in idx:
        if not r["hasLaps"]:
            continue
        s = by_race[(r["year"], r["race"])]
        d = out / "races" / r["slug"]
        _write(d / "meta.json", race_meta(r["year"], r["race"], s, results, prior))
        _write(d / "result.json", race_result(r["year"], r["race"], results))
        for c in lap_chunks(s, pairs, live):
            _write(d / f"laps-{c['chunk']}.json", c)
            files += 1
        files += 2
        n += 1
    return dict(races=n, files=files)
