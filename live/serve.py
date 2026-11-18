"""Laptop live mode: run a race (live or replayed), serve the site + /f1/live/state.json, push race changers to the phone.

  python -m live.serve --replay live/out/2025-12.csv --slug 2025-12 --every 2     # test with a past race
  python -m live.serve --live --year 2026 --round 22 --race "Qatar Grand Prix"    # race day (FastF1 + your F1 TV login)
"""

import argparse
import json
import socket
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from live import push
from live.changers import race_changers
from live.engine import LiveRace, ReplaySource

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "site" / "dist"
DATA = ROOT / "site" / "public" / "data"


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.meta, self.race, self.later, self.alerted = None, None, False, 0

    def snapshot(self):
        with self.lock:
            snap = self.race.snapshot() if self.race else {}
            if self.later:
                snap = dict(snap, laps={})
            return dict(meta=self.meta, later=self.later, **snap)


def prior_from_meta(meta):
    rows = meta.get("prior") or [dict(code=c, p=0.15) for c in meta["grid"]]
    grid = {c: i + 1 for i, c in enumerate(meta["grid"])}
    return pd.DataFrame([dict(Year=meta["year"], Race=meta["race"], Driver=r["code"], Prior=r["p"], GridPosition=grid.get(r["code"], 20)) for r in rows])


def live_meta(index, year, round_, race, laps):
    # race-day meta: exported grid when qualifying is in, else the podium model's projected order; prior from the podium model
    import sys

    sys.path.insert(0, str(ROOT / "app"))
    from future import FutureModel

    b = json.loads((DATA / "podium.json").read_text(encoding="utf8"))
    m = FutureModel(b["model"], pd.DataFrame(b["drivers"]), pd.DataFrame(b["teams"]), pd.DataFrame(b["circuits"]), pd.DataFrame(b["recentGrids"]))
    wg = index.get("weekendGrid")
    if wg and wg["year"] == year and wg["round"] == round_:
        entries = [dict(driver=g["driver"], team=g["team"], grid=g["grid"]) for g in wg["grid"]]
    else:
        teams = m.current_drivers["TeamKey"].to_dict()
        entries = [dict(driver=d, team=teams[d], grid=i + 1) for i, d in enumerate(m.default_order())]
    preds = m.predict(entries)
    colors = {d: v["color"] for t, v in index["teams"][max(index["teams"])].items() for d in v["drivers"]}
    drivers = [dict(code=e["driver"], number="", name=e["driver"], team=e["team"], color=colors.get(e["driver"], "#888888")) for e in entries]
    return dict(slug="live", year=year, round=round_, race=race, circuit=race, laps=laps, drivers=drivers,
                grid=[e["driver"] for e in sorted(entries, key=lambda e: e["grid"])], prior=[dict(code=r["driver"], p=r["p"], baseline=r["baseline"]) for r in preds])


def run(state, source, url):
    for frame in source.frames():
        with state.lock:
            new = state.race.update(frame)
            laps = {int(k): v for k, v in state.race.snapshot()["laps"].items()}
        if not new:
            continue
        found = race_changers(laps, max(laps))
        if len(found) > state.alerted:
            state.alerted = len(found)
            title, body = push.alert_text(found[-1], state.meta["race"], state.later)
            push.send(title, body, url)
    with state.lock:
        state.race.finish()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, state=None, **k):
        self.state = state
        super().__init__(*a, directory=str(DIST), **k)

    def translate_path(self, path):
        p = unquote(urlparse(path).path)
        if p.startswith("/f1/data/"):
            # only files inside the exported data folder; anything else maps to a path that doesn't exist
            root = DATA.resolve()
            t = (root / p[len("/f1/data/"):].lstrip("/\\")).resolve()
            return str(t) if t == root or root in t.parents else str(root / "__forbidden__")
        return super().translate_path(p[len("/f1"):] if p.startswith("/f1/") else p)

    def _send(self, code, body, ctype):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        # watching mode is changed only by POST from the Race page (a plain link or image can't flip it)
        if urlparse(self.path).path != "/f1/live/mode":
            return self._send(404, "Not found", "text/plain")
        body = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 16)).decode(errors="ignore")
        self.state.later = body.strip() == "later"
        return self._send(200, json.dumps({"later": self.state.later}), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/f1/live/state.json":
            return self._send(200, json.dumps(self.state.snapshot()), "application/json")
        if u.path == "/f1/live/vapid":
            return self._send(200, push.vapid_public_key(), "text/plain")
        if u.path == "/f1/live/subscribe" and "s" in q:
            try:
                push.save_subscription(push.parse_subscription(q["s"][0]))
                return self._send(200, "<h1>Race alerts are on for this phone.</h1><p><a href='/f1/#/race'>Open the live race</a></p>", "text/html")
            except ValueError:
                return self._send(400, "Not a valid alert subscription.", "text/plain")
        if u.path in ("/", "/f1"):
            self.send_response(302)
            self.send_header("Location", "/f1/#/race")
            self.end_headers()
            return
        return super().do_GET()


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="CSV of master-table laps for one past race")
    ap.add_argument("--slug", help="race slug in the exported site data (for drivers, grid, prior)")
    ap.add_argument("--every", type=float, default=2.0, help="seconds per lap when replaying")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--year", type=int)
    ap.add_argument("--round", type=int)
    ap.add_argument("--race")
    ap.add_argument("--laps", type=int, help="scheduled race distance")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--later", action="store_true", help="watching later: spoiler-free alerts and a blank Race page")
    a = ap.parse_args()

    state = State()
    if a.replay:
        meta = json.loads((DATA / "races" / a.slug / "meta.json").read_text(encoding="utf8"))
        laps = pd.read_csv(a.replay)

        class Paced(ReplaySource):
            def frames(self):
                for f in super().frames():
                    time.sleep(a.every)
                    yield f

        source = Paced(laps)
    else:
        from live.feed import FastF1Source

        if not (a.year and a.round and a.race and a.laps):
            ap.error("--live needs --year, --round, --race and --laps (the scheduled race distance)")
        meta = live_meta(json.loads((DATA / "index.json").read_text(encoding="utf8")), a.year, a.round, a.race, a.laps)
        source = FastF1Source(a.year, a.round, a.race, ROOT / "live" / "out")
    meta = dict(meta, laps=a.laps or meta["laps"])
    state.meta = meta
    state.later = a.later
    state.race = LiveRace(meta["year"], meta["race"], meta["laps"], prior_from_meta(meta), DATA / "livemodels")
    url = f"http://{lan_ip()}:{a.port}/f1/#/race"
    threading.Thread(target=run, args=(state, source, url), daemon=True).start()
    print(f"Live view: {url}  (phone must be on the same Wi-Fi)")
    print(f"Alert key for phone setup: {push.vapid_public_key()}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), partial(Handler, state=state)).serve_forever()


if __name__ == "__main__":
    main()
