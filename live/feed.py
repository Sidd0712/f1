"""F1 live timing via FastF1 (owner's own F1 TV login, laptop only), converted to master-table lap rows."""

import threading
import time
from pathlib import Path

import pandas as pd

SECONDS = ["Time", "LapTime", "PitInTime", "PitOutTime"]
COLS = ["Year", "Race", "Driver", "LapNumber", "Position", "LapTime", "Time", "Stint", "TyreLife", "TrackStatus", "PitInTime", "PitOutTime", "Compound", "HasPit"]


def to_master(laps, year, race):
    # FastF1 Laps (timedeltas) -> the columns live.features / live.engine expect, in seconds
    d = pd.DataFrame(laps).copy()
    for c in SECONDS:
        if c in d:
            d[c] = pd.to_timedelta(d[c].astype("object"), errors="coerce").dt.total_seconds()
    d["Year"], d["Race"] = year, race
    d["HasPit"] = d["PitOutTime"].notna().astype(int) if "PitOutTime" in d else 0
    for c in COLS:
        if c not in d:
            d[c] = None
    return d[COLS]


class FastF1Source:
    # records the SignalR stream to a file and re-parses it every `every` seconds; FastF1 handles the F1 TV token
    def __init__(self, year, round_, race, out_dir, every=30):
        self.year, self.round, self.race, self.every = year, round_, race, every
        self.file = Path(out_dir) / f"live_{year}_{round_}.txt"
        self.file.parent.mkdir(parents=True, exist_ok=True)

    def _record(self):
        from fastf1.livetiming.client import SignalRClient

        SignalRClient(str(self.file), timeout=3600).start()

    def frames(self):
        import fastf1
        from fastf1.livetiming.data import LiveTimingData

        threading.Thread(target=self._record, daemon=True).start()
        while True:
            time.sleep(self.every)
            if not self.file.exists() or self.file.stat().st_size == 0:
                continue
            try:
                s = fastf1.get_session(self.year, self.round, "R")
                s.load(livedata=LiveTimingData(str(self.file)), telemetry=False, weather=False, messages=False)
                yield to_master(s.laps, self.year, self.race)
            except Exception as e:
                print("live feed not ready yet:", e)
