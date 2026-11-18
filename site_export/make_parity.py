"""Write podium-model parity fixtures from the real Python model (app/future.py)."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from future import FutureModel  # noqa: E402

b = json.loads((ROOT / "site/public/data/podium.json").read_text(encoding="utf8"))
m = FutureModel(b["model"], pd.DataFrame(b["drivers"]), pd.DataFrame(b["teams"]), pd.DataFrame(b["circuits"]), pd.DataFrame(b["recentGrids"]))
order = m.default_order()
teams = m.current_drivers["TeamKey"].to_dict()
base = [dict(driver=d, team=teams[d], grid=i + 1) for i, d in enumerate(order)]
swapped = [dict(e) for e in base]
swapped[0]["grid"], swapped[4]["grid"] = 5, 1
swapped[-1] = dict(driver="ZZZ", team="New Team", grid=len(swapped))
cases = [dict(name="projected-monza", entries=base, mobility=float(m.circuits.set_index("Location")["CircuitMobility"].get("monza", 3.0))),
         dict(name="swapped-unknown", entries=swapped, mobility=2.5),
         dict(name="no-mobility", entries=base, mobility=None)]
for c in cases:
    c["expected"] = [dict(driver=r["driver"], p=r["p"], baseline=r["baseline"]) for r in m.predict(c["entries"], c["mobility"])]
(ROOT / "site/src/test/fixtures/podium_parity.json").write_text(json.dumps(cases, indent=1), encoding="utf8")
print("wrote", len(cases), "cases,", len(base), "drivers")
