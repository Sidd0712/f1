"""
Future-race podium probability for the App: pure numpy/pandas, no Spark or Streamlit.

Mirrors predict_field in notebooks/phase_3_podium.ipynb: logistic score on grid and
recent form, shifted per race so the field's podium probabilities sum to 3, with an
80% range from a bootstrap ensemble.
"""

import json

import numpy as np
import pandas as pd


TEAM_COLORS = {
    "Mercedes": "#27F4D2", "Ferrari": "#E80020", "McLaren": "#FF8000", "Red Bull Racing": "#3671C6", "Racing Bulls": "#6692FF",
    "Aston Martin": "#229971", "Alpine": "#0093CC", "Williams": "#64C4FF", "Haas F1 Team": "#B6BABD", "Audi": "#52E252", "Cadillac": "#9aa0a6",
}


def norm3(z):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).sum() > 3:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


class FutureModel:
    def __init__(self, model_json, drivers, teams, circuits, recent_grids):
        self.m = json.loads(model_json) if isinstance(model_json, str) else model_json
        self.feats = self.m["features"]
        self.mu = np.array([self.m["mu"][f] for f in self.feats])
        self.sd = np.array([self.m["sd"][f] for f in self.feats])
        self.coef = np.array(self.m["coef"])
        self.ens = [(np.array(e["coef"]), e["intercept"]) for e in self.m["ensemble"]]
        self.drivers = drivers.set_index("Driver")
        self.teams = teams.set_index("TeamKey")
        self.circuits = circuits
        self.recent_grids = recent_grids
        self.fills = self.m["fills"]
        self.gap_by_slot = {int(k): v for k, v in self.m["gap_by_slot"].items()}
        self.slot_table = {int(k): v for k, v in self.m["slot_table"].items()}
        order = self.teams["TeamPace"].fillna(-1).rank(ascending=False, method="min")
        self.team_rank = order.astype(int).to_dict()

    @property
    def current_drivers(self):
        return self.drivers[self.drivers["Current"]]

    def default_order(self):
        # current drivers ranked by team pace then driver edge: a plausible grid before qualifying
        d = self.current_drivers.copy()
        d["pace"] = d["TeamKey"].map(self.teams["TeamPace"]).fillna(-1)
        return d.sort_values(["pace", "DriverEdge"], ascending=False).index.tolist()

    def slot_logit(self, g):
        r = min(max(self.slot_table.get(g, self.slot_table[max(self.slot_table)]), 0.01), 0.99)
        return float(np.log(r / (1 - r)))

    def features(self, entries, mobility):
        rows = []
        for e in entries:
            drv = self.drivers.loc[e["driver"]] if e["driver"] in self.drivers.index else None
            tm = self.teams.loc[e["team"]] if e["team"] in self.teams.index else None
            g = int(e["grid"])
            gap = e.get("gap")
            v = {
                "LogGrid": np.log(max(g, 1)),
                "SlotLogit": self.slot_logit(g),
                "QualiGapPct": gap if gap is not None and not pd.isna(gap) else self.gap_by_slot.get(g, self.fills["gap_global"]),
                "TeamPace": tm["TeamPace"] if tm is not None and pd.notna(tm["TeamPace"]) else self.fills["TeamPace"],
                "TeamQuali": tm["TeamQuali"] if tm is not None and pd.notna(tm["TeamQuali"]) else self.fills["TeamQuali"],
                "TeamDnf": tm["TeamDnf"] if tm is not None and pd.notna(tm["TeamDnf"]) else self.fills["TeamDnf"],
                "DriverEdge": drv["DriverEdge"] if drv is not None and pd.notna(drv["DriverEdge"]) else self.fills["DriverEdge"],
                "CircuitMobility": mobility if mobility is not None else self.fills["CircuitMobility"],
            }
            rows.append([v[f] for f in self.feats])
        return (np.array(rows, dtype=float) - self.mu) / self.sd

    def predict(self, entries, mobility=None):
        x = self.features(entries, mobility)
        p = norm3(x @ self.coef + self.m["intercept"])
        ps = np.array([norm3(x @ c + b) for c, b in self.ens])
        base = np.array([self.slot_table.get(int(e["grid"]), min(self.slot_table.values())) for e in entries])
        out = []
        for i, e in enumerate(entries):
            drv = self.drivers.loc[e["driver"]] if e["driver"] in self.drivers.index else None
            tm = self.teams.loc[e["team"]] if e["team"] in self.teams.index else None
            new_team = tm is None or bool(tm["TeamNew"])
            new_driver = drv is None or bool(drv["DriverNew"])
            lo, hi = float(np.percentile(ps[:, i], 10)), float(np.percentile(ps[:, i], 90))
            if new_team or new_driver:
                lo, hi = max(0.0, p[i] - 1.6 * (p[i] - lo)), min(1.0, p[i] + 1.6 * (hi - p[i]))
            out.append(dict(driver=e["driver"], team=e["team"], grid=int(e["grid"]), p=float(p[i]), lo=lo, hi=hi, baseline=float(base[i]),
                            new_team=new_team, new_driver=new_driver, why=self.why(e, drv, tm)))
        return out

    def why(self, e, drv, tm):
        bullets = [f"Starts P{int(e['grid'])}: that slot has finished on the podium {self.slot_table.get(int(e['grid']), 0) * 100:.0f}% of the time since 2019."]
        if tm is None or pd.isna(tm["TeamPace"]):
            bullets.append("No recent results for this team, so it is treated as a back-of-grid car with a wide range.")
        else:
            rk = self.team_rank.get(e["team"])
            bullets.append(f"{e['team']} is {self._ordinal(rk)} of {len(self.team_rank)} teams on recent car pace (weighted toward this season).")
        if drv is None or bool(drv["DriverNew"]):
            bullets.append("Few races on record, so the driver gets a rookie prior and a wider range.")
        elif pd.notna(drv["DriverEdge"]):
            edge = drv["DriverEdge"]
            bullets.append(f"Recently finishing {abs(edge):.1f} points {'ahead of' if edge > 0 else 'behind'} teammates on average.")
        return bullets

    @staticmethod
    def _ordinal(n):
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


CSS = """
<style>
.fu-row{background:var(--surface,#17171C);border:1px solid var(--border,#2A2A32);border-left:5px solid var(--tc,#6b6b76);margin:.35rem 0}
.fu-row.focus{border-color:var(--f1-red,#E10600);box-shadow:0 0 0 1px var(--f1-red,#E10600)}
.fu-row summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:2.4rem 1fr 7.5rem;gap:.2rem .7rem;align-items:center;padding:.45rem .7rem}
.fu-row summary::-webkit-details-marker{display:none}
.fu-grid{font-weight:900;font-size:1.2rem;text-align:center}
.fu-name b{letter-spacing:.03em}.fu-name span{display:block;color:var(--text-dim,#9A9AA5);font-size:.75rem}
.fu-p{text-align:right;font-weight:900;font-size:1.35rem;font-variant-numeric:tabular-nums}
.fu-p small{display:block;font-weight:400;font-size:.7rem;color:var(--text-dim,#9A9AA5)}
.fu-bar{grid-column:2 / 4;height:.35rem;background:#1F1F26;position:relative}
.fu-bar i{position:absolute;top:0;bottom:0;background:var(--tc,#6b6b76)}
.fu-bar u{position:absolute;top:-2px;bottom:-2px;width:2px;background:#fff;opacity:.8}
.fu-detail{padding:.2rem .9rem .6rem 3.4rem;font-size:.8rem;color:var(--text-dim,#9A9AA5)}
.fu-detail li{margin:.15rem 0}
.fu-flag{display:inline-block;font-size:.62rem;font-weight:700;padding:0 .3rem;margin-left:.3rem;border:1px solid #FFD400;color:#FFD400;text-transform:uppercase}
@media (max-width:640px){.fu-row summary{grid-template-columns:2rem 1fr 6rem;padding:.4rem .5rem}}
</style>
"""


def render_results(rows, colors, names, focus=None):
    import html as _h

    out = []
    for r in sorted(rows, key=lambda x: -x["p"]):
        flags = ""
        if r["new_team"]:
            flags += '<span class="fu-flag">new team, wide range</span>'
        if r["new_driver"]:
            flags += '<span class="fu-flag">few races, wide range</span>'
        why = "".join(f"<li>{_h.escape(b)}</li>" for b in r["why"])
        color = colors.get(r["team"], "#6b6b76")
        out.append(
            f'<details class="fu-row{" focus" if focus == r["driver"] else ""}" style="--tc:{color}"><summary>'
            f'<div class="fu-grid" title="Grid position">P{r["grid"]}</div>'
            f'<div class="fu-name"><b>{_h.escape(r["driver"])}</b> {_h.escape(names.get(r["driver"], ""))}{flags}<span>{_h.escape(r["team"])}</span></div>'
            f'<div class="fu-p">{r["p"] * 100:.0f}%<small>{r["lo"] * 100:.0f}&ndash;{r["hi"] * 100:.0f}% model range</small></div>'
            f'<div class="fu-bar"><i style="left:{r["lo"] * 100:.1f}%;width:{max((r["hi"] - r["lo"]) * 100, 0.8):.1f}%;opacity:.45"></i>'
            f'<u style="left:{r["p"] * 100:.1f}%" title="Model estimate"></u></div></summary>'
            f'<div class="fu-detail"><ul>{why}<li>Grid-slot-only baseline: {r["baseline"] * 100:.0f}%.</li></ul></div></details>'
        )
    return "".join(out)
