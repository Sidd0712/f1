"""
Race Replay logic and HTML for the App: full-grid cards, key moments, battle board.

Pure pandas, no Spark or Streamlit imports, so it is unit-testable offline.
Every value shown for lap L uses only laps <= L (spoiler-safe by construction).
Inputs are the f1_replay_lap_state and f1_battle_pairs rows for one race.
"""

import html

import numpy as np
import pandas as pd

COMPOUND_COLORS = {"SOFT": "#FF1E1E", "MEDIUM": "#FFD400", "HARD": "#F5F5F5", "INTERMEDIATE": "#43B02A", "WET": "#0067B1"}
COMPOUND_TEXT = {"SOFT": "#FFFFFF", "MEDIUM": "#15151E", "HARD": "#15151E", "INTERMEDIATE": "#FFFFFF", "WET": "#FFFFFF"}
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
FALLBACK_COLOR = "#6b6b76"
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"

GLOSSARY = {
    "Gap": "Time between the car's rear and the car ahead, measured when this lap was completed.",
    "Under 1 s": "Less than one second behind the car ahead at the end of the lap. A chasing car this close usually gets a speed boost on the straights.",
    "Closing": "How many seconds per lap the gap is shrinking, averaged over the last 3 laps.",
    "Laps to catch": "Gap divided by closing rate: roughly how many laps until the chaser reaches the car ahead, if nothing changes.",
    "Pack": "Three or more cars in a row, each within one second of the one ahead. Overtaking is harder in a pack.",
    "Tyre age": "Laps done on the current set. Older tyres are usually slower, and fresher tyres a few laps newer can be a real pace edge.",
    "Undercut": "Pitting before the car ahead to gain time on fresh tyres, then jumping ahead when they pit.",
    "Overcut": "Staying out longer than a rival while their fresh tyres are slow to warm up, then pitting and rejoining ahead.",
    "Safety Car": "A pace car brings the field together and bunches up the gaps. Cars often pit under it.",
    "DRS": "A flap on the rear wing a chasing car can open within one second on straights. Used in 2018-2025, removed in 2026.",
    "Dirty air": "Turbulence behind a car that reduces the follower's grip and makes close following harder.",
    "Podium": "The top three finishers.",
    "Podium chance": "Model estimate that this car finishes in the top three. Up to half distance it blends pre-race form with the race so far; after that it follows the running order, which no model beat late in a race.",
    "Win chance": "Model estimate that this car wins, shared among the top four. Experimental: it beats 'the leader wins' mid-race, but not provably in the last laps.",
}


def tip(term, label=None):
    return f'<abbr title="{html.escape(GLOSSARY[term], quote=True)}">{html.escape(label or term)}</abbr>'


def team_color(hex_code):
    h = str(hex_code or "").lstrip("#")
    return "#" + h if len(h) == 6 and all(c in "0123456789abcdefABCDEF" for c in h) else FALLBACK_COLOR


def sparkline(values):
    v = [x for x in values if pd.notna(x)]
    if len(v) < 2 or len(v) != len(values):
        return ""
    lo, hi = min(v), max(v)
    if hi - lo < 0.05:
        return SPARK_BLOCKS[3] * len(v)
    return "".join(SPARK_BLOCKS[int((x - lo) / (hi - lo) * 7)] for x in v)


def svg_spark(values):
    v = list(values)
    if len(v) < 2 or any(x is None for x in v):
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i * 30 / (len(v) - 1):.0f},{13 - (x - lo) / span * 11:.1f}" for i, x in enumerate(v))
    return f'<svg class="rp-spark" width="32" height="14" viewBox="-1 0 32 14" aria-hidden="true"><polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>'


def one_in(rate):
    return int(round(1 / rate)) if rate and rate > 0 else None


class RaceView:
    def __init__(self, state, pairs, bands=None):
        s = state.copy()
        self.total = int(s["LapNumber"].max())
        pv = lambda c: s.pivot(index="LapNumber", columns="Driver", values=c).reindex(range(0, self.total + 1))
        self.pos, self.gap, self.age = pv("Position"), pv("GapToAhead"), pv("TyreLife")
        self.comp, self.pit, self.sc = pv("Compound"), pv("HasPit").fillna(0), pv("SC").fillna(False).astype(bool)
        meta = s.drop_duplicates("Driver").set_index("Driver")
        self.meta = {
            d: dict(
                name=" ".join(str(r.BroadcastName or d).replace(".", "").title().split()),
                number=str(r.DriverNumber or ""),
                team=str(r.TeamName or ""),
                color=team_color(r.TeamColor),
            )
            for d, r in meta.iterrows()
        }
        self.drivers = list(self.pos.columns)
        self.last = {d: int(self.pos[d].last_valid_index() or 0) for d in self.drivers}
        self.pairs = pairs.copy() if pairs is not None else pd.DataFrame()
        self.bands = {}
        if bands is not None and len(bands):
            self.bands = {r.Band: dict(lo=r.P_Lo, hi=r.P_Hi, rate=r.ObservedRate, n=r.Pairs) for r in bands.itertuples()}
        self._moments = None

    def is_out(self, d, lap):
        return lap > self.last[d] and self.last[d] < self.total * 0.9

    def order(self, lap):
        lap = max(0, min(int(lap), self.total))
        running, out = [], []
        for d in self.drivers:
            if self.is_out(d, lap):
                out.append(d)
            else:
                p = self.pos.at[min(lap, self.last[d]), d]
                if pd.notna(p):
                    running.append((p, d))
        return [d for _, d in sorted(running)], sorted(out, key=lambda d: -self.last[d])

    def _at(self, frame, lap, d):
        return frame.at[min(lap, self.last[d]), d]

    def closing_rate(self, lap, d):
        if lap < 2 or self.is_out(d, lap):
            return None
        g0, g1 = self.gap.at[lap - 2, d], self.gap.at[lap, d]
        return float((g0 - g1) / 2) if pd.notna(g0) and pd.notna(g1) else None

    def cards(self, lap):
        lap = max(0, min(int(lap), self.total))
        running, out = self.order(lap)
        packs = set()
        chain = []
        for i, d in enumerate(running):
            g = self.gap.at[lap, d] if i else np.nan
            if i and pd.notna(g) and 0 < g <= 1.0:
                chain = chain or [running[i - 1]]
                chain.append(d)
            else:
                if len(chain) >= 3:
                    packs.update(chain)
                chain = []
        if len(chain) >= 3:
            packs.update(chain)
        sc_now = bool(self.sc.loc[lap].any())
        calm = lap > 1 and not sc_now and not (lap > 1 and bool(self.sc.loc[lap - 1].any()))
        cards = []
        for i, d in enumerate(running):
            m = self.meta[d]
            ahead = running[i - 1] if i else None
            g = self.gap.at[lap, d] if i and lap > 0 else np.nan
            rate = self.closing_rate(lap, d)
            arrow = ""
            if lap >= 3 and pd.notna(g) and pd.notna(self.gap.at[lap - 3, d]):
                delta = g - self.gap.at[lap - 3, d]
                arrow = "▼" if delta <= -0.2 else "▲" if delta >= 0.2 else "–"
            catch = None
            if pd.notna(g) and rate is not None and rate > 0.05 and g / rate <= 15:
                catch = int(np.ceil(g / rate))
            comp = self._at(self.comp, lap, d)
            age = self._at(self.age, lap, d)
            rel = ""
            if ahead is not None:
                a_age = self._at(self.age, lap, ahead)
                if pd.notna(age) and pd.notna(a_age) and abs(age - a_age) >= 3:
                    rel = "newer" if age < a_age else "older"
            chips = []
            if lap > 0 and self.pit.at[lap, d] > 0:
                chips.append("PIT")
            if calm and d in packs:
                chips.append("Pack")
            if calm and pd.notna(g) and 0 < g < 1.0:
                chips.append("Under 1 s")
            cards.append(dict(
                code=d, pos=i + 1, out=False, ahead=ahead, gap=None if pd.isna(g) else float(g), arrow=arrow, catch=catch,
                spark=sparkline([self.gap.at[j, d] for j in range(max(1, lap - 2), lap + 1)]) if lap >= 3 else "",
                history=[None if pd.isna(self.gap.at[j, d]) else round(float(self.gap.at[j, d]), 2) for j in range(max(1, lap - 4), lap + 1)] if lap > 0 else [],
                comp=comp if isinstance(comp, str) else "", age=None if pd.isna(age) else int(age), rel=rel, chips=chips,
                ahead_comp=self._at(self.comp, lap, ahead) if ahead else "", ahead_age=None if not ahead or pd.isna(self._at(self.age, lap, ahead)) else int(self._at(self.age, lap, ahead)),
                **m,
            ))
        for d in out:
            cards.append(dict(code=d, pos=None, out=True, ahead=None, gap=None, arrow="", catch=None, spark="", history=[], comp="", age=None, rel="", chips=["OUT"],
                              ahead_comp="", ahead_age=None, **self.meta[d]))
        return cards

    def moments(self):
        if self._moments is not None:
            return self._moments
        ev, seen = [], {}
        nm = lambda d: self.meta[d]["name"].split()[-1]

        def add(lap, kind, text, ds, key=None):
            if key is not None:
                if lap - seen.get((kind, key), -99) < 5:
                    return
                seen[(kind, key)] = lap
            ev.append(dict(lap=lap, kind=kind, text=text, drivers=set(ds)))

        for lap in range(1, self.total + 1):
            sc_now, sc_prev = bool(self.sc.loc[lap].any()), bool(self.sc.loc[lap - 1].any()) if lap > 1 else False
            if sc_now and not sc_prev:
                add(lap, "sc", "Safety car or virtual safety car: gaps bunch up", [])
            if sc_prev and not sc_now:
                add(lap, "sc", "Back to green flag racing", [])
            running, _ = self.order(lap)
            prev_pos = self.pos.loc[lap - 1]
            for a in running:
                pa1, pa0 = self.pos.at[lap, a], prev_pos.get(a, np.nan)
                if pd.isna(pa0) or pa1 >= pa0 or lap == 1:
                    continue
                for b in running:
                    pb1, pb0 = self.pos.at[lap, b], prev_pos.get(b, np.nan)
                    if pd.notna(pb0) and pb0 < pa0 and pb1 > pa1:
                        if sc_now or sc_prev:
                            continue
                        pit = self.pit.at[lap, a] > 0 or self.pit.at[lap - 1, a] > 0 or self.pit.at[lap, b] > 0 or self.pit.at[lap - 1, b] > 0
                        if pit:
                            add(lap, "pit", f"{nm(a)} gets ahead of {nm(b)} through the pit stops", [a, b])
                        else:
                            add(lap, "pass", f"{nm(a)} passes {nm(b)} on track", [a, b])
            for i, d in enumerate(running[1:], 1):
                c = running[i - 1]
                g = self.gap.at[lap, d]
                if pd.isna(g) or g <= 0:
                    continue
                gp = self.gap.at[lap - 1, d] if lap > 1 else np.nan
                if g < 1.0 and not (pd.notna(gp) and gp < 1.0 and self.pos.at[lap - 1, c] + 1 == self.pos.at[lap - 1, d]):
                    add(lap, "under1", f"{nm(d)} is within 1 s of {nm(c)}", [d, c], key=(d, c))
                rate = self.closing_rate(lap, d)
                if rate is not None and rate >= 0.5 and g <= 3.0:
                    add(lap, "closing", f"{nm(d)} is closing on {nm(c)} by {rate:.1f} s a lap", [d, c], key=(d, c))
            if len(running) >= 4:
                d4, d3 = running[3], running[2]
                g4 = self.gap.at[lap, d4]
                g4p = self.gap.at[lap - 1, d4] if lap > 1 else np.nan
                if pd.notna(g4) and 0 < g4 < 1.0 and not (pd.notna(g4p) and g4p < 1.0):
                    add(lap, "podium", f"Podium fight: {nm(d4)} is within 1 s of {nm(d3)} for P3", [d4, d3], key=(d4, d3))
            for d in running:
                if self.pit.at[lap, d] > 0:
                    nxt = self.comp.at[min(lap + 1, self.last[d]), d]
                    add(lap, "pit", f"{nm(d)} pits" + (f" for {nxt.title()}s" if isinstance(nxt, str) and lap < self.last[d] else ""), [d])
        for d in self.drivers:
            if self.last[d] < self.total - 1:
                add(self.last[d], "out", f"{self.meta[d]['name'].split()[-1]} is out of the race", [d])
        self._moments = sorted(ev, key=lambda e: (e["lap"], e["kind"]))
        return self._moments

    def moments_upto(self, lap, follow=None, fewer=False):
        keep = []
        for e in self.moments():
            if e["lap"] > lap:
                continue
            mine = follow is not None and follow in e["drivers"]
            if follow and not mine and e["kind"] not in ("sc", "out", "podium"):
                continue
            if fewer and not mine and e["kind"] in ("pit", "under1"):
                continue
            keep.append(e)
        return keep

    def stints(self, lap):
        running, out = self.order(lap)
        res = []
        for d in running + out:
            end = min(lap, self.last[d])
            segs, start = [], 1
            for j in range(1, end + 1):
                comp = self.comp.at[j, d]
                if j > 1 and self.pit.at[j - 1, d] > 0:
                    segs.append((start, j - 1, self.comp.at[j - 1, d]))
                    start = j
            if end >= start:
                segs.append((start, end, self.comp.at[end, d]))
            res.append((d, segs))
        return res

    def board(self, lap, follow=None, est=False, n=3):
        if self.pairs.empty or lap < 1:
            return []
        p = self.pairs[(self.pairs["LapNumber"] == lap) & (self.pairs["Gap"] <= 3.0)].copy()
        if p.empty:
            return []
        busy = {d for d in self.drivers if lap > 0 and (self.pit.at[lap, d] > 0 or (lap > 1 and self.pit.at[lap - 1, d] > 0))}
        p = p[~p["Trailing"].isin(busy) & ~p["Leading"].isin(busy)]
        if p.empty:
            return []
        p["rate"] = p["Closing"].where(p["Closing"].abs() <= 3.0, np.nan)
        p["catch"] = np.where(p["rate"] > 0.05, p["Gap"] / p["rate"].where(p["rate"] > 0.05), np.nan)
        p["mine"] = (p["Trailing"] == follow) | (p["Leading"] == follow) if follow else False
        p["score"] = (p["P"] if est else -p["Gap"])
        p = p.sort_values(["mine", "score"], ascending=False).head(n)
        out = []
        for r in p.itertuples():
            t, l = r.Trailing, r.Leading
            ta, la = self._at(self.age, lap, t), self._at(self.age, lap, l)
            tc, lc = self._at(self.comp, lap, t), self._at(self.comp, lap, l)
            gap_txt = "under 0.1" if r.Gap < 0.05 else f"{r.Gap:.1f}"
            parts = [f"<b>{html.escape(self.meta[t]['name'])}</b> is {gap_txt} s behind <b>{html.escape(self.meta[l]['name'])}</b>"]
            if r.rate > 0.05:
                parts[0] += f" and catching by {r.rate:.1f} s a lap"
                if r.catch <= 15:
                    n_l = int(np.ceil(r.catch))
                    parts[0] += f", so about {n_l} lap{'s' if n_l != 1 else ''} to reach the car ahead"
            elif r.rate < -0.05:
                parts[0] += f" but dropping back by {-r.rate:.1f} s a lap"
            parts[0] += "."
            if pd.notna(ta) and pd.notna(la):
                fresher = "newer" if ta < la else "older" if ta > la else "the same age"
                parts.append(f"Chaser's tyres: {tc.title() if isinstance(tc, str) else '?'} {int(ta)} laps old ({fresher}); car ahead: {lc.title() if isinstance(lc, str) else '?'} {int(la)} laps old.")
            band = None
            if est and self.bands.get(r.Band):
                b = self.bands[r.Band]
                band = (r.Band, one_in(b["rate"]))
                parts.append(f"Pass chance in the next 5 laps if the fight stays on track: <b>{r.Band}</b> (about 1 in {band[1]} past pairs at this level ended in a pass).")
            out.append(dict(text=" ".join(parts), band=band[0] if band else None))
        return out


CSS = """
<style>
.rp-banner{background:var(--surface-alt,#1F1F26);border-left:3px solid var(--f1-red,#E10600);padding:.55rem .8rem;margin:.2rem 0 .8rem;font-size:.85rem;color:var(--text-dim,#9A9AA5)}
.rp-sc{background:#FFD400;color:#15151E;font-weight:700;padding:.35rem .8rem;margin:.4rem 0;text-transform:uppercase;letter-spacing:.06em;font-size:.8rem}
.rp-lap{font-size:1.6rem;font-weight:900;letter-spacing:.02em}
.rp-grid{display:grid;grid-template-columns:1fr 1fr;gap:.45rem 1rem;margin:.4rem 0 1rem}
.rp-grid>details:nth-child(even){margin-top:.9rem}
.rp-card{background:var(--surface,#17171C);border:1px solid var(--border,#2A2A32);border-left:5px solid var(--tc,#6b6b76);padding:0;min-width:0}
.rp-card.follow{border-color:var(--f1-red,#E10600);box-shadow:0 0 0 1px var(--f1-red,#E10600)}
.rp-card.out{opacity:.45}
.rp-card summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:2.1rem 1fr auto;grid-template-areas:"pos id gap" "pos tyre chips";gap:.1rem .6rem;padding:.4rem .6rem;align-items:center}
.rp-card summary::-webkit-details-marker{display:none}
.rp-pos{grid-area:pos;font-size:1.5rem;font-weight:900;text-align:center}
.rp-id{grid-area:id;min-width:0}
.rp-id b{font-size:1.05rem;letter-spacing:.04em}
.rp-num{color:var(--text-dim,#9A9AA5);font-size:.75rem;margin-left:.3rem}
.rp-name{color:var(--text-dim,#9A9AA5);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rp-gap{grid-area:gap;text-align:right;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.rp-gap .up{color:#9A9AA5}.rp-gap .down{color:#43B02A}
.rp-spark{color:var(--text-dim,#9A9AA5);margin-left:.35rem;vertical-align:middle}
.rp-tyre{grid-area:tyre;font-size:.75rem;color:var(--text-dim,#9A9AA5);white-space:nowrap}
.rp-chip{display:inline-block;min-width:1.3rem;text-align:center;font-weight:700;font-size:.7rem;padding:0 .3rem;border-radius:2px}
.rp-chips{grid-area:chips;text-align:right}
.rp-tag{display:inline-block;font-size:.62rem;font-weight:700;padding:0 .3rem;margin-left:.2rem;border:1px solid var(--border,#2A2A32);text-transform:uppercase;white-space:nowrap}
.rp-tag.pit{background:#0067B1;color:#fff;border-color:#0067B1}.rp-tag.sc{background:#FFD400;color:#15151E;border-color:#FFD400}.rp-tag.u1{border-color:#9A9AA5}.rp-tag.est{background:#2A2A32}
.rp-detail{padding:.3rem .8rem .6rem 3.1rem;font-size:.78rem;color:var(--text-dim,#9A9AA5)}
.rp-board{background:var(--surface,#17171C);border:1px solid var(--border,#2A2A32);border-left:3px solid var(--f1-red,#E10600);padding:.6rem .8rem;margin:.4rem 0;font-size:.9rem;line-height:1.45}
.rp-ev{display:grid;grid-template-columns:3.2rem 1fr;gap:.5rem;padding:.25rem 0;border-bottom:1px solid var(--border,#2A2A32);font-size:.88rem}
.rp-ev .l{color:var(--text-dim,#9A9AA5);font-variant-numeric:tabular-nums}
.rp-stint{display:flex;align-items:center;gap:.5rem;margin:.18rem 0;font-size:.78rem}
.rp-stint .c{width:3rem;font-weight:700}.rp-stint .bar{display:flex;flex:1;height:1rem;background:#1F1F26}
.rp-stint .seg{height:100%;border-right:1px solid #0C0C10;font-size:.62rem;font-weight:700;text-align:center;overflow:hidden}
.rp-legend{font-size:.75rem;color:var(--text-dim,#9A9AA5);margin:.2rem 0 .6rem}
.rp-legend i{display:inline-block;width:.7rem;height:.7rem;margin:0 .25rem 0 .6rem;vertical-align:middle}
abbr[title]{text-decoration:underline dotted;cursor:help}
@media (max-width:640px){.rp-grid{grid-template-columns:1fr}.rp-grid>details:nth-child(even){margin-top:0}.block-container{padding-left:16px!important;padding-right:16px!important}.rp-card summary{padding:.35rem .5rem}}
</style>
"""


def _chip(comp):
    c = comp if comp in COMPOUND_COLORS else ""
    return f'<span class="rp-chip" style="background:{COMPOUND_COLORS.get(c, "#444")};color:{COMPOUND_TEXT.get(c, "#fff")}">{COMPOUND_LETTER.get(c, "?")}</span>'


def render_grid(cards, lap, follow=None, est=None, live=None):
    est, live = est or {}, live or {}
    items = []
    for c in cards:
        color = c["color"]
        if c["out"]:
            gap_html, pos = "", "–"
        elif lap == 0:
            gap_html, pos = '<span title="Qualifying order; pit-lane starts are not adjusted">Grid</span>', c["pos"]
        elif c["gap"] is None:
            gap_html, pos = "Leader" if c["pos"] == 1 else '<span title="Gap not available for this lap">n/a</span>', c["pos"]
        else:
            cls = "down" if c["arrow"] == "▼" else "up"
            gap_html = f'+{c["gap"]:.1f} s <span class="{cls}" title="Change in gap over the last 3 laps">{c["arrow"]}</span>'
            gap_html += svg_spark(c["history"][-3:]) if lap >= 3 else ""
            pos = c["pos"]
        tyre = ""
        if c["comp"]:
            tyre = _chip(c["comp"])
            if c["age"] is not None and lap > 0:
                tyre += f' {c["age"]} lap{"s" if c["age"] != 1 else ""}'
            if c["rel"]:
                tyre += f' &middot; {c["rel"]} than car ahead'
        tags = ""
        for t in c["chips"]:
            cls = {"PIT": "pit", "SC": "sc", "Under 1 s": "u1"}.get(t, "")
            hint = {"PIT": "Pitting this lap", "Pack": GLOSSARY["Pack"], "Under 1 s": GLOSSARY["Under 1 s"], "OUT": "Out of the race"}.get(t, "")
            tags += f'<span class="rp-tag {cls}" title="{html.escape(hint, quote=True)}">{html.escape(t)}</span>'
        if c["code"] in est:
            tags += f'<span class="rp-tag est" title="Model estimate: chance of an on-track pass in the next 5 laps">Pass {est[c["code"]]}</span>'
        if c["code"] in live and not c["out"]:
            podium, win = live[c["code"]]
            tags += f'<span class="rp-tag est" title="{html.escape(GLOSSARY["Podium chance"], quote=True)}">Podium {podium:.0%}</span>'
            if win >= 0.01:
                tags += f'<span class="rp-tag est" title="{html.escape(GLOSSARY["Win chance"], quote=True)}">Win {win:.0%}</span>'
        detail = ""
        if not c["out"] and c["ahead"]:
            hist = ", ".join("?" if h is None else f"{h:.1f}" for h in c["history"])
            ahead_t = f'{c["ahead_comp"].title()} {c["ahead_age"]} lap{"s" if c["ahead_age"] != 1 else ""}' if c["ahead_comp"] and c["ahead_age"] is not None else "unknown"
            catch = f" About {c['catch']} laps to catch." if c["catch"] else ""
            detail = f'Car ahead: {html.escape(c["ahead"])} on {ahead_t}. Last gaps (s): {hist}.{catch}'
        elif c["out"]:
            detail = "Out of the race."
        cls = "rp-card" + (" follow" if follow and c["code"] == follow else "") + (" out" if c["out"] else "")
        num = f'<span class="rp-num">#{html.escape(c["number"])}</span>' if c["number"] else ""
        items.append(
            f'<details class="{cls}" style="--tc:{color}"><summary><div class="rp-pos">{pos}</div>'
            f'<div class="rp-id"><b>{html.escape(c["code"])}</b>{num}<div class="rp-name">{html.escape(c["name"])} &middot; {html.escape(c["team"])}</div></div>'
            f'<div class="rp-gap">{gap_html}</div><div class="rp-tyre">{tyre}</div><div class="rp-chips">{tags}</div></summary>'
            f'<div class="rp-detail">{detail}</div></details>'
        )
    return f'<div class="rp-grid">{"".join(items)}</div>'


def render_legend(view):
    seen, parts = set(), []
    for m in view.meta.values():
        if m["team"] not in seen:
            seen.add(m["team"])
            parts.append(f'<i style="background:{m["color"]}"></i>{html.escape(m["team"])}')
    return f'<div class="rp-legend">{"".join(parts)}</div>'


def render_board(items, est_on, lap):
    if not items:
        return '<div class="rp-board">No two cars are within 3 seconds of each other on this lap.</div>'
    foot = '<div class="rp-legend">Pass chances are learned mostly from DRS-era races (2018-2025). 2026 cars have no DRS, so real battle rates may differ.</div>'
    return "".join(f'<div class="rp-board">{i["text"]}</div>' for i in items) + foot


def live_note(bt):
    # one-line honest summary of the walk-forward backtest vs the running order
    if bt is None or bt.empty:
        return ""
    row = lambda t, w: bt[(bt["Target"] == t) & (bt["Window"] == w)].iloc[0]
    p, w = row("Podium", "0-50%"), row("Winner", "70-100%")
    pts = lambda r: f"{r.Gain * 100:+.1f} pts [{r.CI_Lo * 100:+.1f}, {r.CI_Hi * 100:+.1f}]"
    return (f"Backtest on unseen seasons: in the first half of a race the podium pick beats the running order by {pts(p)} "
            f"({p.Model:.0%} vs {p.RunningOrder:.0%}); after half distance it follows the running order. "
            f"Winner pick in the last 30%: {pts(w)} vs 'the leader wins'" + ("" if w.Ship else ", not yet better than chance, so treat it as experimental") + ".")


def render_moments(events, view):
    if not events:
        return '<div class="rp-board">Nothing yet. Move the lap slider forward.</div>'
    rows = "".join(f'<div class="rp-ev"><span class="l">Lap {e["lap"]}</span><span>{html.escape(e["text"])}</span></div>' for e in reversed(events[-60:]))
    return f"<div>{rows}</div>"


def render_stints(view, lap):
    rows = []
    for d, segs in view.stints(lap):
        bars = "".join(
            f'<div class="seg" style="width:{(b - a + 1) / view.total * 100:.2f}%;background:{COMPOUND_COLORS.get(c, "#444")};color:{COMPOUND_TEXT.get(c, "#fff")}">{COMPOUND_LETTER.get(c, "?")}</div>'
            for a, b, c in segs
        )
        rows.append(f'<div class="rp-stint"><span class="c">{html.escape(d)}</span><div class="bar">{bars}</div></div>')
    return "".join(rows)


def position_chart(view, lap, follow=None):
    import altair as alt

    rows = []
    teams = {}
    for d in view.drivers:
        m = view.meta[d]
        teams.setdefault(m["team"], []).append(d)
        for j in range(0, min(lap, view.last[d]) + 1):
            p = view.pos.at[j, d]
            if pd.notna(p):
                rows.append(dict(Lap=j, Position=int(p), Driver=d, Color=m["color"], Second=teams[m["team"]].index(d) > 0, Follow=d == follow))
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    colors = list(df.drop_duplicates("Driver")["Color"])
    base = alt.Chart(df).mark_line(interpolate="step-after").encode(
        x=alt.X("Lap:Q", scale=alt.Scale(domain=[0, view.total])),
        y=alt.Y("Position:Q", scale=alt.Scale(domain=[20.5, 0.5]), axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Driver:N", scale=alt.Scale(domain=list(df.drop_duplicates("Driver")["Driver"]), range=colors), legend=None),
        strokeDash=alt.condition(alt.datum.Second, alt.value([4, 3]), alt.value([1, 0])),
        strokeWidth=alt.condition(alt.datum.Follow, alt.value(4), alt.value(1.6)),
        tooltip=["Driver", "Lap", "Position"],
    )
    return base.properties(height=380)
