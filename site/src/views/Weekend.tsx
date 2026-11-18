import { useState } from "react";
import type { SiteIndex } from "../data";
import type { PodiumModel } from "../model/podium";
import { GridBuilder } from "../components/GridBuilder";
import { Tower } from "../components/Tower";
import { ValuePair } from "../components/ValuePair";
import { formatDay } from "../components/Shell";
import { TrackOutline } from "./SeasonBrowser";
import { markWatched, weekendState } from "../weekend/state";
import "./weekend.css";

type Colors = Record<string, { color: string; accent: string }>;

export function driverColors(index: SiteIndex): Colors {
  const years = Object.keys(index.teams).sort();
  const out: Colors = {};
  for (const t of Object.values(index.teams[years[years.length - 1]] ?? {})) for (const d of t.drivers) out[d] = { color: t.color, accent: t.accent };
  return out;
}

const day = (iso: string | null) => (iso ? formatDay(iso.slice(0, 10)) : "");

export function Weekend({ index, model, now = new Date() }: { index: SiteIndex; model: PodiumModel | null; now?: Date }) {
  const [, bump] = useState(0);
  const st = weekendState(index.calendar, index.races, index.weekendGrid, now);
  const colors = driverColors(index);

  if (st.kind === "gate") {
    const r = st.gated;
    return (
      <section>
        <h1 className="ev">{r.location} <span>{r.race} · {formatDay(r.date)}</span></h1>
        <div className="gate">
          <div>
            <h2>Haven't watched the {r.race} yet?</h2>
            <p>Start at the grid and step through the race. Results, key moments and the podium stay hidden until you reach them.</p>
            <div className="row">
              <a className="btn" href={`#/replay/${r.slug}`}>Start at lap 0</a>
              <button className="ghost" onClick={() => { markWatched(r.slug); bump((n) => n + 1); }}>I've seen it</button>
            </div>
          </div>
        </div>
      </section>
    );
  }

  if (st.kind === "done")
    return (
      <section>
        <h1 className="ev">Season complete</h1>
        <p className="sub">Every race is in <a href="#/replay">Replay</a>.</p>
      </section>
    );

  const next = st.next;
  const heading = (
    <h1 className="ev">{next.location} <span>Round {next.round} · {next.race} · race {day(next.sessions.race)}</span></h1>
  );
  if (!model) return <section>{heading}<p className="sub">Loading the podium model…</p></section>;
  const mobility = model.mobility(next.circuit, next.location);

  if (st.kind === "grid")
    return (
      <section>
        {heading}
        <p className="sub">Grid set. Tap a car, then another slot, to play what-if.</p>
        <GridBuilder model={model} entries={st.grid.grid} colors={colors} mobility={mobility} />
      </section>
    );

  const preds = model.predict(model.projectedGrid(), mobility);
  const ranked = [...preds].sort((a, b) => b.p - a.p);
  const disagree = [...preds].sort((a, b) => Math.abs(b.p - b.baseline) - Math.abs(a.p - a.baseline)).slice(0, 3);
  return (
    <section>
      {heading}
      <p className="sub">Qualifying {day(next.sessions.quali)}. Here's where our podium read differs most from what recent grid spots alone would say.</p>
      <div className="dis">
        {disagree.map((p) => {
          const diff = Math.round((p.p - p.baseline) * 100);
          return (
            <div className="dc" key={p.driver} data-testid="disagree" data-code={p.driver} style={{ borderTopColor: colors[p.driver]?.color ?? "#888" }}>
              <div className="who"><b>{p.driver}</b><span>{p.team}</span></div>
              <div className="nums">
                <span className="big">{Math.round(p.p * 100)}%</span>
                <span className={diff > 0 ? "up" : "dn"}>{diff > 0 ? "+" : ""}{diff} pts</span>
                <small>grid alone {Math.round(p.baseline * 100)}%</small>
              </div>
              <p>{model.reason(p.driver, p.team, diff > 0)}</p>
            </div>
          );
        })}
      </div>
      <div className="wcols">
        <div className="panel">
          <div className="ph"><h2>Podium chance, whole field</h2><span className="meta">Ours · grid alone in grey · adds up to 3</span></div>
          <Tower
            label="Podium chance by driver"
            columns={[{ key: "team", label: "Team", hideSm: true }, { key: "p", label: "Podium chance", align: "right" }]}
            rows={ranked.map((p, i) => ({ pos: i + 1, code: p.driver, color: colors[p.driver]?.color ?? "#888", cells: { team: <span className="dimtxt">{p.team}</span>, p: <ValuePair model={p.p} base={p.baseline} /> } }))}
          />
        </div>
        <div className="panel trackcard">
          <div className="ph"><h2>{next.circuit}</h2></div>
          <TrackOutline circuit={next.circuit} className="outline" />
          <p className="howto">Projected grid from each driver's last four qualifying spots. Once the real grid is set you can rearrange it. <a href="#/about">How sure are we?</a></p>
        </div>
      </div>
    </section>
  );
}
