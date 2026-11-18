import { useState } from "react";
import { TrackOutline } from "./SeasonBrowser";
import "./weekend.css";

export type Profile = { races: number; years: number[]; laps: number; stops: Record<string, number>; plans: { seq: string; stops: number; share: number; stints: number[] }[] };
const NAME: Record<string, string> = { S: "Soft", M: "Medium", H: "Hard", I: "Inter", W: "Wet" };
const COLOR: Record<string, string> = { S: "var(--soft)", M: "var(--med)", H: "var(--hard)", I: "var(--inter)", W: "var(--wet)" };
const pct = (v: number) => `${Math.round(v * 100)}%`;

const plain = (n: string) => n.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();

export function Lab({ data, initial, locations }: { data: Record<string, Profile>; initial?: { circuit: string; location: string; circuitId?: string | null }; locations: Record<string, string> }) {
  const names = Object.keys(data).sort((a, b) => a.localeCompare(b));
  const start = initial ? (data[initial.circuit] ? initial.circuit : names.find((n) => locations[n] && [initial.circuitId ?? "", initial.location].some((v) => v && plain(locations[n]) === plain(v)))) : undefined;
  const [circ, setCirc] = useState(start ?? names[0]);
  const p = data[circ];
  return (
    <section>
      <h1 className="ev">Strategy Lab <span>What actually worked, circuit by circuit</span></h1>
      <p className="sub">From {p ? `${p.races} races (${p.years[0]}–${p.years[1]})` : "past races"} at this circuit: the strategies top-10 finishers used. A guide to what usually works, not a prediction.</p>
      {initial && !start && <p className="sub" role="status">No past races at {initial.circuit} yet, so here's another circuit to explore.</p>}
      <label className="field labpick">Circuit
        <select aria-label="Circuit" value={circ} onChange={(e) => setCirc(e.target.value)}>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      {p && (
        <div className="wcols">
          <div className="panel">
            <div className="ph"><h2>Most used plans</h2><span className="meta">{p.laps} laps</span></div>
            {p.plans.map((pl) => (
              <div className="plan" key={pl.seq}>
                <div className="plan-h"><b>{pl.seq.split("-").map((c) => NAME[c] ?? c).join(" → ")}</b><span>{pct(pl.share)} of top-10 finishers · {pl.stops} stop{pl.stops === 1 ? "" : "s"}</span></div>
                <div className="stints" aria-hidden="true">
                  {pl.stints.map((len, i) => <i key={i} style={{ flex: len, background: COLOR[pl.seq.split("-")[i]] ?? "var(--dim)" }}>{Math.round(len)}</i>)}
                </div>
                <p className="dimtxt">Typical stints: {pl.stints.map((l) => `${Math.round(l)} laps`).join(", then ")}.</p>
              </div>
            ))}
          </div>
          <div className="panel trackcard">
            <div className="ph"><h2>How many stops</h2></div>
            <ul className="stopbars">
              {Object.entries(p.stops).map(([k, v]) => (
                <li key={k}><span>{k} stop{k === "1" ? "" : "s"}</span><i style={{ width: pct(v) }} /><b>{pct(v)}</b></li>
              ))}
            </ul>
            <TrackOutline circuit={circ} className="outline" />
          </div>
        </div>
      )}
    </section>
  );
}
