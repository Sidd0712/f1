import { useState } from "react";
import tracks from "../tracks.json";
import type { SiteIndex } from "../data";
import { formatDay } from "../components/Shell";
import "./replay.css";

export function TrackOutline({ circuit, className }: { circuit: string; className?: string }) {
  const d = (tracks as Record<string, string>)[circuit];
  return (
    <svg className={className} viewBox="0 0 300 170" aria-hidden="true">
      {d && <path d={d} fill="none" stroke="currentColor" strokeWidth="5" strokeLinejoin="round" />}
    </svg>
  );
}

export function SeasonBrowser({ index }: { index: SiteIndex }) {
  const races = index.races.filter((r) => r.hasLaps);
  const years = [...new Set(races.map((r) => r.year))].sort((a, b) => b - a);
  const [year, setYear] = useState(years[0]);
  return (
    <section>
      <h1 className="ev">Replay <span>Any race since 2018, results hidden until you reach the flag</span></h1>
      <div className="years" role="tablist" aria-label="Season">
        {years.map((y) => (
          <button key={y} role="tab" aria-selected={y === year} className={y === year ? "on" : ""} onClick={() => setYear(y)}>{y}</button>
        ))}
      </div>
      <div className="races">
        {races.filter((r) => r.year === year).map((r) => (
          <a key={r.slug} className="racecard" href={`#/replay/${r.slug}`}>
            <TrackOutline circuit={r.circuit} className="outline" />
            <span className="rnd">Round {r.round} · {formatDay(r.date)}</span>
            <b>{r.race}</b>
            <span className="cir">{r.circuit}</span>
          </a>
        ))}
      </div>
    </section>
  );
}
