import { useEffect, useMemo, useState } from "react";
import type { LapState, RaceMeta, SiteIndex } from "../data";
import { raceChangers, type Laps } from "../replay/logic";
import { Board, ChangerStrip } from "./Board";
import { AlertsSetup } from "./AlertsSetup";
import "./weekend.css";

type LiveState = { meta: RaceMeta; later: boolean; updatedAt: number | null; laps: Record<string, LapState> };
const url = () => `${import.meta.env.BASE_URL}live/state.json`;

export function Race({ index, every = 5000 }: { index: SiteIndex; every?: number }) {
  const [st, setSt] = useState<LiveState | null>(null);
  const [off, setOff] = useState(false);
  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetch(url(), { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then((j: LiveState) => { if (alive && j.meta) { setSt(j); setOff(false); } else if (alive) setOff(true); })
        .catch(() => alive && setOff(true));
    poll();
    const t = setInterval(poll, every);
    return () => { alive = false; clearInterval(t); };
  }, [every]);

  const laps: Laps = useMemo(() => Object.fromEntries(Object.entries(st?.laps ?? {}).map(([k, v]) => [Number(k), v])), [st]);
  const lap = Math.max(0, ...Object.keys(laps).map(Number));
  const changers = useMemo(() => raceChangers(laps, lap), [laps, lap]);

  if (!st) {
    if (!off) return <p className="sub">Looking for the live race…</p>;
    return (
      <section>
        <h1 className="ev">Race <span>Live mode</span></h1>
        <p className="sub">Live mode runs on the owner's laptop during races. Finished races are in <a href="#/replay">Replay</a>.</p>
        <AlertsSetup />
      </section>
    );
  }
  const { meta } = st;
  const setMode = (later: boolean) =>
    fetch(`${import.meta.env.BASE_URL}live/mode`, { method: "POST", body: later ? "later" : "live" }).then(() => setSt({ ...st, later }), () => undefined);
  if (st.later)
    return (
      <section>
        <h1 className="ev">Race <span>Live mode</span></h1>
        <div className="gate">
          <div>
            <h2>You're watching later</h2>
            <p>Nothing about this race is shown here, and alerts only say that something big happened.</p>
            <div className="row"><button className="ghost" onClick={() => setMode(false)}>Switch to live</button></div>
          </div>
        </div>
      </section>
    );
  const stale = st.updatedAt ? Math.floor((Date.now() / 1000 - st.updatedAt) / 60) : 0;
  return (
    <section className="replay">
      <h1 className="ev">{meta.race} <span>Live · every lap</span></h1>
      <ChangerStrip latest={changers[changers.length - 1]} lap={lap} />
      <Board
        meta={meta} laps={laps} lap={lap} last={meta.laps} index={index} changers={changers}
        controls={
          <div className="clock">
            <div className="lapnow">LAP <b data-testid="lap-now">{lap}</b><span> / {meta.laps}</span></div>
            <span className="livechip"><i />Live</span>
            {stale >= 3 && <span className="hid" role="status">No new lap for {stale} minutes</span>}
            <button className="ctl" onClick={() => setMode(true)}>I'm watching later</button>
          </div>
        }
      />
    </section>
  );
}
