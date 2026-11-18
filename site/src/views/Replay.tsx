import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { loadLaps, loadMeta, loadResult, type LapState, type RaceMeta, type RaceResult, type SiteIndex } from "../data";
import { clockReducer, raceChangers, type Laps } from "../replay/logic";
import { Board, ChangerStrip } from "./Board";
import { ValuePair } from "../components/ValuePair";
import { TrackOutline } from "./SeasonBrowser";
import { markWatched } from "../weekend/state";
import "./replay.css";

const CHUNK = 10;
type Key = number | "final";
const chunkOf = (lap: number, last: number): Key => (lap >= last ? "final" : Math.floor(lap / CHUNK));
const SPEEDS = [{ ms: 2000, label: "1 lap / 2s" }, { ms: 5000, label: "1 lap / 5s" }, { ms: 90000, label: "Real time" }];

export function Replay({ slug, index }: { slug: string; index: SiteIndex }) {
  const [meta, setMeta] = useState<RaceMeta | null>(null);
  const [laps, setLaps] = useState<Laps>({});
  const [loaded, setLoaded] = useState<Set<Key>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RaceResult | null>(null);
  const [pauseOnChanger, setPauseOnChanger] = useState(true);
  const [clock, dispatch] = useReducer(clockReducer, { lap: 0, revealed: 0, playing: false, speed: 2000, last: 0 });
  const seenChangers = useRef(0);
  const pending = useRef(new Set<Key>());
  const [resultError, setResultError] = useState(false);

  useEffect(() => {
    loadMeta(slug).then(setMeta, (e: Error) => setError(e.message));
  }, [slug]);

  const last = meta?.laps ?? 0;
  const state = clock;
  const act = (a: Parameters<typeof clockReducer>[1]) => dispatch(a);
  const needKeys: Key[] = [];
  for (let k = 0; k <= Math.floor(Math.min(state.revealed, Math.max(last - 1, 0)) / CHUNK); k++) needKeys.push(k);
  if (last > 0 && state.revealed >= last) needKeys.push("final");
  const need = needKeys.join(",");

  useEffect(() => {
    if (meta) dispatch({ type: "init", last: meta.laps });
  }, [meta]);

  useEffect(() => {
    if (!meta) return;
    for (const k of needKeys) {
      if (loaded.has(k) || pending.current.has(k)) continue;
      pending.current.add(k);
      loadLaps(slug, k).then(
        (c) => {
          setLaps((l) => ({ ...l, ...Object.fromEntries(Object.entries(c.laps).map(([n, v]) => [Number(n), v as LapState])) }));
          setLoaded((s) => new Set(s).add(k));
          pending.current.delete(k);
        },
        () => {
          pending.current.delete(k);
          setError(k === "final" ? "Couldn't load the last lap." : `Couldn't load laps ${k * CHUNK}–${k * CHUNK + CHUNK - 1}.`);
          dispatch({ type: "pause" });
        },
      );
    }
  }, [meta, need, loaded, slug]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (meta && last > 0 && state.revealed >= last && !result) {
      markWatched(slug);
      if (!resultError) loadResult(slug).then(setResult, () => setResultError(true));
    }
  }, [meta, state.revealed, last, result, slug, resultError]);

  const ready = loaded.has(chunkOf(state.lap, last));
  useEffect(() => {
    if (!clock.playing || !ready || !meta) return;
    const t = setTimeout(() => dispatch({ type: "tick" }), clock.speed);
    return () => clearTimeout(t);
  }, [clock.playing, clock.lap, clock.speed, ready, meta]);

  const changers = useMemo(() => raceChangers(laps, state.lap), [laps, state.lap]);
  useEffect(() => {
    if (changers.length > seenChangers.current) {
      seenChangers.current = changers.length;
      if (pauseOnChanger) dispatch({ type: "pause" });
    }
  }, [changers, pauseOnChanger]);

  if (error && !meta) return <p className="sub" role="alert">{error}</p>;
  if (!meta) return <p className="sub">Loading…</p>;

  const entry = index.races.find((r) => r.slug === slug);
  const colors = Object.fromEntries(meta.drivers.map((d) => [d.code, d.color]));
  const latest = changers[changers.length - 1];
  const prior = meta.prior ? [...meta.prior].sort((x, y) => y.p - x.p).slice(0, 5) : null;

  return (
    <section className="replay">
      <a className="back" href="#/replay">← All races</a>
      <h1 className="ev">{entry?.location ?? meta.circuit} <span>{meta.race} · {meta.year} · results hidden</span></h1>

      {prior && state.lap === 0 && (
        <div className="panel prerace">
          <div className="ph"><h2>Before the race, our podium read</h2><span className="meta">Ours · grid slot alone in grey</span></div>
          <ul>{prior.map((p) => <li key={p.code}><i style={{ background: colors[p.code] }} /><b>{p.code}</b><ValuePair model={p.p} base={p.baseline} /></li>)}</ul>
        </div>
      )}

      <ChangerStrip latest={latest} lap={state.lap} />

      <Board
        meta={meta} laps={laps} lap={state.lap} last={last} index={index} changers={changers}
        controls={
          <div className="clock">
            <div className="lapnow">LAP <b data-testid="lap-now">{state.lap}</b><span> / {last}</span></div>
            <button className="ctl primary" onClick={() => act({ type: clock.playing ? "pause" : "play" })}>{clock.playing ? "Pause" : state.lap === 0 ? "Start" : "Play"}</button>
            <button className="ctl" aria-label="Back a lap" onClick={() => act({ type: "step", delta: -1 })}>−1</button>
            <button className="ctl" aria-label="Forward a lap" onClick={() => act({ type: "step", delta: 1 })}>+1</button>
            <button className="ctl" onClick={() => act({ type: "jump" })}>+5 laps</button>
            <select aria-label="Speed" value={clock.speed} onChange={(e) => act({ type: "speed", speed: Number(e.target.value) })}>
              {SPEEDS.map((sp) => <option key={sp.ms} value={sp.ms}>{sp.label}</option>)}
            </select>
            <input type="range" aria-label="Lap" min={0} max={last} value={state.lap} onChange={(e) => act({ type: "seek", lap: Number(e.target.value) })} />
            <span className="hid">{state.revealed < last ? `Laps after ${state.revealed} are hidden` : "Race complete"}</span>
          </div>
        }
        notices={
          <>
            {clock.needsConfirm != null && (
              <div className="confirm" role="alertdialog" aria-label="Skip ahead">
                <span>This will show you laps you haven't seen.</span>
                <button className="ctl" onClick={() => act({ type: "seek", lap: clock.needsConfirm!, confirmed: true })}>Show lap {clock.needsConfirm}</button>
                <button className="ctl" onClick={() => act({ type: "cancel" })}>Stay here</button>
              </div>
            )}
            {error && <p className="sub" role="alert">{error} <button className="ctl" onClick={() => { setError(null); setLoaded(new Set(loaded)); }}>Retry</button></p>}
            {resultError && <p className="sub" role="alert">Couldn't load the result. <button className="ctl" onClick={() => setResultError(false)}>Retry</button></p>}
          </>
        }
        footer={<label className="toggle"><input type="checkbox" checked={pauseOnChanger} onChange={(e) => setPauseOnChanger(e.target.checked)} /> Pause on race changers</label>}
      />

      {result && state.revealed >= last && (
        <div className="panel reveal">
          <div className="ph"><h2>Final result</h2><span className="meta">Next to what we said before the race</span></div>
          <ol>
            {result.finish.map((f) => {
              const pr = meta.prior?.find((p) => p.code === f.code);
              return (
                <li key={f.code}><span className="fp">{f.pos ?? "–"}</span><i style={{ background: colors[f.code] }} /><b>{f.code}</b><span className="st">{f.status}</span>{pr && <ValuePair model={pr.p} base={pr.baseline} />}</li>
              );
            })}
          </ol>
          {changers.length > 0 && <p className="sub">Race changers we flagged: {changers.map((c) => `lap ${c.lap} (${c.reason})`).join(", ")}.</p>}
        </div>
      )}
      <TrackOutline circuit={meta.circuit} className="mini" />
    </section>
  );
}
