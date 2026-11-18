import { useMemo, useState } from "react";
import type { Entry, PodiumModel } from "../model/podium";
import { Car } from "./Car";
import { ValuePair } from "./ValuePair";
import "./grid.css";

type Colors = Record<string, { color: string; accent: string }>;
const GREY = { color: "#888888", accent: "#444444" };

export function GridBuilder({ model, entries, colors, mobility }: { model: PodiumModel; entries: Entry[]; colors: Colors; mobility: number | null }) {
  const start = useMemo(() => [...entries].sort((a, b) => a.grid - b.grid), [entries]);
  const [order, setOrder] = useState(start);
  const [sel, setSel] = useState<number | null>(null);
  const [note, setNote] = useState("Tap a car, then tap another slot to swap them.");
  const preds = useMemo(() => model.predict(order.map((e, i) => ({ ...e, grid: i + 1, gap: null })), mobility), [model, order, mobility]);
  const total = preds.reduce((a, p) => a + p.p, 0);

  const tap = (i: number) => {
    if (sel === null) return setSel(i);
    if (sel !== i) {
      const next = [...order];
      [next[sel], next[i]] = [next[i], next[sel]];
      const moved = next[i].driver;
      const after = model.predict(next.map((e, k) => ({ ...e, grid: k + 1, gap: null })), mobility);
      setOrder(next);
      setNote(`${moved} to P${i + 1}: podium chance now ${Math.round(after[i].p * 100)}%. Everyone else rebalances so the field still adds up to 3.`);
    }
    setSel(null);
  };

  return (
    <div className="gbw">
      <div className="track">
        <div className="cheq" aria-hidden="true" />
        <div className="gg" role="list" aria-label="Starting grid">
          {order.map((e, i) => {
            const c = colors[e.driver] ?? GREY;
            const p = preds[i];
            return (
              <div role="listitem" key={e.driver} className="cell">
                <button
                  className={`slot${sel === i ? " sel" : ""}`}
                  data-code={e.driver}
                  aria-pressed={sel === i}
                  aria-label={`P${i + 1} ${e.driver}, podium chance ${Math.round(p.p * 100)}%, grid slot alone ${Math.round(p.baseline * 100)}%`}
                  onClick={() => tap(i)}
                  onKeyDown={(ev) => ev.key === "Escape" && setSel(null)}
                >
                  <span className="n">P{i + 1}</span>
                  <span className="carbox"><Car className="car" color={c.color} accent={c.accent} /></span>
                  <span className="code" style={{ borderLeftColor: c.color }}>{e.driver}</span>
                  <ValuePair model={p.p} base={p.baseline} />
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <aside className="panel whatif">
        <div className="ph">
          <h2>Your what-if</h2>
          <button className="ghost" onClick={() => { setOrder(start); setSel(null); setNote("Back to the starting grid."); }}>Reset grid</button>
        </div>
        <p className="note" aria-live="polite">{note}</p>
        <p className="legend">Bold is our podium chance. Grey is what that grid slot alone has given since 2019.</p>
        <div className="total"><span>Field total</span><b>{total.toFixed(2)} podium places</b></div>
      </aside>
    </div>
  );
}
