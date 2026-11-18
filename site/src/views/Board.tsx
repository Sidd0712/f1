import { useState, type ReactNode } from "react";
import type { Car, RaceMeta, SiteIndex } from "../data";
import { battlesAt, moments, stopsUpTo, type BattleView, type Changer, type Laps } from "../replay/logic";
import { Tower } from "../components/Tower";
import { ValuePair } from "../components/ValuePair";
import "./replay.css";

export const pct = (v: number) => `${Math.round(v * 100)}%`;
const TYRE: Record<string, string> = { S: "var(--soft)", M: "var(--med)", H: "var(--hard)", I: "var(--inter)", W: "var(--wet)" };

function Tyre({ cmp, age }: { cmp: string | null; age: number | null }) {
  if (!cmp) return <span className="dimtxt">–</span>;
  return <span className="tyre"><i style={{ borderColor: TYRE[cmp] ?? "var(--dim)" }}>{cmp}</i>{age ?? ""}</span>;
}

export function forecastText(b: Pick<BattleView, "forecast" | "leading">) {
  const f = b.forecast;
  if (f.kind === "catch") return `Catches ${b.leading} in ${f.lo === f.hi ? `${f.lo} lap${f.lo === 1 ? "" : "s"}` : `${f.lo} to ${f.hi} laps`}`;
  if (f.kind === "runsOut") return "Race runs out first";
  return "Within 1s now";
}

function Spark({ gaps, color }: { gaps: number[]; color: string }) {
  const mx = Math.max(...gaps, 1.2), mn = 0.6;
  const y = (v: number) => 24 - ((v - mn) / (mx - mn)) * 20;
  const pts = gaps.map((v, i) => `${1 + (i * 94) / Math.max(gaps.length - 1, 1)},${y(v)}`).join(" ");
  return (
    <svg className="spark" viewBox="0 0 96 26" aria-hidden="true">
      <line x1="0" x2="96" y1={y(1)} y2={y(1)} stroke="#ffffff40" strokeDasharray="3 3" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  );
}

function Swings({ laps, upTo, changers, colors }: { laps: Laps; upTo: number; changers: Changer[]; colors: Record<string, string> }) {
  const nums = Object.keys(laps).map(Number).filter((n) => n <= upTo && laps[n].live.length).sort((a, b) => a - b);
  if (nums.length < 2) return <p className="empty">The win-chance chart starts once a few laps have run.</p>;
  const last = laps[nums[nums.length - 1]].live;
  const top = [...last].sort((a, b) => b.win - a.win).slice(0, 4).map((x) => x.d);
  const x = (n: number) => ((n - nums[0]) / Math.max(nums[nums.length - 1] - nums[0], 1)) * 500;
  return (
    <div className="chart">
      <svg viewBox="0 0 500 110" preserveAspectRatio="none" role="img" aria-label="Win chance over the race so far">
        {top.map((d) => (
          <polyline key={d} fill="none" stroke={colors[d] ?? "#888"} strokeWidth="2"
            points={nums.map((n) => `${x(n)},${105 - (laps[n].live.find((v) => v.d === d)?.win ?? 0) * 100}`).join(" ")} />
        ))}
        {changers.map((c) => <line key={c.lap} x1={x(c.lap)} x2={x(c.lap)} y1="0" y2="110" stroke="#e10600" strokeDasharray="4 3" />)}
      </svg>
      <ul className="legend">{top.map((d) => <li key={d}><i style={{ background: colors[d] ?? "#888" }} />{d}</li>)}</ul>
      {changers.map((c) => <div className="item" key={c.lap}><b>Lap {c.lap} · {c.reason}</b><p>{c.driver} win chance {pct(c.from)} → {pct(c.to)}</p></div>)}
    </div>
  );
}

export function ChangerStrip({ latest, lap }: { latest?: Changer; lap: number }) {
  if (!latest || lap - latest.lap > 8) return null;
  return <div className="strip"><span className="k">Race changer · lap {latest.lap}</span><span>{latest.reason}. {latest.driver}'s win chance</span><span className="m">{pct(latest.from)} → {pct(latest.to)}</span></div>;
}

export function Board({ meta, laps, lap, last, index, changers, controls, notices, footer }: {
  meta: RaceMeta; laps: Laps; lap: number; last: number; index: SiteIndex; changers: Changer[]; controls: ReactNode; notices?: ReactNode; footer?: ReactNode;
}) {
  const [tab, setTab] = useState<"b" | "m" | "s">("b");
  const colors = Object.fromEntries(meta.drivers.map((d) => [d.code, d.color]));
  const cur = laps[lap];
  const stops = stopsUpTo(laps, lap);
  const dec = String(Math.min(9, Math.floor((10 * lap) / Math.max(last, 1))));
  const ro = index.runningOrder;
  const live = new Map((cur?.live ?? []).map((x) => [x.d, x]));
  const hasLive = (cur?.live.length ?? 0) > 0;
  const columns = [
    { key: "gap", label: "Gap", align: "right" as const },
    { key: "tyre", label: "Tyre" },
    { key: "stops", label: "Stops", align: "right" as const, hideSm: true },
    ...(hasLive ? [{ key: "podium", label: "Podium", align: "right" as const }, { key: "win", label: "Win", align: "right" as const, hideSm: true }] : []),
  ];
  const fallback: Car[] = meta.grid.map((d, i) => ({ d, pos: i + 1, gap: null, lead: null, cmp: null, age: null, pit: false, out: false }));
  const present = cur?.cars ?? fallback;
  const gone: Car[] = meta.drivers.filter((d) => !present.some((c) => c.d === d.code)).map((d) => ({ d: d.code, pos: null, gap: null, lead: null, cmp: null, age: null, pit: false, out: true }));
  const rows = [...present, ...gone].map((c) => {
    const lv = live.get(c.d);
    const rank = String(c.pos ?? "");
    return {
      pos: c.pos, code: c.d, color: colors[c.d] ?? "#888", dim: c.out,
      cells: {
        gap: c.out ? "Out" : c.pos === 1 ? "Leader" : c.gap != null ? `+${c.gap.toFixed(1)}` : "",
        tyre: <Tyre cmp={c.cmp} age={c.age} />,
        stops: stops[c.d] ?? 0,
        podium: lv ? <ValuePair model={lv.podium} base={ro?.podium[dec]?.[rank] ?? null} /> : "",
        win: lv && lv.win >= 0.01 ? <ValuePair model={lv.win} base={ro?.win[dec]?.[rank] ?? null} /> : "",
      },
    };
  });
  const battles = cur ? battlesAt(laps, lap, last).slice(0, 6) : [];
  const ms = moments(laps, lap, changers);
  return (
    <div className="cols">
      <div className="panel">
        {controls}
        {notices}
        {cur?.sc && <div className="sc">Safety car</div>}
        {lap > 0 && !cur ? <p className="empty">Loading lap {lap}…</p> : <Tower columns={columns} rows={rows} label="Timing" />}
      </div>
      <div className="panel side">
        <div className="seg" role="tablist">
          {([["b", "Battles"], ["m", "Moments"], ["s", "Swings"]] as const).map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
        {tab === "b" && (battles.length ? battles.map((b) => (
          <div className="fc" key={b.trailing}>
            <div className="pair">
              <i style={{ background: colors[b.trailing] }} />{b.trailing} <span className="v">chasing</span> <i style={{ background: colors[b.leading] }} />{b.leading}
              {b.forecast.kind === "close" && b.forecast.p != null && <ValuePair model={b.forecast.p} base={null} />}
            </div>
            <div className="eta">{forecastText(b)}</div>
            <div className="ev">
              <Spark gaps={b.history} color={colors[b.trailing] ?? "#888"} />
              <span>{b.history.map((g) => g.toFixed(1)).join(" › ")}s{b.forecast.kind !== "close" ? `, closing ${b.forecast.rate.toFixed(2)}s a lap` : b.forecast.p != null ? " · pass chance, next 5 laps" : ""}</span>
            </div>
          </div>
        )) : <p className="empty">No car is closing on the one ahead right now.</p>)}
        {tab === "m" && (ms.length ? ms.map((x, i) => <div className="item" key={i}><b>Lap {x.lap}</b><p>{x.text}</p></div>) : <p className="empty">Nothing yet. Moments unlock as you go.</p>)}
        {tab === "s" && <Swings laps={laps} upTo={lap} changers={changers} colors={colors} />}
        {footer}
      </div>
    </div>
  );
}
