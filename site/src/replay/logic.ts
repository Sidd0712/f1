import type { LapState } from "../data";

export type Laps = Record<number, LapState>;
export type Forecast = { kind: "catch"; lo: number; hi: number; rate: number } | { kind: "close"; p: number | null } | { kind: "runsOut"; rate: number };
export type BattleView = { trailing: string; leading: string; gap: number; history: number[]; forecast: Forecast; p: number | null };
export type Changer = { lap: number; driver: string; from: number; to: number; reason: string };

const lapNums = (laps: Laps, upTo: number) => Object.keys(laps).map(Number).filter((n) => n <= upTo).sort((a, b) => a - b);

function slope(ys: number[]): number {
  const n = ys.length, mx = (n - 1) / 2, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  ys.forEach((y, i) => { num += (i - mx) * (y - my); den += (i - mx) ** 2; });
  return den ? num / den : 0;
}

export function forecast(gaps: number[], lapsLeft: number, overtakeP: number | null): Forecast | null {
  if (!gaps.length) return null;
  const g = gaps[gaps.length - 1];
  if (g <= 1.0) return { kind: "close", p: overtakeP };
  if (gaps.length < 3) return null;
  const rate = -slope(gaps);
  if (rate < 0.05 - 1e-9) return null;
  const laps = (g - 1) / rate;
  const lo = Math.max(1, Math.floor(0.8 * laps));
  const hi = Math.max(lo, Math.ceil(1.2 * laps));
  return lo > lapsLeft ? { kind: "runsOut", rate } : { kind: "catch", lo, hi, rate };
}

function aheadOf(state: LapState, code: string): { ahead: string; gap: number } | null {
  const me = state.cars.find((c) => c.d === code);
  if (!me || me.pos == null || me.pos < 2 || me.gap == null) return null;
  const a = state.cars.find((c) => c.pos === me.pos! - 1);
  return a ? { ahead: a.d, gap: me.gap } : null;
}

export function pairHistory(laps: Laps, upTo: number, trailing: string, n = 5): number[] {
  const now = laps[upTo] && aheadOf(laps[upTo], trailing);
  if (!now) return [];
  const out: number[] = [];
  for (let k = upTo; k >= 0 && out.length < n; k--) {
    const s = laps[k] && aheadOf(laps[k], trailing);
    if (!s || s.ahead !== now.ahead) break;
    out.unshift(s.gap);
  }
  return out;
}

export function battlesAt(laps: Laps, lap: number, totalLaps: number): BattleView[] {
  const s = laps[lap];
  if (!s) return [];
  const out: BattleView[] = [];
  for (const c of s.cars) {
    if (c.gap == null || c.gap > 3 || c.pos == null || c.pos < 2) continue;
    const a = aheadOf(s, c.d);
    if (!a) continue;
    const history = pairHistory(laps, lap, c.d);
    const p = s.battles.find((b) => b.t === c.d)?.p ?? null;
    const f = forecast(history, totalLaps - lap, p);
    if (f) out.push({ trailing: c.d, leading: a.ahead, gap: c.gap, history, forecast: f, p });
  }
  return out.sort((x, y) => x.gap - y.gap);
}

export function stopsUpTo(laps: Laps, lap: number): Record<string, number> {
  const out: Record<string, number> = {};
  for (const n of lapNums(laps, lap)) for (const c of laps[n].cars) if (c.pit) out[c.d] = (out[c.d] ?? 0) + 1;
  return out;
}

const scStart = (laps: Laps, n: number) => !!laps[n]?.sc && !laps[n - 1]?.sc;

function eventAt(laps: Laps, n: number): string | null {
  if (scStart(laps, n)) return "Safety car";
  const prev = laps[n - 1], cur = laps[n];
  if (!prev || !cur) return null;
  for (const c of prev.cars) {
    const now = cur.cars.find((x) => x.d === c.d);
    if (c.pos != null && c.pos <= 6 && (!now || now.out)) return `${c.d} retires`;
  }
  if (cur.cars.some((c) => c.pit && c.pos != null && c.pos <= 4)) return "Pit stops";
  return null;
}

export function raceChangers(laps: Laps, upTo: number): Changer[] {
  const found: Changer[] = [];
  const nums = lapNums(laps, upTo).filter((n) => laps[n].live.length);
  const win = (n: number, d: string) => laps[n]?.live.find((x) => x.d === d)?.win;
  for (const n of nums) {
    if (found.length && found[found.length - 1].lap + 5 > n) continue;
    const reason = eventAt(laps, n) ?? eventAt(laps, n - 1);
    if (!reason) continue;
    let best: Changer | null = null;
    const out = new Set(laps[n].cars.filter((c) => c.out).map((c) => c.d));
    for (const x of laps[n].live) {
      if (out.has(x.d)) continue;
      for (const k of [n - 1, n - 2]) {
        const from = win(k, x.d);
        if (from == null) continue;
        const d = Math.abs(x.win - from);
        if (d >= 0.15 && (!best || d > Math.abs(best.to - best.from))) best = { lap: n, driver: x.d, from, to: x.win, reason };
      }
    }
    if (best) found.push(best);
  }
  return found;
}

export function moments(laps: Laps, upTo: number, changers: Changer[]): { lap: number; text: string }[] {
  const out: { lap: number; text: string }[] = [];
  const nums = lapNums(laps, upTo);
  nums.forEach((n, i) => {
    if (!i) return;
    const prev = laps[nums[i - 1]], cur = laps[n];
    const lead = (s: LapState) => s.cars.find((c) => c.pos === 1)?.d;
    if (lead(cur) && lead(prev) && lead(cur) !== lead(prev)) out.push({ lap: n, text: `${lead(cur)} takes the lead` });
    if (scStart(laps, n)) out.push({ lap: n, text: "Safety car" });
    for (const c of cur.cars) if (c.out && prev.cars.find((x) => x.d === c.d && !x.out)) out.push({ lap: n, text: `${c.d} retires` });
  });
  for (const c of changers) out.push({ lap: c.lap, text: `Race changer: ${c.driver} win chance ${Math.round(c.from * 100)}% → ${Math.round(c.to * 100)}%` });
  return out.sort((a, b) => a.lap - b.lap);
}

export type ClockState = { lap: number; revealed: number; playing: boolean; speed: number; last: number; needsConfirm?: number };
export type ClockAction =
  | { type: "tick" } | { type: "play" } | { type: "pause" } | { type: "step"; delta: number } | { type: "jump" }
  | { type: "seek"; lap: number; confirmed?: boolean } | { type: "speed"; speed: number } | { type: "cancel" } | { type: "init"; last: number };

const go = (s: ClockState, lap: number): ClockState => {
  const l = Math.max(0, Math.min(s.last, lap));
  return { ...s, lap: l, revealed: Math.max(s.revealed, l), playing: l >= s.last ? false : s.playing, needsConfirm: undefined };
};

export function clockReducer(s: ClockState, a: ClockAction): ClockState {
  switch (a.type) {
    case "tick": return s.lap >= s.last ? { ...s, playing: false } : go(s, s.lap + 1);
    case "play": return s.lap >= s.last ? s : { ...s, playing: true };
    case "pause": return { ...s, playing: false };
    case "step": return go(s, s.lap + a.delta);
    case "jump": return go(s, s.lap + 5);
    case "speed": return { ...s, speed: a.speed };
    case "cancel": return { ...s, needsConfirm: undefined };
    case "init": return { ...s, last: a.last };
    case "seek":
      if (a.lap <= s.revealed || a.confirmed) return go(s, a.lap);
      return { ...s, needsConfirm: a.lap };
  }
}
