import type { Car, LapState } from "../data";
import { battlesAt, clockReducer, forecast, moments, pairHistory, raceChangers, stopsUpTo, type ClockState } from "../replay/logic";

const car = (d: string, pos: number | null, gap: number | null, extra: Partial<Car> = {}): Car => ({ d, pos, gap, lead: null, cmp: "M", age: 5, pit: false, out: pos === null, ...extra });
const lap = (cars: Car[], extra: Partial<LapState> = {}): LapState => ({ sc: false, cars, battles: [], live: [], ...extra });

test("forecast: closing car gives a lap range", () => {
  const f = forecast([2.8, 2.6, 2.3, 2.1, 1.9], 30, null);
  expect(f).toMatchObject({ kind: "catch" });
  if (f?.kind === "catch") {
    expect(f.rate).toBeCloseTo(0.23, 2);
    expect(f.lo).toBeGreaterThanOrEqual(1);
    expect(f.lo).toBeLessThanOrEqual(f.hi);
  }
});

test("forecast: gap opening or flat gives nothing", () => {
  expect(forecast([1.5, 1.7, 1.9], 30, null)).toBeNull();
  expect(forecast([2.0, 2.0, 2.01], 30, null)).toBeNull();
});

test("forecast: within a second switches to the overtake chance", () => {
  expect(forecast([1.4, 1.1, 0.8], 30, 0.34)).toEqual({ kind: "close", p: 0.34 });
});

test("forecast: race runs out first", () => {
  expect(forecast([2.9, 2.85, 2.8, 2.75, 2.7], 3, null)).toMatchObject({ kind: "runsOut" });
});

test("forecast skips null gaps", () => {
  expect(forecast([], 30, null)).toBeNull();
});

test("forecast ignores non-adjacent laps", () => {
  const laps: Record<number, LapState> = {
    1: lap([car("A", 1, null), car("C", 2, 1.0), car("B", 3, 2.5)]),
    2: lap([car("A", 1, null), car("B", 2, 2.4)]),
    3: lap([car("A", 1, null), car("B", 2, 2.2)]),
  };
  expect(pairHistory(laps, 3, "B")).toEqual([2.4, 2.2]);
});

test("battlesAt lists close closing pairs only", () => {
  const laps: Record<number, LapState> = {};
  [3.6, 3.2, 2.8, 2.4, 2.0].forEach((g, i) => (laps[i + 1] = lap([car("A", 1, null), car("B", 2, g), car("C", 3, 9.0)])));
  const b = battlesAt(laps, 5, 50);
  expect(b.map((x) => [x.trailing, x.leading])).toEqual([["B", "A"]]);
});

test("stops counted from pit flags", () => {
  const laps = { 1: lap([car("A", 1, null, { pit: true })]), 2: lap([car("A", 1, null)]), 3: lap([car("A", 1, null, { pit: true })]) };
  expect(stopsUpTo(laps, 2)).toEqual({ A: 1 });
  expect(stopsUpTo(laps, 3)).toEqual({ A: 2 });
});

const live = (w: Record<string, number>) => Object.entries(w).map(([d, win]) => ({ d, podium: 0.5, win }));

test("race changer needs a big swing tied to an event", () => {
  const laps: Record<number, LapState> = {
    10: lap([car("A", 1, null), car("B", 2, 1)], { live: live({ A: 0.6, B: 0.08 }) }),
    11: lap([car("A", 1, null), car("B", 2, 1)], { sc: true, live: live({ A: 0.3, B: 0.41 }) }),
    12: lap([car("A", 1, null), car("B", 2, 1)], { live: live({ A: 0.3, B: 0.45 }) }),
  };
  const c = raceChangers(laps, 12);
  expect(c).toHaveLength(1);
  expect(c[0]).toMatchObject({ lap: 11, driver: "B", from: 0.08, to: 0.41, reason: "Safety car" });
});

test("no event, no race changer; cooldown of 5 laps", () => {
  const quiet: Record<number, LapState> = {
    1: lap([car("A", 1, null)], { live: live({ A: 0.2 }) }),
    2: lap([car("A", 1, null)], { live: live({ A: 0.5 }) }),
  };
  expect(raceChangers(quiet, 2)).toEqual([]);
  const busy: Record<number, LapState> = {};
  for (let i = 1; i <= 6; i++) busy[i] = lap([car("A", 1, null), car("B", 2, 1)], { sc: i % 2 === 0, live: live({ A: i % 2 ? 0.2 : 0.6, B: 0.1 }) });
  expect(raceChangers(busy, 6).length).toBe(1);
});

test("changers empty without live data", () => {
  expect(raceChangers({ 1: lap([car("A", 1, null)], { sc: true }) }, 1)).toEqual([]);
});

test("moments: lead change, safety car, retirement", () => {
  const laps: Record<number, LapState> = {
    1: lap([car("A", 1, null), car("B", 2, 1), car("C", 3, 1)]),
    2: lap([car("B", 1, null), car("A", 2, 1), car("C", null, null)], { sc: true }),
  };
  const m = moments(laps, 2, []).map((x) => x.text);
  expect(m).toContain("B takes the lead");
  expect(m).toContain("Safety car");
  expect(m).toContain("C retires");
});

test("clock: forward-only seek needs confirmation", () => {
  const s: ClockState = { lap: 5, revealed: 5, playing: false, speed: 2, last: 50 };
  expect(clockReducer(s, { type: "seek", lap: 3 })).toMatchObject({ lap: 3, revealed: 5 });
  expect(clockReducer(s, { type: "seek", lap: 20 })).toMatchObject({ lap: 5, needsConfirm: 20 });
  expect(clockReducer(s, { type: "seek", lap: 20, confirmed: true })).toMatchObject({ lap: 20, revealed: 20 });
  expect(clockReducer({ ...s, lap: 48, revealed: 48 }, { type: "jump" })).toMatchObject({ lap: 50, revealed: 50, playing: false });
  expect(clockReducer({ ...s, playing: true }, { type: "tick" })).toMatchObject({ lap: 6, revealed: 6 });
});

test("clock: init sets the race length so jumps can move", () => {
  const s: ClockState = { lap: 0, revealed: 0, playing: false, speed: 2000, last: 0 };
  const t = clockReducer(clockReducer(s, { type: "init", last: 25 }), { type: "jump" });
  expect(t).toMatchObject({ lap: 5, revealed: 5, last: 25 });
});

import changerCases from "./fixtures/changer_cases.json";
test.each(changerCases.map((c) => [c.name, c] as const))("shared race-changer case: %s", (_, c) => {
  const laps = Object.fromEntries(Object.entries(c.laps).map(([k, v]) => [Number(k), v])) as unknown as Record<number, LapState>;
  expect(raceChangers(laps, c.upTo)).toEqual(c.expected);
});
