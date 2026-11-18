import { markWatched, readWatched, weekendState } from "../weekend/state";
import type { CalendarEntry, RaceEntry } from "../data";

const cal: CalendarEntry[] = [
  { year: 2026, round: 20, race: "Brazilian Grand Prix", location: "Sao Paulo", circuit: "Interlagos", sessions: { quali: "2026-11-07T18:00:00Z", sprint: null, race: "2026-11-08T17:00:00Z" } },
  { year: 2026, round: 21, race: "Las Vegas Grand Prix", location: "Las Vegas", circuit: "Las Vegas Strip Street Circuit", sessions: { quali: "2026-11-21T04:00:00Z", sprint: null, race: "2026-11-22T04:00:00Z" } },
];
const races = [{ slug: "2026-20", year: 2026, round: 20, race: "Brazilian Grand Prix", hasLaps: true }] as RaceEntry[];
const now = new Date("2026-11-17T00:00:00Z");

beforeEach(() => localStorage.clear());

test("latest finished race not watched gates the weekend", () => {
  expect(weekendState(cal, races, null, now)).toMatchObject({ kind: "gate", gated: { slug: "2026-20" } });
});

test("watched race opens the pre-qualifying view", () => {
  markWatched("2026-20");
  expect(readWatched("2026-20")).toBe(true);
  expect(weekendState(cal, races, null, now)).toMatchObject({ kind: "pre", next: { round: 21 } });
});

test("real grid exported for the next race shows the grid view", () => {
  markWatched("2026-20");
  const wg = { year: 2026, round: 21, grid: [{ driver: "NOR", team: "McLaren", grid: 1 }] };
  expect(weekendState(cal, races, wg, now)).toMatchObject({ kind: "grid" });
});

test("season over with everything watched is done", () => {
  markWatched("2026-20");
  expect(weekendState(cal, races, null, new Date("2027-01-01T00:00:00Z")).kind).toBe("done");
});

test("storage failure counts as not watched", () => {
  const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
  expect(readWatched("2026-98")).toBe(false);
  expect(() => markWatched("2026-98")).not.toThrow();
  spy.mockRestore();
});

test("blocked storage still remembers 'I've seen it' for this visit", () => {
  const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
  const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
  markWatched("2026-19");
  expect(readWatched("2026-19")).toBe(true);
  set.mockRestore();
  get.mockRestore();
});
