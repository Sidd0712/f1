import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Replay } from "../views/Replay";
import { SeasonBrowser } from "../views/SeasonBrowser";
import type { LapState, SiteIndex } from "../data";

const LAPS = 25;
const cars = (n: number) => [
  { d: "AAA", pos: 1, gap: null, lead: 0, cmp: "M", age: n, pit: false, out: false },
  { d: "BBB", pos: 2, gap: 3 - n * 0.1, lead: 1, cmp: "H", age: n, pit: false, out: false },
];
const chunk = (k: number) => {
  const laps: Record<string, LapState> = {};
  const ns = k < 0 ? [LAPS] : Array.from({ length: 10 }, (_, i) => k * 10 + i).filter((n) => n < LAPS);
  for (const n of ns) laps[n] = { sc: false, cars: cars(n), battles: [], live: [{ d: "AAA", podium: 0.9, win: 0.7 }, { d: "BBB", podium: 0.8, win: 0.3 }] };
  return { chunk: k < 0 ? "final" : k, from: k < 0 ? LAPS : k * 10, to: k < 0 ? LAPS : k * 10 + 9, laps };
};
const meta = { slug: "2025-01", year: 2025, round: 1, race: "Test Grand Prix", circuit: "Silverstone Circuit", laps: LAPS,
  drivers: [{ code: "AAA", number: "1", name: "A A", team: "T", color: "#FF8000" }, { code: "BBB", number: "2", name: "B B", team: "T", color: "#3671C6" }],
  grid: ["AAA", "BBB"], prior: [{ code: "AAA", p: 0.7, baseline: 0.6 }, { code: "BBB", p: 0.5, baseline: 0.5 }] };
const index: SiteIndex = { asOf: "2026-11-16", calendar: [], teams: {}, runningOrder: { podium: { "0": { "1": 0.6 } }, win: { "0": { "1": 0.5 } } },
  races: [{ slug: "2025-01", year: 2025, round: 1, race: "Test Grand Prix", circuit: "Silverstone Circuit", location: "Silverstone", date: "2025-07-06", laps: LAPS, hasLaps: true, hasLive: true, hasPrior: true }] };

let fetched: string[] = [];
beforeEach(() => {
  fetched = [];
  globalThis.fetch = vi.fn(async (u: string) => {
    const url = String(u);
    fetched.push(url.split("/").pop()!);
    if (url.endsWith("meta.json")) return new Response(JSON.stringify(meta));
    if (url.endsWith("laps-final.json")) return new Response(JSON.stringify(chunk(-1)));
    const m = url.match(/laps-(\d+)\.json$/);
    if (m) return new Response(JSON.stringify(chunk(Number(m[1]))));
    if (url.endsWith("result.json")) return new Response(JSON.stringify({ slug: "2025-01", finish: [{ code: "BBB", pos: 1, status: "Finished", points: 25 }, { code: "AAA", pos: 2, status: "Finished", points: 18 }] }));
    return new Response("", { status: 404 });
  }) as unknown as typeof fetch;
});

test("loads only what has been reached, result only at the flag", async () => {
  render(<Replay slug="2025-01" index={index} />);
  await screen.findByText("Test Grand Prix", { exact: false });
  await waitFor(() => expect(fetched).toContain("laps-0.json"));
  expect(fetched).not.toContain("laps-1.json");
  const jump = screen.getByRole("button", { name: "+5 laps" });
  fireEvent.click(jump);
  fireEvent.click(jump);
  await waitFor(() => expect(fetched).toContain("laps-1.json"));
  expect(fetched).not.toContain("laps-2.json");
  expect(fetched).not.toContain("laps-final.json");
  expect(fetched).not.toContain("result.json");
  for (let i = 0; i < 4; i++) {
    fireEvent.click(screen.getByRole("button", { name: "+5 laps" }));
    await waitFor(() => expect(screen.getByTestId("lap-now").textContent).toBe(String(Math.min(LAPS, 15 + i * 5))));
  }
  await waitFor(() => expect(fetched).toContain("result.json"));
  expect(fetched).toContain("laps-final.json");
  expect(await screen.findByText("Final result")).toBeInTheDocument();
});

test("seeking past what you've seen asks first", async () => {
  render(<Replay slug="2025-01" index={index} />);
  const slider = await screen.findByRole("slider", { name: "Lap" });
  fireEvent.change(slider, { target: { value: "20" } });
  expect(await screen.findByText(/This will show you laps you haven't seen/)).toBeInTheDocument();
  expect(fetched).not.toContain("laps-2.json");
});

test("season browser never shows winners", () => {
  render(<SeasonBrowser index={index} />);
  expect(screen.getByText("Test Grand Prix")).toBeInTheDocument();
  expect(document.body.textContent?.toLowerCase()).not.toMatch(/winner|won |podium/);
});

test("a driver missing from the lap shows as out", async () => {
  const m2 = { ...meta, drivers: [...meta.drivers, { code: "CCC", number: "3", name: "C C", team: "T", color: "#00FF00" }] };
  const f = globalThis.fetch;
  globalThis.fetch = vi.fn(async (u: string) => (String(u).endsWith("meta.json") ? new Response(JSON.stringify(m2)) : (f as any)(u))) as unknown as typeof fetch;
  render(<Replay slug="2025-01" index={index} />);
  await waitFor(() => expect(fetched).toContain("laps-0.json"));
  expect(await screen.findByText("CCC")).toBeInTheDocument();
  expect(screen.getAllByText("Out").length).toBeGreaterThan(0);
});

test("reaching the flag marks the race watched", async () => {
  localStorage.clear();
  render(<Replay slug="2025-01" index={index} />);
  const slider = await screen.findByRole("slider", { name: "Lap" });
  fireEvent.change(slider, { target: { value: String(LAPS) } });
  fireEvent.click(await screen.findByRole("button", { name: `Show lap ${LAPS}` }));
  await waitFor(() => expect(localStorage.getItem("ri.watched.2025-01")).toBe("1"));
});


test("a result that fails to load says so and can be retried", async () => {
  let fail = true;
  const f = globalThis.fetch;
  globalThis.fetch = vi.fn(async (u: string) => (String(u).endsWith("result.json") && fail ? new Response("", { status: 500 }) : (f as any)(u))) as unknown as typeof fetch;
  render(<Replay slug="2025-01" index={index} />);
  const slider = await screen.findByRole("slider", { name: "Lap" });
  fireEvent.change(slider, { target: { value: String(LAPS) } });
  fireEvent.click(await screen.findByRole("button", { name: `Show lap ${LAPS}` }));
  expect(await screen.findByText(/Couldn't load the result/)).toBeInTheDocument();
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText("Final result")).toBeInTheDocument();
});

test("while a lap is loading the tower says so instead of showing the grid", async () => {
  const f = globalThis.fetch;
  globalThis.fetch = vi.fn((u: string) => (String(u).endsWith("laps-1.json") ? new Promise<Response>(() => undefined) : (f as any)(u))) as unknown as typeof fetch;
  render(<Replay slug="2025-01" index={index} />);
  await waitFor(() => expect(fetched).toContain("laps-0.json"));
  const jump = screen.getByRole("button", { name: "+5 laps" });
  fireEvent.click(jump);
  fireEvent.click(jump);
  expect(await screen.findByText("Loading lap 10…")).toBeInTheDocument();
  expect(screen.queryByText("Leader")).toBeNull();
});

import { forecastText } from "../views/Board";
test("catch forecast names the car ahead and reads naturally when the range is one lap", () => {
  expect(forecastText({ leading: "VER", forecast: { kind: "catch", lo: 1, hi: 1 } as never })).toBe("Catches VER in 1 lap");
  expect(forecastText({ leading: "VER", forecast: { kind: "catch", lo: 3, hi: 5 } as never })).toBe("Catches VER in 3 to 5 laps");
});
