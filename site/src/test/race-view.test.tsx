import { render, screen } from "@testing-library/react";
import { Race } from "../views/Race";
import type { SiteIndex } from "../data";

const index = { asOf: "2026-11-16", races: [], calendar: [], teams: {} } as SiteIndex;
const meta = { slug: "live", year: 2026, round: 22, race: "Qatar Grand Prix", circuit: "Lusail International Circuit", laps: 57,
  drivers: [{ code: "NOR", number: "1", name: "N", team: "McLaren", color: "#FF8000" }, { code: "VER", number: "3", name: "V", team: "Red Bull", color: "#3671C6" }],
  grid: ["NOR", "VER"], prior: null };
const lap = (n: number) => ({ lap: n, sc: false, battles: [], cars: [{ d: "NOR", pos: 1, gap: null, lead: 0, cmp: "M", age: n, pit: false, out: false }, { d: "VER", pos: 2, gap: 2.0, lead: 2, cmp: "M", age: n, pit: false, out: false }], live: [{ d: "NOR", podium: 0.9, win: 0.7 }, { d: "VER", podium: 0.8, win: 0.3 }] });

test("shows the latest live lap from the laptop", async () => {
  const now = Date.now() / 1000;
  globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ meta, later: false, updatedAt: now, laps: { "11": lap(11), "12": lap(12) } }))) as unknown as typeof fetch;
  render(<Race index={index} />);
  expect(await screen.findByText(/LAP/)).toBeInTheDocument();
  expect(screen.getByTestId("lap-now").textContent).toBe("12");
  expect(screen.getByText("Leader")).toBeInTheDocument();
});

test("stale feed shows how long since the last lap", async () => {
  globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ meta, later: false, updatedAt: Date.now() / 1000 - 300, laps: { "12": lap(12) } }))) as unknown as typeof fetch;
  render(<Race index={index} />);
  expect(await screen.findByText(/No new lap for 5 minutes/)).toBeInTheDocument();
});

test("without the laptop it explains where live mode runs", async () => {
  globalThis.fetch = vi.fn(async () => new Response("", { status: 404 })) as unknown as typeof fetch;
  render(<Race index={index} />);
  expect(await screen.findByText(/runs on the owner's laptop during races/)).toBeInTheDocument();
});

test("watching later shows nothing about the race and offers to switch", async () => {
  globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ meta, later: true, updatedAt: Date.now() / 1000, laps: { "12": lap(12) } }))) as unknown as typeof fetch;
  render(<Race index={index} />);
  expect(await screen.findByText(/You're watching later/)).toBeInTheDocument();
  expect(screen.queryByText("Leader")).toBeNull();
  expect(screen.queryByText("NOR")).toBeNull();
  expect(screen.getByRole("button", { name: "Switch to live" })).toBeInTheDocument();
});
