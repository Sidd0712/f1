import { render, screen } from "@testing-library/react";
import { App } from "../App";
import { nextRace } from "../data";

const index = {
  asOf: "2026-11-16",
  races: [{ slug: "2026-20", year: 2026, round: 20, race: "Brazilian Grand Prix", circuit: "Interlagos", location: "Sao Paulo", date: "2026-11-08", laps: 71, hasLaps: true, hasLive: true, hasPrior: true }],
  calendar: [
    { year: 2026, round: 20, race: "Brazilian Grand Prix", location: "Sao Paulo", circuit: "Interlagos", sessions: { quali: "2026-11-07T18:00:00Z", sprint: null, race: "2026-11-08T17:00:00Z" } },
    { year: 2026, round: 21, race: "Las Vegas Grand Prix", location: "Las Vegas", circuit: "Las Vegas Strip Street Circuit", sessions: { quali: "2026-11-21T04:00:00Z", sprint: null, race: "2026-11-22T04:00:00Z" } },
  ],
  teams: {},
};

test("nextRace picks the first race not yet finished", () => {
  expect(nextRace(index.calendar, new Date("2026-11-17T00:00:00Z"))?.race).toBe("Las Vegas Grand Prix");
  expect(nextRace(index.calendar, new Date("2027-01-01T00:00:00Z"))).toBeNull();
});

test("home shows the next race from the exported index once the last race is watched", async () => {
  window.location.hash = "";
  localStorage.setItem("ri.watched.2026-20", "1");
  globalThis.fetch = vi.fn(async (u: string) => (String(u).endsWith("index.json") ? new Response(JSON.stringify(index)) : new Response("", { status: 404 }))) as unknown as typeof fetch;
  render(<App now={new Date("2026-11-17T00:00:00Z")} />);
  expect(await screen.findByText("Las Vegas")).toBeInTheDocument();
  expect(screen.getByText("16 Nov 2026")).toBeInTheDocument();
});
