import { fireEvent, render, screen } from "@testing-library/react";
import bundle from "./fixtures/podium_bundle.json";
import { Weekend } from "../views/Weekend";
import { PodiumModel } from "../model/podium";
import type { PodiumBundle, SiteIndex } from "../data";

const model = new PodiumModel(bundle as unknown as PodiumBundle);
const index: SiteIndex = {
  asOf: "2026-11-16", teams: {}, runningOrder: undefined,
  races: [{ slug: "2026-20", year: 2026, round: 20, race: "Brazilian Grand Prix", circuit: "Autódromo José Carlos Pace", location: "Sao Paulo", date: "2026-11-08", laps: 71, hasLaps: true, hasLive: true, hasPrior: true }],
  calendar: [{ year: 2026, round: 21, race: "Las Vegas Grand Prix", location: "Las Vegas", circuit: "Las Vegas Strip Street Circuit", sessions: { quali: "2026-11-21T04:00:00Z", sprint: null, race: "2026-11-22T04:00:00Z" } }],
};
const now = new Date("2026-11-17T00:00:00Z");

beforeEach(() => localStorage.clear());

test("unwatched last race gates the weekend and hides the next race", () => {
  render(<Weekend index={index} model={model} now={now} />);
  expect(screen.getByText(/Haven't watched the Brazilian Grand Prix yet/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Start at lap 0" })).toHaveAttribute("href", "#/replay/2026-20");
  expect(screen.queryByText(/Las Vegas/)).toBeNull();
});

test("marking it watched shows the pre-qualifying read with three disagreement cards", () => {
  render(<Weekend index={index} model={model} now={now} />);
  fireEvent.click(screen.getByRole("button", { name: "I've seen it" }));
  expect(screen.getByRole("heading", { level: 1 }).textContent).toContain("Las Vegas");
  const cards = screen.getAllByTestId("disagree");
  expect(cards).toHaveLength(3);
  const preds = model.predict(model.projectedGrid(), model.mobility("Las Vegas Strip Street Circuit", "Las Vegas"));
  const top = [...preds].sort((a, b) => Math.abs(b.p - b.baseline) - Math.abs(a.p - a.baseline)).slice(0, 3).map((p) => p.driver);
  expect(cards.map((c) => c.getAttribute("data-code"))).toEqual(top);
  expect(screen.getAllByRole("row").length).toBeGreaterThan(20);
});
