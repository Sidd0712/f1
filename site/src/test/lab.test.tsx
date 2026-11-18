import { fireEvent, render, screen } from "@testing-library/react";
import { Lab } from "../views/Lab";

const data = {
  "Autodromo Nazionale di Monza": { races: 9, years: [2018, 2026], laps: 53, stops: { "1": 0.7011, "2": 0.2759, "3": 0.023 },
    plans: [{ seq: "M-H", stops: 1, share: 0.4023, stints: [20, 31] }, { seq: "S-M", stops: 1, share: 0.08, stints: [26, 27] }] },
  "Circuit de Monaco": { races: 7, years: [2018, 2026], laps: 78, stops: { "1": 0.9, "2": 0.1 }, plans: [{ seq: "M-H", stops: 1, share: 0.5, stints: [30, 48] }] },
};

test("shows what top-10 finishers did at the chosen circuit", () => {
  render(<Lab data={data} initial={{ circuit: "Autodromo Nazionale di Monza", location: "Monza" }} locations={{}} />);
  expect(screen.getByText("Medium → Hard")).toBeInTheDocument();
  expect(screen.getByText(/40% of top-10 finishers/)).toBeInTheDocument();
  expect(screen.getByText(/70%/)).toBeInTheDocument();
  expect(screen.getByText(/not a prediction/i)).toBeInTheDocument();
});

test("changing circuit updates the plans", () => {
  render(<Lab data={data} initial={{ circuit: "Autodromo Nazionale di Monza", location: "Monza" }} locations={{}} />);
  fireEvent.change(screen.getByRole("combobox", { name: "Circuit" }), { target: { value: "Circuit de Monaco" } });
  expect(screen.getByText(/50% of top-10 finishers/)).toBeInTheDocument();
  expect(screen.getByText(/78 laps/)).toBeInTheDocument();
});

test("opens on the next race's circuit by location, even when circuit names differ", () => {
  const d = { ...data, "Losail International Circuit": { races: 3, years: [2021, 2025], laps: 57, stops: { "2": 1 }, plans: [{ seq: "M-H-H", stops: 2, share: 0.6, stints: [15, 20, 22] }] } };
  render(<Lab data={d} initial={{ circuit: "Lusail International Circuit", location: "Lusail", circuitId: "losail" }} locations={{ "Losail International Circuit": "losail", "Circuit de Monaco": "monaco" }} />);
  expect(screen.getByRole("combobox", { name: "Circuit" })).toHaveValue("Losail International Circuit");
});

test("a circuit with no history says so instead of silently showing another", () => {
  render(<Lab data={data} initial={{ circuit: "Madring", location: "Madrid" }} locations={{}} />);
  expect(screen.getByText(/No past races at Madring yet/)).toBeInTheDocument();
});
