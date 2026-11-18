import { fireEvent, render, screen } from "@testing-library/react";
import bundle from "./fixtures/podium_bundle.json";
import { PodiumModel } from "../model/podium";
import { GridBuilder } from "../components/GridBuilder";
import type { PodiumBundle } from "../data";

const model = new PodiumModel(bundle as unknown as PodiumBundle);
const entries = model.projectedGrid();
const colors = Object.fromEntries(entries.map((e) => [e.driver, { color: "#FF8000", accent: "#7F4000" }]));
const slots = () => screen.getAllByRole("button", { name: /^P\d+ / });
const codeAt = (i: number) => slots()[i].getAttribute("data-code");

function setup() {
  render(<GridBuilder model={model} entries={entries} colors={colors} mobility={null} />);
}

test("renders every slot and a field total of three", () => {
  setup();
  expect(slots()).toHaveLength(22);
  expect(screen.getByText("3.00 podium places")).toBeInTheDocument();
});

test("tap one slot then another swaps the cars", () => {
  setup();
  const a = codeAt(0), b = codeAt(4);
  fireEvent.click(slots()[0]);
  expect(slots()[0]).toHaveAttribute("aria-pressed", "true");
  fireEvent.click(slots()[4]);
  expect(codeAt(0)).toBe(b);
  expect(codeAt(4)).toBe(a);
  expect(screen.getByText("3.00 podium places")).toBeInTheDocument();
});

test("tapping the selected slot again cancels", () => {
  setup();
  const a = codeAt(0);
  fireEvent.click(slots()[0]);
  fireEvent.click(slots()[0]);
  expect(slots()[0]).toHaveAttribute("aria-pressed", "false");
  expect(codeAt(0)).toBe(a);
});

test("escape clears the selection", () => {
  setup();
  fireEvent.click(slots()[2]);
  fireEvent.keyDown(slots()[2], { key: "Escape" });
  expect(slots()[2]).toHaveAttribute("aria-pressed", "false");
});

test("reset restores the starting grid", () => {
  setup();
  const a = codeAt(0);
  fireEvent.click(slots()[0]);
  fireEvent.click(slots()[9]);
  fireEvent.click(screen.getByRole("button", { name: "Reset grid" }));
  expect(codeAt(0)).toBe(a);
});
