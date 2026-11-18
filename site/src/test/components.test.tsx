import { render, screen } from "@testing-library/react";
import { ValuePair } from "../components/ValuePair";
import { Tower } from "../components/Tower";
import { Shell } from "../components/Shell";

test("value pair shows whole numbers only", () => {
  render(<ValuePair model={0.614} base={0.48} label="grid alone" />);
  expect(screen.getByText("61%")).toBeInTheDocument();
  expect(screen.getByText(/48%/)).toBeInTheDocument();
  expect(screen.getByText(/grid alone/)).toBeInTheDocument();
  expect(document.body.textContent).not.toContain("61.4");
});

test("value pair handles missing baseline", () => {
  render(<ValuePair model={0.05} base={null} />);
  expect(screen.getByText("5%")).toBeInTheDocument();
});

test("tower renders headers and team stripe", () => {
  render(
    <Tower
      columns={[{ key: "gap", label: "Gap", align: "right" }]}
      rows={[{ pos: 1, code: "NOR", color: "#FF8000", cells: { gap: "Leader" } }]}
    />,
  );
  expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Pos", "Driver", "Gap"]);
  expect(screen.getByTestId("stripe-NOR")).toHaveStyle({ background: "#FF8000" });
});

test("shell shows nav, data stamp and active route", () => {
  window.location.hash = "#/replay";
  render(<Shell asOf="2026-11-16">content</Shell>);
  for (const l of ["This weekend", "Race", "Replay", "Lab", "How we know"]) expect(screen.getAllByText(l).length).toBeGreaterThan(0);
  expect(screen.getByText("16 Nov 2026")).toBeInTheDocument();
  expect(screen.getAllByText("Replay")[0].closest("a")).toHaveAttribute("aria-current", "page");
});
