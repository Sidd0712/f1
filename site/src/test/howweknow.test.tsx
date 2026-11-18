import { render, screen } from "@testing-library/react";
import hwk from "./fixtures/howweknow.json";
import { HowWeKnow } from "../views/HowWeKnow";
import { Footer } from "../components/Footer";

test("states each model against its baseline in plain words", () => {
  render(<HowWeKnow data={hwk as any} />);
  expect(screen.getByText(/76% of the time/)).toBeInTheDocument();
  expect(screen.getByText(/73% for the current running order/)).toBeInTheDocument();
  expect(screen.getAllByText(/\+2\.5 points/).length).toBeGreaterThan(0);
  expect(screen.getByText(/\+2\.2 points/)).toBeInTheDocument();
  expect(screen.getByText(/35%/)).toBeInTheDocument();
});

test("never calls anything AI", () => {
  render(<><HowWeKnow data={hwk as any} /><Footer /></>);
  expect(document.body.textContent ?? "").not.toMatch(/\bAI\b|artificial intelligence/i);
});

test("footer credits the data sources and says unofficial", () => {
  render(<Footer />);
  expect(screen.getByText(/OpenF1/)).toBeInTheDocument();
  expect(screen.getByText(/f1-circuits/)).toBeInTheDocument();
  expect(screen.getByText(/Unofficial/)).toBeInTheDocument();
});

test("pre-race claim counts real seasons only, not the pooled row", () => {
  render(<HowWeKnow data={hwk as any} />);
  expect(screen.getByText(/Across 112 races since 2022/)).toBeInTheDocument();
});
