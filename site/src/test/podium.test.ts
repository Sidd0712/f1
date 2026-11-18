import bundle from "./fixtures/podium_bundle.json";
import cases from "./fixtures/podium_parity.json";
import { PodiumModel, norm3 } from "../model/podium";
import type { PodiumBundle } from "../data";

const m = new PodiumModel(bundle as unknown as PodiumBundle);

test.each(cases.map((c) => [c.name, c] as const))("matches the Python model: %s", (_, c) => {
  const out = m.predict(c.entries, c.mobility);
  out.forEach((r, i) => {
    expect(r.driver).toBe(c.expected[i].driver);
    expect(Math.abs(r.p - c.expected[i].p)).toBeLessThan(1e-6);
    expect(r.baseline).toBeCloseTo(c.expected[i].baseline, 9);
  });
  expect(out.reduce((a, r) => a + r.p, 0)).toBeCloseTo(3, 6);
});

test("norm3 sums to three", () => {
  expect(norm3([2, 1, 0, -1, -2]).reduce((a, b) => a + b, 0)).toBeCloseTo(3, 6);
});

test("projected grid ranks every current driver", () => {
  const g = m.projectedGrid();
  expect(g.length).toBe(22);
  expect(g.map((e) => e.grid)).toEqual(Array.from({ length: 22 }, (_, i) => i + 1));
});

test("mobility lookup by circuit name with fallback", () => {
  expect(m.mobility("Autodromo Nazionale di Monza", "Monza")).not.toBeNull();
  expect(m.mobility("Nowhere Ring", "Nowhere")).toBeNull();
});
