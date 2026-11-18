import type { PodiumBundle } from "../data";

export type Entry = { driver: string; team: string; grid: number; gap?: number | null };
export type Pred = { driver: string; team: string; grid: number; p: number; baseline: number };
type Row = Record<string, unknown>;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const plain = (s: string) => s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/_/g, " ").trim();

export function norm3(z: number[]): number[] {
  let lo = -15, hi = 15;
  const sig = (s: number) => z.map((v) => 1 / (1 + Math.exp(-(v + s))));
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (sig(mid).reduce((a, b) => a + b, 0) > 3) hi = mid;
    else lo = mid;
  }
  return sig((lo + hi) / 2);
}

export class PodiumModel {
  private m: Record<string, any>;
  private feats: string[];
  private drivers: Map<string, Row>;
  private teams: Map<string, Row>;
  private slot: Map<number, number>;
  private gapBySlot: Map<number, number>;
  constructor(private b: PodiumBundle) {
    this.m = b.model as Record<string, any>;
    this.feats = this.m.features;
    this.drivers = new Map(b.drivers.map((d) => [d.Driver as string, d]));
    this.teams = new Map(b.teams.map((t) => [t.TeamKey as string, t]));
    this.slot = new Map(Object.entries(this.m.slot_table as Record<string, number>).map(([k, v]) => [Number(k), v]));
    this.gapBySlot = new Map(Object.entries(this.m.gap_by_slot as Record<string, number>).map(([k, v]) => [Number(k), v]));
  }

  slotRate(g: number): number {
    return this.slot.get(g) ?? Math.min(...this.slot.values());
  }

  private slotLogit(g: number): number {
    const r = Math.min(Math.max(this.slot.get(g) ?? this.slot.get(Math.max(...this.slot.keys()))!, 0.01), 0.99);
    return Math.log(r / (1 - r));
  }

  private features(e: Entry, mobility: number | null): number[] {
    const f = this.m.fills as Record<string, number>;
    const d = this.drivers.get(e.driver);
    const t = this.teams.get(e.team);
    const pick = (row: Row | undefined, k: string) => (row && num(row[k]) !== null ? (row[k] as number) : f[k]);
    const g = Math.trunc(e.grid);
    const v: Record<string, number> = {
      LogGrid: Math.log(Math.max(g, 1)),
      SlotLogit: this.slotLogit(g),
      QualiGapPct: num(e.gap) ?? this.gapBySlot.get(g) ?? f.gap_global,
      TeamPace: pick(t, "TeamPace"),
      TeamQuali: pick(t, "TeamQuali"),
      TeamDnf: pick(t, "TeamDnf"),
      DriverEdge: pick(d, "DriverEdge"),
      CircuitMobility: mobility ?? f.CircuitMobility,
    };
    return this.feats.map((k) => (v[k] - this.m.mu[k]) / this.m.sd[k]);
  }

  predict(entries: Entry[], mobility: number | null = null): Pred[] {
    const z = entries.map((e) => this.features(e, mobility).reduce((a, x, i) => a + x * this.m.coef[i], this.m.intercept as number));
    const p = norm3(z);
    return entries.map((e, i) => ({ driver: e.driver, team: e.team, grid: Math.trunc(e.grid), p: p[i], baseline: this.slotRate(Math.trunc(e.grid)) }));
  }

  mobility(circuitName: string, location: string): number | null {
    const rows = this.b.circuits;
    const hit = rows.find((c) => plain(String(c.CircuitName)) === plain(circuitName)) ?? rows.find((c) => plain(String(c.Location)) === plain(location));
    return hit ? num(hit.CircuitMobility) : null;
  }

  reason(driver: string, team: string, up: boolean): string {
    const d = this.drivers.get(driver);
    const edge = d ? num(d.DriverEdge) : null;
    const ranked = [...this.teams.values()].sort((a, b) => (num(b.TeamPace) ?? -1) - (num(a.TeamPace) ?? -1)).map((t) => t.TeamKey);
    const rank = ranked.indexOf(team) + 1;
    const ord = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;
    if (!d || d.DriverNew) return "Few races on record, so we lean on the team's form.";
    if (up && edge != null && edge > 0.5) return `Finishing ${edge.toFixed(1)} points ahead of his teammate on average lately.`;
    if (!up && edge != null && edge < -0.5) return `Finishing ${Math.abs(edge).toFixed(1)} points behind his teammate on average lately.`;
    if (rank > 0) return `${team} is ${ord(rank)} of ${ranked.length} on recent race pace.`;
    return up ? "Recent race results beat his qualifying spots." : "Recent qualifying spots flatter his race results.";
  }

  projectedGrid(): Entry[] {
    const cur = this.b.drivers.filter((d) => d.Current);
    const grids = [...this.b.recentGrids].sort((a, b) => (b.Year as number) - (a.Year as number) || (b.Round as number) - (a.Round as number));
    const avg = (code: string) => {
      const g = grids.filter((r) => r.Driver === code && num(r.GridPosition) !== null).slice(0, 4).map((r) => r.GridPosition as number);
      return g.length ? g.reduce((a, b) => a + b, 0) / g.length : Infinity;
    };
    return cur
      .map((d) => ({ driver: d.Driver as string, team: d.TeamKey as string, a: avg(d.Driver as string) }))
      .sort((x, y) => x.a - y.a || x.driver.localeCompare(y.driver))
      .map((x, i) => ({ driver: x.driver, team: x.team, grid: i + 1 }));
  }
}
