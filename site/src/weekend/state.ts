import { nextRace, type CalendarEntry, type RaceEntry } from "../data";

export type WeekendGrid = { year: number; round: number; grid: { driver: string; team: string; grid: number }[] };
export type WeekendState =
  | { kind: "gate"; gated: RaceEntry; next: CalendarEntry | null }
  | { kind: "grid"; next: CalendarEntry; grid: WeekendGrid }
  | { kind: "pre"; next: CalendarEntry }
  | { kind: "done" };

const KEY = (slug: string) => `ri.watched.${slug}`;
const session = new Set<string>();

export function readWatched(slug: string): boolean {
  if (session.has(slug)) return true;
  try {
    return localStorage.getItem(KEY(slug)) === "1";
  } catch {
    return false;
  }
}

export function markWatched(slug: string): void {
  session.add(slug);
  try {
    localStorage.setItem(KEY(slug), "1");
  } catch {
    /* per-viewer convenience only */
  }
}

export function weekendState(calendar: CalendarEntry[], races: RaceEntry[], weekendGrid: WeekendGrid | null | undefined, now: Date): WeekendState {
  const latest = races.filter((r) => r.hasLaps).sort((a, b) => b.year - a.year || b.round - a.round)[0];
  const next = nextRace(calendar, now);
  if (latest && !readWatched(latest.slug)) return { kind: "gate", gated: latest, next };
  if (!next) return { kind: "done" };
  if (weekendGrid && weekendGrid.year === next.year && weekendGrid.round === next.round && weekendGrid.grid.length) return { kind: "grid", next, grid: weekendGrid };
  return { kind: "pre", next };
}
