export type RaceEntry = { slug: string; year: number; round: number; race: string; circuit: string; location: string; date: string; laps: number | null; hasLaps: boolean; hasLive: boolean; hasPrior: boolean };
export type CalendarEntry = { year: number; round: number; race: string; location: string; circuit: string; circuitId?: string | null; sessions: { quali: string | null; sprint: string | null; race: string | null } };
export type Team = { color: string; accent: string; drivers: string[] };
export type Rates = Record<string, Record<string, number>>;
export type SiteIndex = { asOf: string; races: RaceEntry[]; calendar: CalendarEntry[]; teams: Record<string, Record<string, Team>>; runningOrder?: { podium: Rates; win: Rates }; weekendGrid?: { year: number; round: number; grid: { driver: string; team: string; grid: number }[] } | null };
export type Driver = { code: string; number: string; name: string; team: string; color: string };
export type RaceMeta = { slug: string; year: number; round: number; race: string; circuit: string; laps: number; drivers: Driver[]; grid: string[]; prior: { code: string; p: number; baseline: number | null }[] | null };
export type Car = { d: string; pos: number | null; gap: number | null; lead: number | null; cmp: string | null; age: number | null; pit: boolean; out: boolean };
export type Battle = { t: string; l: string; gap: number; closing: number | null; tyre: number | null; p: number; band: string };
export type LapState = { sc: boolean; cars: Car[]; battles: Battle[]; live: { d: string; podium: number; win: number }[] };
export type LapChunk = { chunk: number | "final"; from: number; to: number; laps: Record<string, LapState> };
export type RaceResult = { slug: string; finish: { code: string; pos: number | null; status: string; points: number }[] };
export type PodiumBundle = { model: Record<string, unknown>; drivers: Record<string, unknown>[]; teams: Record<string, unknown>[]; circuits: Record<string, unknown>[]; recentGrids: Record<string, unknown>[] };

const base = () => `${import.meta.env.BASE_URL}data/`;

async function get<T>(path: string): Promise<T> {
  const r = await fetch(base() + path);
  if (!r.ok) throw new Error(`Could not load ${path} (${r.status})`);
  return r.json() as Promise<T>;
}

export const loadIndex = () => get<SiteIndex>("index.json");
export const loadMeta = (slug: string) => get<RaceMeta>(`races/${slug}/meta.json`);
export const loadLaps = (slug: string, chunk: number | "final") => get<LapChunk>(`races/${slug}/laps-${chunk}.json`);
export const loadResult = (slug: string) => get<RaceResult>(`races/${slug}/result.json`);
export const loadPodium = () => get<PodiumBundle>("podium.json");

export function nextRace(calendar: CalendarEntry[], now: Date): CalendarEntry | null {
  const upcoming = calendar.filter((c) => c.sessions.race && new Date(c.sessions.race).getTime() + 3 * 3600e3 > now.getTime());
  return upcoming.sort((a, b) => a.round - b.round)[0] ?? null;
}

export const loadHowWeKnow = () => get<{ live: Record<string, any>[]; podium: Record<string, any>[]; bands: Record<string, any>[] }>("howweknow.json");
export const loadStrategy = () => get<Record<string, import("./views/Lab").Profile>>("strategy.json");
