import { useEffect, useState, type ReactNode } from "react";
import "./shell.css";

export const ROUTES = [
  { path: "weekend", label: "This weekend", short: "Weekend" },
  { path: "race", label: "Race", short: "Race" },
  { path: "replay", label: "Replay", short: "Replay" },
  { path: "lab", label: "Lab", short: "Lab" },
  { path: "about", label: "How we know", short: "About" },
] as const;

export type Route = (typeof ROUTES)[number]["path"];

export function currentRoute(): Route {
  const p = window.location.hash.replace(/^#\/?/, "").split("/")[0];
  return (ROUTES.find((r) => r.path === p)?.path ?? "weekend") as Route;
}

export function useHash(): string {
  const [h, setH] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setH(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return h;
}

export function useRoute(): Route {
  useHash();
  return currentRoute();
}

export function formatDay(iso: string) {
  return new Date(iso + "T12:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

export function Shell({ asOf, children }: { asOf?: string; children: ReactNode }) {
  const route = useRoute();
  const link = (r: (typeof ROUTES)[number], text: string) => (
    <a key={r.path} href={`#/${r.path}`} aria-current={route === r.path ? "page" : undefined} className={route === r.path ? "on" : ""}>{text}</a>
  );
  return (
    <>
      <header className="bar">
        <div className="brand"><i />Race Intelligence</div>
        <nav className="top" aria-label="Main">{ROUTES.map((r) => link(r, r.label))}</nav>
        {asOf && <div className="asof">Data as of <b>{formatDay(asOf)}</b></div>}
      </header>
      <main className="page">{children}</main>
      <nav className="bottom" aria-label="Main">{ROUTES.map((r) => link(r, r.short))}</nav>
    </>
  );
}
