import type { ReactNode } from "react";
import "./tower.css";

export type Column = { key: string; label: string; align?: "left" | "right"; hideSm?: boolean };
export type TowerRow = { pos: number | null; code: string; color: string; cells: Record<string, ReactNode>; dim?: boolean; hot?: boolean };

export function Tower({ columns, rows, label }: { columns: Column[]; rows: TowerRow[]; label?: string }) {
  return (
    <table className="tower" aria-label={label}>
      <thead>
        <tr>
          <th className="r">Pos</th>
          <th aria-hidden="true"></th>
          <th>Driver</th>
          {columns.map((c) => (
            <th key={c.key} className={`${c.align === "right" ? "r" : ""} ${c.hideSm ? "hide-sm" : ""}`}>{c.label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.code} className={`${r.dim ? "out" : ""} ${r.hot ? "hot" : ""}`}>
            <td className="p">{r.pos ?? "–"}</td>
            <td className="s"><i data-testid={`stripe-${r.code}`} style={{ background: r.color }} /></td>
            <td className="c">{r.code}</td>
            {columns.map((c) => (
              <td key={c.key} className={`${c.align === "right" ? "r" : ""} ${c.hideSm ? "hide-sm" : ""}`}>{r.cells[c.key]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
