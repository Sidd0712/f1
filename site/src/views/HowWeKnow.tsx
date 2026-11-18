import "./weekend.css";

type Row = Record<string, any>;
export type HowData = { live: Row[]; podium: Row[]; bands: Row[] };

const pct = (v: number) => `${Math.round(v * 100)}%`;
const pts = (v: number) => { const t = Math.round(Math.abs(v) * 1000 + 1e-9) / 10; return `${v < 0 ? "-" : "+"}${t.toFixed(1)} points`; };

function LiveCard({ r, what, base }: { r: Row; what: string; base: string }) {
  return (
    <div className="panel claim">
      <h2>{what}</h2>
      <p>We get it right {pct(r.Model)} of the time, against {pct(r.RunningOrder)} for {base}: <b>{pts(r.Gain)}</b> over {r.Races} races.</p>
      <details>
        <summary>Details</summary>
        <p>95% range for the gain, resampling whole races: {pts(r.CI_Lo)} to {pts(r.CI_Hi)}. Every number comes from walk-forward testing: each season is predicted by a model trained only on earlier seasons.</p>
      </details>
    </div>
  );
}

export function HowWeKnow({ data }: { data: HowData }) {
  const live = (t: string, w: string) => data.live.find((r) => r.Target === t && r.Window === w);
  const podium = live("Podium", "0-50%");
  const winner = live("Winner", "70-100%");
  const pool = (m: string) => {
    const rows = data.podium.filter((r) => r.Model === m && r.Year > 0);
    const n = rows.reduce((a, r) => a + r.Races, 0);
    return { n, brier: rows.reduce((a, r) => a + r.brier * r.Races, 0) / Math.max(n, 1), years: rows.map((r) => r.Year) };
  };
  const ours = pool("model"), slot = pool("slot");
  const bands = ["High", "Medium", "Low"].map((b) => data.bands.find((r) => r.Band === b)).filter(Boolean) as Row[];
  return (
    <section className="how">
      <h1 className="ev">How we know <span>Every number here is checked against a simple baseline</span></h1>
      <div className="claims">
        {ours.n > 0 && (
          <div className="panel claim">
            <h2>Podium chances before the race</h2>
            <p>Across {ours.n} races since {Math.min(...ours.years)}, our pre-race podium chances landed closer to what happened than the grid slot alone (lower error: {ours.brier.toFixed(4)} against {slot.brier.toFixed(4)}).</p>
            <details>
              <summary>Details</summary>
              <p>Error is the Brier score, the average squared gap between the chance we gave and what happened. Season by season: {data.podium.filter((r) => r.Model === "model" && r.Year > 0).sort((a, b) => a.Year - b.Year).map((r) => `${r.Year} ${r.brier.toFixed(4)}`).join(", ")}.</p>
            </details>
          </div>
        )}
        {podium && <LiveCard r={podium} what="Who finishes on the podium, first half of the race" base="the current running order" />}
        {winner && <LiveCard r={winner} what="Who wins, last 30% of the race" base="whoever is leading" />}
        {bands.length > 0 && (
          <div className="panel claim">
            <h2>Battles: will he get past in the next 5 laps?</h2>
            <p>{bands.map((b) => `When we said ${b.Band.toLowerCase()} (${pct(b.P_Lo)} to ${pct(b.P_Hi)}), a pass happened ${pct(b.ObservedRate)} of the time`).join(". ")}.</p>
            <details><summary>Details</summary><p>Only clean fights count: no pit stop or safety car in the window. Each band holds about {bands[0].Pairs.toLocaleString("en-GB")} battles.</p></details>
          </div>
        )}
        <div className="panel claim">
          <h2>Strategy Lab</h2>
          <p>Built on 2018–23 cars and not retrained since, so treat it as indicative. Its stint lengths use an even split, because the model we tried did no better.</p>
        </div>
      </div>
    </section>
  );
}
